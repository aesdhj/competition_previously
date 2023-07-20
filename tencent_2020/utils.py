import pandas as pd
import numpy as np
import gensim
import pickle
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
import os
import random
import torch
from torch import nn
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
import collections
from tqdm import tqdm


# https://github.com/bettenW/Tencent2020_Rank1st
# https://zhuanlan.zhihu.com/p/166710532


args = {}
args['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args['seed'] = 123456
args['model_class'] = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}
args['model_type'] = 'roberta'
args['output_dir'] = 'saved_models'
args['block_size'] = 128 # 序列最大长度
args['mlm_probability'] = 0.2 # 序列mask比例
args['train_batch_size'] = 64
args['eval_batch_size'] = 64
args['max_steps'] = 100000
args['gradient_accumulation_steps'] = 1
args['weight_decay'] = 0.01
args['learning_rate'] = 5e-5
args['warmup_steps'] = 10000
args['start_epoch'] = 0
args['start_step'] = 0
args['max_grad_norm'] = 1.0
args['save_steps'] = 10000


def merge_files():
	click_df = pd.read_csv('data/train/click_log.csv')
	click_df = click_df.append(pd.read_csv('data/test/click_log.csv'))
	click_df = click_df.sort_values(by=['time']).drop_duplicates()

	ad_df = pd.read_csv('data/train/ad.csv')
	ad_df = ad_df.append(pd.read_csv('data/test/ad.csv'))
	ad_df = ad_df.drop_duplicates()

	train_user = pd.read_csv('data/train/user.csv')
	# 筛选出有click_log的用户
	train_user = train_user[train_user['user_id'].isin(set(click_df['user_id']))]
	train_user['age'] = train_user['age'] - 1
	train_user['gender'] = train_user['gender'] - 1
	test_user = pd.read_csv('data/test/click_log.csv')[['user_id']].drop_duplicates()
	test_user = test_user.reset_index(drop=True)
	test_user['age'] = -1
	test_user['gender'] = -1

	# click_log合并creative_id特征和用户label,缺失值-1
	click_df = click_df.merge(ad_df, on='creative_id', how='left')
	click_df = click_df.merge(train_user, on='user_id', how='left')
	click_df = click_df.fillna(-1)
	click_df = click_df.replace('\\N', -1)
	for f in click_df:
		click_df[f] = click_df[f].astype(int)
	for i in range(10):
		click_df['age_{}'.format(i)] = (click_df['age'] == i).astype(np.int)
	for i in range(2):
		click_df['gender_{}'.format(i)] = (click_df['gender'] == i).astype(np.int)

	click_df = click_df.reset_index(drop=True)
	train_user = train_user.reset_index(drop=True)
	test_user = test_user.reset_index(drop=True)

	return click_df, train_user, test_user


def get_agg_features(dfs, f1, f2, agg, click_log):
	if type(f1) == str:
		f1 = [f1]
	if agg != 'size':
		data = click_log[f1 + [f2]]
	else:
		data = click_log[f1]
	f_name = '_'.join(f1 + [f2, agg])
	if agg == 'size':
		tmp = data.groupby(f1, as_index=False).size()
	elif agg == 'count':
		tmp = data.groupby(f1, as_index=False)[f2].count()
	elif agg == 'mean':
		tmp = data.groupby(f1, as_index=False)[f2].mean()
	elif agg == 'unique':
		tmp = data.groupby(f1, as_index=False)[f2].nunique()
	elif agg == 'max':
		tmp = data.groupby(f1, as_index=False)[f2].max()
	elif agg == 'min':
		tmp = data.groupby(f1, as_index=False)[f2].min()
	elif agg == 'sum':
		tmp = data.groupby(f1, as_index=False)[f2].sum()
	elif agg == 'std':
		tmp = data.groupby(f1, as_index=False)[f2].std()
	elif agg == 'median':
		tmp = data.groupby(f1, as_index=False)[f2].median()
	else:
		raise 'agg error'
	for df in dfs:
		try:
			df = df.drop(f_name, axis=1)
		except:
			pass
		tmp.columns = f1 + [f_name]
		df[f_name] = df.merge(tmp, on=f1, how='left')[f_name]

	return [f_name]


def sequence_text(dfs, f1, f2, click_log):
	f_name = '_'.join(['sequence_text', f1, f2])
	data = click_log[[f1, f2]]
	temp = data.groupby(f1, as_index=False)[f2].agg(list)
	temp.columns = [f1, f_name]
	temp[f_name] = temp[f_name].apply(lambda x: list(map(int, x)))
	temp[f_name] = temp[f_name].apply(lambda x: list(map(str, x)))
	for df in dfs:
		try:
			df = df.drop(f_name, axis=1)
		except:
			pass
		df[f_name] = df.merge(temp, on=f1, how='left')[f_name]
	return [f_name]


def kfold(dfs, log_data, pivot):
	kfold_features = ['age_{}'.format(i) for i in range(10)] + ['gender_{}'.format(i) for i in range(2)]
	log = log_data[['user_id', pivot, 'fold'] + kfold_features]
	tmps = []
	for fold in range(6):
		tmp = log[(log['fold'] != fold) & (log['fold'] != 5)]
		tmp = tmp.groupby(pivot, as_index=False)[kfold_features].mean()
		tmp.columns = [pivot] + kfold_features
		tmp['fold'] = fold
		tmps.append(tmp)
	tmp = pd.concat(tmps, axis=0).reset_index(drop=True)
	# 每个用户每个字段每折的年龄，性别分布
	tmp = log[['user_id', pivot, 'fold']].merge(tmp, on=[pivot, 'fold'], how='left')
	tmp_mean = tmp.groupby('user_id', as_index=False)[kfold_features].mean()
	tmp_mean.columns = ['user_id'] + ['_'.join([f, pivot, 'mean']) for f in kfold_features]
	for df in dfs:
		tmp = df.merge(tmp_mean, on='user_id', how='left')
		tmp = tmp.fillna(-1)
		for f in ['_'.join([f, pivot, 'mean']) for f in kfold_features]:
			df[f] = tmp[f]
	return ['_'.join([f, pivot, 'mean']) for f in kfold_features]


def kfold_sequence(train_df, test_df, log_data, pivot):
	kfold_features = ['age_{}'.format(i) for i in range(10)] + ['gender_{}'.format(i) for i in range(2)]
	log = log_data[['user_id', pivot, 'fold'] + kfold_features]
	tmps = []
	for fold in range(6):
		tmp = log[(log['fold'] != fold) & (log['fold'] != 5)]
		tmp = tmp.groupby(pivot, as_index=False)[kfold_features].mean()
		tmp.columns = [pivot] + kfold_features
		tmp['fold'] = fold
		tmps.append(tmp)
	tmp = pd.concat(tmps, axis=0).reset_index(drop=True)
	tmp = log[['user_id', pivot, 'fold']].merge(tmp, on=[pivot, 'fold'], how='left')
	tmp = tmp.fillna(-1)
	tmp[pivot + '_fold'] = tmp[pivot] * 10 + tmp['fold']
	tmp[pivot + '_fold'] = tmp[pivot + '_fold'].astype(int)
	kfold_sequence_features = sequence_text([train_df, test_df], 'user_id', pivot + '_fold', tmp)
	tmp = tmp.drop_duplicates([pivot + '_fold']).reset_index(drop=True)
	# 标准化年龄，性别分布，以w2v格式导出pivot + '_fold'对应的向量
	ss = StandardScaler()
	ss.fit(tmp[kfold_features])
	tmp[kfold_features] = ss.transform(tmp[kfold_features])
	for f in kfold_features:
		tmp[f] = tmp[f].apply(lambda x: round(x, 4))
	path = 'data/sequence_text_user_id_{}_fold.{}d'.format(pivot, 12)
	with open(path, 'w') as f:
		f.write(str(len(tmp)) + ' ' + '12' + '\n')
		for item in tmp[[pivot + '_fold'] + kfold_features].values:
			f.write(' '.join([str(int(item[0]))] + [str(x) for x in item[1:]]) + '\n')
	tmp = gensim.models.KeyedVectors.load_word2vec_format(path, binary=False)
	pickle.dump(tmp, open(path, 'wb'))
	return kfold_sequence_features


def w2v(dfs, f, L=32):
	"""
	sg-0,中心词算法，sg-1跳词算法
	hs-0层次softmax,hs-1负采样
	为了保证每次w2v的结果一致，除了固定seed，必须workers=1
	"""
	df = pd.concat(dfs, axis=0, ignore_index=True)
	sentences = list(df[f])
	w2v = Word2Vec(sentences, vector_size=L, window=8, min_count=1, sg=1, hs=0, workers=1, seed=0, epochs=10)
	path = os.path.join('data', f + '.{}d'.format(L))
	pickle.dump(w2v, open(path, 'wb'))


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed) # 为CPU设置随机种子
	torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
	torch.cuda.manual_seed_all(seed) # 为所有GPU设置随机种子
	# 将这个flag设置为TRUE的话，每次返回的卷积算法是默认算法
	# https://blog.csdn.net/byron123456sfsfsfa/article/details/96003317
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


class TextDataset(torch.utils.data.Dataset):
	def __init__(self, args, df, embedding_table):
		self.df_features = [df[feature[1]].values for feature in args['text_features']]
		self.embedding_table = embedding_table
		self.args = args

	def __len__(self):
		return len(self.df_features[0])

	def __getitem__(self, i):
		# 如果广告id和属性其中一个被mask，其他选择不mask, 因为广告id和属性之间可以互相预测
		# 和mask对齐，被遮掩时没有w2v
		text_features = np.zeros((self.args['block_size'], self.args['text_dim']))
		# text_ids 对应args.vocab_dic
		text_ids = np.zeros((self.args['block_size'], len(self.args['text_features'])), dtype=np.int64)
		text_mask = np.zeros(self.args['block_size'])
		# text_label对应args.vocab_list
		text_label = np.zeros((self.args['block_size'], len(self.args['text_features'])), dtype=np.int64) - 100
		# 选择20%的token进行掩码，其中80%设为[mask], 10%设为不变,10%随机选择，20%不在args.vocab_dic用unk代替
		begin_dim = 0
		for idx, feature in enumerate(self.args['text_features']):
			end_dim = begin_dim + feature[2]
			for word_idx, word in enumerate(self.df_features[idx][i][:self.args['block_size']]):
				text_mask[word_idx] = 1
				# 进行遮掩
				if random.random() < self.args['mlm_probability']:
					if word in self.args['vocab_list'][idx]:
						text_label[word_idx, idx] = self.args['vocab_list'][idx][word]
					else:
						text_label[word_idx, idx] = 0
					# mask,此时text_features=0
					if random.random() < 0.8:
						text_ids[word_idx, idx] = self.args['vocab_dic']['mask']
					# 保持不变
					elif random.random() < 0.5:
						text_features[word_idx, begin_dim:end_dim] = self.embedding_table[idx].wv[word]
						try:
							text_ids[word_idx, idx] = self.args['vocab_dic'][(feature[1], word)]
						except:
							text_ids[word_idx, idx] = self.args['vocab_dic']['unk']
					# 替换成任意字符
					else:
						while True:
							random_word = random.sample(list(self.args['vocab_list'][idx]), 1)[0]
							if random_word != word and random_word != 'unk':
								break
						text_features[word_idx, begin_dim:end_dim] = self.embedding_table[idx].wv[random_word]
						try:
							text_ids[word_idx, idx] = self.args['vocab_dic'][(feature[1], random_word)]
						except:
							text_ids[word_idx, idx] = self.args['vocab_dic']['unk']
				# 不进行遮掩
				else:
					text_features[word_idx, begin_dim:end_dim] = self.embedding_table[idx].wv[word]
					try:
						text_ids[word_idx, idx] = self.args['vocab_dic'][(feature[1], word)]
					except:
						text_ids[word_idx, idx] = self.args['vocab_dic']['unk']
			begin_dim = end_dim
		return (
			torch.tensor(text_features), torch.tensor(text_ids),
			torch.tensor(text_mask), torch.tensor(text_label)
		)


class Model(nn.Module):
	def __init__(self, encoder, args, config):
		super(Model, self).__init__()
		self.encoder = encoder
		self.args = args
		self.text_embeddings = nn.Embedding(args['vocab_size_v1'], args['vocab_dim_v1'])
		self.text_embeddings.apply(self._init_weights)
		self.text_linear = nn.Linear(
			args['text_dim'] + args['vocab_dim_v1'] * len(args['text_features']),
			config.hidden_size
		)
		self.text_linear.apply(self._init_weights)
		self.lm_heads = nn.ModuleList([
			nn.Linear(config.hidden_size, vocab_size, bias=False) for vocab_size in args['vocab_size_list']
		])

	# nn.embedding初始化可以快速收敛，https://zhuanlan.zhihu.com/p/466943663
	def _init_weights(self, module):
		if isinstance(module, (nn.Linear, nn.Embedding)):
			module.weight.data.normal_(mean=0.0, std=0.02)

	def forward(self, inputs, inputs_ids, masks, labels):
		"""
		:param inputs: batch_size, block_size, text_dim
		:param inputs_ids: batch_size, block_size, text_features_len
		:param masks: batch_size, block_size
		:param labels: batch_size, block_size, text_features_len
		:return:
		"""
		# 由于是部分mask，所以这里embedding学到的是广告id及其属性的综合特征
		inputs_embedding = self.text_embeddings(inputs_ids)
		# batch_size, block_size, vocab_dim_v1 * text_features_len
		inputs_embedding = inputs_embedding.view(inputs.shape[0], inputs.shape[1], -1)
		# batch_size, block_size, vocab_dim_v1 * text_features_len + text_dim
		inputs = torch.cat([inputs.float(), inputs_embedding], dim=2)
		# batch_size, block_size, hidden_size
		inputs = torch.relu(self.text_linear(inputs))
		# batch_size, block_size, hidden_size
		# roberta输出一个结果集合, https://huggingface.co/docs/transformers/v4.31.0/en/model_doc/roberta
		outputs = self.encoder(inputs_embeds=inputs, attention_mask=masks.float())[0]
		loss = 0
		# 对不同的字段分开计算loss
		for idx, (lm_head, feature) in enumerate(zip(self.lm_heads, self.args['text_features'])):
			if feature[3]:
				# batch_size, hidden_size
				outputs_tmp = outputs[labels[:, :, idx].ne(-100)]
				pred_scores = lm_head(outputs_tmp)
				labels_tmp = labels[:, :, idx]
				labels_tmp = labels_tmp[labels_tmp.ne(-100)].long()
				loss_func = nn.CrossEntropyLoss()
				loss += loss_func(pred_scores, labels_tmp)
		return loss


def evaluate(args, model, eval_dataset):
	eval_dataloader = torch.utils.data.DataLoader(
		eval_dataset,
		batch_size=args['eval_batch_size'],
		num_workers=0
	)
	eval_loss = 0.0
	nb_eval_steps = 0
	model.eval()
	for batch in eval_dataloader:
		inputs, inputs_ids, masks, labels = [item.to(args['device']) for item in batch]
		with torch.no_grad():
			lm_loss = model(inputs, inputs_ids, masks, labels)
			eval_loss += lm_loss.item()
		nb_eval_steps += 1
	eval_loss = eval_loss / nb_eval_steps
	perplexity = torch.exp(torch.tensor(eval_loss))
	result = {
		'perplexity': float(perplexity),
		'eval_loss': eval_loss
	}
	return result


def train(args, train_dataset, dev_dataset, model):
	# 设置dataloader
	train_dataloader = torch.utils.data.DataLoader(
		train_dataset,
		shuffle=True,
		batch_size=args['train_batch_size'],
		num_workers=0
	)
	args['num_train_epochs'] = int(args['max_steps'] / len(train_dataloader))
	print('steps:', args['max_steps'], 'epochs:', args['num_train_epochs']-1)

	# 设置优化器
	model.to(args['device'])
	no_decay = ['bias', 'LayerNorm.weight']
	# bert官方的代码中对于bias项、LayerNorm.bias、LayerNorm.weight项是免于正则化的，https://zhuanlan.zhihu.com/p/524036087
	optimizer_grouped_parameters = [
		{
			'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
			'weight_decay': args['weight_decay']
		},
		{
			'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
			'weight_decay': 0.0
		},
	]
	optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'])
	# lr的warmup,前期以较小的lr使模型的初始化参数变化不至于太大，后期逐渐增大lr正常学习
	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=args['warmup_steps'],
		num_training_steps=args['max_steps']
	)

	# 训练模型
	global_step = 0
	step_accumulation = 0
	model.zero_grad()
	set_seed(args['seed'])

	for idx in range(args['start_epoch'], args['num_train_epochs']):
		tr_loss = 0.0
		for step, batch in tqdm(enumerate(train_dataloader), desc='epoch {}'.format(idx+1)):
			inputs, inputs_ids, masks, labels = [item.to(args['device']) for item in batch]
			step_accumulation += 1
			model.train()
			loss = model(inputs, inputs_ids, masks, labels)
			if args['gradient_accumulation_steps'] > 1:
				loss = loss / args['gradient_accumulation_steps']
			loss.backward()
			# 梯度裁剪
			nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])

			tr_loss += loss.item()

			if (step_accumulation + 1) % args['gradient_accumulation_steps'] == 0:
				optimizer.step()
				optimizer.zero_grad()
				scheduler.step()
				global_step += 1
				step_accumulation = 0

			# 验证
			if (args['save_steps'] > 0 and global_step % args['save_steps'] == 0) or (len(train_dataloader) == step+1):
				checkpoint_prefix = 'checkpoint'
				results = evaluate(args, model, dev_dataset)
				print('epoch: {}, step: {}, train_loss: {:.5f}, eval_loss: {:.5f}'.format(idx+1, global_step, tr_loss/(step+1), results['eval_loss']))
				if results['eval_loss'] <= 15:
					output_dir = os.path.join(args['output_dir'], '{}-{}-{}'.format(checkpoint_prefix, global_step, round(results['perplexity'], 4)))
					if not os.path.exists(output_dir):
						os.makedirs(output_dir)
					# 保存模型
					encoder_to_save = model.encoder
					encoder_to_save.save_pretrained(output_dir)
					linear_to_save = model.text_linear
					torch.save(linear_to_save.state_dict(), os.path.join(output_dir, 'linear.bin'))
					embeddings_to_save = model.text_embeddings
					torch.save(embeddings_to_save.state_dict(), os.path.join(output_dir, 'embeddings.bin'))
					torch.save(model.state_dict(), os.path.join(output_dir, 'model.bin'))

					last_output_dir = os.path.join(args['output_dir'], 'checkpoint-last')
					if not os.path.exists(last_output_dir):
						os.makedirs(last_output_dir)
					encoder_to_save.save_pretrained(last_output_dir)
					torch.save(linear_to_save.state_dict(), os.path.join(last_output_dir, 'linear.bin'))
					torch.save(embeddings_to_save.state_dict(), os.path.join(last_output_dir, 'embeddings.bin'))
					torch.save(model.state_dict(), os.path.join(last_output_dir, 'model.bin'))
					torch.save(optimizer.state_dict(), os.path.join(last_output_dir, 'optimizer.pt'))
					torch.save(scheduler.state_dict(), os.path.join(last_output_dir, 'scheduler.pt'))

					idx_file = os.path.join(last_output_dir, 'idx_file.txt')
					with open(idx_file, 'w') as idx_f:
						idx_f.write(str(idx) + '\n')
					step_file = os.path.join(last_output_dir, 'step_file.txt')
					with open(step_file, 'w') as step_f:
						step_f.write(str(global_step) + '\n')


def main(args):

	# 读取数据
	train_df = pd.read_pickle('data/train_user_features.pkl')
	test_df = pd.read_pickle('data/test_user_features.pkl')
	# MODEL模型对应的训练集，验证集拆分
	dev_data = train_df.sample(n=10000, random_state=args['seed'])
	train_data = train_df[~train_df['user_id'].isin(list(dev_data['user_id']))]
	train_data = train_data.append(test_df)
	print('train_df:', train_df.shape, 'test_df:', test_df.shape)
	print('train_data:', train_data.shape, 'dev_data:', dev_data.shape)

	text_features = [
		[
			'data/sequence_text_user_id_{}.128d'.format(key),
			'sequence_text_user_id_{}'.format(key),
			128,
			True
		] for key in ['ad_id', 'creative_id', 'advertiser_id', 'product_id', 'industry', 'product_category', 'time', 'click_times']
	]
	# 读取每个字段w2v
	embedding_table = []
	for feature in text_features:
		print(feature[0])
		embedding_table.append(pickle.load(open(feature[0], 'rb')))
	# 输入端词表，所有字段统一编码，每个字段最多保留10w个id
	try:
		dic = pickle.load(open(os.path.join(args['output_dir'], 'vocab.pkl'), 'rb'))
	except:
		dic = {}
		dic['pad'] = 0
		dic['mask'] = 1
		dic['unk'] = 2
		for feature in text_features:
			lines = []
			# train_df??
			for line in list(train_df[feature[1]]):
				lines.extend(line)
			lines_add = [feature[1]] * len(lines)
			lines = zip(lines_add, lines)
			result = collections.Counter(lines)
			most_common = result.most_common(100000)
			cont = 0
			for item in most_common:
				if item[1] >= 5:
					dic[item[0]] = len(dic)
					cont += 1
					if cont < 10:
						print(item[0], dic[item[0]])
			print(feature[1], cont)
		pickle.dump(dic, open(os.path.join(args['output_dir'], 'vocab.pkl'), 'wb'))
	args['vocab_dic'] = dic
	args['vocab_size_v1'] = len(dic)
	args['vocab_dim_v1'] = 64
	# 输出端词表，每个字段分别编码，每个域最多保留10w个id
	try:
		vocab_list = pickle.load(open(os.path.join(args['output_dir'], 'vocab_list.pkl'), 'rb'))
	except:
		vocab_list = []
		for feature in text_features:
			lines = []
			# train_data??
			for line in list(train_data[feature[1]]):
				lines.extend(line)
			result = collections.Counter(lines)
			most_common = result.most_common(100000)
			dic = {}
			dic['unk'] = 0
			cont = 0
			for idx, item in enumerate(most_common):
				dic[item[0]] = idx + 1
				cont += 1
			print(feature[1], cont, len(set(lines)))
			vocab_list.append(dic)
			pickle.dump(vocab_list, open(os.path.join(args['output_dir'], 'vocab_list.pkl'), 'wb'))

	# 设置参数
	args['text_features'] = text_features
	args['vocab_list'] = vocab_list
	args['text_dim'] = sum([feature[2] for feature in text_features])
	train_dataset = TextDataset(args, train_data, embedding_table)
	dev_dataset = TextDataset(args, dev_data, embedding_table)
	args['vocab_size_list'] = [len(vocab) for vocab in vocab_list]

	# 设置随机种子等，使神经网络模型结果可重复
	set_seed(args['seed'])
	# BERT属于NLP的预训练模型，encoder出来的特征表达能对应各种下游任务
	# BERT轮子代码可以参考李沐的神经网络课程,https://zh-v2.d2l.ai/chapter_natural-language-processing-pretraining/bert.html
	# RoBERTa相比BERT的改进:
	# -1,BERT对每个序列随机筛选出15%token作为label进行训练，RoBERTa在每个epoch会进行一次随机筛选出15%;
	# -2,RoBERTa淘汰了NSP下一句的预测，替代方案是使用连续多个序列直到MAX_LEN;
	# -3,RoBERTa使用了更大的batch_size和lr，实验证明可以加速模型训练和性能;
	# hugging-face transformers库入门，https://zhuanlan.zhihu.com/p/548336726
	config_class, model_class, tokenizer_class = args['model_class'][args['model_type']]
	# bert_encoder
	config = RobertaConfig()
	config.num_hidden_layers = 12
	config.hidden_size = 512
	config.intermediate_size = config.hidden_size * 4
	config.num_attention_heads = 16
	encoder = model_class(config)

	# 创建模型，对原有字段统一编码后BERT输出特征表达
	model = Model(encoder, args, config)

	# 训练模型
	train(args, train_dataset, dev_dataset, model)


class TextDataset_Last(torch.utils.data.Dataset):
	def __init__(self, args, df):
		self.df = df
		self.label = df['label'].values
		self.args = args
		self.df_features = df[[feature[1] for feature in args['text_features']]].values
		self.df_features_1 = df[[feature[1] for feature in args['text_features_1']]].values
		self.dense_features = df[args['dense_features']].values

	def __len__(self):
		return len(self.label)

	def __getitem__(self, i):
		# 相比TextDataset， 这边用的全部mask
		label = self.label[i]
		# 字段序列bert输入
		text_features = np.zeros((self.args['block_size'], self.args['text_dim']))
		text_masks = np.zeros(self.args['block_size'])
		text_ids = np.zeros((self.args['block_size'], len(self.args['text_features'])), dtype=np.int64)
		begin_dim = 0
		for idx, (embeddings_table, df_feature) in enumerate(zip(self.args['embeddings_tables'], self.df_features[i])):
			end_dim = begin_dim + self.args['text_features'][idx][-1]
			for word_idx, word in enumerate(df_feature[:self.args['block_size']]):
				text_masks[word_idx] = 1
				text_features[word_idx, begin_dim:end_dim] = embeddings_table.wv[word]
				try:
					text_ids[word_idx, idx] = self.args['vocab_dic'][(self.args['text_features'][idx][1], word)]
				except:
					text_ids[word_idx, idx] = self.args['vocab_dic']['unk']
			begin_dim = end_dim
		# kfold target encoding编码后序列的bert输入
		text_features_1 = np.zeros((self.args['block_size'], self.args['text_dim_1']))
		text_masks_1 = np.zeros(self.args['block_size'])
		begin_dim = 0
		for idx, (embeddings_table, df_feature) in enumerate(zip(self.args['embeddings_tables_1'], self.df_features_1[i])):
			end_dim = begin_dim + self.args['text_features_1'][idx][-1]
			for word_idx, word in enumerate(df_feature[:self.args['block_size']]):
				text_masks_1[word_idx] = 1
				text_features_1[word_idx, begin_dim:end_dim] = embeddings_table[word]
			begin_dim = end_dim
		# 连续特征
		dense_features = self.dense_features[i]

		return (
			torch.tensor(label), torch.tensor(dense_features),
			torch.tensor(text_features), torch.tensor(text_ids), torch.tensor(text_masks),
			torch.tensor(text_features_1), torch.tensor(text_masks_1)
		)


class ClassificationHead(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.norm = nn.BatchNorm1d(args['out_size'])
		self.dense = nn.Linear(args['out_size'], args['linear_layer_size'][0])
		self.norm_1 = nn.BatchNorm1d(args['linear_layer_size'][0])
		self.dropout = nn.Dropout(args['hidden_dropout_prob'])
		self.dense_1 = nn.Linear(args['linear_layer_size'][0], args['linear_layer_size'][1])
		self.norm_2 = nn.BatchNorm1d(args['linear_layer_size'][1])
		self.out = nn.Linear(args['linear_layer_size'][1], args['num_label'])

	def forward(self, features):
		x = self.norm(features)
		x = self.dropout(x)

		x = self.dense(x)
		x = torch.relu(self.norm_1(x))
		x = self.dropout(x)

		x = self.dense_1(x)
		x = torch.relu(self.norm_2(x))
		x = self.dropout(x)

		x = self.out(x)
		return x


class Model_Last(nn.Module):
	def __init__(self, args):
		super(Model_Last, self).__init__()
		# vocab_dict对应的embedding
		self.text_embeddings = nn.Embedding.from_pretrained(
			torch.load(os.path.join(args['pretrained_model_path'], 'checkpoint-last', 'embeddings.bin'))['weight'],
			freeze=True
		)
		self.text_linear = nn.Linear(
			args['text_dim']+args['vocab_dim_v1']*len(args['text_features']),
			args['hidden_size']
		)
		self.text_linear.load_state_dict(
			torch.load(os.path.join(args['pretrained_model_path'], 'checkpoint-last', 'linear.bin'))
		)
		self.dropout = nn.Dropout(args['hidden_dropout_prob'])
		config = RobertaConfig.from_pretrained(
			os.path.join(args['pretrained_model_path'], 'checkpoint-last')
		)
		# 已训练的bert模型
		self.text_layer = RobertaModel.from_pretrained(
			os.path.join(args['pretrained_model_path'], 'checkpoint-last'),
			config=config
		)

		# 创建新的BERT模型，训练text_features_1
		self.norm = nn.BatchNorm1d(args['text_dim_1'] + args['hidden_size'])
		self.text_linear_1 = nn.Linear(args['text_dim_1'] + args['hidden_size'], 1024 * args['n'])
		self.text_linear_1.apply(self._init_weights)

		config = RobertaConfig()
		config.num_hidden_layers = 4 * args['n']
		config.hidden_size = 512 * args['n']
		config.intermediate_size = config.hidden_size * 4
		config.num_attention_heads = 16
		self.text_layer_1 = RobertaModel(config=config)
		self.text_layer_1.apply(self._init_weights)

		# 分类器
		self.classifier = ClassificationHead(args)
		self.classifier.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, (nn.Linear, nn.Embedding)):
			module.weight.data.normal_(mean=0.0, std=0.02)

	def forward(self, dense_features, text_features, text_ids, text_masks, text_features_1, text_masks_1, labels=None):
		"""
		:param dense_features: batch_size, dense_features_len
		:param text_features: batch_size, block_size, text_dim
		:param text_ids: batch_size, block_size, text_features_len
		:param text_masks: batch_size, block_size
		:param text_features_1: batch_size, block_size, text_dim_1
		:param text_masks_1: batch_size, block_size
		:param labels: batch_size
		:return:
		"""
		outputs = []
		outputs.append(dense_features.float())

		# 已训练的bert模型
		text_masks = text_masks.float()
		# batch_size, block_size, text_dim_v1 * text_features_len
		text_embedding = self.text_embeddings(text_ids).view(text_ids.shape[0], text_ids.shape[1], -1)
		# batch_size, block_size, text_dim_v1 * text_features_len + text_dim
		text_features = torch.cat([text_features.float(), text_embedding], dim=2)
		# batch_size, block_size, hidden_size
		text_features = torch.relu(self.text_linear(self.dropout(text_features)))
		# batch_size, block_size, hidden_size
		hidden_states = self.text_layer(
			inputs_embeds=text_features, attention_mask=text_masks
		)[0]
		# bert encoder 输出 (batch_size, seq_len, hidden_size)在seq_len取max,mean可以对比一下cls的效果
		# (batch_size, block_size, hidden_size) * (batch_size, block_size, 1) -> (batch_size, hidden_size)
		# (batch_size, hidden_size) / (batch_size, 1) -> (batch_size, hidden_size)
		embed_mean = (hidden_states * text_masks.unsqueeze(-1)).sum(1) / text_masks.sum(1).unsqueeze(-1)
		embed_mean = embed_mean.float()
		embed_max = hidden_states + (1 - text_masks).unsqueeze(-1) * (-1e10)
		embed_max = embed_max.max(1)[0].float()
		outputs.append(embed_mean)
		outputs.append(embed_max)

		# text_features_1获取text_feature的hidden_states代替ids的embedding,
		# 进行bert_encoder,方式和text_feature一样
		text_masks_1 = text_masks_1.float()
		# batch_size, block_size, hidden_size + text_dim_1
		text_features_1 = torch.cat([text_features_1.float(), hidden_states], dim=-1)
		bs, le, dim = text_features_1.shape
		text_features_1 = self.norm(text_features_1.view(-1, dim)).view(bs, le, dim)
		text_features_1 = torch.relu(self.text_linear_1(text_features_1))
		hidden_states = self.text_layer_1(
			inputs_embeds=text_features_1, attention_mask=text_masks_1
		)[0]
		embed_mean = (hidden_states * text_masks_1.unsqueeze(-1)).sum(1) / text_masks_1.sum(1).unsqueeze(-1)
		embed_mean = embed_mean.float()
		embed_max = hidden_states + (1 - text_masks_1).unsqueeze(-1) * (-1e10)
		embed_max = embed_max.max(1)[0].float()
		outputs.append(embed_mean)
		outputs.append(embed_max)

		final_hidden_state = torch.cat(outputs, dim=-1)
		preds_score = self.classifier(final_hidden_state)

		if labels is not None:
			loss_func = nn.CrossEntropyLoss()
			loss = loss_func(preds_score, labels)
			return loss
		else:
			prob = torch.softmax(preds_score, dim=-1)
			age_probs = prob.view(-1, 10, 2).sum(dim=2)
			gender_probs = prob.view(-1, 10, 2).sum(dim=1)
			return age_probs, gender_probs


class Ctr_Net(nn.Module):
	def __init__(self, args):
		super(Ctr_Net, self).__init__()
		self.args = args
		self.model = Model_Last(args)
		self.model.to(args['device'])

	def train(self, train_dataset, dev_dataset=None):
		# 设置dataloader
		train_dataloader = torch.utils.data.DataLoader(
			train_dataset, shuffle=True, batch_size=self.args['train_batch_size']
		)
		self.args['warmup_steps'] = len(train_dataloader)
		self.args['max_steps'] = len(train_dataloader) * self.args['epoch']
		# 设置优化器
		optimizer = AdamW(self.model.parameters(), lr=args['lr'], weight_decay=0.08)
		scheduler = get_linear_schedule_with_warmup(
			optimizer,
			num_warmup_steps=self.args['warmup_steps'],
			num_training_steps=self.args['max_steps']
		)
		# 训练
		global_step = 0
		self.model.zero_grad()
		set_seed(args['seed'])
		best_age_score, best_gender_score = 0, 0

		for idx in range(args['epoch']):
			tr_loss = 0.0
			for step, batch in tqdm(enumerate(train_dataloader)):
				labels, dense_features, text_features, text_ids, text_masks, text_features_1, text_masks_1 = [item.to(args['device']) for item in batch]
				self.model.train()
				loss = self.model(dense_features, text_features, text_ids, text_masks, text_features_1, text_masks_1, labels)
				loss.backward()
				nn.utils.clip_grad_norm_(self.model.parameters(), args['max_grad_norm'])
				tr_loss += loss.item()
				optimizer.step()
				optimizer.zero_grad()
				scheduler.step()
				global_step += 1

				# 打印输出
				if (global_step % args['save_steps'] == 0) and dev_dataset is not None:
					dev_loss, age_score, gender_score, score, _, _ = self.evaluate(dev_dataset)
					print('epoch: {}, step: {}, train_loss: {:.5f}, eval_loss: {:.5f}, age_score: {:.4f}, gender_score: {:.4f}, score: {:.4f}'.format(
						idx + 1, global_step, tr_loss / (step + 1), dev_loss, age_score, gender_score, score))

			# 每个epoch以后验证
			if dev_dataset is not None:
				dev_loss, age_score, gender_score, score, _, _ = self.evaluate(dev_dataset)
				if best_age_score < age_score:
					best_age_score = age_score
					torch.save(self.model.state_dict(), os.path.join(args['output_dir'], 'pytorch_model_age.bin'))
					print('best age model saved, best age score: {:.4f}'.format(best_age_score))
				if best_gender_score < gender_score:
					best_gender_score = gender_score
					torch.save(self.model.state_dict(), os.path.join(args['output_dir'], 'pytorch_model_gender.bin'))
					print('best gender model saved, best gender score: {:.4f}'.format(best_gender_score))

	def evaluate(self, dev_dataset):
		dev_dataloader = torch.utils.data.DataLoader(
			dev_dataset, shuffle=False, batch_size=self.args['train_batch_size']
		)
		dev_loss = 0.0
		nb_dev_steps = 0
		age_probs = []
		gender_probs = []
		self.model.eval()
		for batch in dev_dataloader:
			labels, dense_features, text_features, text_ids, text_masks, text_features_1, text_masks_1 = [item.to(args['device']) for item in batch]
			with torch.no_grad():
				loss = self.model(dense_features, text_features, text_ids, text_masks, text_features_1, text_masks_1, labels)
				age_probs_part, gender_probs_part = self.model(dense_features, text_features, text_ids, text_masks, text_features_1, text_masks_1)
			dev_loss += loss.item()
			nb_dev_steps += 1
			age_probs.append(age_probs_part.cpu().numpy())
			gender_probs.append(gender_probs_part.cpu().numpy())

		age_probs = np.concatenate(age_probs, axis=0)
		gender_probs = np.concatenate(gender_probs, axis=0)
		age_labels = dev_dataset.df['age'].values
		gender_labels = dev_dataset.df['gender'].values
		age_score = np.mean(np.argmax(age_probs, axis=1) == age_labels)
		gender_score = np.mean(np.argmax(gender_probs, axis=1) == gender_labels)
		score = age_score + gender_score
		dev_loss = dev_loss / nb_dev_steps
		return (dev_loss, age_score, gender_score, score, age_probs, gender_probs)

	def reload(self, label):
		self.model.load_state_dict(
			torch.load(os.path.join(args['output_dir'], 'pytorch_model_{}.bin'.format(label)))
		)


















