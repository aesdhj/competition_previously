import pandas as pd
import torch
import numpy as np
from prefetch_generator import BackgroundGenerator
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score
import scipy as sp
from functools import partial
import os
from collections import defaultdict
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report


# https://github.com/shyoulala/CCF_BDCI_2020_DIDI_rank1_solution/tree/master/code
# https://www.datafountain.cn/competitions/466/datasets


pd.options.display.max_columns = None


def id_select():
	topo = pd.read_csv('data/topo.txt', sep='\t', header=None)
	topo.columns = ['id', 'link_id']
	topo['link_id'] = topo['link_id'].apply(lambda x: x.split(','))
	topo = topo.explode('link_id')
	topo['id'] = topo['id'].astype('str')
	topo['link_id'] = topo['link_id'].astype('str')
	# topo_sample = topo.sample(n=1000, random_state=2022)
	# topo_sample_ = topo.sample(n=1000, random_state=2021)
	# topo_sample_ = topo[topo['id'].isin(topo_sample_['id'])]
	# topo_sample_after = topo[topo['id'].isin(topo_sample['id'])]
	# topo_sample_before = topo[topo['link_id'].isin(topo_sample_after['id'])]
	# topo_sample_cat = pd.concat([topo_sample_after, topo_sample_before, topo_sample_], axis=0)
	# id_sample = set(list(topo_sample_cat['id']) + list(topo_sample_cat['link_id']))
	id_sample = set(list(topo['id']) + list(topo['link_id']))
	return id_sample


def add_link_set(links, link_set, add=True):
	n = 0
	for link in links:
		if link not in link_set and add:
			link_set.add(link)
			n += 1
	return link_set, n


def get_next_link(link_id, link_next_dict):
	next_link = link_next_dict.get(link_id, [])
	return (
		[[link_id, next_l] for next_l in next_link],
		len(next_link),
		next_link
	)


def get_before_link(link_id, link_before_dict):
	before_link = link_before_dict.get(link_id, [])
	return (
		[[link_id, before_l] for before_l in before_link],
		len(before_link),
		before_link
	)


def add_link_info(link_add_info, sub_info, num, max_number, link_set):
	for info in sub_info:
		if info[1] in link_set and info[0] in link_set:
			link_add_info.append(info)
		elif info[1] not in link_set and info[0] not in link_set:
			if num > max_number - 2:
				continue
			link_set.add(info[1])
			link_set.add(info[0])
			link_add_info.append(info)
			num += 2
		elif info[1] not in link_set:
			if num > max_number - 1:
				continue
			link_set.add(info[1])
			link_add_info.append(info)
			num += 1
		elif info[0] not in link_set:
			if num > max_number - 1:
				continue
			link_set.add(info[0])
			link_add_info.append(info)
			num += 1
	return link_add_info, num, link_set


def get_model_data(df, feat_columns, feat_value_columns):
	"""
	# category_features 统一序列编码
	# category_features_values 连续特征为原连续特征， 分类特征为1.0
	# recent_split_info (5, 4) recent_features
	# history_split_info (5 * 4, 4) history_features
	"""
	df['category_features'] = [i for i in df[feat_columns].values]
	df['category_features_values'] = [i for i in df[feat_value_columns].values]
	df_data = df[[
		'category_features', 'category_features_values', 'recent_split_info',
		'history_split_info', 'link_index', 'label']].values
	return df_data


class zy_DataSet(torch.utils.data.Dataset):
	def __init__(self, data, graph_dict, max_number):
		self.data = data
		self.graph_dict = graph_dict
		self.max_number = max_number

	def get_graph_feat(self, link_id):
		if link_id not in self.graph_dict:
			return [link_id] + [0] * (self.max_number - 1), np.eye(self.max_number)
		link_graph, link_seq = self.graph_dict[link_id]
		return link_seq, link_graph

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		feature = self.data[index, :]
		category_index = torch.tensor(feature[0], dtype=torch.long)
		category_value = torch.tensor(feature[1], dtype=torch.float32)
		recent_split_info = torch.tensor(feature[2], dtype=torch.float32)
		his_split_info = torch.tensor(feature[3], dtype=torch.float32)

		link_index = feature[4]
		link_seq, link_graph = self.get_graph_feat(link_index)
		link_seq = torch.tensor(link_seq, dtype=torch.long)
		link_graph = torch.tensor(link_graph, dtype=torch.float32)

		label = torch.tensor(feature[5], dtype=torch.long)
		return (
			category_index, category_value, recent_split_info, his_split_info,
			link_seq, link_graph, label
		)

	# batch 是一个list格式，每个元素用__getitem__
	def collate_fn(self, batch):
		category_index = torch.stack([x[0] for x in batch])
		category_value = torch.stack([x[1] for x in batch])
		recent_split_info = torch.stack([x[2] for x in batch])
		his_split_info = torch.stack([x[3] for x in batch])
		link_seq = torch.stack([x[4] for x in batch])
		link_graph = torch.stack([x[5] for x in batch])
		label = torch.stack([x[6] for x in batch])
		return (
			category_index, category_value, recent_split_info, his_split_info,
			link_seq, link_graph, label
		)


class DataLoaderX(torch.utils.data.DataLoader):
	"""
	数据预加载, https://blog.csdn.net/weixin_39870155/article/details/110493548
	"""
	def __iter__(self):
		return BackgroundGenerator(super().__iter__())


def get_loader(df, graph_dict, max_number, batch_size=256, train_mode=False):
	ds_df = zy_DataSet(df, graph_dict, max_number)
	loader = DataLoaderX(
		ds_df,
		batch_size=batch_size,
		shuffle=train_mode,
		num_workers=2,
		collate_fn=ds_df.collate_fn,
		drop_last=train_mode
	)
	return loader


class GraphConvolution(nn.Module):
	def __init__(self, in_features, out_features, bias=True):
		super(GraphConvolution, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
		if bias:
			self.bias = nn.Parameter(torch.Tensor(out_features))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		nn.init.kaiming_uniform_(self.weight)
		if self.bias is not None:
			nn.init.zeros_(self.bias)

	def forward(self, input, adj):
		"""
		:param input: (batch_size, node_num, feat_dim)
		:param adj: (batch_size, node_num, node_num)
		:return:
		"""
		support = torch.matmul(input, self.weight)
		output = torch.matmul(adj, support)
		if self.bias is not None:
			return F.relu(output + self.bias)
		else:
			return F.relu(output)

	def extra_repr(self):
		"""
		设置模块的额外表示信息
		"""
		return 'in_features={}, out_features={}, bias={}'.format(
			self.in_features, self.out_features, self.bias is not None
		)


class GCN(nn.Module):
	def __init__(self, feat_dim, k=2):
		super(GCN, self).__init__()
		self.k = k
		self.gcs = nn.ModuleList([
			GraphConvolution(feat_dim, feat_dim) for _ in range(k)
		])

	def forward(self, h, adj):
		"""
		:param h: (batch_size, node_num, feat_dim)
		:param adj: (batch_size, node_num, node_num)
		:return:
		"""
		for kp in range(self.k):
			h = self.gcs[kp](h, adj)
		return h


class DiDi_Model(nn.Module):
	def __init__(self, embedding_num, embedding_dim, link_embedding_matrix, device):
		super(DiDi_Model, self).__init__()
		self.embedding_num = embedding_num
		self.embedding_dim = embedding_dim
		self.device = device
		# fm二阶
		self.embedding_seq = nn.Embedding(embedding_num, embedding_dim)
		# lstm
		input_dim = 4
		output_dim = 8
		self.lstm_recent = nn.LSTM(input_dim, output_dim, num_layers=2, batch_first=True)
		self.lstm_history = nn.LSTM(input_dim, output_dim, num_layers=2, batch_first=True)
		# link 对应的统一编码
		self.link_or_em = nn.Embedding(link_embedding_matrix.shape[0], link_embedding_matrix.shape[1])
		self.link_or_em.weight.data.copy_(torch.from_numpy(link_embedding_matrix))
		self.link_or_em.requires_grad = False
		# 连续特征embedding
		self.width_em = nn.Linear(1, embedding_dim)
		self.length_em = nn.Linear(1, embedding_dim)
		self.speed_em = nn.Linear(1, embedding_dim)
		# gcn
		self.gcn = GCN(feat_dim=9*32, k=2)
		# dnn
		self.output_dim = embedding_dim + output_dim * 2 + output_dim * 2 * 4 + 9 * 32
		self.mlp = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(self.output_dim, self.output_dim // 2),
			nn.BatchNorm1d(self.output_dim // 2),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(self.output_dim // 2, 3)
		)

	def forward(
			self, category_index, category_value, recent_split_info, his_split_info,
			link_seq, link_graph, label, is_test=False
	):
		batch_size = category_index.shape[0]
		# fm二阶, nmf,https://zhongqiang.blog.csdn.net/article/details/109532267
		# (batch_size, field_num, embed_dim) * (batch_size, field_num, 1)
		seq_em = self.embedding_seq(category_index) * (category_value.unsqueeze(2))
		square_of_sum = torch.pow(torch.sum(seq_em, dim=1), 2)
		sum_of_square = torch.sum(torch.pow(seq_em, 2), dim=1)
		# (batch_size, embed_dim)
		nmf_features = 0.5 * (square_of_sum - sum_of_square)

		# lstm,在seq_len对hidden_size取最大值和平均值
		# (batch_size, seq_len, hidden_size)
		f, _ = self.lstm_recent(recent_split_info)
		fmax = torch.max(f, dim=1)[0]
		fmean = torch.mean(f, dim=1)
		# (batch_size, hidden_size * 2)
		recent_features = torch.cat([fmax, fmean], dim=1)

		# his_features共用一个lstm
		his_features = []
		for i in range(4):
			his_split_info_part = his_split_info[:, i*5:(i+1)*5, :]
			f, _ = self.lstm_history(his_split_info_part)
			fmax = torch.max(f, dim=1)[0]
			fmean = torch.mean(f, dim=1)
			his_features.extend([fmax, fmean])
		# (batch_size, hidden_size * 2 * 4)
		his_features = torch.cat(his_features, dim=1)

		# gcn
		number_node = link_seq.shape[1]
		# link进行embedding, (batch_size, number_node, embed_dim)
		link_feat_link_id = self.embedding_seq(link_seq)
		# (batch_size, number_node, 8)
		link_feat = self.link_or_em(link_seq)
		link_feat_cate = link_feat[:, :, :5].long()
		# (batch_size, number_node, 5, embed_dim)
		link_feat_cate = self.embedding_seq(link_feat_cate)
		# (batch_size, number_node, 5 * embed_dim)
		link_feat_cate = link_feat_cate.reshape(batch_size, number_node, -1)
		# 对数值特征进行线性变换
		link_feat_width = self.width_em(link_feat[:, :, 5].float().unsqueeze(2))
		link_feat_length = self.length_em(link_feat[:, :, 6].float().unsqueeze(2))
		link_feat_speed = self.speed_em(link_feat[:, :, 7].float().unsqueeze(2))
		# gcn原始特征,(batch_size, number_node, 9 * embed_dim)
		link_feat = torch.cat(
			[link_feat_cate, link_feat_length, link_feat_speed, link_feat_width, link_feat_link_id],
			dim=-1
		)
		# (batch_size, max_number, 9 * embed_dim)
		gcn_features = self.gcn(link_feat, link_graph)
		# 第一个link为起始link, (batch_size, 9 * embed_dim)
		gcn_features = gcn_features[:, 0, :].squeeze()

		features = torch.cat([nmf_features, recent_features, his_features, gcn_features], dim=1)
		out = self.mlp(features)

		if not is_test:
			loss_fun = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.3, 0.6]).to(self.device))
			loss = loss_fun(out, label)
			return loss, F.softmax(out, dim=1)
		else:
			loss_fun = nn.CrossEntropyLoss()
			loss = loss_fun(out, label)
			return loss, F.softmax(out, dim=1)


def validation_fn(model, validation_loader, device, is_test=False):
	model.eval()
	preds = []
	labels = []
	loss_all = []
	for batch in validation_loader:
		category_index, category_value, recent_split_info, his_split_info, link_seq, link_graph, label = (
			item.to(device) for item in batch)
		loss, p = model(category_index, category_value, recent_split_info, his_split_info, link_seq, link_graph, label, is_test=True)
		preds.append(p.detach().cpu().numpy())
		labels.append(label.detach().cpu().numpy())
		loss_all.append(loss.item())
	preds = np.concatenate(preds, axis=0)
	labels = np.concatenate(labels, axis=0)
	if not is_test:
		preds = np.argmax(preds, axis=1)
		f1 = f1_score(labels, preds, average=None)
		score = 0.2 * f1[0] + 0.2 * f1[1] + 0.6 * f1[2]
		return np.mean(loss_all), score
	else:
		return preds


def train_model(
		model, train_loader, validation_loader, accumulation_steps,
		early_stop_epochs, epochs, model_save_path, device
):
	losses = []
	no_improve_epochs = 0
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99), weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
	train_len = len(train_loader)
	best_score = -np.inf
	loss_ages = []
	loss_genders = []
	for epoch in range(1, epochs+1):
		model.train()
		bar = tqdm(train_loader)
		for i, batch in enumerate(bar):
			category_index, category_value, recent_split_info, his_split_info, link_seq, link_graph, label = (item.to(device) for item in batch)
			loss, _ = model(category_index, category_value, recent_split_info, his_split_info, link_seq, link_graph, label, is_test=False)
			loss.backward()
			if (i + 1) % accumulation_steps == 0 or (i + 1) == train_len:
				optimizer.step()
				optimizer.zero_grad()
			loss_ages.append(loss.item())
			loss_genders.append(loss.item())
			bar.set_postfix(
				loss_ages=np.array(loss_ages).mean(),
				loss_genders=np.array(loss_genders).mean(),
				epoch=epoch
			)
		scheduler.step()
		# val
		val_loss, score = validation_fn(model, validation_loader, device)
		losses.append(
			f'train_loss:{np.array(loss_ages).mean():.5f}, val_loss:{val_loss:.5f}, score:{score:.5f}, best_score:{best_score:.5f}'
		)
		print(losses[-1])
		if score >= best_score:
			torch.save(model.state_dict(), model_save_path)
			best_score = score
			no_improve_epochs = 0
		else:
			no_improve_epochs += 1
		if no_improve_epochs == early_stop_epochs:
			break
	return losses


def f1_loss(weight, y_hat, y):
	y_hat = y_hat * weight
	y_hat = np.argmax(y_hat, axis=1)
	f1 = f1_score(y, y_hat, average=None)
	score = f1[0] * 0.2 + f1[1] * 0.2 + f1[2] * 0.6
	return -score


def get_weights(y_hat, y):
	"""
	对于多分类不平衡的情况下，argmax(logits)f1-score不是最优解，
	后处理的时候argmax(w * logits),利用验证集scipy.optimize.minimize去求解w
	"""
	size = np.unique(y).shape[0]
	loss_partial = partial(f1_loss, y_hat=y_hat, y=y)
	initial_weights = [1.0 for _ in range(size)]
	weights = sp.optimize.minimize(
		fun=loss_partial,
		x0=initial_weights,
		method='Powell'
	)
	return weights['x']


def get_his_sample(time_start):
	if os.path.exists('data/sample_data.csv'):
		pass
	else:
		a, b = 0, 0
		tmp_df = pd.DataFrame()
		for d in tqdm(range(time_start, 20190731)):
			tmp_file = pd.read_csv('data/traffic/{}.txt'.format(d), header=None, sep=';')
			# tmp_file = tmp_file.sample(frac=0.2, random_state=1)
			tmp_file['day'] = d
			tmp_file['future_slice_id'] = tmp_file[0].apply(lambda x: int(x.split(' ')[3]))
			if d != 20190730:
				# 第一次采样
				# 测试集中future_slice_id10-39累计占比84.5%
				# 对训练集future_slice_id10-39进行采样10-39,在20190730对>40采样5w条数据
				tmp_file = tmp_file[(tmp_file['future_slice_id'] > 10) & (tmp_file['future_slice_id'] < 40)]
				a += len(tmp_file)
			else:
				tmp_file_1 = tmp_file[(tmp_file['future_slice_id'] > 10) & (tmp_file['future_slice_id'] < 40)]
				a += len(tmp_file_1)
				tmp_file_2 = tmp_file[~((tmp_file['future_slice_id'] > 10) & (tmp_file['future_slice_id'] < 40))].sample(n=50000, random_state=1)
				b += len(tmp_file_2)
				tmp_file = tmp_file_1.append(tmp_file_2)
			tmp_file = tmp_file.drop('future_slice_id', axis=1)
			tmp_df = tmp_df.append(tmp_file)
		# print(tmp_df.shape)
		print(a, b)
		tmp_df.to_csv('data/sample_data.csv', index=False, sep=';', header=None)


def day_origin_df(path):
	df = pd.read_csv(path, header=None, sep=';', usecols=[0, 1])
	df['linkid'] = df[0].apply(lambda x: int(x.split(' ')[0]))
	df['label'] = df[0].apply(lambda x: int(x.split(' ')[1]))
	# 官方回应label-4当作3一样处理
	df['label'] = df['label'].apply(lambda x: 3 if x > 3 else x)
	df['current_slice_id'] = df[0].apply(lambda x: int(x.split(' ')[2]))
	df['future_slice_id'] = df[0].apply(lambda x: int(x.split(' ')[3]))
	df['curr_state'] = df[1].apply(lambda x: int(x.split(' ')[-1].split(':')[-1].split(',')[2]))
	df = df.drop([0, 1], axis=1)
	return df


def sample_train_data(origin_train, origin_test):
	origin_train['time_diff'] = origin_train['future_slice_id'] - origin_train['current_slice_id']
	origin_test['time_diff'] = origin_test['future_slice_id'] - origin_test['current_slice_id']

	# 第二次采样，curr_state采样，0按比例不够，全部
	curr_state_dict = (origin_test['curr_state'].value_counts(normalize=True) * len(origin_train)).astype(int).to_dict()
	sample_train = pd.DataFrame()
	for t, group in origin_train.groupby('curr_state'):
		if t == 0:
			sample_tmp = group
		else:
			sample_tmp = group.sample(n=curr_state_dict[t], random_state=1)
		sample_train = sample_train.append(sample_tmp)

	# 第三次采样，time_diff采样,全部不够采样，按照0.65比例缩减
	time_dict = (origin_test['time_diff'].value_counts(normalize=True) * len(sample_train) * 0.65).astype(int).to_dict()
	sample_df = pd.DataFrame()
	for t, group in sample_train.groupby('time_diff'):
		sample_tmp = group.sample(n=time_dict[t], random_state=1)
		sample_df = sample_df.append(sample_tmp)

	# 测试集中['linkid', 'future_slice_id', 'time_diff']不重复
	# 训练集中time_diff集中在11-15，['linkid', 'future_slice_id']相同的数据只保留time_diff较大的
	for j, i in enumerate(range(11, 16)):
		sample_df.loc[sample_df['time_diff'] == i, 'time_diff'] = \
			sample_df.loc[sample_df['time_diff'] == i, 'time_diff'] * (10 + j)
		sample_df = sample_df.sort_values('time_diff').drop_duplicates(subset=['linkid', 'future_slice_id'], keep='last')
	for j, i in enumerate(range(11, 16)):
		sample_df.loc[sample_df['time_diff'] == i * (10 + j), 'time_diff'] = \
			sample_df.loc[sample_df['time_diff'] == i, 'time_diff'] / (10 + j)

	sample_df = sample_df.drop('time_diff', axis=1)
	return sample_df


def split_features(features, index):
	features = features.split(' ')
	mid = [f.split(':')[-1] for f in features]
	result = [float(f.split(',')[index]) for f in mid]
	return result


def split_slice(features):
	features = features.split(' ')
	result = [int(f.split(':')[0]) for f in features]
	return result


def load_traffic_data(df):
	"""
	同一连续时段的特征整理成列表形式
	"""
	df.columns = [
		0, 'recent_feature', 'history_feature_28', 'history_feature_21',
		'history_feature_14', 'history_feature_7', 'day'
	]
	df['linkid'] = df[0].apply(lambda x: int(x.split(' ')[0]))
	df['current_slice_id'] = df[0].apply(lambda x: int(x.split(' ')[2]))
	df['future_slice_id'] = df[0].apply(lambda x: int(x.split(' ')[3]))
	if 20190801 in set(df['day']):
		df['label'] = df[0].apply(lambda x: int(x.split(' ')[1]))
		df['label'] = df['label'].apply(lambda x: 3 if x > 3 else x)
		df['curr_state'] = df['recent_feature'].apply(lambda x: int(x.split(' ')[-1].split(':')[-1].split(',')[2]))

	df['recent_speed'] = df['recent_feature'].apply(lambda x: split_features(x, 0))
	df['recent_eta'] = df['recent_feature'].apply(lambda x: split_features(x, 1))
	df['recent_status'] = df['recent_feature'].apply(lambda x: split_features(x, 2))
	df['recent_vichles_num'] = df['recent_feature'].apply(lambda x: split_features(x, 3))
	df['recent_slices'] = df['recent_feature'].apply(lambda x: split_slice(x))

	for i in [28, 21, 14, 7]:
		df['history_speed_{}'.format(i)] = df['history_feature_{}'.format(i)].apply(lambda x: split_features(x, 0))
		df['history_eta_{}'.format(i)] = df['history_feature_{}'.format(i)].apply(lambda x: split_features(x, 1))
		df['history_status_{}'.format(i)] = df['history_feature_{}'.format(i)].apply(lambda x: split_features(x, 2))
		df['history_vichles_num_{}'.format(i)] = df['history_feature_{}'.format(i)].apply(lambda x: split_features(x, 3))

	df['weekday'] = pd.to_datetime(df['day'].astype(str)).dt.weekday + 1
	df['hour'] = df['future_slice_id'].apply(lambda x: x // 30)
	df = df.drop(0, axis=1)
	return df


def get_ups(path):
	tmp_dict = defaultdict(list)
	with open(path, 'r') as f:
		for line in f.readlines():
			up, downs = line.split('\t')
			for d in downs.split(','):
				tmp_dict[d.strip()].append(up)
	tmp_df = pd.DataFrame(
		[[k, v] for k, v in tmp_dict.items()],
		columns=['linkid', 'target_link_list']
	)
	tmp_df['linkid'] = tmp_df['linkid'].astype(int)
	tmp_df['target_link_list'] = tmp_df['target_link_list'].apply(lambda x: ','.join(x))
	return tmp_df


def gen_ctr_features(d, key, his_df):
	"""
	可以理解为随时间变化link在key上累计的label每个值的强度
	"""
	his_ = his_df[his_df['day'] < d].copy()
	his_ = his_.drop_duplicates(subset=['link_id', 'future_slice_id', 'day', 'label'], keep='last')
	dummy = pd.get_dummies(his_['label'], prefix='label')
	his_ = pd.concat([his_, dummy], axis=1)
	ctr = his_.groupby(key, as_index=False)['label_0', 'label_1', 'label_2'].mean()
	ctr.columns = key + [
		'_'.join(key)+'_label_0_ctr',
		'_'.join(key) + '_label_1_ctr',
		'_'.join(key) + '_label_2_ctr'
	]
	ctr['day'] = d
	return ctr


def gen_label_timedelta(time_slice, status_list, status):
	"""
	对应status最大的time_slice
	"""
	timedelta = [time_slice[i] for i, f in enumerate(status_list) if f in status]
	if len(timedelta) > 0:
		timedelta = np.max(timedelta)
	else:
		timedelta = np.nan
	return timedelta


def gen_label_timedelta_min(time_slice, status_list, status):
	"""
	对应status最小的time_slice
	"""
	timedelta = [time_slice[i] for i, f in enumerate(status_list) if f in status]
	if len(timedelta) > 0:
		timedelta = np.min(timedelta)
	else:
		timedelta = np.nan
	return timedelta


def gen_label_timedelta_diff(time_slice, status_list, status):
	"""
	对应status的time_slice的间隔平均值
	"""
	timedelta = [time_slice[i] for i, f in enumerate(status_list) if f in status]
	if len(timedelta) > 1:
		timedelta = np.mean(np.diff(timedelta))
	else:
		timedelta = np.nan
	return timedelta


def get_topo_info(df, topo_df, slices=30, mode='down'):
	"""
	linkid 的future_slice_id和相连接的target_linkid的current_slice_id的时间差距
	是否在slices的范围内影响linkid的future_status特征
	"""
	if mode == 'down':
		flg = 'down_target_state'
	else:
		flg = 'up_target_state'
	use_ids = set(df['linkid'])
	topo_df['target_link_list'] = topo_df['target_link_list'].apply(lambda x: x.split(','))
	topo_df = topo_df.explode('target_link_list')
	topo_df['target_link_list'] = topo_df['target_link_list'].astype(int)
	topo_df = topo_df[topo_df['linkid'].isin(use_ids)]
	topo_df = topo_df[topo_df['target_link_list'].isin(use_ids)]
	curr_df = topo_df.rename(columns={'target_link_list': 'target_id'})
	# linkid, target_id, future_slice_id
	curr_df = curr_df.merge(df[['linkid', 'future_slice_id']], on='linkid', how='left')
	tmp_df = df[['linkid', 'current_slice_id', 'curr_state']]
	# target_id, current_slice_id, curr_state
	tmp_df = tmp_df.rename(columns={'linkid': 'target_id'})
	# linkid 相连接的linkid 的current_slice_id, curr_state
	curr_df = curr_df.merge(tmp_df, on='target_id', how='left')

	curr_df['{}_diff_slice'.format(flg)] = curr_df['future_slice_id'] - curr_df['current_slice_id']
	curr_df = curr_df[(curr_df['{}_diff_slice'.format(flg)] >= 0) & (curr_df['{}_diff_slice'.format(flg)] <= slices)]

	curr_df = curr_df.drop_duplicates()
	tmp_list = ['{}_diff_slice'.format(flg)]
	curr_df['{}_diff_slice'.format(flg)] = 1 - curr_df['{}_diff_slice'.format(flg)] / slices
	for s in range(5):
		curr_df['{}_{}'.format(flg, s)] = curr_df['curr_state'].apply(lambda x: 1 if x == s else 0)
		curr_df['{}_{}'.format(flg, s)] = curr_df['{}_{}'.format(flg, s)] * curr_df['{}_diff_slice'.format(flg)]
		tmp_list.append('{}_{}'.format(flg, s))
	curr_df = curr_df.groupby(['linkid', 'future_slice_id'], as_index=False)[tmp_list].sum()
	return curr_df


def f1_weight_score(preds, train_data):
	"""
	lightgbm自定义评价函数
	"""
	y_true = train_data.label
	preds = np.argmax(preds.reshape(-1, 3), axis=1)
	report = classification_report(y_true, preds, digits=5, output_dict=True)
	score = report['0.0']['f1-score'] * 0.2 + report['1.0']['f1-score'] * 0.2 + report['2.0']['f1-score'] * 0.6
	return 'f1_weight_score', score, True


class OptimizeKappa:
	"""
	对于多分类不平衡的情况下，argmax(logits)f1-score不是最优解，
	后处理的时候argmax(w * logits),利用验证集scipy.optimize.minimize去求解w
	"""
	def __init__(self):
		self.coef_ = []

	def _kappa_loss(self, coef, x, y):
		x_p = np.copy(x)
		x_p = coef * x_p
		report = classification_report(y, np.argmax(x_p, axis=1), digits=5, output_dict=True)
		score = report['0.0']['f1-score'] * 0.2 + report['1.0']['f1-score'] * 0.2 + report['2.0']['f1-score'] * 0.6
		return -score

	def fit(self, x, y):
		# 偏函数应用，固定位置参数，关键字参数
		# https://zhuanlan.zhihu.com/p/47124891
		loss_partial = partial(self._kappa_loss, x=x, y=y)
		inital_coef = [1.0 for _ in range(len(set(y)))]
		# 非线性规划求极值，-score的最小值就是score的最大值
		# https://www.jianshu.com/p/94817f7cc89b
		self.coef_ = sp.optimize.minimize(loss_partial, inital_coef, method='Powell')

	def predict(self, x, y):
		x_p = np.copy(x)
		x_p = self.coef_['x'] * x_p
		report = classification_report(y, np.argmax(x_p, axis=1), digits=5, output_dict=True)
		score = report['0.0']['f1-score'] * 0.2 + report['1.0']['f1-score'] * 0.2 + report['2.0']['f1-score'] * 0.6
		return score

	def coefficients(self):
		return self.coef_['x']


def run_lgb(train, test, target, k):
	feats = [f for f in train.columns if f not in ['linkid', 'label', 'day']]
	folds = KFold(n_splits=k, shuffle=True, random_state=2020)
	# linkid作为交叉验证的依据
	train_user_id = train['linkid'].unique()
	output_preds = []
	feature_importance = pd.DataFrame()
	offline_score = []
	train['oof_pred_0'] = 0.0
	train['oof_pred_1'] = 0.0
	train['oof_pred_2'] = 0.0

	for i, (train_idx, valid_idx) in enumerate(folds.split(train_user_id), start=1):
		train_x, train_y = train.loc[train['linkid'].isin(train_user_id[train_idx]), feats], train.loc[train['linkid'].isin(train_user_id[train_idx]), target]
		valid_x, valid_y = train.loc[train['linkid'].isin(train_user_id[valid_idx]), feats], train.loc[train['linkid'].isin(train_user_id[valid_idx]), target]
		dtrain, dvalid = lgb.Dataset(train_x, label=train_y), lgb.Dataset(valid_x, label=valid_y)

		parameters = {
			'learning_rate': 0.05,
			'objective': 'multiclass',
			'metric': 'None',
			'num_leaves': 63,
			'num_class': 3,
			'feature_fraction': 0.8,
			'bagging_fraction': 0.8,
			'verbose': -1,
		}
		lgb_model = lgb.train(
			parameters,
			dtrain,
			num_boost_round=5000,
			valid_sets=[dtrain, dvalid],
			early_stopping_rounds=100,
			verbose_eval=100,
			feval=f1_weight_score
		)

		train.loc[train['linkid'].isin(train_user_id[valid_idx]), ['oof_pred_0', 'oof_pred_1', 'oof_pred_2']] =\
			lgb_model.predict(valid_x, num_iteration=lgb_model.best_iteration)
		# 预测值后处理
		op = OptimizeKappa()
		op.fit(train.loc[train['linkid'].isin(train_user_id[valid_idx]), ['oof_pred_0', 'oof_pred_1', 'oof_pred_2']].values, valid_y)
		output_preds.append(op.coefficients() * lgb_model.predict(test[feats], num_iteration=lgb_model.best_iteration))
		offline_score.append(op.predict(lgb_model.predict(valid_x), valid_y))
		# 特征重要性，以降低loss为标准
		fold_importance = pd.DataFrame()
		fold_importance['feature'] = feats
		fold_importance['importance'] = lgb_model.feature_importance(importance_type='gain')
		feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

	print('feature importance:')
	print(feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False).head(15))
	oof_preds = train[['oof_pred_0', 'oof_pred_1', 'oof_pred_2']].copy()
	train = train.drop(['oof_pred_0', 'oof_pred_1', 'oof_pred_2'], axis=1)
	return output_preds, oof_preds, np.mean(offline_score)


































