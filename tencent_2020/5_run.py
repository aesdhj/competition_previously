from utils import *
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


if __name__ == '__main__':
	# 读取word2v
	text_features = [
		[
			'data/sequence_text_user_id_{}.128d'.format(key),
			'sequence_text_user_id_{}'.format(key),
			128
		] for key in ['ad_id', 'creative_id', 'advertiser_id', 'product_id', 'industry', 'product_category', 'time', 'click_times']
	]
	args['text_features'] = text_features
	text_features_1 = [
		[
			'data/sequence_text_user_id_{}_fold.12d'.format(key),
			'sequence_text_user_id_{}_fold'.format(key),
			12
		] for key in ['creative_id', 'ad_id', 'product_id', 'advertiser_id', 'industry']
	]
	args['text_features_1'] = text_features_1

	# 字段w2v特征
	args['embeddings_tables'] = []
	for feature in args['text_features']:
		args['embeddings_tables'].append(pickle.load(open(feature[0], 'rb')))
	args['embeddings_tables_1'] = []
	# 字段kflod target encoding编码w2v特征
	for feature in args['text_features_1']:
		args['embeddings_tables_1'].append(pickle.load(open(feature[0], 'rb')))
	# user_id相关的统计特征
	dense_features = [
		'user_id_{}_{}'.format(a, b) for (a, b) in [
			('', 'size'), ('ad_id', 'unique'), ('creative_id', 'unique'), ('advertiser_id', 'unique'),
			('industry', 'unique'), ('product_id', 'unique'), ('time', 'unique'), ('click_times', 'sum'),
			('click_times', 'mean'), ('click_times', 'std')
		]
	]
	# 字段kflod target encoding特征
	for l in ['age_{}'.format(i) for i in range(10)] + ['gender_{}'.format(i) for i in range(2)]:
		for f in ['creative_id', 'ad_id', 'product_id', 'advertiser_id', 'industry']:
			dense_features.append('{}_{}_mean'.format(l, f))
	args['dense_features'] = dense_features

	# 设置参数
	args['n'] = 1
	args['text_dim'] = sum([feature[-1] for feature in text_features])
	args['pretrained_model_path'] = 'saved_models'
	args['vocab_dic'] = pickle.load(open(os.path.join(args['pretrained_model_path'], 'vocab.pkl'), 'rb'))
	args['text_dim_1'] = sum([feature[-1] for feature in text_features_1])
	args['vocab_dim_v1'] = 64
	args['hidden_size'] = 1024
	args['hidden_dropout_prob'] = 0.2
	args['out_size'] = len(args['dense_features']) + args['hidden_size'] * 2 + 1024 * args['n']
	args['linear_layer_size'] = [1024, 512]
	args['num_label'] = 20
	args['epoch'] = 10
	args['lr'] = 8e-5
	args['index'] = 0

	# 读取数据
	train_df = pd.read_pickle('data/train_user_features.pkl')
	test_df = pd.read_pickle('data/test_user_features.pkl')
	test_df['age'] = 1
	test_df['gender'] = 1
	train_df['label'] = train_df['age'] * 2 + train_df['gender']
	test_df['label'] = test_df['age'] * 2 + test_df['gender']
	df = train_df.append(test_df)
	ss = StandardScaler()
	ss.fit(df[args['dense_features']])
	train_df[args['dense_features']] = ss.transform(train_df[args['dense_features']])
	test_df[args['dense_features']] = ss.transform(test_df[args['dense_features']])
	test_dataset = TextDataset_Last(args, test_df)

	# 建立模型
	skf = StratifiedKFold(n_splits=10, random_state=10, shuffle=True)
	model = Ctr_Net(args)
	for i, (train_index, test_index) in enumerate(skf.split(train_df, train_df['label'])):
		if i == args['index']:
			train_dataset = TextDataset_Last(args, train_df.iloc[train_index])
			dev_dataset = TextDataset_Last(args, train_df.iloc[test_index])
			if os.path.exists(os.path.join(args['output_dir'], 'pytorch_model_age.bin')) and os.path.exists(os.path.join(args['output_dir'], 'pytorch_model_gender.bin')):
				pass
			else:
				model.train(train_dataset, dev_dataset)

	# 输出结果
	accs = []
	for label, num in [('age', 10), ['gender', 2]]:
		model.reload(label)
		if label == 'age':
			accs.append(model.evaluate(dev_dataset)[1])
			test_preds = model.evaluate(test_dataset)[4]
			test_df[label] = np.argmax(test_preds, axis=-1) + 1
		else:
			accs.append(model.evaluate(dev_dataset)[2])
			test_preds = model.evaluate(test_dataset)[5]
			test_df[label] = np.argmax(test_preds, axis=-1) + 1

	output_df = test_df[['user_id', 'age', 'gender']]
	output_df.to_csv(
		'submission_{}_{}.csv'.format(args['index'], round(sum(accs), 5)),
		index=False
	)



























