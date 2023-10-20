import pandas as pd
import numpy as np


def get_bi_feature(candidate, user_aid_list, bigram_count, name):
	type2id = {'clicks': 0, 'carts': 1, 'orders': 2}
	feature_df = pd.DataFrame()
	for t in ['clicks', 'carts']:
		features = []
		for idx, session, aid in candidate[['session', 'aid']].itertuples():
			past_acts = user_aid_list[type2id[t]].get(session, [])
			if len(past_acts) > 0:
				sims = []
				for act in past_acts:
					# candidate和对应session的test aid的相似性数据
					sims.append(bigram_count.get((act, aid), 0))
				features.append([np.sum(sims), np.mean(sims), np.max(sims), np.min(sims), sims[-1]])
			else:
				features.append([-1, -1, -1, -1, -1])
		if len(feature_df) == 0:
			feature_df = pd.DataFrame(
				features,
				columns=[f'{name}_{t}_{pattern}' for pattern in ['sum', 'mean', 'max', 'min', 'last']])
		else:
			feature = pd.DataFrame(
				features,
				columns=[f'{name}_{t}_{pattern}' for pattern in ['sum', 'mean', 'max', 'min', 'last']])
			feature_df = pd.concat([feature, feature_df], axis=1)
	return feature_df


def make_bi_feature(mode):
	type2id = {'clicks': 0, 'carts': 1, 'orders': 2}
	data_path = f'data/train_{mode}/'
	candidate_path = 'data/candidate/'
	feature_path = 'data/feature/'
	train = pd.read_parquet(data_path + 'train.parquet')
	test = pd.read_parquet(data_path + 'test.parquet')
	df = pd.concat([train, test], axis=0, ignore_index=True)
	# bigram特征
	item_cnt = {all: df.groupby('aid').size().to_dict()}
	for t in type2id.keys():
		item_cnt[type2id[t]] = df[df['type'] == type2id[t]].groupby('aid').size().to_dict()
	df['aid_next'] = df.groupby('session')['aid'].shift(-1)
	df = df.dropna()
	bigram_counter = df.groupby(['aid', 'aid_next']).size().to_dict()
	normed_bigram_counter = {}
	# 关联度标准化
	for (a1, a2), cnt in bigram_counter.items():
		normed_bigram_counter[(a1, a2)] = cnt /\
			np.sqrt(item_cnt['all'].get(a1, 1) * item_cnt['all'].get(a2, 1))
	# ???为什么只取order_candidate特征
	candidate = pd.read_parquet(candidate_path + f'orders_candidate_{mode}.parquet')
	user_aid_list = {}
	for t in type2id.keys():
		user_aid_list[type2id[t]] = test[test['type'] == type2id[t]].groupby('session').agg(list).to_dict()
	bigram_feature = get_bi_feature(candidate, user_aid_list, normed_bigram_counter, 'bigram_normed')
	bigram_feature = pd.concat([candidate[['session', 'aid']], bigram_feature], axis=1)
	bigram_feature.to_parquet(f'{feature_path}bigram_feature_{mode}.parquet')


MODE = 'valid'
assert MODE in ['valid', 'test']
make_bi_feature(MODE)

