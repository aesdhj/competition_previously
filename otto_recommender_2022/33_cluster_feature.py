import pandas as pd
import numpy as np


def make_cluster_co_mat(df, test, dim):
	result_all = []
	for i in range(df[f'cluster_{dim}dim'].max()+1):
		# 原代码这里只考虑了click的数据
		row = df[df[f'cluster_{dim}dim'] == i]
		# 类似candidate的过程, 考虑cluster之间的共现概率
		row = row.merge(test, on='session', how='inner')
		result = row.groupby([f'cluster_{dim}dim_x', f'cluster_{dim}dim_y'], as_index=False)['session'].count()
		result = result.rename(columns={'session': f'cluster_{dim}dim_trans_prob'})
		result[f'cluster_{dim}dim_trans_prob'] = result[f'cluster_{dim}dim_trans_prob'] / result[f'cluster_{dim}dim_trans_prob'].sum()
		result_all.append(result)
	cluster_co_matrix = pd.concat(result_all, axis=0, ignore_index=True)
	return cluster_co_matrix


def make_cluster_feature(mode, dim):
	preprocess_path = 'data/preprocess/'
	data_path = f'data/train_{mode}/'
	candidate_path = 'data/candidate/'
	feature_path = 'data/feature/'
	type2id = {'clicks': 0, 'carts': 1, 'orders': 2}

	aid_cluster = pd.read_parquet(f'{preprocess_path}aid_cluster_{dim}_{mode}.parquet')
	aid_cluster.columns = ['aid', f'cluster_{dim}dim']
	train = pd.read_parquet(data_path + 'train.parquet')
	test = pd.read_parquet(data_path + 'test.parquet')
	df = pd.concat([train, test], axis=0, ignore_index=True)
	test = test.merge(aid_cluster, on='aid', how='left')
	df = df.merge(aid_cluster, on='aid', how='left')
	cluster_co_matrix = make_cluster_co_mat(df, test, dim)

	test_last_aid = test.groupby('session', as_index=False)['aid'].last()
	for t in type2id.keys():
		candidate = pd.read_parquet(f'{candidate_path}{t}_candidate_{mode}.parquet')
		candidate = candidate.merge(test_last_aid, on='session', how='left')
		# last aid的cluster
		candidate = candidate.merge(aid_cluster, left_on='aid_x', right_on='aid', how='left')
		# candidate的cluster
		candidate = candidate.merge(aid_cluster, left_on='aid_y', right_on='aid', how='left')
		# last aid 和 candidate cluster之间的共现概率
		candidate = candidate.merge(cluster_co_matrix, on=[f'cluster_{dim}dim_x', f'cluster_{dim}dim_y'], how='left')
		candidate = candidate[['session', 'aid_x', f'cluster_{dim}dim_trans_prob']]
		candidate.columns = ['session', 'aid', f'cluster_{dim}dim_trans_prob']
		candidate.to_parquet(
			f'{feature_path}{t}_cluster_feature_{dim}dim_{mode}.parquet',
			index=False
		)


MODE, DIM = 'valid', 16
assert MODE in ['test', 'valid']
assert DIM in [16, 64]
make_cluster_feature(MODE, DIM)

