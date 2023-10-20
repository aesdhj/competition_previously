import pandas as pd
import numpy as np


def cos_sim(a, b):
	sim = np.sum(a * b, axis=1)
	sim = sim / np.linalg.norm(a, axis=1)
	sim = sim / np.linalg.norm(b, axis=1)
	sim = sim.reshape(-1, 1)
	return sim


def make_action_data(test):
	max_ts = test.groupby('session', as_index=False)['ts'].agg({'max_ts': 'max'})
	test = test.merge(max_ts, on='session', how='left')
	last_action = test[test['ts'] == test['max_ts']]
	last_action = last_action[['session', 'aid']].drop_duplicates().reset_index()
	last_action = last_action.rename(columns={'aid': 'action_aid'})
	last_hour_action = test[(test['ts'] > (test['max_ts'] - 60 * 60)) & (test['ts'] != test['max_ts'])]
	last_hour_action = last_hour_action[['session', 'aid']].drop_duplicates().reset_index()
	last_hour_action = last_hour_action.rename(columns={'aid': 'action_aid'})
	return last_action, last_hour_action


def session_aid_w2v_sim(test, mode, dim, type, chunk_size):
	preprocess_path = 'data/preprocess/'
	candidate_path = 'data/candidate/'
	feature_path = 'data/feature/'

	aid_w2v = pd.read_parquet(f'{preprocess_path}w2v_{dim}_{mode}.parquet')
	session_w2v = test[['aid', 'session']].merge(aid_w2v, on='aid', how='left')
	session_w2v = session_w2v.drop('aid', axis=1)
	session_w2v = session_w2v.groupby('session', as_index=False).mean()
	candidate = pd.read_parquet(f'{candidate_path}{type}_candidate_{mode}.parquet')
	session_list = list(set(candidate['session']))
	chunk_num = int(len(session_list)/chunk_size) + 1
	# candidate中aid和session的cossim
	candidate_feature = []
	for i in range(chunk_num):
		start = i * chunk_size
		end = (i+1) * chunk_size
		chunk_candidate = candidate[candidate['session'].isin(session_list[start:end])][['session', 'aid']]
		aid_vec = chunk_candidate[['aid']].merge(aid_w2v, on='aid', how='left')
		session_vec = chunk_candidate[['session']].merge(session_w2v, on='session', how='left')
		chunk_candidate[f'session_aid_w2v_sim_{dim}dim'] = cos_sim(aid_vec.values, session_vec.values)
		candidate_feature.append(chunk_candidate)
	candidate_feature = pd.concat(candidate_feature, axis=0, ignore_index=True)
	candidate_feature.to_parquet(
		f'{feature_path}{type}_session_aid_w2v_sim_{dim}dim_{mode}.parquet',
		index=False
	)


def aid_aid_w2v_sim(action_dfs, mode, dim, type, chunk_size):
	preprocess_path = 'data/preprocess/'
	candidate_path = 'data/candidate/'
	feature_path = 'data/feature/'

	aid_w2v = pd.read_parquet(f'{preprocess_path}w2v_{dim}_{mode}.parquet')
	candidate = pd.read_parquet(f'{candidate_path}{type}_candidate_{mode}.parquet')
	session_list = list(set(candidate['session']))
	chunk_num = int(len(session_list)/chunk_size) + 1
	# candidate aid和对应session的前一时间段的aid的cosim
	candidate_feature = []
	for idx, action_df in enumerate(action_dfs):
		for i in range(chunk_num):
			start = i * chunk_size
			end = (i+1) * chunk_size
			chunk_candidate = candidate[candidate['session'].isin(session_list[start:end])][['session', 'aid']]
			chunk_candidate = chunk_candidate.merge(action_df, on='session', how='inner')
			aid_vec = chunk_candidate[['aid']].merge(aid_w2v, on='aid', how='left')
			action_aid_vec = chunk_candidate[['action_aid']].merge(aid_w2v, on='aid', how='left')
			chunk_candidate[f'aid_aid_w2v_sim_{dim}dim'] = cos_sim(aid_vec.values, action_aid_vec.values)
			chunk_candidate = chunk_candidate.groupby(['session', 'aid'])[f'aid_aid_w2v_sim_{dim}dim'].mean()
			candidate_feature.append(chunk_candidate)
		candidate_feature = pd.concat(candidate_feature, axis=0, ignore_index=True)
		if idx == 0:
			candidate_feature.to_parquet(
				f'{feature_path}{type}_last_action_aid_aid_w2v_sim_{dim}dim_{mode}.parquet',
				index=False
			)
		else:
			candidate_feature.to_parquet(
				f'{feature_path}{type}_last_hour_action_aid_aid_w2v_sim_{dim}dim_{mode}.parquet',
				index=False
			)


def main(mode, dim, type):
	data_path = f'data/train_{mode}'
	test = pd.read_parquet(data_path + 'test.parquet')
	session_aid_w2v_sim(test, mode, dim, type, chunk_size=15000)
	last_action, last_hour_action = make_action_data(test)
	aid_aid_w2v_sim([last_action, last_hour_action], mode, dim, type, chunk_size=15000)


MODE, DIM = 'valid', 16
assert MODE in ['test', 'valid']
assert DIM in [16, 64]
main(MODE, DIM, type='clicks')
main(MODE, DIM, type='carts')
main(MODE, DIM, type='orders')

