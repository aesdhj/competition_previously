import pandas as pd
import numpy as np
from catboost import CatBoostRanker


def join_features(candidate, type_name, co_dict, mode):
	feature_path = 'data/feature/'
	if type_name == 'clicks_all':
		feature_type_name = 'clicks'
	else:
		feature_type_name = type_name

	# 32-bpr_feature
	bpr_feature = pd.read_parquet(f'{feature_path}{feature_type_name}_bpr_feature_{mode}.parquet')
	candidate = candidate.merge(bpr_feature, on=['session', 'aid'], how='left')
	# 35-w2v_feature
	for dim in [16, 64]:
		session_aid_w2v_sim = pd.read_parquet(
			f'{feature_path}{feature_type_name}_session_aid_w2v_sim_{dim}dim_{mode}.parquet')
		candidate = candidate.merge(session_aid_w2v_sim, on=['session', 'aid'], how='left')
		aid_aid_w2v_sim = pd.read_parquet(
			f'{feature_path}{feature_type_name}_last_action_aid_aid_w2v_sim_{dim}dim_{mode}.parquet')
		candidate = candidate.merge(aid_aid_w2v_sim, on=['session', 'aid'], how='left')
		aid_aid_w2v_sim = pd.read_parquet(
			f'{feature_path}{feature_type_name}_last_hour_action_aid_aid_w2v_sim_{dim}dim_{mode}.parquet')
		candidate = candidate.merge(aid_aid_w2v_sim, on=['session', 'aid'], how='left')
	# 11-comat_feature
	for pattern in co_dict.keys():
		if pattern == 'w2v':
			continue
		else:
			for j in range(len(co_dict[pattern])):
				start_type = co_dict[pattern][j][0]
				end_type = co_dict[pattern][j][1]
				action_pattern_list = co_dict[pattern][j][8]
				for action_pattern in action_pattern_list:
					feature_name = f'{start_type}_{end_type}_{pattern}_{action_pattern}'
					com_feature = pd.read_parquet(f'{feature_path}{feature_name}_{mode}.parquet')
					if 'rank' in com_feature.columns:
						com_feature = com_feature.drop('rank', axis=1)
					candidate = candidate.merge(com_feature, on=['session', 'aid'], how='left')
	same_aid_feature = pd.read_parquet(f'{feature_path}same_aid_feature_{mode}.parquet')
	candidate = candidate.merge(same_aid_feature, on='aid', hoe='left')
	# 33—cluster_feature
	for dim in [16, 64]:
		cluster_feature = pd.read_parquet(f'{feature_path}{feature_type_name}_cluster_feature_{dim}dim_{mode}.parquet')
		candidate = candidate.merge(cluster_feature, on=['session', 'aid'], how='left')
	# 34-session_aid_feature
	session_aid_feature = pd.read_parquet(f'{feature_path}session_aid_feature_{mode}.parquet')
	candidate = candidate.merge(session_aid_feature, on=['session', 'aid'], how='left')
	last_chunk_session_aid_feature = pd.read_parquet(f'{feature_path}last_chunk_session_aid_feature_{mode}.parquet')
	candidate = candidate.merge(last_chunk_session_aid_feature, on=['session', 'aid'], how='left')
	session_feature = pd.read_parquet(f'{feature_path}session_feature_{mode}.parquet')
	candidate = candidate.merge(session_feature, on='session', how='left')
	session_aid_feature = pd.read_parquet(f'{feature_path}session_aid_feature_{mode}.parquet')
	session_aid_feature = session_aid_feature.drop('aid', axis=1)
	candidate = candidate.merge(session_aid_feature, on=['session', 'aid'], how='left')
	session_day_feature = pd.read_parquet(f'{feature_path}session_day_feature_{mode}.parquet')
	# candidate_aid, day_last的特征
	candidate = candidate.merge(session_day_feature, on=['aid', 'day'], how='left')
	session_aid_feature = session_aid_feature.drop('day', axis=1)
	aid_feature = pd.read_parquet(f'{feature_path}aid_feature_{mode}.parquet')
	candidate = candidate.merge(aid_feature, on='aid', how='left')

	candidate = candidate.fillna(0)
	return candidate


def main(type_name):
	candidate_path = 'data/candidate/'
	model_path = 'data/model/'
	output_path = 'data/output/'

	candidate_all = pd.read_parquet(f'{candidate_path}{type_name}_candidate_test.parquet')
	candidate_all = join_features(candidate_all)
	feature_list = [col for col in candidate_all.columns if col not in ['session', 'aid']]
	pred = candidate_all[['session', 'aid']].copy()
	result = np.zeros(len(candidate_all))

	for fold in range(5):
		model_path = f'{model_path}_cb_fold{fold}_{type_name}.cbm'
		ranker = CatBoostRanker()
		ranker.load_model(model_path)
		result += ranker.predict(candidate_all[feature_list]) / 5
	pred['pred'] = result
	pred.to_parquet(
		f'{output_path}prediction_{type_name}.parquet',
		index=False
	)


for t in ['clicks', 'clicks_all', 'carts', 'orders']:
	main(t)

