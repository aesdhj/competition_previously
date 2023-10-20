import pandas as pd
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRanker, Pool


def select_train_sample(candidate):
	"""
	:param candidate: ['session', 'aid', 'target']
	:return:
	"""
	candidate_true = candidate[candidate['target'] == 1]
	candidate_false = candidate[candidate['target'] == 0]
	candidate_false_list = []
	for session in list(set(candidate_true['session'])):
		# 每个正样本负采样对应session20个负样本
		true_count = len(candidate_true[candidate_true['session'] == session])
		candidate_false_sample = candidate_false[candidate_false['session'] == session].sample(n=20 * true_count, random_state=2023)
		candidate_false_list.append(candidate_false_sample)
	candidate_resample = pd.concat([candidate_true] + candidate_false_list, axis=0, ignore_index=True)
	return candidate_resample


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
	feature_path = 'data/feature/'
	model_path = 'data/model/'
	oof_path = 'data/oof/'
	co_dict = {
		'allterm': [
			['click', 'click', 20, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']],
			['click', 'buy', 10, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']],
			['buy', 'click', 10, 101, 100, 200, 250, 200, ['all']],
			['buy', 'buy', 10, 101, 100, 200, 250, 200, ['all']]
		],
		'dup': [
			['click', 'click', 20, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']],
			['click', 'buy', 10, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']],
			['buy', 'click', 10, 101, 100, 200, 250, 200, ['all']],
			['buy', 'buy', 10, 101, 100, 200, 250, 200, ['all']]
		],
		'dup_wlen': [
			['click', 'click', 20, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']],
			['click', 'buy', 10, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']],
			['buy', 'click', 10, 101, 100, 200, 250, 200, ['all']],
			['buy', 'buy', 10, 101, 100, 200, 250, 200, ['all']]
		],
		'dup_hour': [
			['click', 'click', 20, 101, 100, 200, 250, 200, ['last', 'hour']],
			['click', 'buy', 10, 101, 100, 200, 250, 200, ['last', 'hour']],
			['buy', 'click', 10, 101, 100, 200, 250, 200, ['all']],
			['buy', 'buy', 10, 101, 100, 200, 250, 200, ['all']]
		],
		'base': [
			['click', 'click', 20, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']],
			['click', 'buy', 10, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']],
			['buy', 'click', 10, 101, 100, 200, 250, 200, ['all']],
			['buy', 'buy', 10, 101, 100, 200, 250, 200, ['all']]
		],
		'base_wlen': [
			['click', 'click', 20, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']],
			['click', 'buy', 10, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']],
			['buy', 'click', 10, 101, 100, 200, 250, 200, ['all']],
			['buy', 'buy', 10, 101, 100, 200, 250, 200, ['all']]
		],
		'base_hour': [
			['click', 'click', 20, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']],
			['click', 'buy', 10, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']],
			['buy', 'click', 10, 101, 100, 200, 250, 200, ['all']],
			['buy', 'buy', 10, 101, 100, 200, 250, 200, ['all']]
		],
		'w2v': [
			['click', 'click', 20, 60, 50, 50, 50, 50, ['last', 'hour']]
		]
	}

	# 读取候选集数据
	candidate_all = pd.read_parquet(f'{candidate_path}{type_name}_candidate_valid.parquet')
	# 负采样
	candidate = select_train_sample(candidate_all)
	# 合并特征
	candidate = join_features(candidate, type_name, co_dict, mode='valid')
	candidate.to_parquet(
		f'{feature_path}{type_name}_candidate_with_feature_valid.parquet',
		index=False
	)
	candidate = candidate[['session', 'aid', 'target']]

	# catboost入门 https://zhuanlan.zhihu.com/p/540956200
	# catboostranker参数 https://catboost.ai/en/docs/references/training-parameters/
	# ltr入门 https://zhuanlan.zhihu.com/p/138436325
	skf = GroupKFold(n_splits=5)
	result_all = []
	for fold, (train_idx, valid_idx) in enumerate(skf.split(candidate, candidate['target'], groups=candidate['session'])):
		# 交叉验证
		x_train = candidate.loc[train_idx, :]
		x_valid = candidate.loc[valid_idx, :]
		features = pd.read_parquet(
			f'{feature_path}{type_name}_candidate_with_feature_valid.parquet'
		)
		features = features.drop('target', axis=1)
		feature_list = [col for col in features.columns if col not in ['session', 'aid']]
		x_train = x_train.merge(features, on=['session', 'aid'], how='left')
		x_valid = x_valid.merge(features, on=['session', 'aid'], how='left')
		x_train = Pool(
			data=x_train[feature_list],
			label=x_train['target'],
			group_id=x_train['session']
		)
		x_valid = Pool(
			data=x_valid[feature_list],
			label=x_valid['target'],
			group_id=x_valid['session']
		)
		# 模型训练
		params = {
			'loss_function': 'PairLogitPairwise',
			'learning_rate': 0.05,
			'custom_metric': 'RecallAt:top=20',
			'iterations': 10000,
			'depth': 7,
			'use_best_model': True,
			'task_type': 'GPU',
			'metric_period': 100,
			'early_stopping_rounds': 100,
			'random_state': 2023
		}
		ranker = CatBoostRanker(**params)
		ranker.fit(x_train, eval_set=x_valid)
		ranker.save_model(
			f'{model_path}_cb_fold{fold}_{type_name}.cbm',
			format='cbm'
		)
		# 验证全部数据
		candidate_all = pd.read_parquet(f'{candidate_path}{type_name}_candidate_valid.parquet')
		candidate_all = join_features(candidate_all, type_name, co_dict, mode='valid')
		result = ranker.predict(candidate_all[feature_list])
		candidate_all = candidate_all[['session', 'aid', 'target']]
		candidate_all['pred'] = result
		result_all.append(candidate_all)

	result_all = pd.concat(result_all, axis=0, ignore_index=True)
	result_all.to_parquet(
		f'{oof_path}oof_result_{type_name}.parquet',
		index=False
	)


for t in ['clicks', 'clicks_all', 'carts', 'orders']:
	main(t)

