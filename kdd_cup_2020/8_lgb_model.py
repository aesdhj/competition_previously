from utils import *
import warnings
from sklearn.model_selection import KFold
import lightgbm as lgb


pd.options.display.max_columns = None
warnings.filterwarnings('ignore')
recall_source_names = cur_recall_source_names
recall_file_names = '-'.join(recall_source_names + [sum_mode])
cur_stage = CUR_STAGE
mode = cur_mode


def modeling(train_x, train_y, test_x, test_y, cate_list, mode, kfold, weight=None):
	early_stop = 300
	max_rounds = 10000
	lr = 0.005
	params = {
		'metric': 'binary_logloss',
		'objective': 'binary',
		'learning_rate': lr,
		'subsample': 0.85,
		'subsample_freq': 10,
		'colsample_bytree': 0.8,
		'num_leaves': 63,
		'seed': 2022,
		'scale_pos_weight': 1.5 # 对不平衡样本加权，参考值len(negative_samples) / len(positive_samples)
	}
	if mode == 'valid':
		feat_cols = list(train_x.columns)
		dtrain = lgb.Dataset(data=train_x, label=train_y, feature_name=feat_cols, weight=weight)
		dvalid = lgb.Dataset(data=test_x, label=test_y, feature_name=feat_cols)
		model = lgb.train(
			params,
			dtrain,
			categorical_feature=cate_list,
			num_boost_round=max_rounds,
			early_stopping_rounds=early_stop,
			verbose_eval=100,
			valid_sets=[dtrain, dvalid],
			valid_names=['train', 'valid']
		)
		importances = pd.DataFrame({
			'features': model.feature_name(),
			'importances': model.feature_importance()
		})
		importances = importances.sort_values('importances', ascending=False)
		print(importances[:20])
		importances.to_csv(feat_imp_dir+'kfold_{}_imp.csv'.format(kfold+1))
		return model


def evaluate_each_phase(predictions, answers, at=50):
	# ndcg指标解释， https://www.cnblogs.com/by-dream/p/9403984.html
	list_item_degrees = []
	for item, item_degree in answers:
		list_item_degrees.append(item_degree)
	median_item_degree = np.median(list_item_degrees)

	num_cases_full = 0.0
	ndcg_50_full = 0.0
	ndcg_50_half = 0.0
	num_cases_half = 0.0
	hitrate_50_full = 0.0
	hitrate_50_half = 0.0

	row = 0
	for item, item_degree in answers:
		rank = 0
		while rank < at and predictions[row][rank] != item:
			rank += 1

		num_cases_full += 1.0
		if rank < at:
			ndcg_50_full += 1.0 / np.log2(rank+2.0)
			hitrate_50_full += 1.0
		if item_degree <= median_item_degree:
			num_cases_half += 1.0
			if rank < at:
				ndcg_50_half += 1.0 / np.log2(rank+2.0)
				hitrate_50_half += 1.0
		row += 1

	ndcg_50_full /= num_cases_full
	hitrate_50_full /= num_cases_full
	ndcg_50_half /= num_cases_half
	hitrate_50_half /= num_cases_half
	return np.array([
		hitrate_50_full, ndcg_50_full,
		hitrate_50_half, ndcg_50_half
	], dtype=np.float32)


def get_scores(ans, shift, bottom, after_deal):
	print(f'shift: {shift}, bottom: {bottom}, after_deal: {after_deal}')
	phase_item_degree = load_pickle(phase_full_item_degree_path.format(cur_stage))
	df_valid = load_pickle(all_valid_stage_data_path.format(cur_stage))
	phase2valid_item_degree = {}
	# 每个stage对应df_valid中recall_item degree median
	phase2median = {}
	for stage in range(cur_stage+1):
		cur_df_valid = df_valid[df_valid['stage'] == stage]
		items = cur_df_valid['item_id'].values
		item_degree = phase_item_degree[stage]
		list_item_degrees = []
		# recall if re[0] in stage_items:
		for item_id in items:
			list_item_degrees.append(item_degree[item_id])
			phase2valid_item_degree[(stage, item_id)] = item_degree[item_id]
		phase2median[stage] = np.median(list_item_degrees)

	old = False
	# 对预测概率进行后处理
	if after_deal:
		# user不同recall_type item 可能会被召回多次, 取label最大
		ans = ans.groupby(['user', 'item'], as_index=False)['label'].max()
		user2stage = df_valid[['user_id', 'stage']]
		user2stage.columns = ['user', 'stage']
		ans = ans.merge(user2stage, on='user', how='left')
		if old:
			# 用了test部分的item_degree_median
			sta_list = []
			item_list = []
			degree_list = []
			for stage in range(cur_stage+1):
				item_degree = phase_item_degree[stage]
				for item in item_degree.keys():
					sta_list.append(stage)
					item_list.append(item)
					degree_list.append(item_degree[item])
			df_degree = pd.DataFrame({
				'stage': sta_list,
				'item': item_list,
				'degree': degree_list
			})
			ans = ans.merge(df_degree, on=['stage', 'item'], how='left')
			phase_median = ans.groupby('stage', as_index=False)['degree'].median()
			phase_median.columns = ['stage', 'median_degree']
			ans = ans.merge(phase_median, on='stage', how='left')
			ans['is_rare'] = (ans['degree'] <= (ans['median_degree'] + shift))
		else:
			# 用了valid item_degree_median进行比较
			ans['is_rare'] = ans.apply(lambda x: 1 if phase_item_degree[x['stage']][x['item']] <= phase2median[x['stage']] else 0, axis=1)

		# 把地流行度预测的recall_item 提前，对评价函数ndcg-50half的优化
		ans['is_rare'] = ans['is_rare'].astype('float') / bottom
		ans['is_rare'] = ans['is_rare'] + 1.0
		ans['label'] = ans['label'] * ans['is_rare']

	else:
		# user不同recall_type item 可能会被召回多次, 取label最大
		ans = ans.groupby(['user', 'item'], as_index=False)['label'].max()

	ans['label'] = -ans['label']
	ans = ans.sort_values(['user', 'label'])
	user2recall = ans.groupby('user')['item'].agg(list)
	user2pos = df_valid[['user_id', 'item_id']].set_index('user_id')

	all_scores = []
	all_pred_items = {}
	pickup = 500
	for stage in range(cur_stage+1):
		predictions = []
		answers = []
		item_degree = phase_item_degree[stage]
		users = set(df_valid[df_valid['stage'] == stage]['user_id']) & set(ans['user'])
		for user in users:
			pos = user2pos.loc[user].values[0]
			pred = user2recall.loc[user]
			# 不重复，取第一个预测
			new_pred = []
			for p in pred:
				if len(new_pred) < pickup:
					flag = 0
					for k in new_pred:
						if k == p:
							flag = 1
							break
					if flag == 0:
						new_pred.append(p)

			answers.append((pos, item_degree[pos]))
			all_pred_items[user] = []
			for pred in new_pred[:pickup]:
				all_pred_items[user].append(pred)
			predictions.append(new_pred[:50] + [0] * (max(0, 50 - len(new_pred))))

		scores = evaluate_each_phase(predictions, answers, at=50)
		print('stage:', stage)
		print(scores)
		all_scores.append(scores)
	return all_scores


if __name__ == '__main__':
	drop_list = [
		'left_items_list', 'right_items_list', 'left_times_list', 'right_times_list',
		'user', 'time', 'item', 'road_item',
		'recall_type', 'label'
	]
	cate_list = ['stage']

	phase_item_degree = load_pickle(phase_full_item_degree_path.format(cur_stage))
	df_valid = load_pickle(all_valid_stage_data_path.format(cur_stage))
	phase2valid_item_degree = {}
	# 每个stage df_valid item degree median
	phase2median = {}
	for stage in range(cur_stage+1):
		cur_df_valid = df_valid[df_valid['stage'] == stage]
		items = cur_df_valid['item_id'].values
		item_degree = phase_item_degree[stage]
		list_item_degrees = []
		for item_id in items:
			list_item_degrees.append(item_degree[item_id])
			phase2valid_item_degree[(stage, item_id)] = item_degree[item_id]
		phase2median[stage] = np.median(list_item_degrees)
	print(phase2median)

	lgb_model_data_path = os.path.join(feat_dir, 'lgb_model_data.pkl')
	data = load_pickle(lgb_model_data_path)
	# data = load_pickle(lgb_base_pkl.format(recall_file_names, mode, cur_stage))
	t = data.groupby('user', as_index=False)['label'].sum()
	# recall 有命中的 user
	has_pos_users = list(set(t[t['label'] > 0]['user']))
	all_users = data['user'].unique()
	ans = data[['user', 'item', 'time', 'stage']]

	kfold = KFold(n_splits=3, shuffle=True, random_state=2022)
	for j, (train_user_index, test_user_index) in enumerate(kfold.split(X=all_users)):
		print('-'*50)
		print('kfold {}'.format(j+1))
		# 训练集用recall命中的user, 验证集不需要这个条件
		train_users = all_users[train_user_index]
		train_users = list(set(train_users) & set(has_pos_users))
		train_index = data[data['user'].isin(train_users)].index
		train_data = data.loc[train_index]
		test_users = all_users[test_user_index]
		test_index = data[data['user'].isin(test_users)].index
		test_data = data.loc[test_index]

		users = set()
		# 每个user recall_item对应stage的degree
		stage2degree = {stage: [] for stage in range(cur_stage+1)}
		phase_item_degree = np.zeros(len(train_data))
		for i, (user, stage, item, label) in enumerate(zip(
				train_data['user'].values, train_data['stage'].values,
				train_data['item'].values, train_data['label'].values
		)):
			if label == 1:
				phase_item_degree[i] = phase2valid_item_degree[(stage, item)]
				if user not in users:
					users.add(user)
					stage2degree[stage].append(phase_item_degree[i])
			else:
				phase_item_degree[i] = np.nan

		# train_data 中每个stage label==1 的 recall_item 的degree median
		phase2median = {}
		for stage in range(cur_stage+1):
			list_item_degrees = stage2degree[stage]
			phase2median[stage] = np.median(list_item_degrees)
		print(phase2median)

		weights = [6.5, 1] # 参数搜索出来的
		median = train_data['stage'].map(phase2median).values
		weight = np.ones(len(train_data))
		# 对label==1 的recall_item，如果degree < median，进行样本加权
		weight[phase_item_degree <= median] = weights[0]
		weight[phase_item_degree > median] = weights[1]

		train_x = train_data.drop(drop_list, axis=1)
		train_y = train_data['label']
		test_x = test_data.drop(drop_list, axis=1)
		test_y = test_data['label']

		model = modeling(train_x, train_y, test_x, test_y, cate_list, mode, j, weight)
		ans_block = ans.copy()
		ans_block.loc[test_index, 'label'] = model.predict(test_x, num_iteration=model.best_iteration)
		ans.loc[test_index, 'label'] = model.predict(test_x, num_iteration=model.best_iteration)
		get_scores(ans=ans_block.copy(), shift=0.0, bottom=0.25, after_deal=False)
		get_scores(ans=ans_block.copy(), shift=0.0, bottom=0.25, after_deal=True)

	print('all score')
	get_scores(ans=ans.copy(), shift=0.0, bottom=0.25, after_deal=False)
	get_scores(ans=ans.copy(), shift=0.0, bottom=0.25, after_deal=True)

































