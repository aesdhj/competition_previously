from utils import *
import warnings

pd.options.display.max_columns = None
warnings.filterwarnings('ignore')
recall_source_names = cur_recall_source_names
recall_file_names = '-'.join(recall_source_names + [sum_mode])
cur_stage = CUR_STAGE
mode = cur_mode


def feat_user_item_weight_sum_mean(data):
	df = data.copy()
	feat = df[['user', 'item']]
	temp = df.groupby(['user', 'item'])['sim_weight', 'loc_weight', 'time_weight', 'rank_weight'].agg(['sum', 'mean']).reset_index()
	feat_cols = [f'item_{j}_{i}' for i in ['sim_weight', 'loc_weight', 'time_weight', 'rank_weight'] for j in ['sum', 'mean']]
	temp.columns = ['user', 'item'] + feat_cols
	feat = feat.merge(temp, on=['user', 'item'], how='left')
	feat = feat[feat_cols]
	return feat


def feat_weight_sum(data):
	df = data.copy()
	feat = df[['user', 'item']]
	feat['sum_sim_loc_time_weight'] = df['sim_weight'] + df['loc_weight'] + df['time_weight']
	feat = feat[['sum_sim_loc_time_weight']]
	return feat


def feat_road_item_text_cossim_plus(data):
	df = data.copy()
	feat = data[['road_item', 'item']]
	# {item: [vec_text, vec_pic], ...}
	item_feat = load_pickle(item_feat_pkl)

	def func(row):
		item1 = row['road_item']
		item2 = row['item']
		if (item1 in item_feat) and (item2 in item_feat):
			item1_text = item_feat[item1][0]
			item2_text = item_feat[item2][0]
			c = np.dot(item1_text, item2_text)
			a = np.linalg.norm(item1_text)
			b = np.linalg.norm(item2_text)
			return c / (a * b + 1e-9)
		else:
			return np.nan

	def func_1(row):
		item1 = row['road_item']
		item2 = row['item']
		if (item1 in item_feat) and (item2 in item_feat):
			item1_text = item_feat[item1][0]
			item2_text = item_feat[item2][0]
			c = np.dot(item1_text, item2_text)
			return c
		else:
			return np.nan

	def func_2(row):
		item1 = row['road_item']
		item2 = row['item']
		if (item1 in item_feat) and (item2 in item_feat):
			item1_text = item_feat[item1][0]
			item2_text = item_feat[item2][0]
			a = np.linalg.norm(item1_text)
			b = np.linalg.norm(item2_text)
			return a * b
		else:
			return np.nan

	def func_3(row):
		item1 = row['road_item']
		if item1 in item_feat:
			item1_text = item_feat[item1][0]
			a = np.linalg.norm(item1_text)
			return a
		else:
			return np.nan

	feat['road_item_text_cossim'] = df[['road_item', 'item']].apply(func, axis=1)
	feat['road_item_text_dot'] = df[['road_item', 'item']].apply(func_1, axis=1)
	feat['road_item_text_product_norm2'] = df[['road_item', 'item']].apply(func_2, axis=1)
	feat['road_item_norm2'] = df[['road_item', 'item']].apply(func_3, axis=1)
	feat = feat[[
		'road_item_text_cossim', 'road_item_text_dot',
		'road_item_text_product_norm2', 'road_item_norm2'
	]]
	return feat


def feat_road_item_text_eulasim(data):
	df = data.copy()
	feat = data[['road_item', 'item']]
	item_feat = load_pickle(item_feat_pkl)

	def func(row):
		item1 = row['road_item']
		item2 = row['item']
		if (item1 in item_feat) and (item2 in item_feat):
			item1_text = item_feat[item1][0]
			item2_text = item_feat[item2][0]
			a = np.linalg.norm(item1_text - item2_text)
			return a
		else:
			return np.nan

	feat['road_item_text_eulasim'] = df[['road_item', 'item']].apply(func, axis=1)
	feat = feat[['road_item_text_eulasim']]
	return feat


def feat_sim_base_plus(data):
	df = data.copy()
	feat = df[['road_item', 'item', 'left_items_list', 'right_items_list']]

	if mode == 'valid':
		df_train = load_pickle(all_train_data_path.format(cur_stage))
	else:
		df_train = load_pickle(online_all_train_data_path.format(cur_stage))
	user_item_dict = df_train.groupby('user_id')['item_id'].agg(list).to_dict()
	user_time_dict = df_train.groupby('user_id')['time'].agg(list).to_dict()

	sim_item = {}
	item_cnt = defaultdict(int)
	com_item_cnt = {}
	item_set = set()
	item_dict_set = {}

	loc_weights = {}
	time_weights = {}
	record_weights = {}

	for user, items in user_item_dict.items():
		for item in items:
			item_set.add(item)
	for item in item_set:
		item_dict_set[item] = set()

	for user, items in user_item_dict.items():
		times = user_time_dict[user]

		for loc1, item in enumerate(items):
			item_cnt[item] += 1
			sim_item.setdefault(item, {})
			com_item_cnt.setdefault(item, {})
			loc_weights.setdefault(item, {})
			time_weights.setdefault(item, {})
			record_weights.setdefault(item, {})
			for loc2, relate_item in enumerate(items):
				if item == relate_item:
					continue
				item_dict_set[item].add(relate_item)

				t1, t2 = times[loc1], times[loc2]
				sim_item[item].setdefault(relate_item, 0)
				com_item_cnt[item].setdefault(relate_item, 0)
				loc_weights[item].setdefault(relate_item, 0)
				time_weights[item].setdefault(relate_item, 0)
				record_weights[item].setdefault(relate_item, 0)

				time_weight = (1 - abs(t1-t2)*100)
				time_weight = max(time_weight, 0.2)
				loc_weight = 0.9 ** (abs(loc1-loc2)-1)
				loc_weight = max(loc_weight, 0.2)
				# 1.0为可调整系数
				sim_item[item][relate_item] += 1 * 1.0 * loc_weight * time_weight / math.log(1+len(items))
				com_item_cnt[item][relate_item] += 1.0
				loc_weights[item][relate_item] += loc_weight
				time_weights[item][relate_item] += time_weight
				record_weights[item][relate_item] += len(items)

	num = feat.shape[0]
	road_item = feat['road_item'].values
	t_item = feat['item'].values
	left_items_list = feat['left_items_list'].values
	right_items_list = feat['right_items_list'].values

	road_item_cnt = np.zeros(num, dtype=float)
	t_item_cnt = np.zeros(num, dtype=float)
	com_item_cij = np.zeros(num, dtype=float)
	t_com_item_cnt = np.zeros(num, dtype=float)
	com_item_loc_weights_sum = np.zeros(num, dtype=float)
	com_item_time_weights_sum = np.zeros(num, dtype=float)
	com_item_record_weights_sum = np.zeros(num, dtype=float)

	max_i2i_sim_arr = np.zeros(len(feat))
	mean_i2i_sim_arr = np.zeros(len(feat))

	for i in range(num):
		if road_item[i] in item_set:
			road_item_cnt[i] = item_cnt[road_item[i]]
			if t_item[i] in item_dict_set[road_item[i]]:
				com_item_cij[i] = sim_item[road_item[i]][t_item[i]]
				t_com_item_cnt[i] = com_item_cnt[road_item[i]][t_item[i]]
				com_item_loc_weights_sum[i] = loc_weights[road_item[i]][t_item[i]]
				com_item_time_weights_sum[i] = time_weights[road_item[i]][t_item[i]]
				com_item_record_weights_sum[i] = record_weights[road_item[i]][t_item[i]]
			else:
				com_item_cij[i] = np.nan
				t_com_item_cnt[i] = np.nan
				com_item_loc_weights_sum[i] = np.nan
				com_item_time_weights_sum[i] = np.nan
				com_item_record_weights_sum[i] = np.nan
		else:
			road_item_cnt[i] = np.nan
			com_item_cij[i] = np.nan
			t_com_item_cnt[i] = np.nan
			com_item_loc_weights_sum[i] = np.nan
			com_item_time_weights_sum[i] = np.nan
			com_item_record_weights_sum[i] = np.nan
		if t_item[i] in item_set:
			t_item_cnt[i] = item_cnt[t_item[i]]
		else:
			t_item_cnt[i] = np.nan

		seq_i2i_sim = []
		for h_item in left_items_list[i] + right_items_list[i]:
			sim_item[h_item].setdefault(t_item[i], 0)
			seq_i2i_sim.append(sim_item[h_item][t_item[i]])
		max_i2i_sim_arr[i] = max(seq_i2i_sim) if len(seq_i2i_sim) > 0 else np.nan
		mean_i2i_sim_arr[i] = max(seq_i2i_sim) / len(seq_i2i_sim) if len(seq_i2i_sim) > 0 else np.nan


	feat['road_item_cnt'] = road_item_cnt
	feat['item_cnt'] = t_item_cnt
	feat['com_item_cij'] = com_item_cij
	feat['com_item_cnt'] = t_com_item_cnt
	feat['com_item_loc_weights_sum'] = com_item_loc_weights_sum
	feat['com_item_time_weights_sum'] = com_item_time_weights_sum
	feat['com_item_record_weights_sum'] = com_item_record_weights_sum
	feat['com_item_loc_weights_mean'] = feat['com_item_loc_weights_sum'] / feat['com_item_cnt']
	feat['com_item_time_weights_mean'] = feat['com_item_time_weights_sum'] / feat['com_item_cnt']
	feat['com_item_record_weights_mean'] = feat['com_item_record_weights_sum'] / feat['com_item_cnt']

	feat['max_i2i_sim_arr'] = max_i2i_sim_arr
	feat['mean_i2i_sim_arr'] = mean_i2i_sim_arr

	feat = feat[[
		'road_item_cnt', 'item_cnt', 'com_item_cij', 'com_item_cnt',
		'com_item_loc_weights_sum', 'com_item_time_weights_sum', 'com_item_record_weights_sum',
		'com_item_loc_weights_mean', 'com_item_time_weights_mean', 'com_item_record_weights_mean',
		'max_i2i_sim_arr', 'mean_i2i_sim_arr'
	]]
	return feat


def feat_diff_type_score_sum_mean(data):
	df = data.copy()
	feat = df[['user', 'item', 'sim_weight', 'recall_type', 'road_item']]
	feat['i2i_score'] = feat['sim_weight']
	feat['blend_score'] = feat['sim_weight']
	feat['i2i2i_score'] = feat['sim_weight']
	feat.loc[feat['recall_type'] != 0, 'blend_score'] = np.nan
	feat.loc[feat['recall_type'] != 1, 'i2i2i_score'] = np.nan
	feat.loc[feat['recall_type'] != 2, 'i2i_score'] = np.nan

	tmp = feat.groupby(['user', 'item'])['i2i_score', 'blend_score', 'i2i2i_score'].agg(['sum', 'mean']).reset_index()
	tmp.columns = ['user', 'item'] + [f'user_item_{i}_{j}' for i in ['i2i_score', 'blend_score', 'i2i2i_score'] for j in ['sum', 'mean']]
	feat = feat.merge(tmp, on=['user', 'item'], how='left')
	tmp = feat.groupby('item')['i2i_score', 'blend_score', 'i2i2i_score'].agg(['sum', 'mean']).reset_index()
	tmp.columns = ['item'] + [f'item_{i}_{j}' for i in ['i2i_score', 'blend_score', 'i2i2i_score'] for j in ['sum', 'mean']]
	feat = feat.merge(tmp, on=['item'], how='left')
	tmp = feat.groupby('road_item')['i2i_score', 'blend_score', 'i2i2i_score'].agg(['sum', 'mean']).reset_index()
	tmp.columns = ['road_item'] + [f'road_item_{i}_{j}' for i in ['i2i_score', 'blend_score', 'i2i2i_score'] for j in ['sum', 'mean']]
	feat = feat.merge(tmp, on=['road_item'], how='left')
	feat = feat[
		[f'user_item_{i}_{j}' for i in ['i2i_score', 'blend_score', 'i2i2i_score'] for j in ['sum', 'mean']] + \
		[f'item_{i}_{j}' for i in ['i2i_score', 'blend_score', 'i2i2i_score'] for j in ['sum', 'mean']] + \
		[f'road_item_{i}_{j}' for i in ['i2i_score', 'blend_score', 'i2i2i_score'] for j in ['sum', 'mean']]
	]
	return feat


def feat_u2i_road_item_time_diff(data):
	df = data.copy()
	feat = df[['user', 'road_item_loc', 'road_item_time']]
	tmp = feat.groupby(['user', 'road_item_loc'], as_index=False).first()
	tmp_group = tmp.sort_values(['user', 'road_item_loc']).set_index(['user', 'road_item_loc']).groupby('user')
	feat1 = tmp_group['road_item_time'].diff(1)
	feat2 = tmp_group['road_item_time'].diff(-1)
	feat1.name = 'u2i_road_item_time_diff_history'
	feat2.name = 'u2i_road_item_time_diff_future'
	tmp = pd.concat([feat1, feat2], axis=1)
	tmp = tmp.reset_index()
	feat = feat.merge(tmp, on=['user', 'road_item_loc'], how='left')
	feat = feat[['u2i_road_item_time_diff_history', 'u2i_road_item_time_diff_future']]
	return feat


def feat_time_window_cate_count(data):
	if mode == 'valid':
		df_train = load_pickle(all_train_data_path.format(cur_stage))
	else:
		df_train = load_pickle(online_all_train_data_path.format(cur_stage))
	df_train = df_train.sort_values(['item_id', 'time'])
	item2times = df_train.groupby('item_id')['time'].agg(list).to_dict()

	df = data.copy()
	feat = df[['item', 'time']]

	def find_count_round_time(row, mode, delta):
		item, t = row['item'], row['time']
		if mode == 'left':
			left = t - delta
			right = t
		elif mode == 'right':
			left = t
			right = t + delta
		else:
			left = t - delta
			right = t + delta
		click_times = item2times[item]
		count = 0
		for ts in click_times:
			if ts < left:
				continue
			elif ts > right:
				break
			else:
				count += 1
		return count

	feat['item_cnt_around_0.01'] = feat.apply(lambda x: find_count_round_time(x, mode='all', delta=0.01), axis=1)
	feat['item_cnt_before_0.01'] = feat.apply(lambda x: find_count_round_time(x, mode='left', delta=0.01), axis=1)
	feat['item_cnt_after_0.01'] = feat.apply(lambda x: find_count_round_time(x, mode='right', delta=0.01), axis=1)
	feat['item_cnt_around_0.02'] = feat.apply(lambda x: find_count_round_time(x, mode='all', delta=0.02), axis=1)
	feat['item_cnt_before_0.02'] = feat.apply(lambda x: find_count_round_time(x, mode='left', delta=0.02), axis=1)
	feat['item_cnt_after_0.02'] = feat.apply(lambda x: find_count_round_time(x, mode='right', delta=0.02), axis=1)
	feat['item_cnt_around_0.05'] = feat.apply(lambda x: find_count_round_time(x, mode='all', delta=0.05), axis=1)
	feat['item_cnt_before_0.05'] = feat.apply(lambda x: find_count_round_time(x, mode='left', delta=0.05), axis=1)
	feat['item_cnt_after_0.05'] = feat.apply(lambda x: find_count_round_time(x, mode='right', delta=0.05), axis=1)

	feat = feat[[f'item_cnt_{i}_{j}' for i in ['around', 'before', 'after'] for j in [0.01, 0.02, 0.05]]]
	return feat


def feat_automl_recall_type_cate_count_plus(data):
	df = data.copy()
	feat = df[['user', 'item', 'road_item', 'recall_type', 'query_item_loc', 'road_item_loc']]
	feat['loc_diff'] = feat['query_item_loc'] - feat['road_item_loc']
	feat['road_item-item'] = feat['road_item'].astype('str') + '-' + feat['item'].astype('str')
	feat_cols = []
	for cate1 in ['recall_type']:
		for cate2 in ['item', 'road_item', 'road_item-item']:
			tmp = feat.groupby([cate1, cate2], as_index=False).size()
			tmp.columns = [cate1, cate2, f'{cate1}_{cate2}_count']
			feat_cols.append(f'{cate1}_{cate2}_count')
			feat = feat.merge(tmp, on=[cate1, cate2], how='left')
	for cate1 in ['loc_diff']:
		for cate2 in ['item', 'road_item', 'road_item-item', 'recall_type']:
			tmp = feat.groupby([cate1, cate2], as_index=False).size()
			tmp.columns = [cate1, cate2, f'{cate1}_{cate2}_count']
			feat_cols.append(f'{cate1}_{cate2}_count')
			feat = feat.merge(tmp, on=[cate1, cate2], how='left')
	for cate1 in ['user']:
		for cate2 in ['recall_type']:
			for cate3 in ['item', 'road_item', 'road_item-item']:
				tmp = feat.groupby([cate1, cate2, cate3], as_index=False).size()
				tmp.columns = [cate1, cate2, cate3, f'{cate1}_{cate2}_{cate3}_count']
				feat_cols.append(f'{cate1}_{cate2}_{cate3}_count')
				feat = feat.merge(tmp, on=[cate1, cate2, cate3], how='left')
	feat = feat[feat_cols]
	return feat


def feat_i2i_cijs_topk_by_loc_plus(data):
	df = data.copy()
	feat = df[['road_item', 'item']]
	feat['new_keys'] = feat.apply(lambda x: (x['road_item'], x['item']), axis=1)
	new_keys = set(feat['new_keys'])

	i2i_sim_seq = {}
	if mode == 'valid':
		df_train = load_pickle(all_train_data_path.format(cur_stage))
	else:
		df_train = load_pickle(online_all_train_data_path.format(cur_stage))
	user_item_dict = df_train.groupby('user_id')['item_id'].agg(list).to_dict()
	user_time_dict = df_train.groupby('user_id')['time'].agg(list).to_dict()
	for user, items in user_item_dict.items():
		times = user_time_dict[user]
		for loc1, item in enumerate(items):
			for loc2, relate_item in enumerate(items):
				if item == relate_item:
					continue
				# in查询优先set, dict(hash), list查询效率慢
				if (item, relate_item) not in new_keys:
					continue
				t1, t2 = times[loc1], times[loc2]
				i2i_sim_seq.setdefault((item, relate_item), [])
				i2i_sim_seq[(item, relate_item)].append((loc1, loc2, t1, t2, len(items)))

	topk = 3
	result = {}
	result_topk_by_loc = {}
	result_history_loc_diff1_cnt = {}
	result_future_loc_diff1_cnt = {}
	result_history_loc_diff1_time_mean = {}
	result_future_loc_diff1_time_mean = {}

	result_1 = {}
	result_median = {}
	result_mean = {}
	result_topk = {}

	for key in new_keys:
		if key not in i2i_sim_seq:
			continue
		result.setdefault(key, [])
		result_history_loc_diff1_cnt.setdefault(key, 0.0)
		result_future_loc_diff1_cnt.setdefault(key, 0.0)
		result_history_loc_diff1_time_mean.setdefault(key, 0)
		result_future_loc_diff1_time_mean.setdefault(key, 0)
		result_1.setdefault(key, [])

		records = i2i_sim_seq[key]
		for record in records:
			loc1, loc2, t1, t2, record_len = record

			if loc1 - loc2 == 1:
				result_history_loc_diff1_cnt[key] += 1
				result_history_loc_diff1_time_mean[key] += (t1 - t2)
			if loc2 - loc1 == 1:
				result_future_loc_diff1_cnt[key] += 1
				result_future_loc_diff1_time_mean[key] += (t2 - t1)
			time_weight = (1 - abs(t1 - t2) * 100)
			time_weight = max(time_weight, 0.2)
			loc_weight = 0.9 ** (abs(loc1 - loc2) - 1)
			loc_weight = max(loc_weight, 0.2)
			loc_diff = abs(loc1-loc2) - 1
			# 1.0为可调整系数
			result[key].append((loc_diff, 1 * 1.0 * loc_weight * time_weight / math.log(1 + record_len)))
			result_1[key].append(1 * 1.0 * loc_weight * time_weight / math.log(1 + record_len))

		result_history_loc_diff1_time_mean[key] /= (result_history_loc_diff1_cnt[key] + 1e-5)
		result_future_loc_diff1_time_mean[key] /= (result_future_loc_diff1_cnt[key] + 1e-5)
		result_one = sorted(result[key], key=lambda x: x[0])
		result_one_len = len(result_one)
		result_topk_by_loc[key] = [x[1] for x in result_one[:topk]] + [np.nan] * max(0, topk - result_one_len)
		result_median[key] = np.median(result_1[key])
		result_mean[key] = np.mean(result_1[key])
		result_one = sorted(result_1[key], reverse=True)
		result_one_len = len(result_one)
		result_topk[key] = result_one[:topk] + [np.nan] * max(0, topk-result_one_len)

	feat['history_loc_diff1_time_mean'] = feat['new_keys'].map(result_history_loc_diff1_time_mean).fillna(0)
	feat['future_loc_diff1_time_mean'] = feat['new_keys'].map(result_future_loc_diff1_time_mean).fillna(0)
	feat['history_loc_diff1_cnt'] = feat['new_keys'].map(result_history_loc_diff1_cnt).fillna(0)
	feat['future_loc_diff1_cnt'] = feat['new_keys'].map(result_future_loc_diff1_cnt).fillna(0)
	feat['i2i_cijs_median'] = feat['new_keys'].map(result_median)
	feat['i2i_cijs_mean'] = feat['new_keys'].map(result_mean)
	feat_top = []
	for key, value in result_topk_by_loc.items():
		feat_top.append([key[0], key[1]] + value)
	feat_top = pd.DataFrame(
		feat_top,
		columns=['road_item', 'item'] + [f'i2i_cijs_top{k}_by_loc' for k in range(1, topk+1)]
	)
	feat = feat.merge(feat_top, on=['road_item', 'item'], how='left')
	feat_top = []
	for key, value in result_topk.items():
		feat_top.append([key[0], key[1]] + value)
	feat_top = pd.DataFrame(
		feat_top,
		columns=['road_item', 'item'] + [f'i2i_cijs_top{k}_by_cij' for k in range(1, topk+1)]
	)
	feat = feat.merge(feat_top, on=['road_item', 'item'], how='left')

	feat = feat[[
		'history_loc_diff1_time_mean', 'future_loc_diff1_time_mean',
		'history_loc_diff1_cnt', 'future_loc_diff1_cnt',
		'i2i_cijs_median', 'i2i_cijs_mean'
		]+[f'i2i_cijs_top{k}_by_loc' for k in range(1, topk+1)]+[f'i2i_cijs_top{k}_by_cij' for k in range(1, topk+1)]]
	return feat


def item_cnt_in_stage2(data):
	df = data.copy()
	feat = df[['item', 'stage']]
	tmp = feat.groupby(['item', 'stage'], as_index=False).size()
	tmp.columns = ['item', 'stage', 'item_stage_cnt']
	feat = feat.merge(tmp, on=['item', 'stage'], how='left')
	feat = feat[['item_stage_cnt']]
	return feat


def feat_item_cnt_in_different_stage_plus(data):
	if mode == 'valid':
		df_train_stage = load_pickle(all_train_stage_data_path.foramt(cur_stage))
	else:
		df_train_stage = load_pickle(online_all_train_stage_data_path.format(cur_stage))

	df = data.copy()
	feat = df[['item', 'stage', 'user']]

	for stage in range(cur_stage+1):
		tmp = df_train_stage[df_train_stage['stage'] == stage].groupby('item_id', as_index=False).size()
		tmp.columns = ['item'] + ['item_stage_cnt_{}'.format(stage)]
		feat = feat.merge(tmp, on='item', how='left')

	tmp = df_train_stage.groupby('item_id', as_index=False)['stage'].nunique()
	tmp.columns = ['item', 'item_stage_nunique']
	feat = feat.merge(tmp, on='item', how='left')

	tmp = df_train_stage.groupby(['item_id', 'stage'], as_index=False).size()
	tmp.columns = ['item', 'stage', 'item_stage_cnt']
	feat = feat.merge(tmp, on=['item', 'stage'], how='left')
	tmp = feat.groupby('user')['item_stage_cnt'].agg(['mean', 'min', 'max']).reset_index()
	tmp.columns = ['user'] + [f'user_item_stage_cnt_{i}' for i in ['mean', 'min', 'max']]
	feat = feat.merge(tmp, on='user', how='left')

	feat_cols = [f'item_stage_cnt_{stage}' for stage in range(cur_stage+1)] + ['item_stage_nunique'] + \
				[f'user_item_stage_cnt_{i}' for i in ['mean', 'min', 'max']]
	feat = feat[feat_cols]
	return feat


def feat_i2i2i_sim(data):
	df = data.copy()
	feat = df[['road_item', 'item']]
	feat['new_keys'] = feat.apply(lambda x: (x['road_item'], x['item']), axis=1)
	new_keys = set(feat['new_keys'])

	if mode == 'valid':
		df_train = load_pickle(all_train_data_path.format(cur_stage))
	else:
		df_train = load_pickle(online_all_train_data_path.format(cur_stage))
	user_item_dict = df_train.groupby('user_id')['item_id'].agg(list).to_dict()
	user_time_dict = df_train.groupby('user_id')['time'].agg(list).to_dict()

	sim_item_p1 = {}
	sim_item_p2 = {}
	item_cnt = defaultdict(int)
	for user, items in user_item_dict.items():
		times = user_time_dict[user]

		for loc1, item in enumerate(items):
			item_cnt[item] += 1
			sim_item_p2.setdefault(item, {})
			for loc2, relate_item in enumerate(items):
				if item == relate_item:
					continue

				t1, t2 = times[loc1], times[loc2]
				sim_item_p2[item].setdefault(relate_item, 0)

				time_weight = (1 - abs(t1-t2)*100)
				time_weight = max(time_weight, 0.2)
				loc_weight = 0.9 ** (abs(loc1-loc2)-1)
				loc_weight = max(loc_weight, 0.2)
				# 1.0为可调整系数
				sim_item_p2[item][relate_item] += 1 * 1.0 * loc_weight * time_weight / math.log(1+len(items))

	for i, related_items in sim_item_p2.items():
		sim_item_p1[i] = {}
		for j, cij in related_items.items():
			sim_item_p2[i][j] = cij / ((item_cnt[i]*item_cnt[j]) ** 0.2)
			sim_item_p1[i][j] = cij / ((item_cnt[i] * item_cnt[j]))

	for item in sim_item_p2.keys():
		related_items = sim_item_p2[item]
		related_items = sorted(related_items.items(), key=lambda x: x[1], reverse=True)
		rel = {}
		for x in related_items[:50]:
			rel[x[0]] = x[1]
		sim_item_p2[item] = rel

	i2i2i_sim_seq = {}
	for item1 in sim_item_p2.keys():
		for item2 in sim_item_p2[item1].keys():
			if item1 == item2:
				continue
			for item3 in sim_item_p2[item2].keys():
				if item3 == item1 or item3 == item2:
					continue
				if item3 in sim_item_p2[item1]:
					continue
				if (item1, item3) not in new_keys:
					continue
				i2i2i_sim_seq.setdefault((item1, item3), [])
				i2i2i_sim_seq[(item1, item3)].append((
					item2, sim_item_p2[item1][item2], sim_item_p2[item2][item3],
					sim_item_p1[item1][item2], sim_item_p1[item2][item3]
				))

	new_keys = list(new_keys)
	result = np.zeros((len(new_keys), 4))
	for i in range(len(new_keys)):
		key = new_keys[i]
		if key not in i2i2i_sim_seq:
			continue
		records = i2i2i_sim_seq[key]
		result[i, 0] = len(records)
		for record in records:
			item, score1, score2, score3, score4 = record
			result[i, 1] += score1 * score2
			result[i, 2] += score3 * score4
			result[i, 3] += item_cnt[item]

	result[:, 1] /= (result[:, 0] + 1e-9)
	result[:, 2] /= (result[:, 0] + 1e-9)
	result[:, 3] /= (result[:, 0] + 1e-9)

	feat_cols = ['i2i2i_road_cnt', 'i2i2i_score1_mean', 'i2i2i_score2_mean', 'i2i2i_middle_item_cnt_mean']
	result = pd.DataFrame(result, index=new_keys, columns=feat_cols)
	result = result.reset_index()
	result.columns = ['new_keys'] + feat_cols
	feat = feat.merge(result, on='new_keys', how='left')
	feat = feat[feat_cols]
	return feat


def feat_item_qtime_time_diff_plus(data):
	if mode == 'valid':
		df_train = load_pickle(all_train_data_path.format(cur_stage))
	else:
		df_train = load_pickle(online_all_train_data_path.format(cur_stage))
	df_train = df_train.sort_values(['item_id', 'time'])
	item2times = df_train.groupby('item_id')['time'].agg(list).to_dict()
	item_count = df_train['item_id'].value_counts()

	df = data.copy()
	feat = df[['item', 'query_item_time']]
	feat_v = feat.values
	result_history = np.zeros(feat.shape[0]) * np.nan
	result_future = np.zeros(feat.shape[0]) * np.nan
	result = np.zeros(feat.shape[0])
	for i in range(feat.shape[0]):
		item = feat_v[i, 0]
		time = feat_v[i, 1]
		time_list = [0] + item2times[item] + [1]
		for j in range(1, len(time_list)):
			if time < time_list[j]:
				result_history[i] = time - time_list[j-1]
				result_future[i] = time_list[j] - time
				result[i] = j-1
				break

	feat['item_qtime_time_diff_history'] = result_history
	feat['item_qtime_time_diff_future'] = result_future
	feat['item_cumcount'] = result
	# 错过的比例
	feat['item_cumrate'] = feat['item_cumcount'] / (feat['item'].map(item_count)).fillna(1e-5)
	feat = feat[[
		'item_qtime_time_diff_history', 'item_qtime_time_diff_future',
		'item_cumcount', 'item_cumrate'
	]]
	return feat


def feat_road_time_bins_cate_cnt(data):
	df = data.copy()
	cates = ['item', 'road_item', 'user', 'recall_type']
	feat = df[cates + ['road_item_time']]
	feat['loc_diff'] = df['query_item_loc'] - df['road_item_loc']
	cates.append('loc_diff')
	feat['road_time_bins'] = pd.Categorical(pd.cut(feat['road_item_time'], 100)).codes

	feat_cols = []
	for cate in cates:
		tmp = feat.groupby([cate, 'road_time_bins'], as_index=False).size()
		tmp.columns = [cate, 'road_time_bins'] + [f'{cate}_cnt_by_road_time_bins']
		feat = feat.merge(tmp, on=[cate, 'road_time_bins'], how='left')
		feat_cols.append(f'{cate}_cnt_by_road_time_bins')
	feat = feat[feat_cols]
	return feat


def feat_u2i_road_item_before_and_after_query_time_diff(data):
	df = data.copy()
	feat = df[['user', 'road_item_loc', 'road_item_time', 'query_item_time']]
	feat_h = df.loc[feat['road_item_time'] < feat['query_item_time']]
	feat_f = df.loc[feat['road_item_time'] > feat['query_item_time']]
	feat_h = feat_h.groupby(['user', 'road_item_loc'], as_index=False).first()
	feat_f = feat_f.groupby(['user', 'road_item_loc'], as_index=False).first()
	feat_h_group = feat_h.sort_values(['user', 'road_item_loc']).set_index(['user', 'road_item_loc']).groupby('user')
	feat_f_group = feat_f.sort_values(['user', 'road_item_loc']).set_index(['user', 'road_item_loc']).groupby('user')

	feat1 = feat_h_group['road_item_time'].diff(1)
	feat2 = feat_h_group['road_item_time'].diff(-1)
	feat3 = feat_f_group['road_item_time'].diff(1)
	feat4 = feat_f_group['road_item_time'].diff(-1)
	feat1.name = 'u2i_road_item_before_query_time_diff_history'
	feat2.name = 'u2i_road_item_before_query_time_diff_future'
	feat3.name = 'u2i_road_item_after_query_time_diff_history'
	feat4.name = 'u2i_road_item_after_query_time_diff_future'
	tmp = pd.concat([feat1, feat2, feat3, feat4], axis=1)

	feat = feat.merge(tmp, on=['user', 'road_item_loc'], how='left')
	feat = feat[[
		'u2i_road_item_before_query_time_diff_history',
		'u2i_road_item_before_query_time_diff_future',
		'u2i_road_item_after_query_time_diff_history',
		'u2i_road_item_after_query_time_diff_future'
	]]
	return feat


def feat_item_seq_sim_cossim_text(data):
	df = data.copy()
	feat = df[['left_items_list', 'right_items_list', 'item']]

	item_feat = load_pickle(item_feat_pkl)
	item_np = np.zeros((120000, 128))
	for k, v in item_feat.items():
		item_np[k, :] = v[0]
	item_np = item_np/(np.linalg.norm(item_np, axis=1, keepdims=True) + 1e-9	)
	all_items = np.array(sorted(item_feat.keys()))

	batch_size = 100
	n = len(feat)
	batch_num = n // batch_size if n % batch_size == 0 else n // batch_size + 1

	feat['left_len'] = feat['left_items_list'].apply(len)
	feat_left = feat.sort_values('left_len')
	feat_left_len = feat_left['left_len'].values
	feat_left_items_list = feat_left['left_items_list'].values
	feat_left_items = feat_left['item'].values
	left_result = np.zeros((len(feat_left), 2))
	left_result_len = np.zeros(len(feat_left))

	for i in range(batch_num):
		cur_batch_size = len(feat_left_len[i*batch_size:(i+1)*batch_size])
		max_len = feat_left_len[i*batch_size:(i+1)*batch_size].max()
		max_len = max(max_len, 1)
		left_items = np.zeros((cur_batch_size, max_len), dtype='int32')
		for j, arr in enumerate(feat_left_items_list[i*batch_size:(i+1)*batch_size]):
			left_items[j, :len(arr)] = arr

		left_result_len[i*batch_size:(i+1)*batch_size] = np.isin(left_items, all_items).sum(axis=1)
		vec1 = item_np[left_items]
		vec2 = item_np[feat_left_items[i*batch_size:(i+1)*batch_size]]
		vec2 = vec2.reshape(-1, 1, 128)
		sim = np.sum(vec1 * vec2, axis=-1)
		left_result[i*batch_size:(i+1)*batch_size, 0] = sim.max(axis=1)
		left_result[i*batch_size:(i+1)*batch_size, 1] = sim.mean(axis=1)

	df_left = pd.DataFrame(
		left_result,
		index=feat_left.index,
		columns=['left_allitem_item_textsim_max','left_allitem_item_textsim_sum']
	)
	df_left['left_allitem_textsim_len'] = left_result_len

	feat['right_len'] = feat['right_items_list'].apply(len)
	feat_right = feat.sort_values('right_len')
	feat_right_len = feat_right['right_len'].values
	feat_right_items_list = feat_right['right_items_list'].values
	feat_right_items = feat_right['item'].values
	right_result = np.zeros((len(feat_right), 2))
	right_result_len = np.zeros(len(feat_right))

	for i in range(batch_num):
		cur_batch_size = len(feat_right_len[i * batch_size:(i + 1) * batch_size])
		max_len = feat_right_len[i * batch_size:(i + 1) * batch_size].max()
		max_len = max(max_len, 1)
		right_items = np.zeros((cur_batch_size, max_len), dtype='int32')
		for j, arr in enumerate(feat_right_items_list[i * batch_size:(i + 1) * batch_size]):
			right_items[j, :len(arr)] = arr

		right_result_len[i * batch_size:(i + 1) * batch_size] = np.isin(right_items, all_items).sum(axis=1)
		vec1 = item_np[right_items]
		vec2 = item_np[feat_right_items[i * batch_size:(i + 1) * batch_size]]
		vec2 = vec2.reshape(-1, 1, 128)
		sim = np.sum(vec1 * vec2, axis=-1)
		right_result[i * batch_size:(i + 1) * batch_size, 0] = sim.max(axis=1)
		right_result[i * batch_size:(i + 1) * batch_size, 1] = sim.mean(axis=1)

	df_right = pd.DataFrame(
		right_result,
		index=feat_right.index,
		columns=['right_allitem_item_textsim_max', 'right_allitem_item_textsim_sum']
	)
	df_right['right_allitem_textsim_len'] = right_result_len

	feat = pd.concat([df_left, df_right], axis=1)
	feat['allitem_item_textsim_max'] = feat[['left_allitem_item_textsim_max', 'right_allitem_item_textsim_max']].max(axis=1)
	feat['allitem_item_textsim_sum'] = feat[['left_allitem_item_textsim_sum', 'right_allitem_item_textsim_sum']].sum(axis=1)
	feat['allitem_item_textsim_len'] = feat[['left_allitem_textsim_len', 'right_allitem_textsim_len']].sum(axis=1)
	feat['allitem_item_textsim_mean'] = feat['allitem_item_textsim_sum'] / (feat['allitem_item_textsim_len'] + 1e-9)
	feat = feat[['allitem_item_textsim_max','allitem_item_textsim_mean']]
	return feat


def feat_item_seq_sim_cossim_image(data):
	df = data.copy()
	feat = df[['left_items_list', 'right_items_list', 'item']]

	item_feat = load_pickle(item_feat_pkl)
	item_np = np.zeros((120000, 128))
	for k, v in item_feat.items():
		item_np[k, :] = v[1]
	item_np = item_np / (np.linalg.norm(item_np, axis=1, keepdims=True) + 1e-9)
	all_items = np.array(sorted(item_feat.keys()))

	batch_size = 100
	n = len(feat)
	batch_num = n // batch_size if n % batch_size == 0 else n // batch_size + 1

	feat['left_len'] = feat['left_items_list'].apply(len)
	feat_left = feat.sort_values('left_len')
	feat_left_len = feat_left['left_len'].values
	feat_left_items_list = feat_left['left_items_list'].values
	feat_left_items = feat_left['item'].values
	left_result = np.zeros((len(feat_left), 2))
	left_result_len = np.zeros(len(feat_left))

	for i in range(batch_num):
		cur_batch_size = len(feat_left_len[i * batch_size:(i + 1) * batch_size])
		max_len = feat_left_len[i * batch_size:(i + 1) * batch_size].max()
		max_len = max(max_len, 1)
		left_items = np.zeros((cur_batch_size, max_len), dtype='int32')
		for j, arr in enumerate(feat_left_items_list[i * batch_size:(i + 1) * batch_size]):
			left_items[j, :len(arr)] = arr

		left_result_len[i * batch_size:(i + 1) * batch_size] = np.isin(left_items, all_items).sum(axis=1)
		vec1 = item_np[left_items]
		vec2 = item_np[feat_left_items[i * batch_size:(i + 1) * batch_size]]
		vec2 = vec2.reshape(-1, 1, 128)
		sim = np.sum(vec1 * vec2, axis=-1)
		left_result[i * batch_size:(i + 1) * batch_size, 0] = sim.max(axis=1)
		left_result[i * batch_size:(i + 1) * batch_size, 1] = sim.mean(axis=1)

	df_left = pd.DataFrame(
		left_result,
		index=feat_left.index,
		columns=['left_allitem_item_imagesim_max', 'left_allitem_item_imagesim_sum']
	)
	df_left['left_allitem_textsim_len'] = left_result_len

	feat['right_len'] = feat['right_items_list'].apply(len)
	feat_right = feat.sort_values('right_len')
	feat_right_len = feat_right['right_len'].values
	feat_right_items_list = feat_right['right_items_list'].values
	feat_right_items = feat_right['item'].values
	right_result = np.zeros((len(feat_right), 2))
	right_result_len = np.zeros(len(feat_right))

	for i in range(batch_num):
		cur_batch_size = len(feat_right_len[i * batch_size:(i + 1) * batch_size])
		max_len = feat_right_len[i * batch_size:(i + 1) * batch_size].max()
		max_len = max(max_len, 1)
		right_items = np.zeros((cur_batch_size, max_len), dtype='int32')
		for j, arr in enumerate(feat_right_items_list[i * batch_size:(i + 1) * batch_size]):
			right_items[j, :len(arr)] = arr

		right_result_len[i * batch_size:(i + 1) * batch_size] = np.isin(right_items, all_items).sum(axis=1)
		vec1 = item_np[right_items]
		vec2 = item_np[feat_right_items[i * batch_size:(i + 1) * batch_size]]
		vec2 = vec2.reshape(-1, 1, 128)
		sim = np.sum(vec1 * vec2, axis=-1)
		right_result[i * batch_size:(i + 1) * batch_size, 0] = sim.max(axis=1)
		right_result[i * batch_size:(i + 1) * batch_size, 1] = sim.mean(axis=1)

	df_right = pd.DataFrame(
		right_result,
		index=feat_right.index,
		columns=['right_allitem_item_imagesim_max', 'right_allitem_item_imagesim_sum']
	)
	df_right['right_allitem_imagesim_len'] = right_result_len

	feat = pd.concat([df_left, df_right], axis=1)
	feat['allitem_item_imagesim_max'] = feat[['left_allitem_item_imagesim_max', 'right_allitem_item_imagesim_max']].max(
		axis=1)
	feat['allitem_item_imagesim_sum'] = feat[['left_allitem_item_imagesim_sum', 'right_allitem_item_imagesim_sum']].sum(
		axis=1)
	feat['allitem_item_imagesim_len'] = feat[['left_allitem_imagesim_len', 'right_allitem_imagesim_len']].sum(axis=1)
	feat['allitem_item_imagesim_mean'] = feat['allitem_item_imagesim_sum'] / (feat['allitem_item_imagesim_len'] + 1e-9)
	feat = feat[['allitem_item_imagesim_max', 'allitem_item_imagesim_mean']]
	return feat


if __name__ == '__main__':
	good_funcs = [
		# user的recall_item 四个 weight 的 sum mean
		feat_user_item_weight_sum_mean,
		# user的recall_item loc_time_weight sum
		feat_weight_sum,
		# road_item, recall_item text_feat_cossim,
		# road_item, recall_item text_feat向量点积
		# road_item, recall_item text_feat向量模积
		# road_item text_feat 的模
		feat_road_item_text_cossim_plus,
		# road_item, recall_item text_feat向量差的模(距离)
		feat_road_item_text_eulasim,
		# road_item, recall_item cnt(df_train)
		# road_item, recall_item sim, pair_cnt(df_train)
		# road_item, recall_item loc_weight_sum, time_weight_sum, itme_list_len_sum(df_train)
		# road_item, recall_item loc_weight_mean, time_weight_mean, itme_list_len_mean(/road_item, recall_item cnt)(df_train)
		# left_right_item_list 中和 recall_item 的 sim_max, sim_mean(df_train)
		feat_sim_base_plus,
		# user的recall_item 每种recall_sim 的 sum mean(df_train)
		# recall_item 每种recall_sim 的 sum mean(df_train)
		# road_item 每种recall_sim 的 sum mean(df_train)
		feat_diff_type_score_sum_mean,
		# user的road_item_time_diff
		feat_u2i_road_item_time_diff,
		# qtime的一定范围内(left,right,all)recall_item多少次被点击(df_train)
		feat_time_window_cate_count,
		# cnt feature
		feat_automl_recall_type_cate_count_plus,
		# road_item recall_item history/future time diff mean(df_train)
		# road_item recall_item history/future cnt(df_train)
		# road_item recall_item sim median mean(df_train)
		# road_item recall_item topk(loc_diff) sim(df_train)
		feat_i2i_cijs_topk_by_loc_plus,
		# recall_item stage cnt
		item_cnt_in_stage2,
		# recall_item stage cnt(df_train)
		# recall_item stage nunique(df_train)
		# user recall_item_stage_cnt min mean max(df_train)
		feat_item_cnt_in_different_stage_plus,
		# road_item recall_item i2i2i new_item cnt, sim scores mean, list_len_mean(df_train)
		feat_i2i2i_sim,
		# qtime 在 recall_item_time_list before/after time diff(df_train)
		# recall_item_time_list miss rate(df_train)
		feat_item_qtime_time_diff_plus,
		# road_item_time_bins cnt
		feat_road_time_bins_cate_cnt,
		# user road_item_time before/after(qtime) diff
		feat_u2i_road_item_before_and_after_query_time_diff,
		# recall_item left_right_item_list text_sim mean sum
		feat_item_seq_sim_cossim_text,
		# recall_item left_right_item_list pic_sim mean sum
		feat_item_seq_sim_cossim_image
	]
	data = load_pickle(lgb_base_pkl.format(recall_file_names, mode, cur_stage))
	data = data.reset_index(drop=True)
	print(data.shape)

	for func in tqdm(good_funcs, desc='feature'):
		feat_path = os.path.join(feat_dir, func.__name__+'.pkl')
		if os.path.exists(feat_path):
			continue
		feat = func(data)
		dump_pickle(feat, feat_path)

	feat_paths = [os.path.join(feat_dir, func.__name__+'.pkl') for func in good_funcs]
	feat_list = [data]
	for feat_path in tqdm(feat_paths, desc='feature_merge'):
		feat = load_pickle(feat_path)
		feat_list.append(feat)
	lgb_model_data = pd.concat(feat_list, axis=1).reset_index(drop=True)
	lgb_model_data_path = os.path.join(feat_dir, 'lgb_model_data.pkl')
	dump_pickle(lgb_model_data, lgb_model_data_path)
	print(data.shape)

