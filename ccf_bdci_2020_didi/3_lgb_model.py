from utils import *


if __name__ == '__main__':
	# ------数据集整理
	# 为什么从20190705开始？
	get_his_sample(20190705)
	train_path = 'data/sample_data.csv'
	test_path = 'data/20190801_testdata.txt'
	topo_path = 'data/topo.txt'

	tmp_train = day_origin_df(train_path)
	tmp_test = day_origin_df(test_path)
	# 按照测试集对训练集采样
	supp_train = sample_train_data(tmp_train, tmp_test)

	train_df = pd.read_csv(train_path, header=None, sep=';')
	test = pd.read_csv(test_path, header=None, sep=';')
	# test = test.sample(frac=0.1, random_state=1)
	test['day'] = 20190801
	train_df = load_traffic_data(train_df)
	test = load_traffic_data(test)
	train_df = supp_train.merge(train_df, on=['linkid', 'future_slice_id', 'current_slice_id'], how='left')
	data = pd.concat([train_df, test], axis=0, ignore_index=True)

	topo = pd.read_csv(topo_path, sep='\t', header=None, names=['linkid', 'target_link_list'])
	ups = get_ups(topo_path)
	topo['down_cnt'] = topo['target_link_list'].apply(lambda x: x.count(',')+1)
	ups['up_cnt'] = ups['target_link_list'].apply(lambda x: x.count(',')+1)

	if os.path.exists('data/his_df.csv'):
		his_df = pd.read_csv('data/his_df.csv')
	else:
		his_df = []
		for d in tqdm(range(20190701, 20190731), desc='his_df'):
			tmp_file = pd.read_csv('data/traffic/{}.txt'.format(d), header=None, sep=';')
			# tmp_file = tmp_file.sample(frac=0.2, random_state=1)
			tmp_file.columns = [
				0, 'recent_feature', 'history_feature_28', 'history_feature_21',
				'history_feature_14', 'history_feature_7']
			tmp_file['day'] = d
			tmp_file['linkid'] = tmp_file[0].apply(lambda x: int(x.split(' ')[0]))
			tmp_file['label'] = tmp_file[0].apply(lambda x: int(x.split(' ')[1]))
			tmp_file['label'] = tmp_file['label'].apply(lambda x: 3 if x > 3 else x)
			tmp_file['current_slice_id'] = tmp_file[0].apply(lambda x: int(x.split(' ')[2]))
			tmp_file['future_slice_id'] = tmp_file[0].apply(lambda x: int(x.split(' ')[3]))
			tmp_file['weekday'] = pd.to_datetime(tmp_file['day'].astype(str)).dt.weekday + 1
			tmp_file['hour'] = tmp_file['future_slice_id'].apply(lambda x: x // 30)
			tmp_file = tmp_file.drop(
				[0, 'recent_feature', 'history_feature_28', 'history_feature_21', 'history_feature_14', 'history_feature_7'],
				axis=1
			)
			his_df.append(tmp_file)
		his_df = pd.concat(his_df, axis=0, ignore_index=True)
		his_df.to_csv('data/his_df.csv', index=False)

	# ------特征
	# 连续特征特征列表的统计特征
	numeric_list = [
		'recent_speed', 'recent_eta', 'recent_vichles_num',
		'history_speed_28', 'history_speed_21', 'history_speed_14', 'history_speed_7',
		'history_eta_28', 'history_eta_21', 'history_eta_14', 'history_eta_7',
		'history_vichles_num_28', 'history_vichles_num_21', 'history_vichles_num_14', 'history_vichles_num_7']
	for col in tqdm(numeric_list, desc='numeric_list'):
		data[col + '_mean'] = data[col].apply(lambda x: np.mean(x))
		data[col + '_max'] = data[col].apply(lambda x: np.max(x))
		data[col + '_min'] = data[col].apply(lambda x: np.min(x))
		data[col + '_std'] = data[col].apply(lambda x: np.std(x))
		data[col + '_median'] = data[col].apply(lambda x: np.median(x))
	status_list = ['recent_status', 'history_status_28', 'history_status_21', 'history_status_14', 'history_status_7']
	# status是分类特征，也可以看成连续递增的强度特征
	for col in tqdm(numeric_list, desc='status_list'):
		data[col + '_mean'] = data[col].apply(lambda x: np.mean(x))
		data[col + '_max'] = data[col].apply(lambda x: np.max(x))
		data[col + '_std'] = data[col].apply(lambda x: np.std(x))
		for i in range(0, 5):
			data[col + '_bef{}'.format(i+1)] = data[col].apply(lambda x: x[i])
		for i in range(0, 5):
			data[col + '_{}_cnt'.format(i)] = data[col].apply(lambda x: x.count(i))

	# label特征
	link_ctr = []
	for d in tqdm(list(range(20190702, 20190731)) + [20190801], desc='link_ctr'):
		link_ctr.append(gen_ctr_features(d, ['linkid'], his_df))
	link_ctr = pd.concat(link_ctr, axis=0, ignore_index=True)
	data = data.merge(link_ctr, on=['linkid', 'day'], how='left')

	future_slice_ctr = []
	for d in tqdm(list(range(20190702, 20190731)) + [20190801], desc='future_slice_ctr'):
		future_slice_ctr.append(gen_ctr_features(d, ['future_slice_id'], his_df))
	future_slice_ctr = pd.concat(future_slice_ctr, axis=0, ignore_index=True)
	data = data.merge(future_slice_ctr, on=['future_slice_ctr', 'day'], how='left')

	link_hour_ctr = []
	for d in tqdm(list(range(20190702, 20190731)) + [20190801], desc='link_hour_ctr'):
		link_hour_ctr.append(gen_ctr_features(d, ['linkid', 'hour'], his_df))
	link_hour_ctr = pd.concat(link_hour_ctr, axis=0, ignore_index=True)
	data = data.merge(link_hour_ctr, on=['linkid', 'hour', 'day'], how='left')

	link_hour_weekday_ctr = []
	for d in tqdm(list(range(20190702, 20190731)) + [20190801], desc='link_hour_weekday_ctr'):
		link_hour_weekday_ctr.append(gen_ctr_features(d, ['linkid', 'hour', 'weekday'], his_df))
	link_hour_weekday_ctr = pd.concat(link_hour_weekday_ctr, axis=0, ignore_index=True)
	data = data.merge(link_hour_weekday_ctr, on=['linkid', 'hour', 'day', 'weekday'], how='left')

	# label相关time_slice特征
	data['label_0_max_slice'] = data['future_slice_id'] - data.apply(
		lambda x: gen_label_timedelta(x['recent_slices'], x['recent_status'], [1]), axis=1)
	data['label_1_max_slice'] = data['future_slice_id'] - data.apply(
		lambda x: gen_label_timedelta(x['recent_slices'], x['recent_status'], [2]), axis=1)
	data['label_2_max_slice'] = data['future_slice_id'] - data.apply(
		lambda x: gen_label_timedelta(x['recent_slices'], x['recent_status'], [3, 4]), axis=1)
	data['label_0_min_slice'] = data['future_slice_id'] - data.apply(
		lambda x: gen_label_timedelta_min(x['recent_slices'], x['recent_status'], [1]), axis=1)
	data['label_1_min_slice'] = data['future_slice_id'] - data.apply(
		lambda x: gen_label_timedelta_min(x['recent_slices'], x['recent_status'], [2]), axis=1)
	data['label_2_min_slice'] = data['future_slice_id'] - data.apply(
		lambda x: gen_label_timedelta_min(x['recent_slices'], x['recent_status'], [3, 4]), axis=1)
	data['label_0_diff_slice'] = data['future_slice_id'] - data.apply(
		lambda x: gen_label_timedelta_diff(x['recent_slices'], x['recent_status'], [1]), axis=1)
	data['label_1_diff_slice'] = data['future_slice_id'] - data.apply(
		lambda x: gen_label_timedelta_diff(x['recent_slices'], x['recent_status'], [2]), axis=1)
	data['label_2_diff_slice'] = data['future_slice_id'] - data.apply(
		lambda x: gen_label_timedelta_diff(x['recent_slices'], x['recent_status'], [3, 4]), axis=1)

	# topo特征
	train = data[data['day'] <= 20190730]
	test = data[data['day'] == 20190801]
	train['label'] = train['label'] - 1
	drop_cols = numeric_list + status_list +\
			['recent_feature', 'history_feature_28', 'history_feature_21', 'history_feature_14', 'history_feature_7', 'recent_slices']
	train = train.drop(drop_cols, axis=1)
	test = test.drop(drop_cols, axis=1)

	down_df_trn = get_topo_info(train, topo, slices=15, mode='down')
	up_df_trn = get_topo_info(train, ups, slices=15, mode='up')
	down_df_tst = get_topo_info(test, topo, slices=15, mode='down')
	up_df_tst = get_topo_info(test, ups, slices=15, mode='up')
	train = train.merge(down_df_trn, on=['linkid', 'future_slice_id'], how='left')
	train = train.merge(up_df_trn, on=['linkid', 'future_slice_id'], how='left')
	test = test.merge(down_df_tst, on=['linkid', 'future_slice_id'], how='left')
	test = test.merge(up_df_tst, on=['linkid', 'future_slice_id'], how='left')

	# ------建模
	lgb_preds, lgb_oof, lgb_score = run_lgb(train, test, 'label', 5)
	lgb_sub = test[['linkid', 'current_slice_id', 'future_slice_id']].copy()
	lgb_sub = lgb_sub.rename(columns={'linkid': 'link'})
	for i in range(3):
		lgb_sub['lgb_pred_{}'.format(i)] = np.mean(lgb_preds, axis=0)[:, i]
	lgb_sub.to_csv('data/lgb_sub_prob.csv', index=False)
























