# https://zhuanlan.zhihu.com/p/191595907
# https://github.com/aister2020/KDDCUP_2020_Debiasing_1st_Place
# https://tianchi.aliyun.com/competition/entrance/231785/information
# https://zhuanlan.zhihu.com/p/149424540


import os
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import math


data_dir = 'data/'
item_feat_csv = data_dir + 'train/underexpose_item_feat.csv'
user_feat_csv = data_dir + 'train/underexpose_user_feat.csv'


pkl_dir = 'user_data/pkl_data/'
if not os.path.exists(pkl_dir):
	os.makedirs(pkl_dir)
item_feat_pkl = pkl_dir + 'item_feat.pkl'
user_feat_pkl = pkl_dir + 'user_feat.pkl'
item_blend_sim_path = pkl_dir + 'item_blend_sim.text'
train_item_click_csv = data_dir + 'train/underexpose_train_click-{}.csv'
test_item_click_csv = data_dir + 'test/underexpose_test_click-{}.csv'
test_qtime_csv = data_dir + 'test/underexpose_test_qtime-{}.csv'

all_train_data_path = pkl_dir + 'all_train_data_{}.pkl'
all_train_stage_data_path = pkl_dir + 'all_train_stage_data_{}.pkl'
all_valid_data_path = pkl_dir + 'all_valid_data_{}.pkl'
all_valid_stage_data_path = pkl_dir + 'all_valid_stage_data_{}.pkl'
all_test_data_path = pkl_dir + 'all_test_data_{}.pkl'

online_all_train_data_path = pkl_dir + 'online_all_train_data_{}.pkl'
online_all_train_stage_data_path = pkl_dir + 'online_all_train_stage_data_{}.pkl'
online_all_test_data_path = pkl_dir + 'online_all_test_data_{}.pkl'

full_item_degree_path = pkl_dir + 'full_item_degree_{}.pkl'
phase_full_item_degree_path = pkl_dir + 'phase_full_item_degree_{}.pkl'

item2time_path = pkl_dir + 'item2time_{}_{}.pkl'
item2times_path = pkl_dir + 'item2times_{}_{}_{}.pkl'
item_pair2time_diff_path = pkl_dir + 'item_pair2times_{}_{}.pkl'
item_pair2time_seq_path = pkl_dir + 'item_pair2time_seq_{}_{}.pkl'


CUR_STAGE = 9
using_last_num = 'one'
online = 'offline' # ['online', 'offline']
cur_mode = 'valid' # ['valid', 'test']


big_or_small_or_history = 'small'
if big_or_small_or_history == 'small':
	i2i_sim_limit = 100
	b2b_sim_limit = 100
	i2i2i_sim_limit = 50
	i2i2i_new_sim_limit = 50
	b2b2b_sim_limit = 50
	i2i2b_i_sim_limit = 50
	i2i2b_b_sim_limit = 50
	b2b2i_i_sim_limit = 50
	b2b2i_b_sim_limit = 50
get_sum = False # 同一种召回方法的话同一个item是否合并
sim_item_path = pkl_dir + 'sim_item_{}_{}.pkl'
test_sim_item_path = pkl_dir + 'test_sim_item_{}.pkl'
i2i_w02_recall_scoure_path = pkl_dir + 'i2i_w02_recall_source_{}_{}_{}.pkl'
answer_source_path = pkl_dir + 'answer_source_{}_{}.pkl'
i2i_w10_recall_scoure_path = pkl_dir + 'i2i_w10_recall_source_{}_{}_{}.pkl'
b2b_recall_scoure_path = pkl_dir + 'b2b_recall_source_{}_{}_{}.pkl'
i2i2i_new_recall_scoure_path = pkl_dir + 'i2i2i_new_recall_source_{}_{}_{}.pkl'

sum_mode = 'nosum'
lgb_base_pkl = pkl_dir + 'lgb_base_{}_{}_{}.pkl'
cur_recall_source_names = ['b2b', 'i2i2i_new', 'i2i_w10']

feat_dir = 'user_data/feat_data/'
if not os.path.exists(feat_dir):
	os.makedirs(feat_dir)

feat_imp_dir = 'user_data/feat_imp/'
if not os.path.exists(feat_imp_dir):
	os.makedirs(feat_imp_dir)


def dump_pickle(obj, file_path):
	return pickle.dump(obj, open(file_path, 'wb'))


def load_pickle(file_path):
	return pickle.load(open(file_path, 'rb'))


def write_sim(sim, file_path):
	with open(file_path, 'w') as f:
		for sim_item in sim:
			src = sim_item[0]
			tgt = sim_item[1]
			for i, (item, score) in enumerate(tgt):
				tgt[i] = str(item) + ',' + str(score)
			text = str(src) + ' ' + ' '.join(tgt) + '\n'
			f.write(text)


def load_sim(file_path):
	"""
	[(item, [(sim_item, sim_score), ...]), ...]
	"""
	sim_item = []
	with open(file_path, 'r') as f:
		for line in f:
			line = line.strip('\n')
			line = line.split(' ')
			src = int(line[0])
			tgt = []
			for item in line[1:]:
				item_score = item.split(',')
				itm = int(item_score[0])
				score = float(item_score[1])
				tgt.append((itm, score))

			sim_item.append((src, tgt))
	return sim_item


def get_train_item_click_file(stage):
	return train_item_click_csv.format(stage)


def get_test_item_click_file(stage):
	return test_item_click_csv.format(stage)


def get_test_qtime_file(stage):
	return test_qtime_csv.format(stage)


def gen_stage_data(stage):
	dfs_train = []
	dfs_valid = []
	dfs_qtest = []
	for i in range(stage+1):
		df_train = pd.read_csv(
			get_train_item_click_file(i),
			names=['user_id', 'item_id', 'time']
		)
		df_test = pd.read_csv(
			get_test_item_click_file(i),
			names=['user_id', 'item_id', 'time']
		)
		print('before drop train:', df_train.shape)
		df_train = df_train.drop_duplicates()
		print('after drop train:', df_train.shape)
		print('before drop test:', df_test.shape)
		df_test = df_test.drop_duplicates()
		print('after drop test:', df_test.shape)

		df_train = df_train.sort_values(['user_id', 'time'])
		df_test = df_test.sort_values(['user_id', 'time'])
		if using_last_num == 'one':
			df_valid = df_test.groupby('user_id', as_index=False).last()
		else:
			df_valid = df_test.groupby('user_id', as_index=False).tail(2)
			df_valid = df_valid.groupby('user_id', as_index=False).head()

		dfs_train.append(df_train)
		dfs_train.append(df_test)
		dfs_valid.append(df_valid)

		df_qtest = pd.read_csv(get_test_qtime_file(i), names=['user_id', 'time'])
		dfs_qtest.append(df_qtest)

	# 训练集，每个stage去重
	df_train = pd.concat(dfs_train, axis=0)
	print('before drop trains:', df_train.shape)
	df_train = df_train.drop_duplicates()
	print('after drop trains:', df_train.shape)

	# 验证集，每个stage去重
	df_valid = pd.concat(dfs_valid, axis=0)
	print('before drop valids:', df_valid.shape)
	df_valid = df_valid.drop_duplicates()
	print('after drop valids:', df_valid.shape)

	# 测试集，每个stage去重
	df_qtest = pd.concat(dfs_qtest, axis=0)
	print('before drop qtests:', df_qtest.shape)
	df_qtest = df_qtest.drop_duplicates()
	print('after drop qtests:', df_qtest.shape)

	for i, cur_df_train in enumerate(dfs_train):
		cur_df_train['stage'] = int(i/2)
		dfs_train[i] = cur_df_train
	for i, cur_df_valid in enumerate(dfs_valid):
		cur_df_valid['stage'] = i
		dfs_valid[i] = cur_df_valid
	for i, cur_df_qtest in enumerate(dfs_qtest):
		cur_df_qtest['stage'] = i
		dfs_qtest[i] = cur_df_qtest

	df_train_stage = pd.concat(dfs_train, axis=0)
	print('before drop trains_stage:', df_train_stage.shape)
	df_train_stage = df_train_stage.drop_duplicates()
	print('after drop trains_stage:', df_train_stage.shape)

	df_valid_stage = pd.concat(dfs_valid, axis=0)
	print('before drop valids_stage:', df_valid_stage.shape)
	df_valid_stage = df_valid_stage.drop_duplicates()
	print('after drop valids_stage:', df_valid_stage.shape)

	df_qtest_stage = pd.concat(dfs_qtest, axis=0)
	print('before drop qtests_stage:', df_qtest_stage.shape)
	df_qtest_stage = df_qtest_stage.drop_duplicates()
	print('after drop qtests_stage:', df_qtest_stage.shape)

	# 对应user在验证集里出现的item在训练集里面删除
	print('before drop trains for valids:', df_train.shape)
	df_train['drop_index'] = np.arange(len(df_train))
	drop_index = df_train.merge(df_valid, on=['user_id', 'item_id'], how='inner')['drop_index']
	df_train = df_train[~df_train['drop_index'].isin(drop_index)]
	df_train = df_train.drop('drop_index', axis=1)
	print('after drop trains for valids:', df_train.shape)
	df_train = df_train.sort_values(['user_id', 'time']).reset_index(drop=True)
	df_train['time'] = (df_train['time'] - 0.98) * 100
	df_train['index'] = df_train.index.astype('str') + '_train'

	print('before drop trains_stage for valids:', df_train_stage.shape)
	df_train_stage['drop_index'] = np.arange(len(df_train_stage))
	drop_index = df_train_stage.merge(df_valid, on=['user_id', 'item_id'], how='inner')['drop_index']
	df_train_stage = df_train_stage[~df_train_stage['drop_index'].isin(drop_index)]
	df_train_stage = df_train_stage.drop('drop_index', axis=1)
	print('after drop trains_stage for valids:', df_train_stage.shape)
	df_train_stage = df_train_stage.sort_values(['stage', 'user_id', 'time']).reset_index(drop=True)
	df_train_stage['time'] = (df_train_stage['time'] - 0.98) * 100
	df_train_stage['index'] = df_train_stage.index.astype('str') + '_train_stage'

	df_valid = df_valid.sort_values(['user_id', 'time']).reset_index(drop=True)
	df_valid['time'] = (df_valid['time'] - 0.98) * 100
	df_valid['index'] = df_valid.index.astype('str') + '_valid'

	df_qtest = df_qtest.sort_values(['user_id', 'time']).reset_index(drop=True)
	df_qtest['time'] = (df_qtest['time'] - 0.98) * 100
	df_qtest['index'] = df_qtest.index.astype('str') + '_test'

	df_valid_stage = df_valid_stage.sort_values(['stage', 'user_id', 'time']).reset_index(drop=True)
	df_valid_stage['time'] = (df_valid_stage['time'] - 0.98) * 100
	df_valid_stage['index'] = df_valid_stage.index.astype('str') + '_valid_stage'

	dump_pickle(df_train, all_train_data_path.format(stage))
	dump_pickle(df_train_stage, all_train_stage_data_path.format(stage))
	dump_pickle(df_valid, all_valid_data_path.format(stage))
	dump_pickle(df_valid_stage, all_valid_stage_data_path.format(stage))
	dump_pickle(df_qtest, all_test_data_path.format(stage))


def gen_stage_data_online(stage):
	dfs_train = []
	dfs_qtest = []
	for i in range(stage+1):
		df_train = pd.read_csv(
			get_train_item_click_file(i),
			names=['user_id', 'item_id', 'time']
		)
		df_test = pd.read_csv(
			get_test_item_click_file(i),
			names=['user_id', 'item_id', 'time']
		)
		print('before drop train:', df_train.shape)
		df_train = df_train.drop_duplicates()
		print('after drop train:', df_train.shape)
		print('before drop test:', df_test.shape)
		df_test = df_test.drop_duplicates()
		print('after drop test:', df_test.shape)

		df_train = df_train.sort_values(['user_id', 'time'])
		df_test = df_test.sort_values(['user_id', 'time'])

		dfs_train.append(df_train)
		dfs_train.append(df_test)

		df_qtest = pd.read_csv(get_test_qtime_file(i), names=['user_id', 'time'])
		dfs_qtest.append(df_qtest)

	df_train = pd.concat(dfs_train, axis=0)
	print('before drop trains:', df_train.shape)
	df_train = df_train.drop_duplicates()
	print('after drop trains:', df_train.shape)

	df_qtest = pd.concat(dfs_qtest, axis=0)
	print('before drop qtests:', df_qtest.shape)
	df_qtest = df_qtest.drop_duplicates()
	print('after drop qtests:', df_qtest.shape)

	for i, cur_df_train in enumerate(dfs_train):
		cur_df_train['stage'] = int(i/2)
		dfs_train[i] = cur_df_train
	for i, cur_df_qtest in enumerate(dfs_qtest):
		cur_df_qtest['stage'] = i
		dfs_qtest[i] = cur_df_qtest

	df_train_stage = pd.concat(dfs_train, axis=0)
	print('before drop trains_stage:', df_train_stage.shape)
	df_train_stage = df_train_stage.drop_duplicates()
	print('after drop trains_stage:', df_train_stage.shape)

	df_qtest_stage = pd.concat(dfs_qtest, axis=0)
	print('before drop qtests_stage:', df_qtest_stage.shape)
	df_qtest_stage = df_qtest_stage.drop_duplicates()
	print('after drop qtests_stage:', df_qtest_stage.shape)

	df_train = df_train.sort_values(['user_id', 'time']).reset_index(drop=True)
	df_train['time'] = (df_train['time'] - 0.98) * 100
	df_train['index'] = df_train.index.astype('str') + '_train'

	df_train_stage = df_train_stage.sort_values(['stage', 'user_id', 'time']).reset_index(drop=True)
	df_train_stage['time'] = (df_train_stage['time'] - 0.98) * 100
	df_train_stage['index'] = df_train_stage.index.astype('str') + '_train_stage'

	df_qtest = df_qtest.sort_values(['user_id', 'time']).reset_index(drop=True)
	df_qtest['time'] = (df_qtest['time'] - 0.98) * 100
	df_qtest['index'] = df_qtest.index.astype('str') + '_test'

	dump_pickle(df_train, online_all_train_data_path.format(stage))
	dump_pickle(df_train_stage, online_all_train_stage_data_path.format(stage))
	dump_pickle(df_qtest, online_all_test_data_path.format(stage))


def gen_item_degree(stage):
	phase_item_deg = {}
	item_deg = defaultdict(int)
	for phase_id in range(stage+1):
		with open(get_train_item_click_file(phase_id)) as f:
			for line in f:
				user_id, item_id, timestamp = line.split(',')
				user_id, item_id, timestamp = int(user_id), int(item_id), float(timestamp)
				item_deg[item_id] += 1
		with open(get_test_item_click_file(phase_id)) as f:
			for line in f:
				user_id, item_id, timestamp = line.split(',')
				user_id, item_id, timestamp = int(user_id), int(item_id), float(timestamp)
				item_deg[item_id] += 1

		phase_item_deg[phase_id] = dict(item_deg)

	dump_pickle(dict(item_deg), full_item_degree_path.format(stage))
	dump_pickle(phase_item_deg, phase_full_item_degree_path.format(stage))


def item2time(stage):
	# for mode in ['valid', 'test']:
	for mode in ['valid']:
		if mode == 'valid':
			all_train_data = load_pickle(all_train_data_path.format(stage))
		else:
			all_train_data = load_pickle(online_all_train_data_path.format(stage))
		item_with_time = all_train_data.sort_values(['item_id', 'time'])
		item2time = item_with_time.groupby('item_id')['time'].agg(list).to_dict()
		dump_pickle(item2time, item2time_path.format(mode, stage))


def item_pair2time_diff(stage):
	# for mode in ['valid', 'test']:
	for mode in ['valid']:
		if mode == 'valid':
			all_train_data = load_pickle(all_train_data_path.format(stage))
		else:
			all_train_data = load_pickle(online_all_train_data_path.format(stage))

		all_item_seqs = all_train_data.groupby('user_id')['item_id'].agg(list).to_list()
		all_time_seqs = all_train_data.groupby('user_id')['time'].agg(list).to_list()

		deltas = [0.01, 0.03, 0.05, 0.07, 0.1]
		item2times = {}
		for delta in deltas:
			item2times[delta] = {}

		for item_seq, time_seq in tqdm(zip(all_item_seqs, all_time_seqs), desc='item_pair2time_diff'):
			length = len(item_seq)
			for i in range(length):
				for j in range(i+1, length):
					item_a, item_b = item_seq[i], item_seq[j]
					time_a, time_b = time_seq[i], time_seq[j]
					time_diff = abs(time_a-time_b)
					pair = tuple(sorted([item_a, item_b]))
					for delta in deltas:
						if time_diff < delta:
							item2times[delta].setdefault(pair, 0)
							item2times[delta][pair] += 1

		for delta in deltas:
			dump_pickle(item2times, item2times_path.format(mode, stage, delta))


def item_pair2time_seq(stage):
	# for mode in ['valid', 'test']:
	for mode in ['valid']:
		if mode == 'valid':
			all_train_data = load_pickle(all_train_data_path.format(stage))
		else:
			all_train_data = load_pickle(online_all_train_data_path.format(stage))

		all_item_seqs = all_train_data.groupby('user_id')['item_id'].agg(list).to_list()
		all_time_seqs = all_train_data.groupby('user_id')['time'].agg(list).to_list()

		item_pair2time_diff = {}
		item_pair2time_seq = {}

		for item_seq, time_seq in tqdm(zip(all_item_seqs, all_time_seqs), desc='item_pair2time_seq'):
			length = len(item_seq)
			for i in range(length):
				for j in range(i + 1, length):
					item_a, item_b = item_seq[i], item_seq[j]
					time_a, time_b = time_seq[i], time_seq[j]
					time_diff = abs(time_a - time_b)
					times = tuple(sorted([time_a, time_b]))
					pair = tuple(sorted([item_a, item_b]))

					item_pair2time_diff.setdefault(pair, [])
					item_pair2time_diff[pair].append(time_diff)
					item_pair2time_seq.setdefault(pair, [])
					item_pair2time_seq[pair].append(times)

		dump_pickle(item_pair2time_diff, item_pair2time_diff_path.format(mode, stage))
		dump_pickle(item_pair2time_seq, item_pair2time_seq_path.format(mode, stage))


def get_sim_item(df_train, p):
	user_item_dict = df_train.groupby('user_id')['item_id'].agg(list).to_dict()
	user_time_dict = df_train.groupby('user_id')['time'].agg(list).to_dict()

	sim_item = {}
	item_cnt = defaultdict(int)
	for user, items in tqdm(user_item_dict.items(), desc='get_sim_item'):
		times = user_time_dict[user]

		for loc1, item in enumerate(items):
			item_cnt[item] += 1
			sim_item.setdefault(item, {})
			for loc2, relate_item in enumerate(items):
				if item == relate_item:
					continue

				t1, t2 = times[loc1], times[loc2]
				sim_item[item].setdefault(relate_item, 0)

				time_weight = (1 - abs(t1-t2)*100)
				time_weight = max(time_weight, 0.2)
				loc_weight = 0.9 ** (abs(loc1-loc2)-1)
				loc_weight = max(loc_weight, 0.2)
				# 1.0为可调整系数,item_cf公式变形，增加了loc_weight,time_weight的影响
				sim_item[item][relate_item] += 1 * 1.0 * loc_weight * time_weight / math.log(1+len(items))

	for i, related_items in sim_item.items():
		for j, cij in related_items.items():
			sim_item[i][j] = cij / ((item_cnt[i]*item_cnt[j]) ** p)
	for item in sim_item.keys():
		related_items = sim_item[item]
		related_items = sorted(related_items.items(), key=lambda x: x[1], reverse=True)
		rel = {}
		for x in related_items[:i2i2i_new_sim_limit]:
			rel[x[0]] = x[1]
		sim_item[item] = rel
	dump_pickle(sim_item, sim_item_path.format(i2i2i_new_sim_limit, p))
	return sim_item


def recommend(sim_item, user_item_dict, user_time_dict, user_id, qtime, loc_coff, recall_type):
	if big_or_small_or_history == 'small':
		if recall_type == 'i2iw10' or recall_type == 'i2iw02':
			recall_max_road_num = 10
			recall_max_num_each_road = 100
		elif recall_type == 'b2b':
			recall_max_road_num = 10
			recall_max_num_each_road = 100
		elif recall_type == 'i2i2i_new':
			recall_max_road_num = 10
			recall_max_num_each_road = 100
		else:
			print(1/0)
	else:
		print('big_or_small_or_history error')
		print(1/0)

	# 原test_click数据集user不止一条数据(>=2)，所以user必定有相对应的items_list, times_list
	# test_qtime, test_click中的user一一对应
	interacted_items = user_item_dict[user_id]
	interacted_times = user_time_dict[user_id]

	# 定位预测时间qtime在行为序列里面的位置
	qtime_loc = 0
	while qtime_loc < len(interacted_items) and qtime >= interacted_times[qtime_loc]:
		qtime_loc += 1

	l_border = max(0, qtime_loc-recall_max_road_num)
	r_border = min(len(interacted_items), qtime_loc+recall_max_road_num)
	cans_loc = list(range(l_border, r_border))

	if get_sum:
		multi_road_result = {}
	else:
		multi_road_result = []
	for loc in cans_loc:
		item = interacted_items[loc]
		time = interacted_times[loc]
		each_road_result = []

		if loc >= qtime_loc:
			loc_weight = loc_coff ** (abs(loc-qtime_loc))
		else:
			loc_weight = loc_coff ** (abs(loc - qtime_loc)-1)
		loc_weight = max(0.1, loc_weight)
		time_weight = (1 - abs(time-qtime)*100)
		time_weight = max(0.1, time_weight)

		if item not in sim_item:
			continue
		for related_item, wij in sim_item[item].items():
			sim_weight = wij
			# item_cf, 在每个user中历史商品找出K个最相似的商品，然后计算这些商品的累计相似性进行排序
			rank_weight = sim_weight * loc_weight * time_weight
			each_road_result.append((
				related_item, sim_weight, loc_weight, time_weight, rank_weight,
				item, loc, time, qtime_loc, qtime
			))
		each_road_result.sort(key=lambda x: x[1], reverse=True)
		each_road_result = each_road_result[0: recall_max_num_each_road]
		# 是否按照传统item_cf对被关联的item-sim进行累加
		if get_sum:
			for idx, k in enumerate(each_road_result):
				if k[0] not in multi_road_result:
					multi_road_result[k[0]] = k[1:]
				else:
					t1 = multi_road_result[k[0]]
					t2 = k[1:]
					multi_road_result[k[0]] = (
						t1[0]+t2[0], t1[1], t1[2], t1[3]+t2[3], t1[4],
						t1[5], t1[6], t1[7], t1[8]
					)
		else:
			multi_road_result += each_road_result

	if get_sum:
		multi_road_result_sorted = sorted(multi_road_result.items(), key=lambda x: x[1][3], reverse=True)
		multi_road_result = []
		for q in multi_road_result_sorted:
			multi_road_result.append((q[0], )+q[1])
		else:
			multi_road_result.sort(key=lambda x: x[4], reverse=True)
	return multi_road_result


def i2i_w02_recall(df_train, df_train_stage, df, df_stage, cur_stage):
	# item_sim
	if cur_mode == 'valid':
		if os.path.exists(sim_item_path.format(i2i_sim_limit, 0.2)):
			sim_item = load_pickle(sim_item_path.format(i2i_sim_limit, 0.2))
		else:
			sim_item = get_sim_item(df_train, 0.2)
	else:
		if os.path.exists(test_sim_item_path.format(i2i_sim_limit)):
			sim_item = load_pickle(test_sim_item_path.format(i2i_sim_limit))
		else:
			sim_item = get_sim_item(df_train)

	user_item_dict = df_train.groupby('user_id')['item_id'].agg(list).to_dict()
	user_time_dict = df_train.groupby('user_id')['time'].agg(list).to_dict()

	user2recall = {}
	# 在user已有的序列里定位qtime的loc, 根据loc左右一定范围内的每个loc对应的item， 参照item_sim，计算出出related_item的特征
	for user, qtime in tqdm(zip(df['user_id'], df['time']), desc='i2i_w02_recall'):
		user2recall[(user, qtime)] = recommend(sim_item, user_item_dict, user_time_dict, user, qtime, 0.7, 'i2iw02')

	phase_ndcg_pred_answer = []
	answers_source = []
	phase_item_degree = load_pickle(phase_full_item_degree_path.format(cur_stage))
	for predict_stage in range(cur_stage+1):
		preds = []
		pos = []
		df_now = df_stage[df_stage['stage'] == predict_stage]
		df_train = df_train_stage[df_train_stage['stage'] == predict_stage]
		stage_items = set(df_train['item_id'])
		cur_user_item_dict = user_item_dict
		for user, it, qtime in zip(df_now['user_id'], df_now['item_id'], df_now['time']):
			recall_items = user2recall[(user, qtime)]
			new_recall = []
			for re in recall_items:
				if re[0] == it:
					new_recall.append(re)
				# not ((user in cur_user_item_dict) and (re[0] in cur_user_item_dict[user]))
				# 条件1 related_item 不在user对应的item_dict
				elif (user not in cur_user_item_dict) or (re[0] not in cur_user_item_dict[user]):
					# 条件2 related_item 在对应的stage的df_train里面
					if re[0] in stage_items:
						new_recall.append(re)

			preds.append(new_recall)
			pos.append(it)

		item_degree = phase_item_degree[predict_stage]
		if cur_mode == 'valid':
			answers = [(p, item_degree[p]) for p in pos]
		else:
			answers = [(p, np.nan) for p in pos]

		phase_ndcg_pred_answer.append(preds)
		answers_source.append(answers)

	if get_sum:
		dump_pickle(phase_ndcg_pred_answer, i2i_w02_recall_scoure_path.format(cur_mode, cur_stage, 'sum'))
	else:
		dump_pickle(phase_ndcg_pred_answer, i2i_w02_recall_scoure_path.format(cur_mode, cur_stage, 'nosum'))
	dump_pickle(answers_source, answer_source_path.format(cur_mode, cur_stage))


def i2i_w10_recall(df_train, df_train_stage, df, df_stage, cur_stage):
	if cur_mode == 'valid':
		if os.path.exists(sim_item_path.format(i2i_sim_limit, 1)):
			sim_item = load_pickle(sim_item_path.format(i2i_sim_limit, 1))
		else:
			sim_item = get_sim_item(df_train, 1)
	else:
		if os.path.exists(test_sim_item_path.format(i2i_sim_limit)):
			sim_item = load_pickle(test_sim_item_path.format(i2i_sim_limit))
		else:
			sim_item = get_sim_item(df_train)

	user_item_dict = df_train.groupby('user_id')['item_id'].agg(list).to_dict()
	user_time_dict = df_train.groupby('user_id')['time'].agg(list).to_dict()

	user2recall = {}
	# 在user已有的序列里定位qtime的loc, 根据loc左右一定范围内的每个loc对应的item， 参照item_sim，计算出出related_item的特征
	for user, qtime in tqdm(zip(df['user_id'], df['time']), desc='i2i_w10_recall'):
		user2recall[(user, qtime)] = recommend(sim_item, user_item_dict, user_time_dict, user, qtime, 0.7, 'i2iw10')

	phase_ndcg_pred_answer = []
	answers_source = []
	phase_item_degree = load_pickle(phase_full_item_degree_path.format(cur_stage))
	for predict_stage in range(cur_stage+1):
		preds = []
		pos = []
		df_now = df_stage[df_stage['stage'] == predict_stage]
		df_train = df_train_stage[df_train_stage['stage'] == predict_stage]
		stage_items = set(df_train['item_id'])
		cur_user_item_dict = user_item_dict
		for user, it, qtime in zip(df_now['user_id'], df_now['item_id'], df_now['time']):
			recall_items = user2recall[(user, qtime)]
			new_recall = []
			for re in recall_items:
				if re[0] == it:
					new_recall.append(re)
				# not ((user in cur_user_item_dict) and (re[0] in cur_user_item_dict[user]))
				# 条件1 related_item 不在user对应的item_dict
				elif (user not in cur_user_item_dict) or (re[0] not in cur_user_item_dict[user]):
					# 条件2 related_item 在对应的stage的df_train里面
					if re[0] in stage_items:
						new_recall.append(re)

			preds.append(new_recall)
			pos.append(it)

		item_degree = phase_item_degree[predict_stage]
		if cur_mode == 'valid':
			answers = [(p, item_degree[p]) for p in pos]
		else:
			answers = [(p, np.nan) for p in pos]

		phase_ndcg_pred_answer.append(preds)
		answers_source.append(answers)

	if get_sum:
		dump_pickle(phase_ndcg_pred_answer, i2i_w10_recall_scoure_path.format(cur_mode, cur_stage, 'sum'))
	else:
		dump_pickle(phase_ndcg_pred_answer, i2i_w10_recall_scoure_path.format(cur_mode, cur_stage, 'nosum'))


def b2b_recall(df_train, df_train_stage, df, df_stage, cur_stage):
	blend_sim = load_sim(item_blend_sim_path)
	blend_score = {}
	for q in blend_sim:
		item = q[0]
		blend_score.setdefault(item, {})
		for related_item, score in q[1][:b2b_sim_limit]:
			blend_score[item][related_item] = score

	user_item_dict = df_train.groupby('user_id')['item_id'].agg(list).to_dict()
	user_time_dict = df_train.groupby('user_id')['time'].agg(list).to_dict()

	user2recall = {}
	for user, qtime in tqdm(zip(df['user_id'], df['time']), desc='b2b_recall'):
		user2recall[(user, qtime)] = recommend(blend_score, user_item_dict, user_time_dict, user, qtime, 0.7, 'b2b')

	phase_ndcg_pred_answer = []
	answers_source = []
	phase_item_degree = load_pickle(phase_full_item_degree_path.format(cur_stage))
	for predict_stage in range(cur_stage+1):
		preds = []
		pos = []
		df_now = df_stage[df_stage['stage'] == predict_stage]
		df_train = df_train_stage[df_train_stage['stage'] == predict_stage]
		stage_items = set(df_train['item_id'])
		cur_user_item_dict = user_item_dict
		for user, it, qtime in zip(df_now['user_id'], df_now['item_id'], df_now['time']):
			recall_items = user2recall[(user, qtime)]
			new_recall = []
			for re in recall_items:
				if re[0] == it:
					new_recall.append(re)
				elif (user not in cur_user_item_dict) or (re[0] not in cur_user_item_dict[user]):
					if re[0] in stage_items:
						new_recall.append(re)

			preds.append(new_recall)
			pos.append(it)

		item_degree = phase_item_degree[predict_stage]
		if cur_mode == 'valid':
			answers = [(p, item_degree[p]) for p in pos]
		else:
			answers = [(p, np.nan) for p in pos]

		phase_ndcg_pred_answer.append(preds)
		answers_source.append(answers)

	if get_sum:
		dump_pickle(phase_ndcg_pred_answer, b2b_recall_scoure_path.format(cur_mode, cur_stage, 'sum'))
	else:
		dump_pickle(phase_ndcg_pred_answer, b2b_recall_scoure_path.format(cur_mode, cur_stage, 'nosum'))


def i2i2i_new_recall(df_train, df_train_stage, df, df_stage, cur_stage):
	if cur_mode == 'valid':
		if os.path.exists(sim_item_path.format(i2i2i_new_sim_limit, 0.2)):
			sim_item = load_pickle(sim_item_path.format(i2i2i_new_sim_limit, 0.2))
		else:
			sim_item = get_sim_item(df_train, 0.2)
	else:
		if os.path.exists(test_sim_item_path.format(i2i2i_new_sim_limit)):
			sim_item = load_pickle(test_sim_item_path.format(i2i2i_new_sim_limit))
		else:
			sim_item = get_sim_item(df_train)

	sim_item_p2 = {}
	for item1 in tqdm(sim_item.keys(), desc='sim_item_p2'):
		sim_item_p2.setdefault(item1, {})
		for item2 in sim_item[item1].keys():
			# item 和 related_item 不相同
			if item1 == item2:
				continue
			for item3 in sim_item[item2].keys():
				# item 和 related_item 不相同
				if item3 == item1 or item3 == item2:
					continue
				# related_item 不应该在item的related_item list中
				if item3 in sim_item[item1]:
					continue
				sim_item_p2[item1].setdefault(item3, 0)
				sim_item_p2[item1][item3] += sim_item[item1][item2] * sim_item[item2][item3]

	user_item_dict = df_train.groupby('user_id')['item_id'].agg(list).to_dict()
	user_time_dict = df_train.groupby('user_id')['time'].agg(list).to_dict()

	user2recall = {}
	for user, qtime in tqdm(zip(df['user_id'], df['time']), desc='i2i2i_new_recall'):
		user2recall[(user, qtime)] = recommend(sim_item_p2, user_item_dict, user_time_dict, user, qtime, 0.7, 'i2i2i_new')

	phase_ndcg_pred_answer = []
	answers_source = []
	phase_item_degree = load_pickle(phase_full_item_degree_path.format(cur_stage))
	for predict_stage in range(cur_stage+1):
		preds = []
		pos = []
		df_now = df_stage[df_stage['stage'] == predict_stage]
		df_train = df_train_stage[df_train_stage['stage'] == predict_stage]
		stage_items = set(df_train['item_id'])
		cur_user_item_dict = user_item_dict
		for user, it, qtime in zip(df_now['user_id'], df_now['item_id'], df_now['time']):
			recall_items = user2recall[(user, qtime)]
			new_recall = []
			for re in recall_items:
				if re[0] == it:
					new_recall.append(re)
				elif (user not in cur_user_item_dict) or (re[0] not in cur_user_item_dict[user]):
					if re[0] in stage_items:
						new_recall.append(re)

			preds.append(new_recall)
			pos.append(it)

		item_degree = phase_item_degree[predict_stage]
		if cur_mode == 'valid':
			answers = [(p, item_degree[p]) for p in pos]
		else:
			answers = [(p, np.nan) for p in pos]

		phase_ndcg_pred_answer.append(preds)
		answers_source.append(answers)

	if get_sum:
		dump_pickle(phase_ndcg_pred_answer, i2i2i_new_recall_scoure_path.format(cur_mode, cur_stage, 'sum'))
	else:
		dump_pickle(phase_ndcg_pred_answer, i2i2i_new_recall_scoure_path.format(cur_mode, cur_stage, 'nosum'))


def gen_data(df, df_stage, mode, cur_stage):
	answer_source = load_pickle(answer_source_path.format(mode, cur_stage, sum_mode))
	b2b_recall_source = load_pickle(b2b_recall_scoure_path.format(mode, cur_stage, sum_mode))
	i2i2i_new_recall_source = load_pickle(i2i2i_new_recall_scoure_path.format(mode, cur_stage, sum_mode))
	i2i_w10_recall_source = load_pickle(i2i_w10_recall_scoure_path.format(mode, cur_stage, sum_mode))
	recall_sources = [b2b_recall_source, i2i2i_new_recall_source, i2i_w10_recall_source]
	recall_source_names = cur_recall_source_names
	recall_file_names = '-'.join(recall_source_names + [sum_mode])

	user2stage = {}
	for idx, row in df_stage.iterrows():
		user2stage[row['user_id']] = row['stage']
	user2index = {}
	for stage in range(cur_stage+1):
		tmp = df_stage[df_stage['stage'] == stage]['user_id'].values
		for i in range(tmp.shape[0]):
			user2index[tmp[i]] = i

	left_items_list = []
	left_times_list = []
	right_items_list = []
	right_times_list = []
	user_list = []
	time_list = []
	item_list = []
	sim_weight_list = []
	loc_weight_list = []
	time_weight_list = []
	rank_weight_list = []
	road_item_list = []
	road_item_loc_list = []
	road_item_time_list = []
	query_item_loc_list = []
	query_item_time_list = []
	recall_type_list = []
	stage_list = []
	label_list = []

	for user, group in tqdm(df.groupby('user_id'), desc='gen_data'):
		items = group['item_id'].values
		times = group['time'].values
		index = group['index'].values

		for i in range(len(items)):
			if index[i].endswith(mode):
				left_items = []
				left_times = []
				right_items = []
				right_times = []
				for k in range(i-1, -1, -1):
					if not index[k].endswith(mode):
						left_items.append(items[k])
						left_times.append(times[k])
				for k in range(i+1, len(items)):
					if not index[k].endswith(mode):
						right_items.append(items[k])
						right_times.append(times[k])

				# 检查数据
				if mode == 'valid':
					if items[i] != answer_source[user2stage[user]][user2index[user]][0]:
						print(user, user2index[user], items[i], answer_source[user2stage[user]][user2index[user]][0])
						print('召回出来的数据对应的pos数据对应不上')
						print(1/0)

				for idx, recall_source in enumerate(recall_sources):
					# (related_item, sim_weight, loc_weight, time_weight, rank_weight, item, loc, time, qtime_loc, qtime)
					recall = recall_source[user2stage[user]][user2index[user]]
					for j in range(len(recall)):
						user_list.append(user)
						# 被预测商品
						item_list.append(recall[j][0])
						if recall[j][0] == items[i]:
							label_list.append(1)
						else:
							label_list.append(0)
						sim_weight_list.append(recall[j][1])
						loc_weight_list.append(recall[j][2])
						time_weight_list.append(recall[j][3])
						rank_weight_list.append(recall[j][4])
						# 关联关系的起始商品
						road_item_list.append(recall[j][5])
						road_item_loc_list.append(recall[j][6])
						road_item_time_list.append(recall[j][7])
						query_item_loc_list.append(recall[j][8])
						query_item_time_list.append(recall[j][9])
						recall_type_list.append(idx)
						stage_list.append(user2stage[user])
						left_items_list.append(left_items)
						right_items_list.append(right_items)
						left_times_list.append(left_times)
						right_times_list.append(right_times)
						time_list.append(times[i])

	data = {}
	data['left_items_list'] = left_items_list # >=qtime左边item list X
	data['right_items_list'] = right_items_list # >qtime右边item list X
	data['left_times_list'] = left_times_list # >=qtime左边time list X
	data['right_times_list'] = right_times_list # >qtime右边time list X
	data['user'] = user_list # user list X
	data['time'] = time_list # qtime list X
	data['item'] = item_list # recall item list X
	data['sim_weight'] = sim_weight_list # recall item list
	data['loc_weight'] = loc_weight_list # recall item list
	data['time_weight'] = time_weight_list # recall item list
	data['rank_weight'] = rank_weight_list # recall item list
	data['road_item'] = road_item_list # road item list X
	data['road_item_loc'] = road_item_loc_list # road item list
	data['road_item_time'] = road_item_time_list # road item list
	data['query_item_loc'] = query_item_loc_list # qtime
	data['query_item_time'] = query_item_time_list # qtime
	data['recall_type'] = recall_type_list # idx of recall source X
	data['stage'] = stage_list # user stage
	data['label'] = label_list # recall item = target item X
	data = pd.DataFrame(data)

	dump_pickle(data, lgb_base_pkl.format(recall_file_names, mode, cur_stage))
