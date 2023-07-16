import pandas as pd
import warnings
from utils import *
from tqdm import tqdm
import numpy as np
from scipy import sparse
import sys
import torch
from torch import nn
from torch.nn import functional as F
import os


pd.options.display.max_columns = None
warnings.filterwarnings('ignore')
train_data_path = 'data/train_{}.pkl'
test_data_path = 'data/test_new.pkl'
attr_path = 'data/attr.txt'
topo_path = 'data/topo.txt'


if __name__ == '__main__':
	# 读取data_prepare的训练集和测试集
	train_list = []
	for i in range(1, 31):
		df = pd.read_pickle(train_data_path.format(i))
		train_list.append(df)
	train = pd.concat(train_list, axis=0, ignore_index=True)
	test = pd.read_pickle(test_data_path)
	test = test.reset_index(drop=True)
	cols = ['link', 'label', 'current_slice_id', 'future_slice_id']
	for col in cols:
		train[col] = train[col].astype(int)
		test[col] = test[col].astype(int)
	print('train, test describe')
	print(train.shape, test.shape)
	print(len(set(train['link'])), len(set(test['link'])))

	# 读取link的attr特征
	attr_df = pd.read_csv(attr_path, header=None, sep='\t')
	attr_df.columns = ['link', 'length', 'direction', 'path_class', 'speed_class', 'lane_num', 'speed_limit', 'level', 'width']
	for col in ['length', 'speed_limit', 'width']:
		mean = attr_df[col].mean()
		std = attr_df[col].std()
		attr_df[col] = (attr_df[col] - mean) / std
	train = train.merge(attr_df, on='link', how='left')
	test = test.merge(attr_df, on='link', how='left')

	# 时间相关特征，slice_id为负为前一天
	train['week_day'] = train['day'].apply(lambda x: x % 7)
	test['week_day'] = 4
	train['hour'] = train['future_slice_id'].apply(lambda x: x//30 if x >= 0 else (x+720)//30)
	test['hour'] = test['future_slice_id'].apply(lambda x: x//30 if x >= 0 else (x+720)//30)
	train['time_gap'] = train['future_slice_id'] - train['current_slice_id']
	test['time_gap'] = test['future_slice_id'] - test['current_slice_id']

	# topo
	topo_df = pd.read_csv(topo_path, header=None, sep='\t')
	topo_df.columns = ['link', 'next_link']
	topo_df['next_link'] = topo_df['next_link'].apply(lambda x: [int(i) for i in x.split(',')])
	train = train.merge(topo_df, on='link', how='left')
	test = test.merge(topo_df, on='link', how='left')
	topo_df_explode = topo_df.explode('next_link')
	all_topo_link = set(list(topo_df_explode['link']) + list(topo_df_explode['next_link']))
	print('all_topo_link', len(all_topo_link))

	# 对attr,时间特征进行统一的序列编码，方便embedding
	col_thre_dict = {
		'link': 0.5, 'direction': 0.5, 'path_class': 0.5, 'speed_class': 0.5, 'lane_num': 0.5, 'level': 0.5,
		'continuous_length': 0.5, 'continuous_speed_limit': 0.5, 'continuous_width': 0.5,
		'week_day': 0, 'hour': 0, 'time_gap': 0,
		'current_slice_id': 0.5, 'future_slice_id': 0.5,
	}
	ids_indexs = {}
	mp_col_ids_indexs = {}
	ids_indexs['padding'] = 0
	for col, thre in col_thre_dict.items():
		mp = {}
		# thre=0默认unkown=None
		unknow = None
		if 'continuous_' in col:
			mp[col] = len(ids_indexs)
			ids_indexs[col] = len(ids_indexs)
			unknow = len(ids_indexs)
			ids_indexs[col + '_unknow'] = len(ids_indexs)
			mp_col_ids_indexs[col] = [mp, unknow]
		else:
			if col == 'link':
				all_ids = list(set(attr_df['link']))
			else:
				t = train[col].value_counts().reset_index()
				all_ids = t[t[col] > thre]['index'].values
			curr_len = len(ids_indexs)
			for i, ids in enumerate(all_ids):
				ids_indexs[col + '_' + str(ids)] = curr_len + i
				mp[ids] = curr_len + i
			if thre != 0:
				unknow = len(ids_indexs)
				ids_indexs[col + '_unknow'] = len(ids_indexs)
			mp_col_ids_indexs[col] = [mp, unknow]

	# attr特征由统一的序列编码同义替换，不同的连续数值是相同的序列编码
	attr_feat_cols = [
		'direction', 'path_class', 'speed_class', 'lane_num',
		'level', 'width', 'length', 'speed_limit']
	for col in attr_feat_cols[:5]:
		attr_df[col] = attr_df[col].map(mp_col_ids_indexs[col][0]).fillna(mp_col_ids_indexs[col][1])
	# link对应的统一编码
	link_embedding_matrix = attr_df[attr_feat_cols].values
	link_embedding_matrix = np.concatenate(
		[np.zeros(len(attr_feat_cols)).reshape(1, -1), link_embedding_matrix],
		axis=0
	)

	# col_index_name 用统一序列编码进行替换
	# col_value_name 连续特征为原连续特征， 分类特征为1.0
	# 后期对index_name embedding 以后 * value_name
	feat_columns = []
	feat_value_columns = []
	for col, thre in col_thre_dict.items():
		col_index_name = '{}_index'.format(col)
		col_value_name = '{}_value'.format(col)
		feat_columns.append(col_index_name)
		feat_value_columns.append(col_value_name)
		real_col = col
		if 'continuous_' in col:
			real_col = real_col.replace('continuous_', '')
			train[col_index_name] = mp_col_ids_indexs[col][0][col]
			train[col_value_name] = train[real_col].values
			test[col_index_name] = mp_col_ids_indexs[col][0][col]
			test[col_value_name] = test[real_col].values
		else:
			mp = mp_col_ids_indexs[col][0]
			unknow = mp_col_ids_indexs[col][1]
			if unknow is not None:
				train[col_index_name] = train[real_col].map(mp).fillna(unknow)
				test[col_index_name] = test[real_col].map(mp).fillna(unknow)
			else:
				train[col_index_name] = train[real_col].map(mp)
				test[col_index_name] = test[real_col].map(mp)
			train[col_value_name] = 1.0
			test[col_value_name] = 1.0

	# topo特征
	mp = mp_col_ids_indexs['link'][0]
	unknow = mp_col_ids_indexs['link'][1]
	# 上游link
	link_before_dict = {}
	for index, row in topo_df.iterrows():
		link = row['link']
		link = mp[link]
		next_link = row['next_link']
		for next_l in next_link:
			next_l = mp[next_l]
			if next_l not in link_before_dict:
				link_before_dict[next_l] = []
			if link not in link_before_dict[next_l]:
				link_before_dict[next_l] = link_before_dict[next_l] + [link]
	# 下游link
	link_next_dict = {}
	for index, row in topo_df.iterrows():
		link = row['link']
		link = mp[link]
		next_link = row['next_link']
		if link not in link_next_dict:
			link_next_dict[link] = []
		for next_l in next_link:
			next_l = mp[next_l]
			link_next_dict[link] = link_next_dict[link] + [next_l]

	train_test_link = []
	for link in set(list(train['link']) + list(test['link'])):
		if link in all_topo_link:
			train_test_link.append(mp[link])
	print('train_test_link', len(train_test_link))

	# gcn图卷积网络,处理拓扑结构数据,https://zhuanlan.zhihu.com/p/452519312
	# gcn数据准备
	link_topo_sub_graph = {}
	# link上游下游相可连接的最大层数
	num_add_graph = 4
	# link上游下游相可连接的最大link数
	max_number = 200
	# 在num_add_graph，max_number 的限制下link可连接的link数
	nums = []
	for link_id in tqdm(train_test_link, desc='link_matrix'):
		# 记录边信息，格式[link, link(next or before)]
		link_info = []
		# 记录连接的link
		link_set = set([link_id])
		num = 1
		next_link = link_next_dict.get(link_id, [])
		before_link = link_before_dict.get(link_id, [])
		# link_id第一层上下游link，去重
		link_set, n = add_link_set(next_link, link_set)
		num += n
		link_set, n = add_link_set(before_link, link_set)
		num += n
		link_info.extend([[link_id, next_l] for next_l in next_link])
		link_info.extend([[link_id, before_l] for before_l in before_link])
		# 由第一层上下游link继续扩展，上游下游的link都可以往上,往下扩展
		link_add_info = []
		for graph_id in range(num_add_graph):
			if num == max_number:
				break
			next_link_next = []
			before_link_before = []
			# 从下游link开始扩展
			for sub_link_id in next_link:
				# 由sub_link_id扩展出的下游link
				sub_info, sub_num, sub_next_link = get_next_link(sub_link_id, link_next_dict)
				# 根据扩展出的link更新link_info, link_set, num
				link_add_info, num, link_set = add_link_info(link_add_info, sub_info, num, max_number, link_set)
				next_link_next.extend(sub_next_link)
				sub_info, sub_num, sub_next_link = get_before_link(sub_link_id, link_before_dict)
				link_add_info, num, link_set = add_link_info(link_add_info, sub_info, num, max_number, link_set)
				next_link_next.extend(sub_next_link)
			# 从上游link开始扩展
			for sub_link_id in before_link:
				sub_info, sub_num, sub_next_link = get_next_link(sub_link_id, link_next_dict)
				link_add_info, num, link_set = add_link_info(link_add_info, sub_info, num, max_number, link_set)
				before_link_before.extend(sub_next_link)
				sub_info, sub_num, sub_next_link = get_before_link(sub_link_id, link_before_dict)
				link_add_info, num, link_set = add_link_info(link_add_info, sub_info, num, max_number, link_set)
				before_link_before.extend(sub_next_link)
			next_link = next_link_next
			before_link = before_link_before
		nums.append(num)
		link_info.extend(link_add_info)

		# 生成gcn A, D矩阵,对每个link_id 计算D-1/2*A*D-1/2
		edges = np.array(link_info)
		link_map = {id: idx for idx, id in enumerate(list(link_set))}
		number = len(link_map)
		edges = np.array(
			list(map(link_map.get, edges.flatten())),
			dtype=np.int32).reshape(edges.shape)
		# 转换稀疏连接矩阵
		adj = sparse.coo_matrix(
			(np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
			shape=(number, number),
			dtype=np.float32
		).toarray()
		# 对adj进行max_number填充，原adj在左上角，只对右边，下边进行填充
		adj = np.pad(
			adj,
			((0, max_number - number), (0, max_number - number)),
			mode='constant',
			constant_values=(0)
		)
		# 边信息用[a, b]填充[b,a],转化为对称矩阵
		adj += adj.T + np.eye(adj.shape[0])
		adj = (adj > 0).astype(np.int)
		# 度矩阵
		deg = np.diag(1 / np.sqrt(np.sum(adj, axis=1)))
		# D-1/2*A*D-1/2
		h = np.dot(deg, adj)
		h = np.dot(h, deg)
		link_ids = list(link_set) + [0] * (max_number - number)
		link_topo_sub_graph[link_id] = (h, link_ids)

	print(np.max(nums), np.min(nums), np.mean(nums))
	num_df = pd.DataFrame({'num': nums})
	print(num_df.describe([0.9, 0.95, 0.98, 0.99]))

	# 对recent_feature， history_feature标准化
	scaled_features = []
	for col in train.columns:
		if 'feature' in col:
			scaled_features.append(col)
	means = np.mean(train[scaled_features].values, axis=0)
	stds = np.std(train[scaled_features].values, axis=0)
	for i, col in enumerate(scaled_features):
		train.loc[:, col] = (train.loc[:, col] - means[i]) / stds[i]
		test.loc[:, col] = (test.loc[:, col] - means[i]) / stds[i]

	# recent_feature
	recent_cols = []
	for i in range(1, 6):
		recent_col = [col for col in train.columns if 'recent_feature_{}'.format(i) in col]
		recent_cols.extend(recent_col)
	train['recent_split_info'] = [i.reshape(5, 4) for i in train[recent_cols].values]
	test['recent_split_info'] = [i.reshape(5, 4) for i in test[recent_cols].values]
	# history_feature
	history_cols = []
	for i in range(1, 6):
		history_col = [col for col in train.columns if 'history_feature_cycle{}'.format(i) in col]
		history_cols.extend(history_col)
	train['history_split_info'] = [i.reshape(20, 4) for i in train[history_cols].values]
	test['history_split_info'] = [i.reshape(20, 4) for i in test[history_cols].values]

	# label
	train['label'] = train['label'].apply(lambda x: x-1 if x != 4 else 2)
	test['label'] = 0

	# 提取四大类特征, 封装成数据集
	train_data = get_model_data(train[train['day'] != 30], feat_columns, feat_value_columns)
	valid_data = get_model_data(train[train['day'] == 30], feat_columns, feat_value_columns)
	test_data = get_model_data(test, feat_columns, feat_value_columns)
	train_loader = get_loader(train_data, link_topo_sub_graph, max_number, batch_size=256, train_mode=True)
	test_loader = get_loader(test_data, link_topo_sub_graph, max_number, batch_size=256, train_mode=False)
	validation_loader = get_loader(valid_data, link_topo_sub_graph, max_number, batch_size=256, train_mode=False)

	# 建模
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = DiDi_Model(
		embedding_num=len(ids_indexs), embedding_dim=32,
		link_embedding_matrix=link_embedding_matrix, device=device
	).to(device)

	accumulation_steps, early_stop_epochs, epochs = 1, 3, 6
	model_save_path = 'nn_model.pkl'
	if os.path.exists(model_save_path):
		model.load_state_dict(torch.load(model_save_path, map_location=device))
	else:
		losses = train_model(
			model, train_loader, validation_loader, accumulation_steps,
			early_stop_epochs, epochs, model_save_path, device)

	# 后处理预测值
	valid_preds = validation_fn(model, validation_loader, device, is_test=True)
	valid_y = train[train['day'] == 30]['label'].values
	weights = get_weights(valid_preds, valid_y)
	print(weights)
	# [5.04463918 1.63468615 0.95875032]
	preds = validation_fn(model, test_loader, is_test=True)
	preds = preds * weights
	nn_sub = test[['link', 'current_slice_id', 'future_slice_id']].copy()
	for i in range(3):
		nn_sub['nn_pred_{}'.format(i)] = preds[:, i]
	nn_sub.to_csv('data/nn_sub_prob.csv', index=False)









