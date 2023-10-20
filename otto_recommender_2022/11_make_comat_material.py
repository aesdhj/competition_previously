import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np


def make_action(test):
	# session 最后一个aid
	max_ts = test.groupby('session', as_index=False)['ts'].max()
	last_action = test.merge(max_ts, on='session', how='left')
	last_action = last_action[last_action['ts_x'] == last_action['ts_y']][['session', 'aid']].drop_duplicates()
	# session 最多click的aid, 以分位数为依据
	top_click = test[test['type'] == 0].groupby(['session', 'aid'], as_index=False)['ts'].count()
	top_click = top_click.rename(columns={'ts': 'n'})
	top_click['share'] = top_click.groupby('session')['n'].rank(ascending=False, pct=True, method='max')
	top_click = top_click[top_click['share'] <= 0.3]
	top_click['weight'] = 1 - top_click['share']
	top_click = top_click[['session', 'aid']].drop_duplicates()
	# 最后时段的记录
	last_ts = test.groupby('session', as_index=False)['ts'].max()
	last_ts['ts_hour'] = last_ts['ts'] - (1 * 60 * 60)
	last_ts['ts_day'] = last_ts['ts'] - (24 * 60 * 60)
	last_ts = last_ts.rename(columns={'ts': 'ts_max'})
	last_actions = test.merge(last_ts, on='session', how='left')
	last_actions = last_actions.drop_duplicates()
	last_hour_actions = last_actions[
		(last_actions['ts'] >= last_action['ts_hour']) & (last_actions['ts'] != last_action['ts_max'])]
	last_day_actions = last_actions[
		(last_actions['ts'] >= last_action['ts_day']) & (last_actions['ts'] < last_action['ts_hour'])]
	# 非click数据
	buy_actions = test[test['type'] != 0]

	return last_action, top_click, last_hour_actions, last_day_actions, buy_actions


def w2v_co_matrix(w2v_path, knn=60):
	w2v = pd.read_parquet(w2v_path)
	model_knn = NearestNeighbors(n_neighbors=knn, metric='cosine')
	model_knn.fit(w2v.iloc[:, 1:])
	distances, indices = model_knn.kneighbors(w2v.iloc[:, 1:])
	co_matrix = pd.DataFrame(np.array([aid] * knn for aid in list(w2v['aid'])), columns='aid_x')
	co_matrix['aid_y'] = indices.reshape(-1)
	co_matrix['aid_y'] = co_matrix.apply(lambda x: list(w2v['aid'])[x['aid_y']], axis=1)
	co_matrix['share'] = distances.reshape(-1)
	co_matrix = co_matrix[co_matrix['aid_x'] != co_matrix['aid_y']]
	return co_matrix


def make_action_datamart(co_matrix, action_df, feature_name, rank, save_path, prefix, w2v=False):
	if w2v:
		action_df = action_df.merge(co_matrix, left_on='aid', right_on='aid_x', how='inner')
		action_df = action_df.groupby(['session', 'aid_y'], as_index=False)['share'].mean()
		action_df = action_df.sort_values(['session', 'share'], ascending=[True, False]).reset_index(drop=True)
	else:
		action_df = action_df.merge(co_matrix, left_on='aid', right_on='aid_x', how='inner')
		action_df = action_df.groupby(['session', 'aid_y'], as_index=False).agg({
			'share':'sum', 'cart': 'mean'
		})
		action_df = action_df.sort_values(
			by=['session', 'share', 'cart_cvr'],
			ascending=[True, False, False]
		)
	action_df = action_df[['session', 'aid_y', 'share']]
	action_df['rank'] = 1
	action_df['rank'] = action_df.groupby('session')['rank'].cusum()
	action_df = action_df[action_df['rank'] <= rank]
	action_df.columns = ['session', 'aid', feature_name, 'rank']
	action_df.to_parquet(f'{save_path}{feature_name}_{prefix}.parquet', index=False)


def make_action_hour_day_datamart(co_matrix, action_df, feature_name, rank, save_path, prefix, w2v=False):
	chunk = 20000
	session_list = list(set(action_df['session']))
	chunk_num = int(len(session_list)/chunk) + 1
	datamart_list = []
	for i in range(chunk_num):
		start = i * chunk
		end = (i+1) * chunk
		action_df_part = action_df[action_df['session'].isin(session_list[start:end])]
		action_df_part = action_df_part.merge(co_matrix, left_on='aid', right_on='aid_x', how='left')
		if w2v:
			action_df_part = action_df_part.groupby(['session', 'aid_y'], as_index=False)['share'].mean()
			action_df_part = action_df_part.sort_values(['session', 'share'], ascending=[True, False])
		else:
			action_df_part = action_df_part.groupby(['session', 'aid_y'], as_index=False)['share'].sum()
			action_df_part = action_df_part.sort_values(['session', 'share'], ascending=[True, False])
		action_df_part['rank'] = 1
		action_df_part['rank'] = action_df_part.groupby('session')['rank'].cusum()
		action_df_part = action_df_part[action_df_part['rank'] <= rank]
		action_df_part.columns = ['session', 'aid', feature_name, 'rank']
		datamart_list.append(action_df_part)
	action_df = pd.concat(datamart_list, axis=0, ignore_index=True)
	action_df.to_parquet(f'{save_path}{feature_name}_{prefix}.parquet', index=False)


def make_cart_cvr(df):
	chunk = 7000
	chunk_num = int(len(set(df['aid']))/chunk) + 1
	aid_list = list(set(df['aid']))
	cvr_list = []
	for i in range(chunk_num):
		start = i * chunk
		end = (i+1) * chunk
		click_actions = df[df['aid'].isin(aid_list[start:end]) & (df['type'] == 0)]
		click_actions = click_actions[['session', 'aid']].drop_duplicates()
		click_all = click_actions.groupby('aid', as_index=False)['session'].count()
		click_all = click_all.rename(columns={'session': 'click_n'})
		cart_actions = df[df['aid'].isin(aid_list[start:end]) & (df['type'] == 1) & df['session'].isin(list(click_actions['session']))]
		cart_actions = cart_actions[['session', 'aid']].drop_duplicates()
		cart_all = cart_actions.groupby('aid', as_index=False)['session'].count()
		cart_all = cart_all.rename(columns={'session': 'cart_n'})
		click_all = click_all.merge(cart_all, on='aid', how='left')
		click_all['cart_cvr'] = click_all['cart_n'] / click_all['click_n']
		cvr_list.append(click_all)
	cart_cvr = pd.concat(cvr_list, axis=0, ignore_index=True)
	cart_cvr = cart_cvr.fillna(0)
	mean_cvr = cart_cvr['cart_cvr'].mean()
	# 对于click_n<4的随机性太大， 降低cvr的权重
	cart_cvr.loc[cart_cvr['click_n'] < 4, 'cart_cvr'] = cart_cvr['cart_cvr'] * mean_cvr
	return cart_cvr


def get_use_aids(pattern, df, start_type, end_type, cutline, chunk):
	aid_count_df =[]
	aid_list = list(set(df['aid']))
	chunk_num = int(len(aid_list)/chunk) + 1
	for i in range(chunk_num):
		start = i * chunk
		end = (i+1) * chunk
		# 各种type行为互相交叉生成配对
		if start_type == 'click':
			row = df[df['aid'].isin(aid_list[start:end]) & df['type'] == 0]
		else:
			row = df[df['aid'].isin(aid_list[start:end]) & df['type'] == 1]
		if end_type == 'click':
			row = row.merge(df[df['type'] == 0], on='session', how='inner')
		else:
			row = row.merge(df[df['type'] != 0], on='session', how='inner')
		# 根据pattern对配对进行筛选
		if pattern == 'allterm':
			row = row[['session', 'aid_x', 'aid_y']].drop_duplicates()
		if pattern == 'base':
			row = row[row['ts_y'] - row['ts_x'] >= 0]
			row = row[['session', 'aid_x', 'aid_y']].drop_duplicates()
		if pattern == 'base_wlen':
			row = row[row['ts_y'] - row['ts_x'] >= 0]
		if pattern == 'base_hour':
			row = row[row['ts_y'] - row['ts_x'] >= 0]
			row = row[row['ts_y'] - row['ts_x'] <= 3600]
			row = row[['session', 'aid_x', 'aid_y']].drop_duplicates()
		if pattern == 'dup':
			pass
		if pattern == 'dup_wlen':
			pass
		if pattern == 'dup_hour':
			row = row[row['ts_y'] - row['ts_x'] >= 0]
			row = row[row['ts_y'] - row['ts_x'] <= 3600]
		# 对关联aid进行计数作为热度
		aid_count = row.groupby('aid_x', as_index=False)['session'].count()
		aid_count.columns = ['aid', 'n']
		aid_count_df.append(aid_count)

	aid_count_df = pd.concat(aid_count_df, axis=0, ignore_index=True)
	low_count_aids = list(aid_count_df[aid_count_df['n'] < cutline]['aid'])
	high_count_aids = list(aid_count_df[aid_count_df['n'] >= cutline]['aid'])

	return low_count_aids, high_count_aids, aid_count_df


def make_chunk_co_matrix(pattern, df, use_aids, cart_cvr_df, start_type, end_type, same_feature_name, cut_rank, chunk):
	co_matrix_df = []
	co_matrix_same_df = []
	chunk_num = int(len(use_aids)/chunk) + 1
	for i in range(chunk_num):
		start = i * chunk
		end = (i+1) * chunk
		# 各种type行为互相交叉生成配对
		if start_type == 'click':
			row = df[df['aid'].isin(use_aids[start:end]) & df['type'] == 0]
		else:
			row = df[df['aid'].isin(use_aids[start:end]) & df['type'] == 1]
		if end_type == 'click':
			row = row.merge(df[df['type'] == 0], on='session', how='inner')
		else:
			row = row.merge(df[df['type'] != 0], on='session', how='inner')
		# 根据pattern对配对进行筛选
		if pattern == 'allterm':
			# 以去重以后的配对数作为权重
			row = row[['session', 'aid_x', 'aid_y']].drop_duplicates()
			row = row.groupby(['aid_x', 'aid_y'], as_index=False)['session'].count()
		if pattern == 'base':
			# 以aid时间之后的去重配对数作为权重
			row = row[row['ts_y'] - row['ts_x'] >= 0]
			row = row[['session', 'aid_x', 'aid_y']].drop_duplicates()
			row = row.groupby(['aid_x', 'aid_y'], as_index=False)['session'].count()
		if pattern == 'base_wlen':
			# 以aid时间之后的时间差升序排序倒数作为权重
			row = row[row['ts_y'] - row['ts_x'] >= 0]
			row['ts_diff'] = np.abs(row['ts_y'] - row['ts_x'])
			row['diff_rank'] = row.groupby(['session', 'aid_x'])['ts_diff'].rank(method='min')
			row['diff_weight'] = 1 / row['diff_rank']
			row = row.groupby(['aid_x', 'aid_y'], as_index=False)['diff_weight'].sum()
		if pattern == 'base_hour':
			# 以aid一个小时之内去重配对数作为权重
			row = row[row['ts_y'] - row['ts_x'] >= 0]
			row = row[row['ts_y'] - row['ts_x'] <= 3600]
			row = row[['session', 'aid_x', 'aid_y']].drop_duplicates()
			row = row.groupby(['aid_x', 'aid_y'], as_index=False)['session'].count()
		if pattern == 'dup':
			# 以配对数作为权重
			row = row.groupby(['aid_x', 'aid_y'], as_index=False)['session'].count()
		if pattern == 'dup_wlen':
			# 以时间差升序排序倒数作为权重
			row['ts_diff'] = np.abs(row['ts_y'] - row['ts_x'])
			row['diff_rank'] = row.groupby(['session', 'aid_x'])['ts_diff'].rank(method='min')
			row['diff_weight'] = 1 / row['diff_rank']
			row = row.groupby(['aid_x', 'aid_y'], as_index=False)['diff_weight'].sum()
		if pattern == 'dup_hour':
			# 以aid一个小时之内配对数作为权重
			row = row[row['ts_y'] - row['ts_x'] >= 0]
			row = row[row['ts_y'] - row['ts_x'] <= 3600]
			row = row.groupby(['aid_x', 'aid_y'], as_index=False)['session'].count()

		row.columns = ['aid_x', 'aid_y', 'n']
		total = row.groupby('aid_x', as_index=False)['n'].sum()
		total.columns = ['aid_x', 'n_total']
		row = row.merge(total, on='aid_x', how='left')
		row['share'] = row['n'] / row['n_total']
		row = row[['aid_x', 'aid_y', 'share']].merge(
			cart_cvr_df[['aid', 'cart_cvr']],
			left_on='aid_y',
			right_on='aid'
		)
		row = row[['aid_x', 'aid_y', 'share', 'cart_cvr']].sort_values(
			['aid_x', 'share', 'cart_cvr'],
			[True, False, False]
		)
		row['rank'] = 1
		row['rank'] = row.groupby('aid_x')['rank'].cumsum()
		# aid自身之间的配对权重作为特征输出
		row_same_aid = row[row['aid_x'] == row['aid_y']]
		row = row[row['aid_x'] == row['aid_y']]
		row = row[row['rank'] <= cut_rank]
		co_matrix_df.append(row)
		co_matrix_same_df.append(row_same_aid)

	co_matrix_df = pd.concat(co_matrix_df, axis=0, ignore_index=True)
	co_matrix_same_df = pd.concat(co_matrix_same_df, axis=0, ignore_index=True)
	co_matrix_same_df = co_matrix_same_df[['aid_x', 'share']]
	co_matrix_same_df.columns = ['aid', same_feature_name]
	return co_matrix_df, co_matrix_same_df


def aug_data(co_matrix_df, base_aug, cart_cvr_df):
	high_asso_df = base_aug[base_aug['rank'] <= 2]
	high_rank_df = co_matrix_df[co_matrix_df['rank'] <= 3]
	# 做了aid之间的二次关联
	aug_data = high_rank_df.merge(high_asso_df, left_on='aid_y', right_on='aid_x', how='inner')
	aug_data = aug_data[aug_data['aid_x_x'] != aug_data['aid_y_y']]
	aug_data['share'] = aug_data['share_x'] * aug_data['share_y']
	aug_data = aug_data.groupby(['aid_x_x', 'aid_y_y'], as_index=False)['share'].sum()
	aug_data.columns = ['aid_x', 'aid_y', 'share']
	co_matrix_df = pd.concat([co_matrix_df[['aid_x', 'aid_y', 'share']], aug_data], axis=0)
	co_matrix_df = co_matrix_df.groupby(['aid_x', 'aid_y'], as_index=False)['share'].max()
	co_matrix_df = co_matrix_df.merge(cart_cvr_df, left_on='aid_y', right_on='aid', how='left')
	co_matrix_df = co_matrix_df.reset_index(drop=True)
	return co_matrix_df


def main(co_dict):
	MODE, DIM = 'valid', 16
	assert MODE in ['test', 'valid']
	assert DIM in [16, 64]
	data_path = f'data/train_{MODE}/'
	preprocess_path = 'data/preprocess/'
	feature_path = 'data/feature/'
	w2v_path = f'{preprocess_path}w2v_{DIM}_{MODE}.parquet'

	# 读取数据
	train = pd.read_parquet(data_path + 'train.parquet')
	test = pd.read_parquet(data_path + 'test.parquet')
	df = pd.concat([train, test], axis=0, ignore_index=True)
	last_action, top_click, last_hour_actions, last_day_actions, buy_actions = make_action(test)
	same_aid_df = df[['aid']].drop_duplicates()

	for pattern in co_dict.keys():
		for j in range(len(co_dict[pattern])):
			start_type = co_dict[pattern][j][0]
			end_type = co_dict[pattern][j][1]
			cut_line = co_dict[pattern][j][2]
			cut_rank = co_dict[pattern][j][3]
			cut_datamart_last = co_dict[pattern][j][4]
			cut_datamart_top = co_dict[pattern][j][5]
			cut_datamart_hour = co_dict[pattern][j][6]
			cut_datamart_day = co_dict[pattern][j][7]
			action_pattern_list = co_dict[pattern][j][8]

			# 以w2v作为关联依据
			if pattern == 'w2v':
				# 以w2v建立aid2aid的相关性，相关系数是w2v的余弦相似性
				co_matrix = w2v_co_matrix(w2v_path)
				if 'last' in action_pattern_list:
					feature_name = f'{start_type}_{end_type}_{pattern}_last_w2v'
					# 以last_action的aid作为起点，利用co_matrix关联aid
					make_action_datamart(
						co_matrix, last_action, feature_name, cut_datamart_last,
						feature_path, MODE, w2v=True
					)
				if 'hour' in action_pattern_list:
					feature_name = f'{start_type}_{end_type}_{pattern}_hour_w2v'
					# 以last_hour_actions的aid作为起点，利用co_matrix关联aid
					# w2v判断session对相同关联的aid的share用mean还是sum
					make_action_hour_day_datamart(
						co_matrix, last_hour_actions, feature_name, cut_datamart_hour,
						feature_path, MODE, w2v=True
					)
			else:
				same_feature_name = f'same_{start_type}_{end_type}_{pattern}'
				# aid在session中都有click_cart记录的比值
				cart_cvr_df = make_cart_cvr(df)
				# 在click,cart相互交叉配对，根据不同模式筛选，统计aid的计数
				low_count_aids, high_count_aids, aid_count_df = get_use_aids(
					pattern, df, start_type, end_type, cut_line, chunk=20000
				)
				# -----high_count_aids
				# 建立aid2aid的相关性，相关系数是各种模型下的配对计数或者时间差排序倒数
				high_co_matrix, high_same_aids = make_chunk_co_matrix(
					pattern, df, high_count_aids, cart_cvr_df,
					start_type, end_type, same_feature_name, cut_rank, chunk=20000
				)
				# 以high_co_matrix作为基准做aid之间的二次关联
				base_aug = high_co_matrix.copy()
				high_co_matrix = aug_data(high_co_matrix, base_aug, cart_cvr_df)
				# -----low_count_aids
				low_co_matrix, low_same_aids = make_chunk_co_matrix(
					pattern, df, low_count_aids, cart_cvr_df,
					start_type, end_type, same_feature_name, cut_rank, chunk=20000
				)
				low_co_matrix = aug_data(low_co_matrix, base_aug, cart_cvr_df)
				# -----合并相关性数据
				co_matrix = pd.concat([high_co_matrix, low_co_matrix], axis=0, ignore_index=True)
				same_aids = pd.concat([high_same_aids, low_same_aids], axis=0, ignore_index=True)
				same_aid_df = same_aid_df.merge(same_aids, on='aid', how='left')
				# 格式 ['session', 'aid', feature_name, 'rank']
				if 'last' in action_pattern_list:
					feature_name = f'{start_type}_{end_type}_{pattern}_last'
					make_action_datamart(
						co_matrix, last_action, feature_name, cut_datamart_last,
						feature_path, MODE
					)
				if 'top' in action_pattern_list:
					feature_name = f'{start_type}_{end_type}_{pattern}_top'
					make_action_datamart(
						co_matrix, top_click, feature_name, cut_datamart_top,
						feature_path, MODE
					)
				if 'hour' in action_pattern_list:
					feature_name = f'{start_type}_{end_type}_{pattern}_hour'
					make_action_hour_day_datamart(
						co_matrix, last_hour_actions, feature_name, cut_datamart_hour,
						feature_path, MODE
					)
				if 'day' in action_pattern_list:
					feature_name = f'{start_type}_{end_type}_{pattern}_day'
					make_action_hour_day_datamart(
						co_matrix, last_day_actions, feature_name, cut_datamart_day,
						feature_path, MODE
					)
				if 'all' in action_pattern_list:
					feature_name = f'{start_type}_{end_type}_{pattern}_all'
					make_action_datamart(
						co_matrix, buy_actions, feature_name, 200,
						feature_path, MODE
					)

	same_aid_df.to_parquet(f'{feature_path}same_aid_feature_{MODE}.parquet', index=False)


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
main(co_dict)


