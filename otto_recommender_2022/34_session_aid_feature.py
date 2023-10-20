import pandas as pd
import numpy as np


def session_day_feature(mode, test, feature_path):
	# to_datetime unit=ns, 默认
	# ???原代码中test['ts']+2 * 60* 60 加两个小时
	test['date'] = pd.to_datetime(test['ts'] * 1e9)
	test['dow'] = test.date.dt.dayofweek
	test['day'] = test.date.dt.dayofyear
	test['hour'] = test.date.dt.hour
	
	# aid在每个day中的占比
	aid_day_n = test.groupby(['aid', 'day'], as_index=False)['session'].count()
	aid_day_n = aid_day_n.renmae(columns={'session': 'aid_day_n'})
	day_n = test.groupby('day', as_index=False)['session'].count()
	day_n = day_n.renmae(columns={'session': 'day_n'})
	aid_day_n = aid_day_n.merge(day_n, on='day', how='left')
	aid_day_n['aid_day_share'] = aid_day_n['aid_day_n'] / aid_day_n['day_n']
	aid_day_n = aid_day_n[['aid', 'day', 'aid_day_share']]
	
	# 去重以后aid在每个day中的占比
	aid_day_nunique = test.groupby(['aid', 'day'], as_index=False)['session'].nunique()
	aid_day_nunique = aid_day_nunique.renmae(columns={'session': 'aid_day_nunique'})
	day_nunique = test.groupby('day', as_index=False)['session'].nunique()
	day_nunique = day_nunique.renmae(columns={'session': 'day_nunique'})
	aid_day_nunique = aid_day_nunique.merge(day_nunique, on='day', how='left')
	aid_day_nunique['aid_day_nunique_share'] = aid_day_nunique['aid_day_nunique'] / aid_day_nunique['day_nunique']
	aid_day_nunique = aid_day_nunique[['aid', 'day', 'aid_day_nunique_share']]

	aid_day_cart_nunique = test[test['type'] == 1].groupby(['aid', 'day'], as_index=False)['session'].nunique()
	aid_day_cart_nunique = aid_day_cart_nunique.renmae(columns={'session': 'aid_day_cart_nunique'})
	day_cart_nunique = test[test['type'] == 1].groupby('day', as_index=False)['session'].nunique()
	day_cart_nunique = day_cart_nunique.renmae(columns={'session': 'day_cart_nunique'})
	aid_day_cart_nunique = aid_day_cart_nunique.merge(day_cart_nunique, on='day', how='left')
	aid_day_cart_nunique['aid_day_cart_nunique_share'] = aid_day_cart_nunique['aid_day_cart_nunique'] / aid_day_cart_nunique['day_cart_nunique']
	aid_day_cart_nunique = aid_day_cart_nunique[['aid', 'day', 'aid_day_cart_nunique_share']]

	aid_day_order_nunique = test[test['type'] == 2].groupby(['aid', 'day'], as_index=False)['session'].nunique()
	aid_day_order_nunique = aid_day_order_nunique.renmae(columns={'session': 'aid_day_order_nunique'})
	day_order_nunique = test[test['type'] == 2].groupby('day', as_index=False)['session'].nunique()
	day_order_nunique = day_order_nunique.renmae(columns={'session': 'day_order_nunique'})
	aid_day_order_nunique = aid_day_order_nunique.merge(day_order_nunique, on='day', how='left')
	aid_day_order_nunique['aid_day_order_nunique_share'] = aid_day_order_nunique['aid_day_order_nunique'] / aid_day_order_nunique['day_order_nunique']
	aid_day_order_nunique = aid_day_order_nunique[['aid', 'day', 'aid_day_order_nunique_share']]

	aid_day_n = aid_day_n.merge(aid_day_nunique, on=['aid', 'day'], how='left')
	aid_day_n = aid_day_n.merge(aid_day_cart_nunique, on=['aid', 'day'], how='left')
	aid_day_n = aid_day_n.merge(aid_day_order_nunique, on=['aid', 'day'], how='left')
	aid_day_n = aid_day_n.fillna(0)
	aid_day_n.to_parquet(
		f'{feature_path}session_day_feature_{mode}.parquet',
		index=False
	)


def session_feature(mode, test, feature_path):
	test['date'] = pd.to_datetime(test['ts'] * 1e9)
	test['dow'] = test.date.dt.dayofweek
	test['day'] = test.date.dt.dayofyear
	test['hour'] = test.date.dt.hour

	# session各种type的占比
	type2id = {'clicks': 0, 'carts': 1, 'orders': 2}
	session_type_n = test.grpuby(['session', 'type'], as_index=False).size()
	session_type_n = session_type_n.rename(columns={'size': 'session_type_n'})
	session_n = test.grpuby('session', as_index=False).size()
	session_n = session_n.rename(columns={'size': 'session_n'})
	for t in type2id.keys():
		tmp = session_type_n[session_type_n['type'] == type2id[t]][['session', 'session_type_n']]
		session_n = session_n.merge(tmp, on='session', how='left')
		session_n = session_n.rename(columns={'session_type_n': f'session_{t}_n'})
		session_n[f'{t}_ratio'] = session_n[f'session_{t}_n'] / session_n['session_n']
		if t in ['carts', 'orders']:
			session_n[f'session_{t}_cvr'] = session_n[f'session_{t}_n'] / session_n['session_clicks_n']
		session_n = session_n.drop(f'session_{t}_n', axis=1)

	max_ts = test.groupby('session', as_index=False)['ts'].max()
	max_ts = max_ts.rename(columns={'ts': 'max_ts'})
	test = test.merge(max_ts, on='session', how='left')
	test_lastday = test[test['ts'] >= (test['max_ts'] - 24*60*60)]
	session_type_lastday_n = test_lastday.grpuby(['session', 'type'], as_index=False).size()
	session_type_lastday_n = session_type_lastday_n.rename(columns={'size': 'session_type_lastday_n'})
	session_lastday_n = test_lastday.grpuby('session', as_index=False).size()
	session_lastday_n = session_lastday_n.rename(columns={'size': 'session_lastday_n'})
	for t in type2id.keys():
		tmp = session_type_lastday_n[session_type_lastday_n['type'] == type2id[t]][['session', 'session_type_lastday_n']]
		session_lastday_n = session_lastday_n.merge(tmp, on='session', how='left')
		session_lastday_n = session_lastday_n.rename(columns={'session_type_lastday_n': f'session_{t}_lastday_n'})
		session_lastday_n[f'{t}_lastday_ratio'] = session_lastday_n[f'session_{t}_lastday_n'] / session_n['session_lastday_n']
		if t in ['carts', 'orders']:
			session_lastday_n[f'session_{t}_lastday_cvr'] = session_lastday_n[f'session_{t}_lastday_n'] / session_lastday_n['session_clicks_lastday_n']
		session_lastday_n = session_lastday_n.drop(f'session_{t}_lastday_n', axis=1)
	session_lastday_n = session_lastday_n.drop('session_lastday_n', axis=1)

	# session不同的aid, ts去重计数
	session_nunqiue = test.groupby('session', as_index=False).agg({'aid': 'nunique', 'ts': 'nunique'})
	session_nunqiue = session_nunqiue.rename(columns={'aid': 'aid_nunique', 'ts': 'ts_nunique'})
	# session时间长度
	session_ts_length = test.groupby('session', as_index=False)['ts'].agg({'ts_max': 'max', 'ts_min': 'min'})
	session_ts_length['ts_length'] = session_ts_length['ts_max'] - session_ts_length['ts_min']
	session_ts_length = session_ts_length[['session', 'ts_length']]
	# session aid之间的时间差
	session_ts_unique = test[['session', 'ts']].drop_duplicates()
	session_ts_unique['ts_diff'] = session_ts_unique.groupby('session')['ts'].diff()
	session_last_ts_diff = session_ts_unique.groupby('session', as_index=False)['ts_diff'].last()
	session_last_ts_diff = session_last_ts_diff.rename({'ts_diff': 'ts_last_diff'})
	session_ts_diff_mean = session_ts_unique.groupby('session', as_index=False)['ts_diff'].mean()
	session_ts_diff_mean = session_ts_diff_mean.rename({'ts_diff': 'ts_diff_mean'})
	# session最后一条记录
	session_last = test.groupby('session', as_index=False)['aid', 'day', 'hour', 'dow', 'type'].last()
	# day用来merge sesson_day的特征
	session_last.columns = ['session', 'aid', 'day', 'hour_last', 'dow_last', 'type']

	session_n = session_n.merge(session_lastday_n, on='session', how='left')
	session_n = session_n.merge(session_nunqiue, on='session', how='left')
	session_n = session_n.merge(session_ts_length, on='session', how='left')
	session_n = session_n.merge(session_last_ts_diff, on='session', how='left')
	session_n = session_n.merge(session_ts_diff_mean, on='session', how='left')
	# session 每条记录的时间间隔
	session_n['n_per_ts'] = session_n['ts_length'] / session_n['session_n']
	# session aid的重复度
	session_n['n_per_aid_nunique'] = session_n['aid_nunique'] / session_n['session_n']
	# session 记录的集中度
	session_n['ts_per_length'] = session_n['ts_length'] / session_n['ts_nunique']
	session_n = session_n.merge(session_last, on='session', how='left')
	session_n = session_n.fillna(0)

	session_n.to_parquet(
		f'{feature_path}session_feature_{mode}.parquet',
		index=False
	)


def aid_feature(mode, df, feature_path):
	id2type = {0: 'clicks', 1: 'carts', 2: 'orders'}
	aid_feature = df[['aid']].drop_duplicates().reset_index()

	# aid在session中从click到cart或者order的转化率，aid必须在session中有click
	row_click = df[df['type'] == 0]
	row_cart = df[(df['type'] == 1) & df['session'].isin(set(row_click['session']))]
	row_order = df[(df['type'] == 2) & df['session'].isin(set(row_click['session']))]
	click_all = row_click.groupby('aid', as_index=False)['session'].nunique()
	click_all = click_all.rename(columns={'session': 'click_n'})
	cart_all = row_cart.groupby('aid', as_index=False)['session'].nunique()
	cart_all = cart_all.rename(columns={'session': 'cart_n'})
	order_all = row_order.groupby('aid', as_index=False)['session'].nunique()
	order_all = order_all.rename(columns={'session': 'order_n'})
	click_all = click_all.merge(cart_all, on='aid', how='left')
	click_all = click_all.merge(order_all, on='aid', how='left')
	click_all = click_all.fillna(0)
	click_all['cart_hit_cvr'] = click_all['cart_n'] / click_all['click_n']
	click_all['order_hit_cvr'] = click_all['order_n'] / click_all['click_n']
	# aid在session中从click到cart或者order的转化率，aid无论在session中有没有click
	row_cart = df[(df['type'] == 1)]
	row_order = df[(df['type'] == 2)]
	cart_all = row_cart.groupby('aid', as_index=False)['session'].nunique()
	cart_all = cart_all.rename(columns={'session': 'cart_true_n'})
	order_all = row_order.groupby('aid', as_index=False)['session'].nunique()
	order_all = order_all.rename(columns={'session': 'order_true_n'})
	click_all = click_all.merge(cart_all, on='aid', how='left')
	click_all = click_all.merge(order_all, on='aid', how='left')
	click_all = click_all.fillna(0)
	click_all['cart_true_hit_cvr'] = click_all['cart_true_n'] / click_all['click_n']
	click_all['order_true_hit_cvr'] = click_all['order_true_n'] / click_all['click_n']
	click_all = click_all.drop(['click_n', 'cart_n', 'order_n', 'cart_true_n', 'order_true_n'], axis=1)

	# session type前后type配对的记录数占比
	df['ts_before'] = df.groupby('session')['ts'].shift()
	df['type_'] = df['type'].map(id2type)
	df['type_before'] = df.grouby('session')['type_'].shift()
	type_pairs = df.dropna()
	type_pairs['type_cvr'] = type_pairs['type_before'] + '_' + type_pairs['type']
	type_pairs['ts_diff'] = type_pairs['ts'] - type_pairs['ts_before']
	type_pairs_count = type_pairs.groupby(['aid', 'type_cvr'], as_index=False).size()
	type_pairs_count = type_pairs_count.pivot_table(
		index=['aid'], columns=['type_cvr'], values=['size'], fill_value=0
	).reset_index()
	type_pairs_count.columns = ['aid'] + [f'{item}[1]_ratio' for item in type_pairs.columns]
	type_pairs_count['cur_sum'] = type_pairs_count[[col for col in type_pairs_count.columns if col != 'aid']].sum(axis=1)
	for col in [col for col in type_pairs_count.columns if col != 'aid']:
		type_pairs_count[col] = type_pairs_count[col] / type_pairs_count['cur_sum']
	type_pairs_count['cur_sum'] = type_pairs_count['cur_sum'] / type_pairs_count['cur_sum'].sum()
	df = df.drop('type_', axis=1)
	# session type前后type配对的时间差平均值
	type_pairs_tsdiff_mean = type_pairs.groupby(['aid', 'type_cvr'], as_index=False)['ts_diff'].mean()
	type_pairs_tsdiff_mean = type_pairs_tsdiff_mean.pivot_data(
		index=['aid'], columns=['type_cvr'], values=['ts_diff'], fill_value=0
	).reset_index()
	type_pairs_tsdiff_mean.columns = ['aid'] + [f'{item}[1]_tsdiff_mean' for item in type_pairs_tsdiff_mean.columns]

	# aid中session直接cart没有click的占比
	cart_nunique = df[df['type'] == 1].groupby('aid', as_index=False)['session'].nunique()
	cart_nunique = cart_nunique.rename(columns={'session': 'cart_n'})
	cart_session = set(df[df['type'] == 1]['session'])
	cart_skip = df[df['session'].isin(cart_session)]
	cart_skip = cart_skip.pivot_table(
		index=['session', 'aid'], columns=['type'], values=['ts'], aggfunc='count', fill_value=0
	).reset_index()
	cart_skip.columns = ['session', 'aid'] + [f'{item}[1]' for item in cart_skip.columns]
	cart_skip = cart_skip[(cart_skip['1'] - cart_skip['0']) > 0]
	cart_skip = cart_skip.groupby('aid', as_index=False)['session'].nunique()
	cart_skip = cart_skip.rename(columns={'session': 'cart_click_skip_n'})
	cart_nunique = cart_nunique.merge(cart_skip, on='aid', how='left')
	cart_nunique = cart_nunique.fillna(0)
	cart_nunique['cart_click_skip_ratio'] = cart_nunique['cart_click_skip_n'] / cart_nunique['cart_n']
	cart_nunique = cart_nunique[['aid', 'cart_click_skip_ratio']]
	# aid中session直接order没有click的占比
	# aid中session直接order没有cart的占比
	order_nunique = df[df['type'] == 2].groupby('aid', as_index=False)['session'].nunique()
	order_nunique = order_nunique.rename(columns={'session': 'order_n'})
	order_session = set(df[df['type'] == 2]['session'])
	order_skip = df[df['session'].isin(order_session)]
	order_skip = order_skip.pivot_table(
		index=['session', 'aid'], columns=['type'], values=['ts'], aggfunc='count', fill_value=0
	).reset_index()
	order_skip.columns = ['session', 'aid'] + [f'{item}[1]' for item in cart_skip.columns]
	order_click_skip = cart_skip[(cart_skip['2'] - cart_skip['0']) > 0]
	order_click_skip = order_click_skip.groupby('aid', as_index=False)['session'].nunique()
	order_click_skip = order_click_skip.rename(columns={'session': 'order_click_skip_n'})
	order_cart_skip = cart_skip[(cart_skip['2'] - cart_skip['1']) > 0]
	order_cart_skip = order_cart_skip.grouby('aid', as_index=False)['session'].nunique()
	order_cart_skip = order_cart_skip.rename(columns={'session': 'order_cart_skip_n'})
	order_nunique = order_nunique.merge(order_click_skip, on='aid', hoe='left')
	order_nunique = order_nunique.merge(order_cart_skip, on='aid', how='left')
	order_nunique['order_click_skip_ratio'] = order_nunique['order_click_skip_n'] / order_nunique['order_n']
	order_nunique['order_cart_skip_ratio'] = order_nunique['order_cart_skip_n'] / order_nunique['order_n']
	order_nunique = order_nunique[['aid', 'order_click_skip_ratio', 'order_cart_skip_ratio']]

	# aid在session的重复度
	aid_session_nunique = df.groupby(['aid', 'type'], as_index=False)['session'].nunique()
	aid_session_nunique = aid_session_nunique.rename(columns={'session': 'nunique'})
	aid_count = df.groupby(['aid', 'type'], as_index=False).size()
	aid_count = aid_count.rename(columns={'size': 'n'})
	repeat_feature = aid_count.merge(aid_session_nunique, on=['aid', 'type'], how='left')
	repeat_feature = repeat_feature.fillna(0)
	repeat_feature['repeat_ratio'] = repeat_feature['nunique'] / repeat_feature['n']
	repeat_feature = repeat_feature.pivot_table(
		index=['aid'], columns=['type'], values=['repeat_ratio'], fill_value=0
	).reset_index()
	repeat_feature.columns = ['aid'] + [f'{id2type}[int({item}[1])]_repeat_ratio' for item in repeat_feature.columns if item[0] != 'aid']

	# 合并特征
	aid_feature = aid_feature.merge(click_all, on='aid', how='left')
	aid_feature = aid_feature.merge(type_pairs_count, on='aid', how='left')
	aid_feature = aid_feature.merge(type_pairs_tsdiff_mean, on='aid', how='left')
	aid_feature = aid_feature.merge(cart_nunique, on='aid', how='left')
	aid_feature = aid_feature.merge(order_nunique, on='aid', how='left')
	aid_feature = aid_feature.merge(repeat_feature, on='aid', how='left')
	aid_feature = aid_feature.fillna(0)

	aid_feature.to_parquet(
		f'{feature_path}aid_feature_{mode}.parquet',
		index=False
	)


def last_chunk_session_aid(mode, test, feature_path):
	id2type = {0: 'clicks', 1: 'carts', 2: 'orders'}

	last_chunk = test.copy()
	last_chunk['ts_diff'] = last_chunk.groupby('session')['ts'].diff()
	last_chunk = last_chunk.dropna()
	last_chunk['chunk_flag'] = np.where(last_chunk['ts_diff'] < 3600, 0, 1)
	# 连续间隔在一小时之内
	last_chunk['chunk'] = last_chunk.groupby('session')['chunk_flag'].cumsum()
	max_chunk = last_chunk.groupby('session', as_index=False)['chunk'].max()
	max_chunk = max_chunk.rename(columns={'chunk': 'max_chunk'})
	last_chunk = last_chunk.merge(max_chunk, on='session', how='left')
	chunk_count = last_chunk.groupby(['session', 'chunk'], as_index=False).size()
	chunk_count = chunk_count.rename(columns={'size': 'chunk_counts'})
	# session 每个连续一小时间隔平均多少条记录
	chunk_count = chunk_count.groupby('session', as_index=False)['chunk_counts'].agg({'session_counts_mean': 'mean', 'session_counts_min': 'min'})

	end_chunk = last_chunk[last_chunk['chunk'] == last_chunk['max_chunk']]
	end_chunk = end_chunk.pivot_table(
		index=['session', 'aid'], columns=['type'], values=['ts'], aggfunc='count', fill_value=0
	).reset_index()
	# 在session最后一个连续一小时间隔内, aid有多少条记录
	end_chunk.columns = ['session', 'aid'] + [f'end_chunk_{id2type}[int({item}[1])]' for item in end_chunk.columns if item[0] not in ['session', 'aid']]
	end_chunk['end_chunk_aid_total'] = end_chunk.iloc[:, 2:].sum(axis=1)
	end_chunk = end_chunk.merge(max_chunk, on='session', how='left')
	end_chunk = end_chunk.merge(chunk_count, on='session', how='left')
	end_chunk_total = end_chunk.groupby('session', as_index=False)['end_chunk_aid_total'].agg({'end_chunk_total': 'sum'})
	end_chunk = end_chunk.merge(end_chunk_total, on='session', how='left')
	end_chunk['end_chunk_aid_ratio'] = end_chunk['end_chunk_aid_total'] / end_chunk['end_chunk_total']
	end_chunk = end_chunk.fillna(0)

	end_chunk.to_parquet(
		f'{feature_path}last_chunk_session_aid_feature_{mode}.parquet',
		index=False
	)


def session_aid_feature(mode, test, feature_path):
	id2type = {0: 'clicks', 1: 'carts', 2: 'orders'}

	session_aid_feature = test.groupby(['session', 'aid'], as_index=False).size()
	session_aid_feature = session_aid_feature.rename(columns={'size': 'session_aid_n'})
	session_n = test.groupby('session', as_index=False).size()
	session_n = session_n.rename(columns={'size': 'session_n'})
	session_aid_feature = session_aid_feature.merge(session_n, on='session', how='left')
	# session 中aid的记录数占比
	session_aid_feature['session_aid_share'] = session_aid_feature['session_aid_n'] / session_feature['session_n']
	session_aid_feature = session_aid_feature.drop('session_n', axis=1)
	# session aid中各种type的记录数占比
	for t in id2type.keys():
		session_aid_t_n = test[test['type'] == t].groupby(['session', 'aid'], as_index=False).size()
		session_aid_t_n = session_aid_t_n.rename(columns={'size': 'session_aid_{t}_n'})
		session_aid_feature = session_aid_feature.merge(session_aid_t_n, on=['session', 'aid'], how='left')
		session_aid_feature = session_aid_feature.fillna(0)
		session_aid_feature[f'session_aid_{id2type}[t]_share'] = session_aid_feature['session_aid_t_n'] / session_aid_feature['session_aid_n']
		session_aid_feature = session_aid_feature.drop('session_aid_t_n', axis=1)

	# 最后时间段各种type的记录数
	max_ts = test.groupby('session', as_index=False)['ts'].agg({'ts_max': 'max'})
	test = test.merge(max_ts, on='session', how='left')
	time_periods = {'last_hour': 60 * 60, 'last_day': 24 * 60 * 60, 'last_week': 7 * 24 * 60 * 60}
	for tp in time_periods:
		for t in id2type.keys():
			tmp = test[(test['type'] == t) & (test['ts'] >= (test['max_ts'] - time_periods[tp]))]
			tmp = tmp.groupby(['session', 'aid'], as_index=False).size()
			tmp = tmp.rename(columns={'size': f'session_aid_{id2type}[t]_{tp}_n'})
			session_aid_feature = session_aid_feature.merge(tmp, on=['session', 'aid'], how='left')
	session_aid_feature = session_aid_feature.fillna(0)

	# session max_ts 和session 中aid的max_ts的时间差
	session_aid_max_ts = test.groupby(['session', 'aid'], as_index=False)['ts'].agg({'session_aid_max_ts': 'max'})
	session_aid_max_ts = session_aid_max_ts.merge(max_ts, on='session', how='left')
	session_aid_max_ts['last_action_ts_diff'] = session_aid_max_ts['max_ts'] - session_aid_max_ts['session_aid_max_ts']
	session_aid_feature = session_aid_feature.merge(session_aid_max_ts, on=['session', 'aid'], how='left')
	session_aid_feature = session_aid_feature.fillna(0)

	session_aid_feature.to_parquet(
		f'{feature_path}session_aid_feature_{mode}.parquet',
		index=False
	)


def session_action_feature(mode, test, feature_path):
	aid_feature = pd.read_parquet(f'{feature_path}aid_feature_{mode}.parquet')
	test = test.drop_duplicates(['session', 'aid', 'ts']).drop_duplicates().reset_index()
	aid_feature = aid_feature[['aid', 'cart_click_skip_ratio', 'order_click_skip_ratio', 'order_cart_skip_ratio']]
	test = test.merge(aid_feature, on='aid', how='left')
	session_aid_mean = test.groupby('session', as_index=False).agg({
		'cart_click_skip_ratio': 'mean',
		'order_click_skip_ratio': 'mean',
		'order_cart_skip_ratio': 'mean',
	})
	session_aid_mean.columns = ['session'] + [f'session_{col}' for col in session_aid_mean.columns if col != 'aid']
	session_aid_mean.to_parquet(
		f'{feature_path}session_action_feature_{mode}.parquet',
		index=False
	)


def main(mode):
	feature_path = 'data/feature/'
	data_path = 'data/train_{mode}/'
	test = pd.read_parquet(data_path + 'test.parquet')
	train = pd.read_parquet(data_path + 'train.parquet')
	df = pd.concat([train, test], axis=0, ignore_index=True)

	session_day_feature(mode, test, feature_path)
	session_feature(mode, test, feature_path)
	aid_feature(mode, df, test, feature_path)
	last_chunk_session_aid(mode, test, feature_path)
	session_aid_feature(mode, test, feature_path)
	session_action_feature(mode, test, feature_path)


MODE = 'valid'
assert MODE in ['test', 'valid']
main(MODE)




















