import pandas as pd
import numpy as np
from contextlib import contextmanager
import time
import faiss
from scipy import sparse
from typing import Tuple
import pickle


# https://github.com/ZonW/HM-Personalized-Fashion-Recommender
# https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324084
# https://zhuanlan.zhihu.com/p/513522302


transactions = pd.read_pickle('data/transactions_train.pkl')
users = pd.read_pickle('data/users.pkl')
items = pd.read_pickle('data/items.pkl')
tmp = transactions[['user', 'item']].merge(users[['user', 'age']], on='user', how='left')
# 24-26为购买记录最多的连续年龄组, 以某个年龄为中心扩展直到记录数和24-26一样
# 消除记录数不平均带来的popular_item的差异
age_volume_threshold = len(tmp.query('24 <= age <=26'))
age_volumes = {age: len(tmp.query('age == @age')) for age in range(16, 100)}
age_shifts = {}
for age in range(16, 100):
	for i in range(0, 100):
		low = age - i
		high = age + i
		age_volume = 0
		for j in range(low, high + 1):
			age_volume += age_volumes.get(j, 0)
		if age_volume >= age_volume_threshold:
			age_shifts[age] = i
			break
# candidates params
REPURCHASE_NUM_ITEMS = 60
POPULAR_WEEKS, POPULAR_NUM_ITEMS = 1, 60
AGE_POPULAR_WEEKS, AGE_POPULAR_NUM_ITEMS = 1, 12
COOC_WEEKS, COOC_THRESHOLD = 32, 15		# 150
OHE_DISTANCE_WEEKS, OHE_DISTANCE_NUM_ITEMS = 20, 60
CATEGORY_WEEKS, CATEGORY_NUM_ITEMS = 1, 6
# feature params
USER_TRANSACTION_FEATURES_WEEKS, ITEM_TRANSACTION_FEATURES_WEEKS = 50, 16
ITEM_VOLUME_FEATURES_WEEKS, USER_VOLUME_FEATURES_WEEKS = 20, 50
USER_ITEM_VOLUME_FEATURES_WEEKS = 16
AGE_VOLUME_FEATURES_WEEKS = 1


@ contextmanager
def timer(name):
	"""
	上下文管理器 https://zhuanlan.zhihu.com/p/385045365
	:param name:
	:return:
	"""
	start_time = time.time()
	yield
	print(f'[{name}] {time.time() - start_time:.3f} s')


def create_candidates_repurchase(
		strategy: str,
		transactions: pd.DataFrame,
		target_users: np.ndarray,
		week_start: int,
		max_items_per_user: int=1234567890
) -> pd.DataFrame:
	tr = transactions.query('week >= @week and user in @target_users')
	tr = tr[['user', 'item', 'week', 'days']].drop_duplicates(ignore_index=True)
	gr_day = tr.groupby(['user', 'item'], as_index=False)['days'].min()
	gr_week = tr.groupby(['user', 'item'], as_index=False)['week'].min()
	gr_volume = tr.groupby(['user', 'item'], as_index=False).size()
	gr_volume = gr_volume.rename(columns={'size': 'volume'})
	gr_day['day_rank'] = gr_day.groupby('user')['days'].rank()
	gr_week['week_rank'] = gr_week.groupby('user')['week'].rank()
	gr_volume['volume_rank'] = gr_volume.groupby('user')['volume'].rank(ascending=False)

	candidates = gr_day.merge(gr_week, on=['user', 'item'], how='left')
	candidates = candidates.merge(gr_volume, on=['user', 'item'], how='left')
	# 在排序中如果day_rank一样在按照volume_rank排序
	candidates['rank_meta'] = 10**9 * candidates['day_rank'] + candidates['volume_rank']
	candidates['rank_meta'] = candidates.groupby('user')['rank_meta'].rank(method='min')
	# 对每个用户的candidates数量限制
	candidates = candidates.query('rank_meta <= @max_items_per_user').reset_index(drop=True)
	candidates = candidates[['user', 'item', 'week_rank', 'volume_rank', 'rank_meta']]
	candidates = candidates.rename(columns={
		'week_rank': f'{strategy}_week_rank',
		'volume_rank': f'{strategy}_volume_rank'
	})
	candidates['strategy'] = strategy
	return candidates.drop_duplicates(ignore_index=True)


def create_candidates_popular(
		transactions: pd.DataFrame,
		target_users: np.ndarray,
		week_start: int,
		num_weeks: int,
		num_items: int,
) -> pd.DataFrame:
	tr = transactions.query('@week_start <= week <= @week_start + @num_weeks')
	tr = tr[['user', 'item']].drop_duplicates(ignore_index=True)
	popular_items = tr['item'].value_counts().index.values[:num_items]
	popular_items = pd.DataFrame({
		'item': popular_items,
		'rank': range(1, num_items+1),
		'crossjoinkey': 1
	})
	candidates = pd.DataFrame({
		'user': target_users,
		'crossjoinkey': 1
	})
	candidates = candidates.merge(popular_items, on='crossjoinkey', how='inner')
	candidates = candidates.drop('crossjoinkey', axis=1)
	candidates = candidates.rename(columns={'rank': 'pop_rank'})
	candidates['strategy'] = 'pop'
	return candidates.drop_duplicates(ignore_index=True)


def create_candidates_age_popular(
		transactions: pd.DataFrame,
		users: pd.DataFrame,
		target_users: np.ndarray,
		week_start: int,
		num_weeks: int,
		num_items: int,
) -> pd.DataFrame:
	tr = transactions.query('@week_start <= week <= @week_start + @num_weeks')
	tr = tr[['user', 'item']].drop_duplicates(ignore_index=True)
	tr = tr.merge(users[['user', 'age']], on='user', how='left')
	pops = []
	for age in range(16, 100):
		low = age - age_shifts[age]
		high = age + age_shifts[age]
		pop = tr.query('@low <= age <= @high')
		pop = pop['item'].value_counts().index.values[:num_items]
		pops.append(pd.DataFrame({
			'age': age,
			'item': pop,
			'age_popular_rank': range(1, num_items+1)
		}))
	pops = pd.concat(pops, axis=0, ignore_index=True)

	candidates = users[['user', 'age']].dropna().query('user in @target_users').reset_index(drop=True)
	candidates = candidates.merge(pops, on='age', how='inner')
	candidates = candidates.drop('age', axis=1)
	candidates['strategy'] = 'age_pop'
	return candidates.drop_duplicates(ignore_index=True)


def create_candidates_cooc(
		transactions: pd.DataFrame,
		base_candidates: pd.DataFrame,
		week_start: int,
		num_weeks: int,
		pair_count_threshold: int
) -> pd.DataFrame:
	tr = transactions.query('@week_start <= week <= @week_start + @num_weeks')
	tr = tr[['user', 'item', 'week']].drop_duplicates(ignore_index=True)
	tr_with = tr.copy()
	tr_with = tr_with.rename(columns={'item': 'item_with', 'week': 'week_with'})
	tr = tr.merge(tr_with, on='user', how='inner')
	# 原代码是week <= week_with,时间线上前面的item关联出后面的item_with
	tr = tr.query('item != item_with and week >= week_with')[['item', 'item_with']].reset_index(drop=True)
	gr_item_count = tr.groupby('item', as_index=False).size()
	gr_item_count = gr_item_count.rename(columns={'size': 'item_count'})
	gr_pair_count = tr.groupby(['item', 'item_with'], as_index=False).size()
	gr_pair_count = gr_pair_count.rename(columns={'size': 'pair_count'})
	item2item = gr_pair_count.merge(gr_item_count, on='item', how='left')
	item2item['ratio'] = item2item['pair_count'] / item2item['item_count']
	# ['item', 'item_with', 'pair_count', 'item_count', 'ratio']
	item2item = item2item.query('pair_count > @pair_count_threshold').reset_index(drop=True)
	# ['user', 'item', 'repurchase_week_rank', 'repurchase_volume_rank', 'rank_meta', 'strategy']
	candidates = base_candidates.merge(item2item, on='item', how='inner')
	candidates = candidates.drop(['item', 'pair_count'], axis=1)
	# ['user', 'item', 'repurchase_week_rank', 'repurchase_volume_rank', 'rank_meta', 'strategy']
	# + ['item_count', 'ratio']
	candidates = candidates.rename(columns={'item_with': 'item'})
	candidates = candidates.rename(columns={col: f'cooc_{col}' for col in candidates.columns if col not in ['user', 'item']})
	candidates['strategy'] = 'cooc'
	return candidates.drop_duplicates(ignore_index=True)


def create_candidates_same_product_code(
		items: pd.DataFrame,
		base_candidates: pd.DataFrame
) -> pd.DataFrame:
	items = items[['item', 'product_code']]
	items_with = items.copy()
	items_with = items_with.rename(columns={'item': 'item_with'})
	item2item = items.merge(items_with, on='product_code', how='inner')
	item2item = item2item.query('item != item_with')[['item', 'item_with']].reset_index(drop=True)

	candidates = base_candidates.merge(item2item, on='item', how='inner')
	candidates = candidates.drop('item', axis=1)
	candidates = candidates.rename(columns={'item_with': 'item'})
	# 如果user的被推荐item有重复，选被关联item rank_meta最小的
	candidates['min_rank_meta'] = candidates.groupby(['user', 'item'])['rank_meta'].transform('min')
	# ['user', 'item', 'repurchase_week_rank', 'repurchase_volume_rank', 'rank_meta', 'strategy']
	# + ['min_rank_meta']
	candidates = candidates.query('rank_meta == min_rank_meta').reset_index(drop=True)
	candidates = candidates.rename(columns={col: f'same_product_code_{col}' for col in candidates.columns if col not in ['user', 'item']})
	candidates['strategy'] = 'same_product_code'
	return candidates.drop_duplicates(ignore_index=True)


def create_candidates_ohe_distance(
		transactions: pd.DataFrame,
		users: pd.DataFrame,
		items: pd.DataFrame,
		target_users: np.ndarray,
		week_start: int,
		num_weeks: int,
		num_items: int
) -> pd.DataFrame:
	users_with_ohe = users[['user']].query('user in @target_users').reset_index(drop=True)
	cols = [col for col in items.columns if col.endswith('_idx')]
	for col in cols:
		feature = pd.read_pickle(f'data/user_features/user_onehot_mean_{col}_{week_start}week.pkl')
		users_with_ohe = users_with_ohe.merge(feature, on='user', how='left')
	users_with_ohe = users_with_ohe.dropna().reset_index(drop=True)
	limited_users = users_with_ohe['user'].values
	a_users = users_with_ohe.drop('user', axis=1).values.astype('float32')
	a_users = np.ascontiguousarray(a_users)

	recent_items = transactions.query('@week_start <= week <= @week_start + @num_weeks')['item'].unique()
	items_with_ohe = items[['item']].query('item in @recent_items').reset_index(drop=True)
	for col in cols:
		feature = pd.get_dummies(items[['item', col]], columns=[col])
		items_with_ohe = items_with_ohe.merge(feature, on='item', how='left')
	items_with_ohe = items_with_ohe.dropna().reset_index(drop=True)
	limited_items = items_with_ohe['item'].values
	a_items = items_with_ohe.drop('item', axis=1).values.astype('float32')
	a_items = np.ascontiguousarray(a_items)

	# faiss 入门 https://zhuanlan.zhihu.com/p/357414033
	# faiss 原理 https://zhuanlan.zhihu.com/p/432317877
	# windows 只支持faiss-cpu, 不支持faiss-gpu
	index = faiss.index_factory(a_items.shape[1], 'Flat', faiss.METRIC_INNER_PRODUCT)
	index.add(a_items)
	distances, idxs = index.search(a_users, num_items)
	return pd.DataFrame({
		'user': np.repeat(limited_users, num_items),
		'item': limited_items[idxs.flatten()],
		'ohe_distance': distances.flatten(),
		'strategy': 'ohe_distance'
	})


def create_candidates_category_popular(
		transactions:pd.DataFrame,
		items: pd.DataFrame,
		base_candidates: pd.DataFrame,
		week_start: int,
		num_weeks: int,
		num_items: int,
		category: str
) -> pd.DataFrame:
	tr = transactions.query('@week_start <= week <= @week_start + @num_weeks')
	tr = tr[['user', 'item']].drop_duplicates(ignore_index=True)
	tr = tr.merge(items[['item', category]], on='item', how='left')
	tr = tr.groupby(['item', category], as_index=False).size()
	tr = tr.rename(columns={'size': 'volume'})
	tr['cat_volume_rank'] = tr.groupby(category)['volume'].rank(ascending=False, method='min')
	tr = tr.query('cat_volume_rank <= @num_items')
	tr = tr[['item', category, 'cat_volume_rank']].reset_index(drop=True)

	candidates = base_candidates[['user', 'item']].merge(items[['item', category]], on='item', how='left')
	candidates = candidates.groupby(['user', category], as_index=False).size()
	candidates = candidates.rename(columns={'size': 'cat_volume'})
	candidates = candidates.merge(tr, on=category, how='inner')
	candidates = candidates.drop(category, axis=1)
	candidates['strategy'] = 'cat_pop'
	return candidates.drop_duplicates(ignore_index=True)


def drop_common_user_item(
		candidates_target: pd.DataFrame,
		candidates_reference: pd.DataFrame
) -> pd.DataFrame:
	tmp = candidates_reference[['user', 'item']].reset_index(drop=True)
	tmp['flag'] = 1
	candidates_target = candidates_target.merge(tmp, on=['user', 'item'], how='left')
	candidates_target = candidates_target.query('flag != 1').drop('flag', axis=1).reset_index(drop=True)
	return candidates_target


def create_candidates(
		transactions: pd.DataFrame,
		users: pd.DataFrame,
		items: pd.DataFrame,
		target_users: np.ndarray,
		week: int
) -> pd.DataFrame:
	with timer('repurchase'):
		# 以user过去的各自购买记录时间间隔越小越容易购买，在同一天购买的以购买天数为准
		# ['user', 'item', 'repurchase_week_rank', 'repurchase_volume_rank', 'rank_meta', 'strategy']
		candidates_repurchase = create_candidates_repurchase('repurchase', transactions, target_users, week, REPURCHASE_NUM_ITEMS)
		print(candidates_repurchase.shape)
	with timer('popular'):
		# 过去一段时间购买用户数最多的商品推荐给每个用户
		# ['user', 'item', 'pop_rank', 'strategy']
		candidates_popular = create_candidates_popular(transactions, target_users, week, POPULAR_WEEKS, POPULAR_NUM_ITEMS)
		print(candidates_popular.shape)
	with timer('age popular'):
		# 给用户推荐年龄(段)内购买用户数最多的商品
		# ['user', 'item', 'age_popular_rank', 'strategy']
		candidates_age_popular = create_candidates_age_popular(transactions, users, target_users, week, AGE_POPULAR_WEEKS, AGE_POPULAR_NUM_ITEMS)
		print(candidates_age_popular.shape)
	with timer('item2item'):
		candidates_item2item = create_candidates_repurchase('item2item', transactions, target_users, week, 24)
		print(candidates_item2item.shape)
	with timer('item2item2'):
		candidates_item2item2 = create_candidates_repurchase('item2item2', transactions, target_users, week, 12)
		print(candidates_item2item2.shape)
	with timer('cooccurrence'):
		# 在同一个user中共现来推荐item, 与item-cf的区别只是归一化不一样
		# candidates_repurchase.columns + ['item_count', 'ratio']
		candidates_cooc = create_candidates_cooc(transactions, candidates_item2item, week, COOC_WEEKS, COOC_THRESHOLD)
		print(candidates_cooc.shape)
		candidates_cooc = drop_common_user_item(candidates_cooc, candidates_repurchase)
		print(candidates_cooc.shape)
	with timer('same_product_code'):
		# 在product_code中共现来推荐item
		# candidates_repurchase.columns + ['min_rank_meta']
		candidates_same_product_code = create_candidates_same_product_code(items, candidates_item2item2)
		print(candidates_same_product_code.shape)
		candidates_same_product_code = drop_common_user_item(candidates_same_product_code, candidates_repurchase)
		print(candidates_same_product_code.shape)
	with timer('ohe distance'):
		# user和item相似性来推荐item
		# ['user', 'item', 'ohe_distance', 'strategy']
		candidates_ohe_distance = create_candidates_ohe_distance(transactions, users, items, target_users, week, OHE_DISTANCE_WEEKS, OHE_DISTANCE_NUM_ITEMS)
		print(candidates_ohe_distance.shape)
		candidates_ohe_distance = drop_common_user_item(candidates_ohe_distance, candidates_repurchase)
		print(candidates_ohe_distance.shape)
	with timer('category popular'):
		# 在category中共现来推荐item
		# ['user', 'item', 'cat_volume', 'cat_volume_rank', 'strategy']
		candidates_dept = create_candidates_category_popular(transactions, items, candidates_item2item2, week, CATEGORY_WEEKS, CATEGORY_NUM_ITEMS, 'department_no_idx')
		print(candidates_dept.shape)
		candidates_dept = drop_common_user_item(candidates_dept, candidates_repurchase)
		print(candidates_dept.shape)

	# 原代码保留了所有字段？？？
	candidates = [
		candidates_repurchase[['user', 'item', 'strategy']],
		candidates_popular[['user', 'item', 'strategy']],
		candidates_age_popular[['user', 'item', 'strategy']],
		candidates_cooc[['user', 'item', 'strategy']],
		candidates_same_product_code[['user', 'item', 'strategy']],
		candidates_ohe_distance[['user', 'item', 'strategy']],
		candidates_dept[['user', 'item', 'strategy']],
	]
	candidates = pd.concat(candidates, axis=0, ignore_index=True)
	print(
		len(candidates),
		len(candidates.drop_duplicates(['user', 'item'])),
		len(candidates.drop_duplicates(['user', 'item']))/len(candidates)
	)
	print(candidates.groupby('strategy', as_index=False).size())
	return candidates


def calc_embeddings(
		model_type: str,
		week: int,
		dim: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
	# 加载model, user_features, item_features
	assert model_type in ['i_i', 'if_i', 'i_if', 'if_if']
	save_path = f'data/lfm/{model_type}_{week}week_{dim}dim'
	model_path = f'{save_path}_model.pkl'
	user_features_path = f'{save_path}_user_features.npz'
	item_features_path = f'{save_path}_item_features.npz'
	with open(model_path, 'rb') as f:
		model = pickle.load(f)
	user_features = None
	item_features = None
	if model_type == 'i_i':
		pass
	elif model_type == 'if_i':
		user_features = sparse.load_npz(user_features_path)
	elif model_type == 'i_if':
		item_features = sparse.load_npz(item_features_path)
	elif model_type == 'if_if':
		user_features = sparse.load_npz(user_features_path)
		item_features = sparse.load_npz(item_features_path)
	else:
		raise NotImplementedError

	# https://making.lyst.com/lightfm/docs/lightfm.html
	biases, embeddings = model.get_user_representations(user_features)
	n_user = len(biases)
	a = np.hstack([embeddings, biases.reshape(n_user, 1)])
	user_embeddings = pd.DataFrame(a, columns=[f'user_rep_{i}' for i in range(dim+1)])
	user_df = pd.DataFrame({'user': range(n_user)})
	user_embeddings = pd.concat([user_embeddings, user_df], axis=1)

	biases, embeddings = model.get_item_representations(item_features)
	n_item = len(biases)
	a = np.hstack([embeddings, biases.reshape(n_item, 1)])
	item_embeddings = pd.DataFrame(a, columns=[f'item_rep_{i}' for i in range(dim+1)])
	item_df = pd.DataFrame({'item': range(n_item)})
	item_embeddings = pd.concat([item_embeddings, item_df], axis=1)

	return user_embeddings, item_embeddings


def attach_features(
		transactions: pd.DataFrame,
		users: pd.DataFrame,
		items: pd.DataFrame,
		candidates: pd.DataFrame,
		week: int,
) -> pd.DataFrame:

	print('before attaching features:', candidates.shape)

	with timer('user static features'):
		user_features = ['age']
		candidates = candidates.merge(users[['user'] + user_features], on='user', how='left')
	with timer('item static features'):
		item_features = [col for col in items.columns if col.endswith('_idx')]
		candidates = candidates.merge(items[['item'] + item_features], on='item', how='left')
	with timer('user dynamic transactions features'):
		week_end = week + USER_TRANSACTION_FEATURES_WEEKS
		tmp = transactions.query('@week <= week <= @week_end')
		tmp = tmp.groupby('user')['price', 'sales_channel_id'].agg(['mean', 'std'])
		tmp.columns = [f'user_{a[0]}_{a[1]}' for a in tmp.columns.to_flat_index()]
		tmp = tmp.reset_index()
		candidates = candidates.merge(tmp, on='user', how='left')
	with timer('item dynamic transactions features'):
		week_end = week + ITEM_TRANSACTION_FEATURES_WEEKS
		tmp = transactions.query('@week <= week <= @week_end')
		tmp = tmp.groupby('item')['price', 'sales_channel_id'].agg(['mean', 'std'])
		tmp.columns = [f'item_{a[0]}_{a[1]}' for a in tmp.columns.to_flat_index()]
		tmp = tmp.reset_index()
		candidates = candidates.merge(tmp, on='item', how='left')
	with timer('item freshness features'):
		# item最近出现的间隔天数
		tmp = transactions.query('week >= @week').groupby('item', as_index=False)['days'].agg({'item_day_min': 'min'})
		tmp['item_day_min'] -= transactions.query('week == @week')['days'].min()
		candidates = candidates.merge(tmp, on='item', how='left')
	with timer('user freshness features'):
		# 用户最近出现的间隔天数
		tmp = transactions.query('week >= @week').groupby('user', as_index=False)['days'].agg({'user_day_min': 'min'})
		tmp['user_day_min'] -= transactions.query('week == @week')['days'].min()
		candidates = candidates.merge(tmp, on='user', how='left')
	with timer('user_item freshness features'):
		# user_item最近出现的间隔天数
		tmp = transactions.query('week >= @week').groupby(['user', 'item'], as_index=False)['days'].agg({'user_item_day_min': 'min'})
		tmp['user_item_day_min'] -= transactions.query('week == @week')['days'].min()
		candidates = candidates.merge(tmp, on=['user', 'item'], how='left')
	with timer('item volume features'):
		week_end = week + ITEM_VOLUME_FEATURES_WEEKS
		tmp = transactions.query('@week <= week <= @week_end')
		tmp = tmp.groupby('item', as_index=False).size()
		tmp = tmp.rename(columns={'size': 'item_volume'})
		candidates = candidates.merge(tmp, on='item', how='left')
	with timer('user volume features'):
		week_end = week + USER_VOLUME_FEATURES_WEEKS
		tmp = transactions.query('@week <= week <= @week_end')
		tmp = tmp.groupby('user', as_index=False).size()
		tmp = tmp.rename(columns={'size': 'user_volume'})
		candidates = candidates.merge(tmp, on='user', how='left')
	with timer('user_item volume features'):
		week_end = week + USER_ITEM_VOLUME_FEATURES_WEEKS
		tmp = transactions.query('@week <= week <= @week_end')
		tmp = tmp.groupby(['user', 'item'], as_index=False).size()
		tmp = tmp.rename(columns={'size': 'user_item_volume'})
		candidates = candidates.merge(tmp, on=['user', 'item'], how='left')
	with timer('item age volume features'):
		week_end = week + AGE_VOLUME_FEATURES_WEEKS
		# 在各个age(包含age_shift)的user item被购买的次数的排名
		tr = transactions.query('@week <= week <= @week_end')
		tr = tr.merge(users[['user', 'age']], on='user', how='left')
		item_age_volumes = []
		for age in range(16, 100):
			low = age - age_shifts[age]
			high = age + age_shifts[age]
			tmp = tr.query('@low <= age <= @high')
			tmp = tmp.groupby('item', as_index=False).size()
			tmp = tmp.rename(columns={'size': 'age_volume'})
			tmp['age_volume'] = tmp['age_volume'].rank(ascending=False)
			tmp['age'] = age
			item_age_volumes.append(tmp)
		item_age_volumes = pd.concat(item_age_volumes, axis=0, ignore_index=True)
		candidates = candidates.merge(item_age_volumes, on=['item', 'age'], how='left')
	with timer('user category most frequent'):
		for col in ['department_no_idx']:
			feature = pd.read_pickle(f'data/user_features/user_onehot_mean_{col}_{week}week.pkl')
			feature_cols = [c for c in feature.columns if col != 'user']
			feature[feature_cols] = feature[feature_cols] / feature[feature_cols].mean()
			feature[f'{col}_most_freq_idx'] = np.argmax(feature[feature_cols].values, axis=1)
			feature = feature[['user', f'{col}_most_freq_idx']]
			candidates = candidates.merge(feature, on='user', how='left')
	with timer('ohe dot product'):
		# ohe特征 user_item 相似性
		item_idx_col = [col for col in items.columns if col.endswith('_idx')]
		for col in item_idx_col:
			item_feature = pd.get_dummies(items[['item'] + [col]], columns=[col])
			feature_cols = [c for c in item_feature.columns if c != 'item']
			item_feature[feature_cols] = item_feature[feature_cols] / item_feature[feature_cols].mean()
			item_feature = candidates[['item']].merge(item_feature, on='item', how='left')
			item_feature = item_feature.drop('item', axis=1)
			user_feature = pd.read_pickle(f'data/user_features/user_onehot_mean_{col}_{week}week.pkl')
			feature_cols = [c for c in user_feature.columns if c != 'user']
			user_feature[feature_cols] = user_feature[feature_cols] / user_feature[feature_cols].mean()
			user_feature = candidates[['user']].merge(user_feature, on='user', how='left')
			user_feature = user_feature.drop('user', axis=1)
			candidates[f'{col}_ohe_score'] = np.sum(item_feature.values * user_feature.values, axis=1)
	with timer('lfm features'):
		user_reps, item_reps = calc_embeddings('i_i', week, 16)
		candidates = candidates.merge(user_reps, on='user', how='left')
		candidates = candidates.merge(item_reps, on='item', how='left')

	print('after attaching features:', candidates.shape)
	return candidates

