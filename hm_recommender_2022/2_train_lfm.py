import pandas as pd
from scipy import sparse
from lightfm import LightFM
from pathlib import Path
import pickle


def lfm(week: int, dim: int, use_user_features: bool, use_item_features: bool):
	transactions = pd.read_pickle('data/transactions_train.pkl')
	users = pd.read_pickle('data/users.pkl')
	items = pd.read_pickle('data/items.pkl')
	n_user = len(users)
	n_item = len(items)
	a = transactions.query('week >= @week')[['user', 'item']].drop_duplicates(ignore_index=True)
	a_train = sparse.lil_matrix((n_user, n_item))
	# 不同稀疏矩阵的区别
	# https://blog.csdn.net/weiwei935707936/article/details/109527517
	a_train[a['user'], a['item']] = 1

	# 是否使用用户特征或者商品特征
	if use_user_features:
		# 'club_member_status','fashion_news_frequency'可以加入尝试
		tmp = users[['age']].reset_index(drop=True)
		tmp['age_nan'] = tmp['age'].isna()
		tmp['age'] = tmp['age'].fillna(0.0)
		tmp = tmp.astype('float32')
		id = sparse.identity(n_user, dtype='f', format='csr')
		user_features = sparse.hstack([id, tmp.values]).astype('float32')
	if use_item_features:
		cols = [col for col in items.columns if col.endswith('_idx')]
		tmp = pd.concat([pd.get_dummies(items[col], prefix=col) for col in cols], axis=1).astype('float32')
		id = sparse.identity(n_item, dtype='f', format='csr')
		item_features = sparse.hstack([id, tmp.values]).astype('float32')

	# lightfm原理 混合协同过滤算法
	# https://zhuanlan.zhihu.com/p/627736906
	# https://mp.weixin.qq.com/s/xY5IYxRd2nufyJcEfa_fmw
	# lightfm 1.17 fit 进程已结束，退出代码为-1073741819(0xC 0000005), 1.16无错误
	lightfm_parmas = LIGHT_PARAMS.copy()
	lightfm_parmas['no_components'] = dim
	model = LightFM(**lightfm_parmas)
	if use_user_features and use_item_features:
		model.fit(a_train, user_features=user_features, item_features=item_features, epochs=EPOCHS, verbose=True)
		save_path = SAVE_DIR / f'if_if_{week}week_{dim}dim'
		sparse.save_npz(f'{save_path}_user_features.npz', user_features)
		sparse.save_npz(f'{save_path}_item_features.npz', item_features)
	elif use_user_features:
		model.fit(a_train, user_features=user_features, epochs=EPOCHS, verbose=True)
		save_path = SAVE_DIR / f'if_i_{week}week_{dim}dim'
		sparse.save_npz(f'{save_path}_user_features.npz', user_features)
	elif use_item_features:
		model.fit(a_train, item_features=item_features, epochs=EPOCHS, verbose=True)
		save_path = SAVE_DIR / f'i_if_{week}week_{dim}dim'
		sparse.save_npz(f'{save_path}_item_features.npz', item_features)
	else:
		model.fit(a_train, epochs=EPOCHS, verbose=True)
		save_path = SAVE_DIR / f'i_i_{week}week_{dim}dim'
	# 保存模型
	with open(f'{save_path}_model.pkl', 'wb') as f:
		pickle.dump(model, f)


LIGHT_PARAMS = {
	'learning_schedule': 'adadelta',
	'loss': 'bpr',
	'learning_rate': 0.005,
	'random_state': 2023
}
EPOCHS = 100
SAVE_DIR = Path('data/lfm')
SAVE_DIR.mkdir(exist_ok=True, parents=True)
for week in [1]:
# for week in [0, 1, 2, 3, 4, 5, 6, 7]:
	for dim in [16]:
		for use_user_features in [True, False]:
			for use_item_features in [True, False]:
				print(week, dim, use_user_features, use_item_features)
				if not use_user_features and not use_item_features:
					lfm(week, dim, use_user_features, use_item_features)

