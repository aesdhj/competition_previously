import pandas as pd
import lightgbm
import catboost
import numpy as np
from pathlib import Path
import pickle
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from utils import create_candidates, attach_features


# candidates = pd.read_pickle('data/candidates/candidates.pkl')
# n_user = len(set(candidates['user']))
# train_user_list = set(candidates[['user']].drop_duplicates().sample(n=int(n_user*0.8), random_state=2023)['user'])
# train = candidates.query('user in @train_user_list').reset_index(drop=True)
# valid = candidates.query('user not in @train_user_list').reset_index(drop=True)
# train.to_pickle('data/candidates/train.pkl')
# valid.to_pickle('data/candidates/valid.pkl')
# print(candidates.shape, train.shape, valid.shape)


def get_query_group(df):
	sequence = df['user'].values
	com_sequence_index, = np.concatenate(([True], sequence[1:] != sequence[:-1], [True])).nonzero()
	return list(np.ediff1d(com_sequence_index))


def apk(actual, predicted, k=12):
	if len(predicted) > 12:
		predicted = predicted[:k]
	score = 0.0
	num_hits = 0.0
	for i, p in enumerate(predicted):
		# p not in predicted[:i], 防止predicted中有重复
		if p in actual and p not in predicted[:i]:
			num_hits += 1.0
			score += num_hits/(i+1.0)
	# 实际为空值
	if not actual:
		return 0.0
	return score / min(len(actual), k)


def mapk(actual, predicted, k=12):
	"""
	# mapk 详解 https://blog.csdn.net/weixin_42690752/article/details/102827308
	"""
	return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


MODE_TYPE = 'lightgbm'
assert MODE_TYPE in ['lightgbm', 'catboost']
SAVE_DIR = Path('data/model')
SAVE_DIR.mkdir(exist_ok=True, parents=True)
transactions = pd.read_pickle('data/transactions_train.pkl')
users = pd.read_pickle('data/users.pkl')
items = pd.read_pickle('data/items.pkl')
mp_user = pd.read_pickle('data/mp_customer_id.pkl')
mp_item = pd.read_pickle('data/mp_article_id.pkl')

# 数据准备
train = pd.read_pickle('data/candidates/train.pkl')
train = train.sort_values('query_group').reset_index(drop=True)
valid = pd.read_pickle('data/candidates/valid.pkl')
valid = valid.sort_values('query_group').reset_index(drop=True)
feature_cols = [col for col in train.columns if col not in ['user', 'item', 'strategy', 'y', 'week', 'query_group']]
cat_features = [col for col in feature_cols if col.endswith('_idx')]

# 训练模型
if MODE_TYPE == 'lightgbm':
	group_train = get_query_group(train)
	group_valid = get_query_group(valid)
	train_dataset = lightgbm.Dataset(
		train[feature_cols], train['y'],
		categorical_feature=cat_features,
		group=group_train
	)
	valid_dataset = lightgbm.Dataset(
		valid[feature_cols], valid['y'],
		categorical_feature=cat_features,
		group=group_valid,
		reference=train_dataset
	)

	# lgbmranker入门 https://www.zhihu.com/question/341082668/answer/2810498340
	# lightgbm参数 https://lightgbm.readthedocs.io/en/latest/Parameters.htm
	params = {
		'objective': 'xendcg',
		'learning_rate': 1e-6,
		'num_leaves': 255,
		'min_data_in_leaf': 100,
		'metric': 'map',
		'eval_at': 12,
		'seed': 2023
	}
	model = lightgbm.train(
		params,
		train_dataset,
		valid_sets=[train_dataset, valid_dataset],
		num_boost_round=200,		# 1000
		callbacks=[lightgbm.early_stopping(10)],		# 20
		categorical_feature=cat_features,
	)
	lightgbm.plot_importance(model, importance_type='gain', figsize=(8, 16))
	plt.show()
elif MODE_TYPE == 'catboost':
	# catboost不能处理分类特征缺失值
	# catboost入门 https://zhuanlan.zhihu.com/p/460986009
	train['department_no_idx_most_freq_idx'] = train['department_no_idx_most_freq_idx'].fillna(0).astype('int')
	valid['department_no_idx_most_freq_idx'] = valid['department_no_idx_most_freq_idx'].fillna(0).astype('int')

	train_dataset = catboost.Pool(
		data=train[feature_cols],
		label=train['y'],
		group_id=train['query_group'],
		cat_features=cat_features
	)
	valid_dataset = catboost.Pool(
		data=valid[feature_cols],
		label=valid['y'],
		group_id=valid['query_group'],
		cat_features=cat_features
	)
	params = {
		'loss_function': 'YetiRank',
		'use_best_model': True,
		'one_hot_max_size': 300,
		'iterations': 200,		# 10000
		'early_stopping_rounds': 10,
		'random_state': 2023,
		'nan_mode': 'Min',
		'eval_metric': 'MAP',
	}
	model = catboost.CatBoost(params)
	model.fit(train_dataset, eval_set=valid_dataset)
	plt.plot(model.get_evals_result()['learn']['MAP'])
	plt.show()
	feature_importance = model.get_feature_importance(train_dataset)
	sorted_idx = np.argsort(feature_importance)
	plt.figure(figsize=(16, 8))
	plt.yticks(range(len(feature_cols)), np.array(feature_cols)[sorted_idx])
	plt.barh(range(len(feature_cols)), feature_importance[sorted_idx])
	plt.show()
else:
	raise NotImplementedError
save_path = SAVE_DIR / f'{MODE_TYPE}.pkl'
with open(save_path, 'wb') as f:
	pickle.dump(model, f)

# valid真实评分
valid_pred = valid[['user', 'item']].reset_index(drop=True)
valid_pred['pred'] = model.predict(valid[feature_cols])
valid_pred = valid_pred.groupby(['user', 'item'], as_index=False)['pred'].max()
valid_pred = valid_pred.sort_values(['user', 'pred'], ascending=False).reset_index(drop=True)
pred = valid_pred.groupby('user', as_index=False)['item'].agg(list)
valid_user = set(valid['user'])
gt = transactions.query('user in @valid_user and week==0')
gt = gt.groupby('user', as_index=False)['item'].agg(list)
gt = gt.rename(columns={'item': 'gt'})
merged = gt.merge(pred, on='user', how='left')
merged['item'] = merged['item'].fillna('').apply(list)
print('map@12:', mapk(merged['gt'], merged['item']))

# 全量数据重新训练模型
train_all = pd.read_pickle('data/candidates/train_all.pkl')
train_all = train_all.sort_values('query_group')
if MODE_TYPE == 'lightgbm':
	group_train_all = get_query_group(train_all)
	train_all_dataset = lightgbm.Dataset(
		train_all[feature_cols], train_all['y'],
		categorical_feature=cat_features,
		group=group_train_all
	)
	best_iteration = model.best_iteration
	# 不是lightgbm增量训练(keep_training_booster=True决定)
	model = lightgbm.train(
		params,
		train_all_dataset,
		categorical_feature=cat_features,
		num_boost_round=best_iteration
	)
elif MODE_TYPE == 'catboost':
	train_all['department_no_idx_most_freq_idx'] = train_all['department_no_idx_most_freq_idx'].fillna(0).astype('int')
	train_all_dataset = catboost.Pool(
		data=train_all[feature_cols],
		label=train_all['y'],
		group_id=train_all['query_group'],
		cat_features=cat_features
	)
	params['iterations'] = model.get_best_iteration()
	params['use_best_model'] = False
	model = catboost.CatBoost(params)
	# 不是catboost增量训练(init_mode决定)
	model.fit(train_all_dataset)
else:
	raise NotImplementedError
save_path = SAVE_DIR / f'submission_{MODE_TYPE}.pkl'
with open(save_path, 'wb') as f:
	pickle.dump(model, f)

# submission
# 比赛要求，无论是否提供purchase data, 都要预测未来一周的该买
all_users = users['user'].values
# 所有user的candidates，和前面的candidates的区别是user是前一周有购买记录的
candidates = create_candidates(transactions, users, items, all_users, 0)
candidates = attach_features(transactions, users, items, candidates, 0)
candidates['pred'] = model.predict(candidates[feature_cols])
pred = candidates.groupby(['user', 'item'], as_index=False)['pred'].max()
pred = pred.sort_values(['user', 'pred'], ascending=False).reset_index(drop=True)
pred = pred.groupby('user', as_index=False)['item'].agg(list)
pred['item'] = pred.apply(lambda x: x['item'][:12], axis=1)
a_user = mp_user['val'].values
a_item = mp_item['val'].values
pred['customer_id'] = pred['user'].ppaly(lambda x: a_user[x])
pred['prediction'] = pred['item'].apply(lambda x: list(map(lambda y: a_item[y], x)))
pred['prediction'] = pred['prediction'].apply(lambda x: ' '.join(map(str, x)))
submission = pred[['customer_id', 'prediction']]
SUBMISSION_DIR = Path('data/submission')
SUBMISSION_DIR.mkdir(exist_ok=True, parents=True)
sub_path = SUBMISSION_DIR / 'submission.csv'
submission.to_csv(sub_path, index=False)









