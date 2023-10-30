import pandas as pd
import warnings
from pathlib import Path
from utils import create_candidates, attach_features


warnings.filterwarnings('ignore')


def merge_labels(
		candidates: pd.DataFrame,
		transactions: pd.DataFrame,
		week: int
) -> pd.DataFrame:
	labels = transactions.query('week == @week')[['user', 'item']].drop_duplicates(ignore_index=True)
	labels['y'] = 1
	original_positives = len(labels)
	candidates = candidates.merge(labels, on=['user', 'item'], how='left')
	candidates = candidates.fillna(0)
	remaining_positives = candidates.drop_duplicates(['user', 'item', 'y'], ignore_index=True)['y'].sum()
	recall = remaining_positives / original_positives
	print(f'week: {week}, {remaining_positives}, {original_positives}, recall: {recall}')

	strategy_positives = candidates.groupby('strategy', as_index=False)['y'].sum()
	volumes = candidates.groupby('strategy', as_index=False).size()
	volumes = volumes.rename(columns={'size': 'volume'})
	strategy_positives = strategy_positives.merge(volumes, on='strategy', how='left')
	strategy_positives['recall'] = strategy_positives['y'] / original_positives
	strategy_positives['hit_ratio'] = strategy_positives['y'] / strategy_positives['volume']
	print(strategy_positives)

	return candidates


def drop_trivial_users(candidates):
	bef = len(candidates)
	tmp = candidates[['user', 'y']].drop_duplicates(ignore_index=True)
	tmp = tmp.groupby('user', as_index=False).size()
	keep_users = set(tmp.query('size == 2')['user'])
	candidates_keep = candidates.query('user in @keep_users').reset_index(drop=True)
	aft = len(candidates_keep)
	print(f'drop trivial before {bef}, after {aft}')
	return candidates_keep


transactions = pd.read_pickle('data/transactions_train.pkl')
users = pd.read_pickle('data/users.pkl')
items = pd.read_pickle('data/items.pkl')

# 每个week时段user的candidates
candidates = []
for week in [0]:
# for week in [0, 1, 2, 3, 4, 5, 6]:
	target_users = transactions.query('week == @week')['user'].unique()
	candidates.append(create_candidates(transactions, users, items, target_users, week+1))

TRAIN_WEEKS = 6
SAVE_DIR = Path('data/candidates')
SAVE_DIR.mkdir(exist_ok=True, parents=True)
for idx in range(len(candidates)):
	# 标记label
	candidates[idx] = merge_labels(candidates[idx], transactions, idx)
for idx in range(len(candidates)):
	candidates[idx]['week'] = idx
for idx in range(len(candidates)):
	# 只保留正负样本的user
	candidates[idx] = drop_trivial_users(candidates[idx])
for idx in range(len(candidates)):
	# 特征
	candidates[idx] = attach_features(transactions, users, items, candidates[idx], idx+1)
for idx in range(len(candidates)):
	candidates[idx]['query_group'] = candidates[idx]['week'].astype('str') + '_' + candidates[idx]['user'].astype('str')

# candidates = pd.concat(candidates, axis=0, ignore_index=True)
# candidates_path = SAVE_DIR / 'candidates.pkl'
# candidates.to_pickle(candidates_path)
# train_all_path = SAVE_DIR / 'train_all.pkl'
# candidates.to_pickle(train_all_path)

# week=0作为valid, 其他week合并作为train
valid = candidates[0]
train = pd.concat([candidates[idx] for idx in range(1, 1+TRAIN_WEEKS)], axis=0, ignore_index=True)
train_all = pd.concat([candidates[idx] for idx in range(0, 0+TRAIN_WEEKS)], axis=0, ignore_index=True)
train_path = SAVE_DIR / 'train.pkl'
valid_path = SAVE_DIR / 'valid.pkl'
train_all_path = SAVE_DIR / 'train_all.pkl'
train.to_pickle(f'{SAVE_DIR} / train.pkl')
valid.to_pickle(f'{SAVE_DIR} / valid.pkl')
train_all.to_pickle(train_all_path)


