import pandas as pd
from pathlib import Path
from tqdm import tqdm


def get_user_features(week):
	transactions = pd.read_pickle('data/transactions_train.pkl')
	users = pd.read_pickle('data/users.pkl')
	items = pd.read_pickle('data/items.pkl')
	tr = transactions.query('week >= @week')[['user', 'item']]
	# print(transactions.shape, tr.shape)
	cols = [col for col in items.columns if col.endswith('_idx')]
	for col in tqdm(cols, desc='features creating'):
		tmp = pd.get_dummies(items[['item', col]], columns=[col])
		tmp = tr.merge(tmp, on='item', how='left')
		tmp = tmp.drop('item', axis=1)
		tmp = tmp.groupby('user', as_index=False).mean()
		user_features = users[['user']].merge(tmp, on='user', how='left')
		user_features = user_features.rename(columns={
			col: f'user_onehot_mean_{col}' for col in user_features.columns if col != 'user'
		})
		save_path = SAVE_DIR / f'user_onehot_mean_{col}_{week}week.pkl'
		user_features.to_pickle(save_path)


SAVE_DIR = Path('data/user_features')
SAVE_DIR.mkdir(exist_ok=True, parents=True)
for week in [1]:
# for week in [0, 1, 2, 3, 4, 5, 6, 7]:
	get_user_features(week)

