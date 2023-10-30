import pandas as pd
from typing import Any, Dict
from pathlib import Path
import shutil
from tqdm import tqdm


# 3.9以后list, dict能起到Dict, List同样的效果
# https://zhuanlan.zhihu.com/p/636582050
def _dict_to_dataframe(mp: Dict[Any, int]) -> pd.DataFrame:
	return pd.DataFrame(mp.items(), columns=['val', 'idx'])


def _add_idx_column(df: pd.DataFrame, col_name_from: str, col_name_to: str, mp: Dict[Any, int]):
	df[col_name_to] = df[col_name_from].apply(lambda x: mp[x]).astype('int64')


def _count_encoding_dict(df: pd.DataFrame, col_name: str) -> Dict[Any, int]:
	v = df.groupby(col_name).size().reset_index(name='size').sort_values(by='size', ascending=False)[col_name].tolist()
	return {x: idx for idx, x in enumerate(v)}


transactions = pd.read_csv(
	'data/transactions_train_sample.csv',
	parse_dates=['t_dat']
)
articles = pd.read_csv('data/articles.csv')
print(articles.shape)
articles = articles[articles['article_id'].isin(set(transactions['article_id']))]
print(articles.shape)
customers = pd.read_csv('data/customers.csv')
print(customers.shape)
customers = customers[customers['customer_id'].isin(set(transactions['customer_id']))]
print(customers.shape)
"""
customers_sample = transactions[['customer_id']].drop_duplicates().sample(n=50000, random_state=2023)
customers_sample = set(customers_sample['customer_id'])
transactions_sample = transactions[transactions['customer_id'].isin(customers_sample)]
(105542, 25)
(46032, 25)
(1371980, 7)
(10000, 7)
"""

# 把id和一些分类特征label_encoder
customer_ids = customers['customer_id'].values
mp_customer_id = {x: idx for idx, x in enumerate(customer_ids)}
_dict_to_dataframe(mp_customer_id).to_pickle('data/mp_customer_id.pkl')
article_ids = articles['article_id'].values
mp_article_id = {x: idx for idx, x in enumerate(article_ids)}
_dict_to_dataframe(mp_article_id).to_pickle('data/mp_article_id.pkl')

# customers
_add_idx_column(customers, 'customer_id', 'user', mp_customer_id)
customers['FN'] = customers['FN'].fillna(0).astype('int64')
customers['Active'] = customers['Active'].fillna(0).astype('int64')
label_encoding_columns = [
	'club_member_status',
	'fashion_news_frequency',
]
for col in tqdm(label_encoding_columns, desc='customers label encoding'):
	customers[col] = customers[col].fillna('NULL')
	mp = _count_encoding_dict(customers, col)
	_add_idx_column(customers, col, f'{col}_idx', mp)
customers.to_pickle('data/users.pkl')

# articles
_add_idx_column(articles, 'article_id', 'item', mp_article_id)
label_encoding_columns = [
	'product_type_no',
	'product_group_name',
	'graphical_appearance_no',
	'colour_group_code',
	'perceived_colour_value_id',
	'perceived_colour_master_id',
	'department_no',
	'index_code',
	'index_group_no',
	'section_no',
	'garment_group_no',
]
for col in tqdm(label_encoding_columns, desc='articles label encoding'):
	mp = _count_encoding_dict(articles, col)
	_add_idx_column(articles, col, f'{col}_idx', mp)
articles.to_pickle('data/items.pkl')

# transactions
_add_idx_column(transactions, 'customer_id', 'user', mp_customer_id)
_add_idx_column(transactions, 'article_id', 'item', mp_article_id)
transactions['sales_channel_id'] = transactions['sales_channel_id'] - 1
transactions['week'] = (transactions['t_dat'].max() - transactions['t_dat']).dt.days // 7
transactions['days'] = (transactions['t_dat'].max() - transactions['t_dat']).dt.days
transactions.to_pickle('data/transactions_train.pkl')

# images
mp = pd.read_pickle('data/mp_article_id.pkl')
dct = dict(zip(mp['val'], mp['idx']))
paths = list(Path('data/images').glob('*/*.jpg'))
for path in paths:
	name = path.name[-4:]
	if name not in dct:
		continue
	idx = dct[name]
	path_to = f'data/images/{idx}.jpg'
	shutil.copyfile(path, path_to)

