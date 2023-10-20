import pandas as pd
import numpy as np
from scipy import sparse
import implicit
import pickle


MODE, SEED = 'valid', 2023
USER_ID, ITEM_ID = 'session', 'aid'
assert MODE in ['test', 'valid']
data_path = f'data/train_{MODE}/'
preprocess_path = 'data/preprocess/'

train = pd.read_parquet(data_path + 'train.parquet')
test = pd.read_parquet(data_path + 'test.parquet')
df = pd.concat([train, test], axis=0, ignore_index=True)
df['user_label'], user_idx = pd.factorize(df[USER_ID])
df['item_label'], item_idx = pd.factorize(df[ITEM_ID])
user_item_matrix = sparse.csr_matrix((np.ones(len(df)), (df['user_label'], df['item_label'])))
epochs, emb_size = 5000, 64
# bpr https://blog.csdn.net/m0_56689123/article/details/118809971
model = implicit.bpr.BayesianPersonalizedRanking(
	factors=emb_size,
	regularization=0.001,
	iterations=epochs,
	random_state=SEED
)
model.fit(user_item_matrix)
u2emb = dict(zip(user_idx, model.user_factors.to_numpy()))
i2emb = dict(zip(item_idx, model.item_factors.to_numpy()))
with open(f'{preprocess_path}u2emb_{MODE}.pkl', 'wb') as f:
	pickle.dump(u2emb, f)
with open(f'{preprocess_path}i2emb_{MODE}.pkl', 'wb') as f:
	pickle.dump(i2emb, f)

