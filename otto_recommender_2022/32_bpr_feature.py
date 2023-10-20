import pandas as pd
import pickle
import numpy as np


def make_bpr_feature(mode):
	preprocess_path = 'data/preprocess/'
	candidate_path = 'data/candidate/'
	feature_path = 'data/feature/'
	type2id = {'clicks': 0, 'carts': 1, 'orders': 2}

	with open(f'{preprocess_path}u2emb_{mode}.pkl', 'wb') as f:
		u2emb = pickle.load(f)
	with open(f'{preprocess_path}i2emb_{mode}.pkl', 'wb') as f:
		i2emb = pickle.load(f)
	for t in type2id.keys():
		candidate = pd.read_parquet(f'{candidate_path}{t}_candidate_{mode}.parquet')
		candidate['bpr'] = candidate.apply(lambda x: np.sum(u2emb[x['session']] * i2emb[x['aid']], axis=1), axis=1)
		candidate.to_parquet(
			f'{feature_path}{t}_bpr_feature_{mode}.parquet',
			index=False
		)


MODE = 'valid'
assert MODE in ['valid', 'test']
make_bpr_feature(MODE)

