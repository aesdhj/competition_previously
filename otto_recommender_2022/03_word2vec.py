import pandas as pd
from gensim.models import Word2Vec


MODE, SEED = 'valid', 2023
assert MODE in ['test', 'valid']
data_path = f'data/train_{MODE}/'
preprocess_path = 'data/preprocess/'

train = pd.read_parquet(data_path + 'train.parquet')
test = pd.read_parquet(data_path + 'test.parquet')
df = pd.concat([train, test], axis=0, ignore_index=True)
# 可以考虑click, cart&order分开生成特征
sentences = df.groupby('session', as_index=False)['aid'].agg(list)
sentences = list(sentences['aid'])
for dim in [16, 64]:
	w2vec = Word2Vec(
		sentences=sentences,
		vector_size=dim,
		window=5,
		min_count=1,
		workers=1,
		seed=SEED
	)
	w2v_df = pd.DataFrame(w2vec.wv.index_to_key, columns='aid')
	w2v_vec_df = pd.DataFrame(w2vec.wv.vectors).add_prefix('vec_')
	w2v_df = pd.concat([w2v_df, w2v_vec_df], axis=1, ignore_index=True)
	w2v_df.to_parquet(f'{preprocess_path}w2v_{dim}_{MODE}.parquet', index=False)

