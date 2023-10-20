import pandas as pd
import scanpy as sc
import numpy as np


MODE, DIM = 'valid', 16
assert MODE in ['test', 'valid']
assert DIM in [16, 64]
data_path = f'data/train_{MODE}/'
preprocess_path = 'data/preprocess/'

w2v = pd.read_parquet(f'{preprocess_path}w2v_{DIM}_{MODE}.parquet')
# scanpy入门 https://blog.csdn.net/weixin_43314378/article/details/129819365
x_all = sc.AnnData(X=w2v.iloc[:, 1:].values)
# use_rep='X' 使用自身的特征进行聚类
sc.pp.neighbors(x_all, use_rep='X', n_neighbors=64, method='umap')
# 在聚类的基础上优化聚类，是聚类更明显
sc.tl.leiden(x_all)
aid_df = w2v[['aid']].copy()
aid_df['cluster'] = list(x_all.obs['leiden'])
aid_df['cluster'] = aid_df['cluster'].astype(np.int)
aid_df.to_parquet(f'{preprocess_path}aid_cluster_{DIM}_{MODE}.parquet')

