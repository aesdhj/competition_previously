from utils import *
import pandas as pd
import numpy as np


"""
PageRank入门,https://blog.csdn.net/gamer_gyt/article/details/47443877
ITEM-CF, USER-CF入门,https://github.com/ericens/RecSys
方案详解,https://zhuanlan.zhihu.com/p/191595907
方案重点总结:
	1,item2item相似度召回对流行度不敏感，可以用来缓解流行度偏差
	2,商品流行度长尾分布，验证集召回商品有一半流行度较高
"""


# 与训练的文本特征和图片特征
item2vecs = {}
with open(item_feat_csv) as f:
	for line in f:
		line = line.strip('\n')
		index = line.index(',')
		item_id = int(line[:index])
		index_2 = line.index(']')
		# eval将字符串转化为里面的内容或者公式
		vec1 = np.array(eval(line[index+1:index_2+1]))
		vec2 = np.array(eval(line[index_2+2:]))
		item2vecs[item_id] = [vec1, vec2]
dump_pickle(item2vecs, item_feat_pkl)

# 用户特征
df_user = pd.read_csv(
	user_feat_csv,
	names=['user_id', 'user_age_level', 'user_gender', 'user_city_level']
)
df_user['user_id'] = df_user['user_id'].astype(np.int32)
df_user['user_age_level'] = df_user['user_age_level'].fillna(0).astype(np.int32)
df_user['user_gender'] = df_user['user_gender'].map({'M': 1, 'F': 2}).fillna(0).astype(np.int32)
df_user['user_city_level'] = df_user['user_city_level'].fillna(0).astype(np.int32)
dump_pickle(df_user, user_feat_pkl)













