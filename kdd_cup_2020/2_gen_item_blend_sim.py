from utils import *
import numpy as np
from tqdm import tqdm


# 根据文本和图片相似性的融合求出每个item前500的item
item_feat = load_pickle(item_feat_pkl)
feat_item_set = set(item_feat.keys())
item_vec1 = np.zeros((max(feat_item_set)+1, 128), dtype='float32')
item_vec2 = np.zeros((max(feat_item_set)+1, 128), dtype='float32')

for k, v in item_feat.items():
	item_vec1[k] = v[0]
	item_vec2[k] = v[1]

split_size = 1000
split_num = int(item_vec1.shape[0]/split_size)
if item_vec1.shape[0] % split_size != 0:
	split_num += 1
all_idx = []
all_score = []
l2norm1 = np.linalg.norm(item_vec1, ord=2, axis=1, keepdims=True)
item_vec1 = item_vec1 / (l2norm1 + 1e-9)
l2norm2 = np.linalg.norm(item_vec2, ord=2, axis=1, keepdims=True)
item_vec2 = item_vec2 / (l2norm2 + 1e-9)
vec1_trans = np.transpose(item_vec1)
vec2_trans = np.transpose(item_vec2)
for i in tqdm(range(split_num)):
	# numpy内存泄露问题，用copy或者列表取值解决，https://zhuanlan.zhihu.com/p/80689571
	vec1_part = item_vec1[i*split_size:(i+1)*split_size].copy()
	vec2_part = item_vec2[i*split_size:(i+1)*split_size].copy()
	text_sim = vec1_part.dot(vec1_trans)
	image_sim = vec2_part.dot(vec2_trans)
	blend_sim = 0.95 * text_sim + 0.05 * image_sim

	idx = (-blend_sim).argsort(axis=1)
	blend_sim = (- blend_sim)
	blend_sim.sort(axis=1)
	idx = idx[:, :500].copy()
	score = blend_sim[:, :500].copy()
	score = (- score)
	all_idx.append(idx)
	all_score.append(score)

idx = np.concatenate(all_idx, axis=0)
score = np.concatenate(all_score, axis=0)

sim = []
for i in range(idx.shape[0]):
	if i in feat_item_set:
		sim_i = []
		for j, item in enumerate(idx[i]):
			if item in feat_item_set:
				sim_i.append((item, score[i][j]))
		sim.append((i, sim_i))

write_sim(sim, item_blend_sim_path)




































