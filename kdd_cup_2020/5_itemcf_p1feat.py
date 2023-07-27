from utils import *


cur_stage = CUR_STAGE
mode = cur_mode

recall_diff_road_list = ['i2i_w02', 'i2i_w10', 'b2b', 'i2i2i_new']

if mode == 'valid':
	df_train = load_pickle(all_train_data_path.format(cur_stage))
	df_train_stage = load_pickle(all_train_stage_data_path.format(cur_stage))
	df = load_pickle(all_valid_data_path.format(cur_stage))
	df_stage = load_pickle(all_valid_stage_data_path.format(cur_stage))
else:
	df_train = load_pickle(online_all_train_data_path.format(cur_stage))
	df_train_stage = load_pickle(online_all_train_stage_data_path.format(cur_stage))
	df = load_pickle(online_all_test_data_path.format(cur_stage))
	df['item_id'] = np.nan
	df_stage = load_pickle(online_all_test_data_path.format(cur_stage))
	df_stage['item_id'] = np.nan

# 利用item 之间的相似性召回最可能相关的related_item
if 'i2i_w02' in recall_diff_road_list:
	i2i_w02_recall(df_train, df_train_stage, df, df_stage, cur_stage)
if 'i2i_w10' in recall_diff_road_list:
	i2i_w10_recall(df_train, df_train_stage, df, df_stage, cur_stage)
if 'b2b' in recall_diff_road_list:
	b2b_recall(df_train, df_train_stage, df, df_stage, cur_stage)
if 'i2i2i_new' in recall_diff_road_list:
	i2i2i_new_recall(df_train, df_train_stage, df, df_stage, cur_stage)

































