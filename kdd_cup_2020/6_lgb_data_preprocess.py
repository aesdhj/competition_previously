from utils import *


mode = cur_mode
cur_stage = CUR_STAGE

# recall_item 相关特征整理成表格数据，用于下一步生成特征
if __name__ == '__main__':
	if mode == 'valid':
		df_train = load_pickle(all_train_data_path.format(cur_stage))
		df_valid = load_pickle(all_valid_data_path.format(cur_stage))
		df_stage = load_pickle(all_valid_stage_data_path.format(cur_stage))
		df = pd.concat([df_train, df_valid], axis=0)
		df = df.sort_values(['user_id', 'time'])
		gen_data(df, df_stage, mode, cur_stage)
