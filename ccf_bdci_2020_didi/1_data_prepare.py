import pandas as pd
pd.options.display.max_columns = None
from utils import *


def get_data(path, id_sample):
	columns = ['link', 'label', 'current_slice_id', 'future_slice_id']
	for i in range(1, 6):
		columns.append(f'recent_feature_{i}')
	for i in range(1, 5):
		for j in range(1, 6):
			columns.append(f'history_feature_cycle{i}_gap{j}')
	data = pd.read_csv(path, header=None, sep=';')
	# id
	for i in range(4):
		data[columns[i]] = data[0].apply(lambda x: x.split()[i])
	# recent, history_feature
	for i in range(1, 6):
		data['temp'] = data[i].apply(lambda x: x.split())
		for j in range(5):
			data[columns[4 + 5 * (i - 1) + j]] = data['temp'].apply(lambda x: x[j])

	data = data[columns]
	save_columns = []
	for col in data.columns:
		if 'feature' in col:
			data['temp'] = data[col].apply(lambda x: x.split(','))
			data[col + '_speed'] = data['temp'].apply(lambda x: float(x[0].split(':')[1]))
			data[col + '_eta'] = data['temp'].apply(lambda x: float(x[1]))
			data[col + '_status'] = data['temp'].apply(lambda x: float(x[2]))
			data[col + '_num_car'] = data['temp'].apply(lambda x: float(x[3]))
			save_columns.extend([col + '_speed', col + '_eta', col + '_status', col + '_num_car'])
	data = data[columns[:4] + save_columns]
	# 保证link_id有topo特征
	data_sample = data[data['link'].isin(id_sample)]
	print(path, data.shape, data_sample.shape)
	return data_sample


"""
数据结构
link label current_slice_id future_slice_id + 
以current_slice_id为结尾的连续5段数据特征 +
以future_slice_id为开始的历史同期的5段数据特征
数据特征的结构 slice_id:speed,eta,status,num_car
预测future_slice_id的status
"""
if __name__ == '__main__':
	data_path = 'data/traffic/{}.txt'
	save_data_path = 'data/train_{}.pkl'
	test_data_path = 'data/20190801_testdata.txt'
	test_data_save_path = 'data/test_new.pkl'

	id_sample = id_select()
	# print(len(id_sample))
	for i in range(1, 31):
		num = 20190700 + i
		train_df = get_data(data_path.format(num), id_sample)
		train_df['day'] = i
		train_df.to_pickle(save_data_path.format(i))

	test = get_data(test_data_path, id_sample)
	test['day'] = 32
	test.to_pickle(test_data_save_path)














