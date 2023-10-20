import pandas as pd


def make_candidate_row(type_dict):
	candidate_all = pd.DataFrame()
	for file_name in type_dict.keys():
		rank = type_dict[file_name]
		candidate = pd.read_parquet('data/preprocess/' + file_name)
		candidate = candidate[candidate['rank'] <= rank]
		if len(candidate_all) == 0:
			candidate_all = candidate[['session', 'aid']]
		else:
			candidate_all = pd.concat([candidate_all, candidate[['session', 'aid']]], axis=0)
	candidate_all = candidate_all.drop_duplicates().reset_index(drop=True)
	return candidate_all
		

def make_all_click_data():
	train = pd.read_parquet('data/train_valid/' + 'train.parquet')
	test = pd.read_parquet('data/train_valid/' + 'test.parquet')
	merge = pd.concat([train, test], axis=0, ignore_index=True)
	merge = merge.groupby('session', as_index=False)['ts'].max()
	merge = merge.rename(columns={'ts', 'ts_max'})
	train_all = pd.read_parquet('data/train_valid/' + 'test.parquet')
	train_all = train_all.merge(merge, on='session', how='left')
	train_all = train_all[train_all['ts'] > train_all['ts_max']]
	train_all = train_all[['session', 'aid']].drop_duplicates()
	train_all['target'] = 1
	return train_all


def main(type, type_dict, mode, calc_recall=False):
	type2id = {'clicks': 0, 'carts': 1, 'orders': 2}
	data_path = f'data/train_{mode}/'
	candidate_path = 'data/candidate/'
	test = pd.read_parquet(data_path + 'test.parquet')
	hist_all = test[['session', 'aid']].drop_duplicates()
	candidate_all = make_candidate_row(type_dict)
	# 可以理解为添加召回自身aid
	candidate_all = pd.concat([candidate_all, hist_all], axis=0)
	candidate_all = candidate_all.drop_duplicates().reset_index(drop=True)
	if mode == 'valid':
		if type != 'click_all':
			target = pd.read_parquet(data_path + 'test_label.parquet')
			# session, type, ground_type -> list
			target = target[target['type'] == type2id[type]]
			target = target.explore('ground_type')
			target = target.rename(columns={'ground_type': 'aid'})
			target['target'] = 1
			target = target[['session', 'aid', 'target']],
			candidate_all = candidate_all.merge(target, on=['session', 'aid'], how='left')
			candidate_all = candidate_all.fillna(0)
		else:
			# 官方click只取最接近的一个，click_all默认所有
			target = make_all_click_data()
			candidate_all = candidate_all.merge(target, on=['session', 'aid'], how='left')
	else:
		pass
	candidate_all.to_parquet(f'{candidate_path}{type}_candidate_{mode}.parquet', index=False)
	if calc_recall:
		# 怎么确定每种模式召回的数量
		# 遍历召回数量，但hits不明显增加的时候确定
		pred = candidate_all.groupby('session', as_index=False)['aid'].agg(list)
		pred = pred.rename(columns={'aid': 'aid_pred'})
		target = target.groupby('session', as_index=False)['aid'].agg(list)
		target = target.merge(pred, on='session', how='left')
		target['hits'] = target.apply(lambda x: len(set(x['aid_pred']) & set(x['aid'])), axi=1)
		target['aid_len'] = target['aid'].apply(len)
		target['aid_len_adjust'] = target['aid_len'].apply(lambda x: 20 if x > 20 else x)
		print(target['hits'].sum(), target['aid_len_adjust'].sum(), target['hits'].sum()/target['aid_len_adjust'].sum())


MODE = 'valid'
assert MODE in ['test', 'valid']
order_dict = []
order_dict.append(zip(
	[f'click_click_allterm_{pattern}_{MODE}.parquet' for pattern in ['last', 'top', 'hour', 'day']],
	[100, 20, 100, 30]))
order_dict.append(zip(
	[f'click_buy_allterm_{pattern}_{MODE}.parquet' for pattern in ['last', 'top', 'hour', 'day']],
	[40, 40, 40, 10]))
order_dict.append(zip(
	[f'buy_click_allterm_all_{MODE}.parquet', f'buy_buy_allterm_all_{MODE}.parquet'],
	[40, 40]))
order_dict.append(zip(
	[f'click_click_dup_{pattern}_{MODE}.parquet' for pattern in ['last', 'top', 'hour']],
	[20, 10, 20]))
order_dict.append(zip(
	[f'click_buy_dup_{pattern}_{MODE}.parquet' for pattern in ['last', 'top', 'hour']],
	[20, 10, 20]))
order_dict.append(zip(
	[f'buy_click_dup_all_{MODE}.parquet', f'buy_buy_dup_all_{MODE}.parquet'],
	[20, 20]))
order_dict.append(zip(
	[f'click_click_dup_wlen_{pattern}_{MODE}.parquet' for pattern in ['last', 'hour']],
	[20, 20]))
order_dict.append(zip(
	[f'click_buy_dup_wlen_{pattern}_{MODE}.parquet' for pattern in ['last', 'hour']],
	[20, 20]))
order_dict.append(zip(
	[f'click_click_dup_{pattern}_{MODE}.parquet' for pattern in ['last', 'top', 'hour']],
	[50, 10, 50]))
order_dict.append(zip(
	[f'buy_click_base_all_{MODE}.parquet', f'buy_buy_base_all_{MODE}.parquet'],
	[40, 40]))
order_dict.append(zip(
	[f'click_click_base_wlen_{pattern}_{MODE}.parquet' for pattern in ['last', 'top', 'hour']],
	[40, 10, 30]))
order_dict.append(zip(
	[f'buy_click_base_wlen_all_{MODE}.parquet', f'buy_buy_base_wlen_all_{MODE}.parquet'],
	[20, 20]))
order_dict.append(zip(
	[f'click_click_base_hour_{pattern}_{MODE}.parquet' for pattern in ['last', 'hour']],
	[15, 15]))
order_dict.append(zip(
	[f'click_click_dup_hour_{pattern}_{MODE}.parquet' for pattern in ['last', 'hour']],
	[5, 5]))
order_dict = dict(order_dict)
cart_dict = order_dict.copy()
click_dict = []
click_dict.append(zip(
	[f'click_click_dup_wlen_{pattern}_{MODE}.parquet' for pattern in ['last', 'hour', 'day']],
	[70, 50, 10]))
click_dict.append(zip(
	[f'click_click_base_hour_{pattern}_{MODE}.parquet' for pattern in ['last', 'top', 'hour', 'day']],
	[90, 5, 60, 20]))
click_dict.append(zip(
	[f'click_click_dup_hour_{pattern}_{MODE}.parquet' for pattern in ['last', 'hour']],
	[30, 30]))
click_dict.append(zip(
	[f'click_click_base_{pattern}_{MODE}.parquet' for pattern in ['last', 'top', 'hour', 'day']],
	[30, 5, 30, 10]))
click_dict.append(zip(
	[f'click_click_allterm_{pattern}_{MODE}.parquet' for pattern in ['last', 'top', 'hour', 'day']],
	[30, 5, 30, 10]))
click_dict.append(zip(
	[f'click_click_dup_{pattern}_{MODE}.parquet' for pattern in ['last', 'top', 'hour', 'day']],
	[30, 5, 30, 10]))
click_dict.append(zip(
	['click_click_w2v_last_w2v.parquet', 'click_click_w2v_hour_w2v.parquet'],
	[10, 5]
))
click_dict = dict(click_dict)

main('orders', order_dict, MODE)
main('carts', cart_dict, MODE)
main('clicks', click_dict, MODE)
main('clicks_all', click_dict, MODE)








