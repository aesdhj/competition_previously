from utils import *
import pandas as pd
pd.options.display.max_columns = None


if __name__ == '__main__':
	click_log = pd.read_pickle('data/click.pkl')
	train_df = pd.read_pickle('data/train_user.pkl')
	test_df = pd.read_pickle('data/test_user.pkl')

	# user_id相关的统计特征
	print('extracting aggregate feature...')
	agg_features = []
	agg_features += get_agg_features([train_df, test_df], 'user_id', '', 'size', click_log)
	agg_features += get_agg_features([train_df, test_df], 'user_id', 'ad_id', 'unique', click_log)
	agg_features += get_agg_features([train_df, test_df], 'user_id', 'creative_id', 'unique', click_log)
	agg_features += get_agg_features([train_df, test_df], 'user_id', 'advertiser_id', 'unique', click_log)
	agg_features += get_agg_features([train_df, test_df], 'user_id', 'industry', 'unique', click_log)
	agg_features += get_agg_features([train_df, test_df], 'user_id', 'product_id', 'unique', click_log)
	agg_features += get_agg_features([train_df, test_df], 'user_id', 'time', 'unique', click_log)
	agg_features += get_agg_features([train_df, test_df], 'user_id', 'click_times', 'sum', click_log)
	agg_features += get_agg_features([train_df, test_df], 'user_id', 'click_times', 'mean', click_log)
	agg_features += get_agg_features([train_df, test_df], 'user_id', 'click_times', 'std', click_log)
	# print(train_df[agg_features].isna().sum())
	# print(test_df[agg_features].isna().sum())
	train_df[agg_features] = train_df[agg_features].fillna(-1)
	test_df[agg_features] = test_df[agg_features].fillna(-1)
	print(agg_features)

	# user_id序列特征list形式，后期在Model进行BERT提取特征
	print('extracting sequence feature...')
	text_features = []
	text_features += sequence_text([train_df, test_df], 'user_id', 'ad_id', click_log)
	text_features += sequence_text([train_df, test_df], 'user_id', 'creative_id', click_log)
	text_features += sequence_text([train_df, test_df], 'user_id', 'advertiser_id', click_log)
	text_features += sequence_text([train_df, test_df], 'user_id', 'product_id', click_log)
	text_features += sequence_text([train_df, test_df], 'user_id', 'industry', click_log)
	text_features += sequence_text([train_df, test_df], 'user_id', 'product_category', click_log)
	text_features += sequence_text([train_df, test_df], 'user_id', 'time', click_log)
	text_features += sequence_text([train_df, test_df], 'user_id', 'click_times', click_log)
	print(text_features)
	# print(train_df[text_features].isna().sum())
	# print(test_df[text_features].isna().sum())

	# kfold target encoding，求出user_id对应不同pivot字段背后年龄，性别平均分布
	print('extracting kfold feature...')
	kfold_features = []
	log_data = click_log.drop_duplicates(['user_id', 'creative_id']).reset_index(drop=True)
	train_df['fold'] = train_df.index % 5
	test_df['fold'] = 5
	df = train_df.append(test_df)[['user_id', 'fold']].reset_index(drop=True)
	log_data = log_data.merge(df, on='user_id', how='left')
	for pivot in ['creative_id', 'ad_id', 'product_id', 'advertiser_id', 'industry']:
		kfold_features += kfold([train_df, test_df], log_data, pivot)
	print(kfold_features)
	# print(train_df[kfold_features].isna().sum())
	# print(test_df[kfold_features].isna().sum())

	# kfold target encoding，pivot_fold的年龄，性别平均分布以w2v导出
	# user_id 对应pivot_fold list形式在MODEL_LAST进行BERT提取特征
	print('extracting kfold sequence feature...')
	kfold_sequence_features = []
	for pivot in ['creative_id', 'ad_id', 'product_id', 'advertiser_id', 'industry']:
		kfold_sequence_features += kfold_sequence(train_df, test_df, log_data, pivot)
	print(kfold_sequence_features)
	# print(train_df[kfold_sequence_features].isna().sum())
	# print(test_df[kfold_sequence_features].isna().sum())

	print(train_df.shape, test_df.shape)
	print(train_df.head())
	train_df.to_pickle('data/train_user_features.pkl')
	test_df.to_pickle('data/test_user_features.pkl')



































