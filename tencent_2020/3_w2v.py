from utils import *
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
	train_df = pd.read_pickle('data/train_user_features.pkl')
	test_df = pd.read_pickle('data/test_user_features.pkl')

	# user_id对应个字段的list输出w2v
	for key in tqdm(
			[
				'ad_id', 'creative_id', 'advertiser_id', 'product_id', 'industry',
				'product_category', 'time', 'click_times'
			]):
		w2v([train_df, test_df], 'sequence_text_user_id_{}'.format(key), L=128)
