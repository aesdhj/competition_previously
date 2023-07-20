from utils import *

if __name__ == '__main__':
	click_df, train_user, test_user = merge_files()
	# print(click_df.shape, train_user.shape, test_user.shape)
	click_df.to_pickle('data/click.pkl')
	train_user.to_pickle('data/train_user.pkl')
	test_user.to_pickle('data/test_user.pkl')

