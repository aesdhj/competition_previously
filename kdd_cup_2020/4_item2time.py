from utils import *


if __name__ == '__main__':
	cur_stage = CUR_STAGE
	# 每个item出现的时间集合列表，格式{item:[time1, time2, ...], ...}
	item2time(cur_stage)
	# 在一定时间间隔内，两个item在同一user中出现的次数，格式{time_delta:{(itme_a, item_b):n,...}
	item_pair2time_diff(cur_stage)
	# 两个item在同一user中出现的时间差，和具体时间差列表和时间对列表
	# 时间差格式{(item_a, item_b): [time_diff_1, ...], ...}
	# 时间对格式{(item_a, item_b): [(time_a_1, time_b_2), ...], ...}
	item_pair2time_seq(cur_stage)

























