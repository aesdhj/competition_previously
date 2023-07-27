from utils import *


if __name__ == '__main__':
	# 整合每个stage的训练集，验证集，测试集
	stage = int(CUR_STAGE)

	if online == 'offline':
		gen_stage_data(stage)
	elif online == 'online':
		gen_stage_data_online(stage)
	else:
		print('online or offline error')
		print(1/0)

	gen_item_degree(stage)




































