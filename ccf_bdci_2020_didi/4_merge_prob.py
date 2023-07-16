import pandas as pd
import numpy as np


nn_sub = pd.read_csv('data/nn_sub_prob.csv')
lgb_sub = pd.read_csv('data/lgb_sub_prob.csv')
test_sub = nn_sub.merge(lgb_sub, on='link', how='left')
for i in range(3):
	test_sub['pred_{}'.format(i)] = test_sub['lgb_pred_{}'.format(i)] * 0.3 + test_sub['nn_pred_{}'.format(i)] * 0.7
test_sub['label'] = np.argmax(test_sub[['pred_0', 'pred_1', 'pred_2']].values, axis=1)
test_sub = test_sub[['link', 'current_slice_id', 'future_slice_id', 'label']]
test_sub.to_csv('data/result.csv', index=False)
