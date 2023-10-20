from beartype import beartype
import json
from pandas.io.json._json import JsonReader
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from typing import List
from copy import deepcopy
import random


@beartype		# beartype检查参数类型， https://www.codenong.com/15299878/
def get_max_ts(sessions_file: Path) -> int:
	# path库入门 https://blog.csdn.net/m0_71986704/article/details/130658599
	max_ts = float('-inf')
	with open(sessions_file) as f:
		for line in tqdm(f, desc='finding max timestamp'):
			# dumps dict->str, loads str->dict
			session = json.loads(line)
			max_ts = max(max_ts, session['events'][-1]['ts'])
	return max_ts


class setEncoder(json.JSONEncoder):
	def default(self, obj):
		return list(obj)


@beartype
def trim_session(session: dict, max_ts: int) -> dict:
	session['events'] = [event for event in session['events'] if event['ts'] < max_ts]
	return session


@beartype
def train_test_split(
		session_chunks: JsonReader,
		train_file: Path,
		test_file: Path,
		max_ts: int,
		test_days: int,
):
	split_ts = max_ts - test_days * 24 * 60 * 60 * 1000		# unix timestamp精确到毫秒

	# https://github.com/otto-de/recsys-dataset
	# 为了防止数据泄露，官方规定了训练集和测试集的划分
	# 整体在split_ts之后的,划分到test，其他的数据切掉split_ts之后的的数据划分到train

	# parents=True可以创建多级目录
	Path(train_file).parent.mkdir(parents=True, exist_ok=True)
	train_file = open(train_file, 'w')
	Path(test_file).parent.mkdir(parents=True, exist_ok=True)
	test_file = open(test_file, 'w')
	for chunk in tqdm(session_chunks, desc='splitting sessions'):
		for _, session in chunk.iterrows():
			session = session.to_dict()
			if session['events'][0]['ts'] > split_ts:
				# dict, set不能序列化，可以提前转化为list在序列化
				# https://geek-docs.com/python/python-ask-answer/400_python_set_object_is_not_json_serializable.html
				test_file.write(json.dumps(session, cls=setEncoder) + '\n')
			else:
				session = trim_session(session, max_ts)
				train_file.write(json.dumps(session, cls=setEncoder) + '\n')
	train_file.close()
	test_file.close()


@beartype
def ground_truth(events: List[dict]):
	# 由当前时间的行为引发后续的三种状态的行为
	prev_labels = {'clicks': None, 'carts': set(), 'orders': set()}

	for event in reversed(events):
		event['labels'] = {}
		for label in ['clicks', 'carts', 'orders']:
			if prev_labels[label]:
				if label != 'clicks':
					event['labels'][label] = prev_labels[label].copy()
				else:
					event['labels'][label] = prev_labels[label]
		if event['type'] == 'clicks':
			prev_labels['clicks'] = event['aid']
		if event['type'] == 'carts':
			prev_labels['carts'].add(event['aid'])
		elif event['type'] == 'orders':
			prev_labels['orders'].add(event['aid'])
	return events[:-1]


@beartype
def split_events(events: List[dict], split_idx=None):
	test_events = ground_truth(deepcopy(events))
	if not split_idx:
		split_idx = random.randint(1, len(test_events))
	test_events = test_events[:split_idx]
	labels = test_events[-1]['labels']
	for event in test_events:
		del event['labels']
	return test_events, labels


@beartype
def create_kaggle_testset(
		sessions: pd.DataFrame,
		session_output: Path,
		labels_output: Path
):
	last_labels = []
	splitted_sessions = []

	for _, session in tqdm(sessions.iterrows(), desc='creating trimmed testset', total=len(sessions)):
		session = session.to_dict()
		splitted_events, labels = split_events(session['events'])
		last_labels.append({'session': session['session'], 'labels': labels})
		splitted_sessions.append(
			{'session': session['session'], 'events': splitted_events}
		)

	with open(session_output, 'w') as f:
		for session in splitted_sessions:
			f.write(json.dumps(session) + '\n')
	with open(labels_output, 'w') as f:
		for label in last_labels:
			f.write(json.dumps(label, cls=setEncoder) + '\n')


def json_to_pq(file, type2id, output_path):
	chunks = pd.read_json(file, lines=True, chunksize=100000)
	event_dict = {'session': [], 'aid': [], 'ts': [], 'type': []}
	for chunk in tqdm(chunks, desc='json trans to parquet'):
		for session, events in zip(chunk['session'].tolist(), chunk['events'].tolist()):
			for event in events:
				event_dict['session'].append(session)
				event_dict['aid'].append(event['aid'])
				event_dict['ts'].append(int(event['ts']/1000))
				event_dict['type'].append(type2id[event['type']])
	pd.DataFrame(event_dict).to(output_path, index=False)


def json_to_pq_y(file, output_path):
	# lines=True 行读取
	df = pd.read_json(file, lines=True)
	event_dict = {'session': [], 'type': [], 'ground_type': []}
	for session, labels in df.values:
		for k, v in labels.items():
			event_dict['session'].append(session)
			event_dict['type'].append(k)
			event_dict['ground_type'].append(list(v))
	pd.DataFrame(event_dict).to_parquet(output_path, index=False)


def train_valid_split():
	# https://www.kaggle.com/code/radek1/howto-full-dataset-as-parquet-csv-files?scriptVersionId=115190031
	id2type = ['clicks', 'carts', 'orders']
	# 读取原始数据
	session_chunks = pd.read_json('data/train.jsonl', lines=True, chunksize=100000)
	max_ts = get_max_ts(Path('data/train.jsonl'))
	# 划分训练集和验证集
	train_test_split(
		session_chunks,
		Path('data/train_valid/train.jsonl'),
		Path('data/train_valid/valid.jsonl'),
		max_ts,
		7
	)
	# 把训练集转化为parquet格式
	json_to_pq(
		'data/train_valid/train.jsonl',
		id2type,
		'data/train_valid/train.parquet')
	# 验证集划分出label
	valid_df = pd.read_json('data/train_valid/valid.jsonl')
	create_kaggle_testset(
		valid_df,
		Path('data/train_valid/test.jsonl'),
		Path('data/train_valid/test_label.jsonl')
	)
	# 把验证集转化为parquet格式
	json_to_pq(
		'data/train_valid/valid.jsonl',
		id2type,
		'data/train_valid/test.parquet')
	# 把label转化为parquet格式
	json_to_pq_y(
		'data/train_valid/test_label.jsonl',
		'data/train_valid/test_label.parquet'
	)


def train_test_process():
	# https://www.kaggle.com/datasets/radek1/otto-full-optimized-memory-footprint/data
	id2type = ['clicks', 'carts', 'orders']
	# 把训练集转化为parquet格式
	json_to_pq(
		'data/train.jsonl',
		id2type,
		'data/train_test/train.parquet')
	# 把测试集转化为parquet格式
	json_to_pq(
		'data/test.jsonl',
		id2type,
		'data/train_test/test.parquet')


# type2id = {'clicks': 0, 'carts': 1, 'orders': 2}
train_valid_split()
train_test_process()

