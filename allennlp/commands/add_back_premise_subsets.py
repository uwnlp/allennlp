import pandas as pd
from allennlp.commands.config import *
from tqdm import tqdm

if __name__ == '__main__':
	for dataset in tqdm(DATASETS):
		if DATASETS[dataset].get('easy_tampered') is not None:
			df = pd.read_json(DATASETS[dataset]['easy_tampered'], lines=True)
			import ipdb; ipdb.set_trace()
			df = df.rename(columns={'sentence1': 'null_token'})
			df = df.rename(columns={'real_premise': 'sentence1'})
			df.to_json(DATASETS[dataset]['easy_untampered'], lines=True, orient='records')
		if DATASETS[dataset].get('hard_tampered') is not None:
			df = pd.read_json(DATASETS[dataset]['hard_tampered'], lines=True)
			df = df.rename(columns={'sentence1': 'null_token'})
			df = df.rename(columns={'real_premise': 'sentence1'})
			df.to_json(DATASETS[dataset]['hard_untampered'], lines=True, orient='records')