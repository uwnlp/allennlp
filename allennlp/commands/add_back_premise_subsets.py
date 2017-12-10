import pandas as pd
from allennlp.commands.config import *


if __name__ == '__main__':
	for dataset in DATASETS:
		df = pd.read_json(DATASETS[dataset]['easy_tampered'], lines=True)
		df = df.rename(columns={'sentence1': 'null_token'})
		df = df.rename(columns={'real_premise': 'sentence1'})
		master.to_json(DATASETS[dataset]['easy_untampered'], lines=True, orient='records')
		df = pd.read_json(DATASETS[dataset]['hard_tampered'], lines=True)
		df = df.rename(columns={'sentence1': 'null_token'})
		df = df.rename(columns={'real_premise': 'sentence1'})
		master.to_json(DATASETS[dataset]['hard_untampered'], lines=True, orient='records')