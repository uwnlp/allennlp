import pandas as pd
from allennlp.commands.config import *
from tqdm import tqdm

if __name__ == '__main__':
	for dataset in tqdm(DATASETS):
		if DATASETS[dataset].get('easy_tampered') is not None:
			df = pd.read_json(DATASETS[args.dataset]['original'])
			df = df[['sentence1', 'pairID']]
			easy = pd.read_json(DATASETS[dataset]['easy_tampered'], lines=True)
			master = easy.merge(df, on='pairID')
			master = master.rename(columns={'sentence1_x': 'sentence1', 'sentence1_y':'real_premise'})
			master = master.rename(columns={'sentence1': 'null_token'})
			master = master.rename(columns={'real_premise': 'sentence1'})
			master.to_json(DATASETS[dataset]['easy_untampered'], lines=True, orient='records')
		if DATASETS[dataset].get('hard_tampered') is not None:
			df = pd.read_json(DATASETS[args.dataset]['original'])
			df = df[['sentence1', 'pairID']]
			easy = pd.read_json(DATASETS[dataset]['hard_tampered'], lines=True)
			master = easy.merge(df, on='pairID')
			master = master.rename(columns={'sentence1_x': 'sentence1', 'sentence1_y':'real_premise'})
			master = master.rename(columns={'sentence1': 'null_token'})
			master = master.rename(columns={'real_premise': 'sentence1'})
			master.to_json(DATASETS[dataset]['hard_untampered'], lines=True, orient='records')