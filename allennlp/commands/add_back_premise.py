import pandas as pd
from allennlp.commands.config import *


if __name__ == '__main__':
	for dataset in DATASETS:
		df = pd.read_json(DATASETS[dataset]['original'], lines=True)
		tampered = pd.read_json(DATASETS[dataset]['full_tampered'], lines=True)
		master = tampered.merge(df[['sentence1', 'pairID']], on='pairID')
		master = master.rename(columns={'sentence1_x': 'sentence1', 'sentence1_y':'real_premise'})
		master = master.rename(columns={'sentence1':'null_token'})
		master = master.rename(columns={'real_premise':'sentence1'})
		master.to_json(DATASETS[dataset]['full_untampered'], lines=True, orient='records')
		tampered = pd.read_json(DATASETS[dataset]['easy_tampered'], lines=True)
		master = tampered.merge(df[['sentence1', 'pairID']], on='pairID')
		master = master.rename(columns={'sentence1_x': 'sentence1', 'sentence1_y':'real_premise'})
		master = master.rename(columns={'sentence1':'null_token'})
		master = master.rename(columns={'real_premise':'sentence1'})
		master.to_json(DATASETS[dataset]['easy_untampered'], lines=True, orient='records')
		tampered = pd.read_json(DATASETS[dataset]['hard_tampered'], lines=True)
		master = tampered.merge(df[['sentence1', 'pairID']], on='pairID')
		master = master.rename(columns={'sentence1_x': 'sentence1', 'sentence1_y':'real_premise'})
		master = master.rename(columns={'sentence1':'null_token'})
		master = master.rename(columns={'real_premise':'sentence1'})
		master.to_json(DATASETS[dataset]['hard_untampered'], lines=True, orient='records')