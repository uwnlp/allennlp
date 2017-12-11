import pandas as pd
from allennlp.commands.config import *
import os

# TODO: 1-line-missing problem

def make_splits(corpus):
	if corpus == 'mnli':
		dataset = 'multinli_0.9_train'
		half_1 = 'mnli_train_half_1'
		half_2 = 'mnli_train_half_2'
	else:
		dataset = 'snli_1.0_train'
		half_1 = 'snli_train_half_1'
		half_2 = 'snli_train_half_2'
	df = pd.read_json(DATASETS[dataset]['original'], lines=True)
	df = df.sample(frac=1, random_state=1)
	half_size = int(df.shape[0]/2)
	half_1 = df.head(n=half_size)
	half_2 = df.tail(n=half_size)
	half_1.to_json(DATASETS[half_1]['original'], lines=True, orient='records')
	half_2.to_json(DATASETS[half_2]['original'], lines=True, orient='records')