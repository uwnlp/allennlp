import pandas as pd
from allennlp.commands.config import *
from tqdm import tqdm 

if __name__ == '__main__':
	for dataset in tqdm(DATASETS):
		df = pd.read_json(DATASETS[dataset]['original'], lines=True)
		df = df.rename(columns={'sentence1':'real_premise'})
		df['sentence1'] = ['<NULL>'] * df.shape[0]
		df.to_json(DATASETS[dataset]['full_tampered'], lines=True, orient='records')