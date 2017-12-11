import pandas as pd
from config import *

if __name__ == '__main__':
	if args.corpus == 'mnli':
		half_1 = MNLI_TRAIN_HALF_1
		half_2 = MNLI_TRAIN_HALF_2
	elif args.corpus == 'snli':
		half_1 = SNLI_TRAIN_HALF_1
		half_2 = SNLI_TRAIN_HALF_2
	for subset in ['hard', 'easy']
		df = pd.read_json(half_1['hard'], lines=True)                                                                                
		df1 = pd.read_json(half_2['hard'], lines=True)                                                                               
		master = pd.concat([df, df1], axis=0)                                                                                                                     
		assert master.shape[0] == (df.shape[0] + df1.shape[0])
		master.to_json('{}/{}_train_hard.jsonl'.format(DATA_DIR, args.corpus), lines=True, orient='records')
		df = pd.read_json(half_1['easy'] , lines=True)                                                                                
		df1 = pd.read_json(half_2['easy'], lines=True)                                                                               
		master = pd.concat([df, df1], axis=0)
		assert master.shape[0] == (df.shape[0] + df1.shape[0])                                                                                                           
		master.to_json('{}/{}_train_easy.jsonl'.format(DATA_DIR, args.corpus), lines=True, orient='records')
