import pandas as pd
from config import *

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='concat hard and easy train halfs')
	parser.add_argument('--corpus')
	args = parser.parse_args()
	if args.corpus == 'mnli':
		half_1 = MNLI_TRAIN_HALF_1
		half_2 = MNLI_TRAIN_HALF_2
	elif args.corpus == 'snli':
		half_1 = SNLI_TRAIN_HALF_1
		half_2 = SNLI_TRAIN_HALF_2
	for subset in ['hard', 'easy']:
		df = pd.read_json(half_1[subset], lines=True)                                                                                
		df1 = pd.read_json(half_2[subset], lines=True)                                                                               
		master = pd.concat([df, df1], axis=0)                                                                                                                     
		assert master.shape[0] == (df.shape[0] + df1.shape[0])
		master.to_json('{}/{}_train_{}.jsonl'.format(DATA_DIR, args.corpus, subset), lines=True, orient='records')
		