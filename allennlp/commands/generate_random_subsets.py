import pandas as pd
from allennlp.commands.config import *

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='concat hard and easy train halfs')
	parser.add_argument('--corpus')
	args = parser.parse_args()
	if args.corpus == 'mnli':
		train = MNLI_TRAIN
	elif args.corpus == 'snli':
		train = SNLI_TRAIN
	hard_df = pd.read_json(train['hard'], lines=True)                                                                                
	easy_df = pd.read_json(train['easy'], lines=True)
	train_df = pd.read_json(train['original'], lines=True)
	random_hard = train_df.sample(n=hard_df.shape[0], random_state=1)
	random_easy = train_df.sample(n=easy_df.shape[0], random_state=1)         
	random_hard.to_json('{}/{}_random_hard.jsonl'.format(DATA_DIR, args.corpus), lines=True, orient='records')
	random_easy.to_json('{}/{}_random_easy.jsonl'.format(DATA_DIR, args.corpus), lines=True, orient='records')
