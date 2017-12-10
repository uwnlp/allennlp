from allennlp.commands.config import *
import argparse
import os
import subprocess

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Test and Subset')
	parser.add_argument('--dataset')
	args = parser.parse_args()
	easy_fp = DATASETS[args.dataset]['easy_tampered']
	hard_fp = DATASETS[args.dataset]['hard_tampered']
	df = pd.read_json(DATASETS[args.dataset]['original'])
	df = df[['sentence1', 'pairID']]
	master = dev_hard.merge(, on='pairID')
	master = master.rename(columns={'sentence1_x': 'sentence1', 'sentence1_y':'real_premise'})
	print(" ".join(['mv', "easy_subset.json", easy_fp]))
	print(" ".join(['mv', "hard_subset.json", hard_fp]))
