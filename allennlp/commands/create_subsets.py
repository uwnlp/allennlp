from allennlp.commands.config import *
import argparse
import os
import subprocess

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Test and Subset')
	parser.add_argument('--dataset')
	args = parser.parse_args()
	easy_fp = os.path.join(DATA_DIR, DATASETS[args.dataset]['easy_tampered'])
	hard_fp = os.path.join(DATA_DIR, DATASETS[args.dataset]['hard_tampered'])
	print(" ".join(['mv', "easy_subset.json", easy_fp]))
	print(" ".join(['mv', "hard_subset.json", hard_fp]))
