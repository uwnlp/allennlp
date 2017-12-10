from allennlp.commands.config import *
import argparse
import os
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Test and Subset')
	parser.add_argument('--dataset')
	args = parser.parse_args()
	easy_fp = os.path.join(DATA_DIR, DATASETS[args.dataset]['easy_tampered'])
	hard_fp = os.path.join(DATA_DIR, DATASETS[args.dataset]['hard_tampered'])
	subprocess.Popen(['mv', "easy_subset.json", easy_fp], shell=True)
	subprocess.Popen(['mv', "hard_subset.json", hard_fp], shell=True)
