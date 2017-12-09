import subprocess
from config import *
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Test and Subset')
	parser.add_argument('--model')
	parser.add_argument('--dataset')
	parser.add_argument('--gpu')
	args = parser.parse_args()
	gpu = 'CUDA_VISIBLE_DEVICES={}'.format(args.gpu)
	archive_file = config[args.model]['archive_file']
	evaluation_data_file = config[args.dataset]['tampered']
	subprocess.Popen([gpu,
					  "python -m allennlp.run evaluate",
					  "--archive-file",
					  archive_file,
					  "--evaluation-data-file",
					  evaluation_data_file,
					  "--subset"])
	easy_fp = os.path.join(DATA_DIR, EASY_FILE)
	hard_fp = os.path.join(DATA_DIR, HARD_FILE)
	subprocess.Popen(['mv', "easy_subset.json", easy_fp])
	subprocess.Popen(['mv', "hard_subset.json", hard_fp])