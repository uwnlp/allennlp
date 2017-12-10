from subprocess import Popen, PIPE
from allennlp.commands.config import *
import argparse
import os
import subprocess

def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True, shell=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Test and Subset')
	parser.add_argument('--model')
	parser.add_argument('--dataset')
	parser.add_argument('--gpu')
	args = parser.parse_args()
	gpu = 'CUDA_VISIBLE_DEVICES={}'.format(args.gpu)
	archive_file = MODELS[args.model]['archive_file']
	evaluation_data_file = DATASETS[args.dataset]['full_tampered']
	command = [gpu,
			   "python -m allennlp.run evaluate",
			   "--archive-file",
			   archive_file,
			   "--evaluation-data-file",
			   evaluation_data_file,
			   "--subset"]
	print(" ".join(command))