import subprocess
import argparse
import json
from allennlp.commands.config import *
import pandas as pd
import os
import arrow

base_config = {
    "dataset_reader": {
      "type": "snli",
      "token_indexers": {
        "tokens": {
          "type": "single_id",
          "lowercase_tokens": True
        }
      },
      "tokenizer": {
        "end_tokens": ["@@NULL@@"]
      }
    },
    "train_data_path": None,
    "validation_data_path": None,
    "model": {
      "type": "decomposable_attention",
      "text_field_embedder": {
        "tokens": {
          "type": "embedding",
          "projection_dim": 200,
          "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.300d.txt.gz",
          "embedding_dim": 300,
          "trainable": False
        }
      },
      "attend_feedforward": {
        "input_dim": 200,
        "num_layers": 2,
        "hidden_dims": 200,
        "activations": "relu",
        "dropout": 0.2
      },
      "similarity_function": {"type": "dot_product"},
      "compare_feedforward": {
        "input_dim": 400,
        "num_layers": 2,
        "hidden_dims": 200,
        "activations": "relu",
        "dropout": 0.2
      },
      "aggregate_feedforward": {
        "input_dim": 400,
        "num_layers": 2,
        "hidden_dims": [200, 3],
        "activations": ["relu", "linear"],
        "dropout": [0.2, 0.0]
      },
       "initializer": [
        [".*linear_layers.*weight", {"type": "xavier_normal"}],
        [".*token_embedder_tokens\._projection.*weight", {"type": "xavier_normal"}]
       ]
     },
    "iterator": {
      "type": "bucket",
      "sorting_keys": [["premise", "num_tokens"], ["hypothesis", "num_tokens"]],
      "batch_size": 64
    },
  
    "trainer": {
      "num_epochs": 140,
      "patience": 20,
      "cuda_device": 0,
      "grad_clipping": 5.0,
      "validation_metric": "+accuracy",
      "no_tqdm": True,
      "optimizer": {
        "type": "adagrad"
      }
    }
  }

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
    df_half_1 = df.head(n=half_size)
    df_half_2 = df.tail(n=half_size)
    log = "SIZE OF HALVES:{} // TOTAL_SIZE: {}".format(half_size, df.shape[0])
    df_half_1.to_json(DATASETS[half_1]['original'], lines=True, orient='records')
    df_half_2.to_json(DATASETS[half_2]['original'], lines=True, orient='records')
    return log
def execute(cmd):
    popen = subprocess.Popen(cmd,
                             stdout=subprocess.PIPE,
                             shell=True,
                             universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Execute training')
    parser.add_argument('--corpus', choices=['mnli', 'snli'], type=str)
    parser.add_argument('--split', choices=['half', 'full'], type=str)
    parser.add_argument('--half', choices=[0, 1, 2], type=int, default=0)
    parser.add_argument('--gpu', choices=[0, 1, 2], type=int)
    args = parser.parse_args()
    if args.split == 'half':
        print("splitting...")
        split_log = make_splits(args.corpus)
    if args.corpus == 'mnli' and args.split == 'full':
        model_name = "multinli_0.9_train"
        base_config['train_data_path'] = "multinli_0.9_train"
        base_config['validation_data_path'] = "multinli_0.9_dev_matched"
    elif args.corpus == 'mnli' and args.split == 'half':
        if args.half == 1:
            model_name = "mnli_train_half_1"
            base_config['train_data_path'] = "mnli_train_half_1"
            base_config['validation_data_path'] = "multinli_0.9_dev_matched"
        elif args.half == 2:
            model_name = "mnli_train_half_2"
            base_config['train_data_path'] = "mnli_train_half_2"
            base_config['validation_data_path'] = "multinli_0.9_dev_matched"
        else:
            raise Exception("invalid argument")
    elif args.corpus == 'snli' and args.split == 'full':
        model_name = "snli_1.0_train"
        base_config['train_data_path'] = "snli_1.0_train"
        base_config['validation_data_path'] = "snli_1.0_dev"  
    elif args.corpus == 'snli' and args.split == 'half':
        if args.half == 1:
            model_name = "snli_train_half_1"
            base_config['train_data_path'] = "snli_train_half_1"
            base_config['validation_data_path'] = "snli_1.0_dev"  
        elif args.half == 2:
            model_name = "snli_train_half_2"
            base_config['train_data_path'] = "snli_train_half_2"
            base_config['validation_data_path'] = "snli_1.0_dev"
        else:
            raise Exception("invalid argument")
    else:
        raise Exception("invalid argument") 
    base_config['train_data_path'] = DATASETS[base_config['train_data_path']]['original']
    base_config['validation_data_path'] = DATASETS[base_config['validation_data_path']]['original']
    config_file = "/home/sg01/allennlp/training_config/c1_train_configs/{}_{}_{}.json".format(args.corpus, args.split, args.half)
    with open(config_file, 'w+') as f:
        out = json.dumps(base_config)
        f.write(out)
    serialization_dir = "/home/sg01/allennlp/final_logs/{}".format(model_name) 
    command = ["CUDA_VISIBLE_DEVICES={}".format(args.gpu),
               "python -m allennlp.run train",
               config_file, 
               "--serialization-dir",
               serialization_dir,
               "--c1"]
    
    
    now = arrow.utcnow()
    log_file = './execute_train_logs/{}_{}_{}_{}.log'.format(args.corpus, args.split, args.half, now)
    with open(log_file, 'w+') as f:
        if split_log:
            f.write(split_log)
        f.write(" ".join(command)+ "\n")
        f.write("ARGS: {} {} {}\n".format(args.corpus, args.split, args.half))
        f.write("TRAIN_DATA_PATH: {}\n".format(base_config['train_data_path']))
        f.write("VALIDATION_DATA_PATH: {}\n".format(base_config['validation_data_path']))
        f.write("CONFIG_FILE: {}\n".format(config_file))
        f.write("SERIALIZATION DIR: {}\n".format(serialization_dir))
    # for path in execute(command):
    #     print(path, end="")
