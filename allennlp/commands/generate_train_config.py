base_config = {
    "dataset_reader": {
      "type": "snli",
      "token_indexers": {
        "tokens": {
          "type": "single_id",
          "lowercase_tokens": true
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
          "trainable": false
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
      "no_tqdm": true,
      "optimizer": {
        "type": "adagrad"
      }
    }
  }

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Test and Subset')
  parser.add_argument('--corpus', choice=['mnli', 'snli'], type=str)
  parser.add_argument('--split', choice=['half', 'full'], type=str)
  parser.add_argument('--half', choice=[1, 2], type=int, required=False)
  if args.corpus == 'mnli' and args.split == 'full':
    base_config['train_data_path'] = 
    base_config['validation_data_path'] = 
  elif args.corpus == 'mnli' and args.split == 'half':
    if args.half == 1:
      base_config['train_data_path'] = 
      base_config['validation_data_path'] = 
    elif args.half == 2:
      base_config['train_data_path'] = 
      base_config['validation_data_path'] = 
    else:
      raise Exception("invalid argument")
  elif args.corpus == 'snli' and args.split == 'full':
    base_config['train_data_path'] = 
    base_config['validation_data_path'] =     
  elif args.corpus == 'snli' and args.split == 'half':
    if args.half == 1:
      base_config['train_data_path'] = 
      base_config['validation_data_path'] = 
    elif args.half == 2:
      base_config['train_data_path'] = 
      base_config['validation_data_path'] = 
    else:
      raise Exception("invalid argument")
  else:
    raise Exception("invalid argument")    