import subprocess
import argparse
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
    config_file = "/home/sg01/allennlp/training_config/c1_train_configs/{}_{}_{}.json".format(args.corpus, args.split, args.half)
    pd.to_json(config_file, lines=True, orient='records')  
    serialization_dir = "/home/sg01/allennlp/final_logs/{}".format(model_name) 
    command = ["CUDA_VISIBLE_DEVICES={}".format(args.gpu),
               "python -m allennlp.run train",
               config_file, 
               "--serialization-dir",
               serialization_dir,
               "--c1"]
    print(" ".join(command))
    for path in execute(command):
        print(path, end="")
