import subprocess

def execute(cmd):
    popen = subprocess.Popen(cmd,
                             stdout=subprocess.PIPE,
                             universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Execute training')
    parser.add_argument('--corpus', choice=['mnli', 'snli'], type=str)
    parser.add_argument('--split', choice=['half', 'full'], type=str)
    parser.add_argument('--half', choice=[0, 1, 2], type=int, default=0)
    parser.add_argument('--gpu', choice=[0, 1, 2], type=int)

    config_file = "/home/sg01/allennlp/training_config/c1_train_configs/{}_{}_{}".format(args.corpus, args.split, args.half)
    if args.corpus == 'mnli' and args.split == 'full':
        model_name = "multinli_0.9_train"
    elif args.corpus == 'mnli' and args.split == 'half':
        if args.half == 1:
            model_name = "mnli_train_half_1"
        elif args.half == 2:
            model_name = "mnli_train_half_2"
        else:
            raise Exception("invalid argument: ", args.half)
    elif args.corpus == 'snli' and args.split == 'full':
        model_name = "snli_1.0_train"
    elif args.corpus == 'snli' and args.split == 'half':
        if args.half == 1:
            model_name = "snli_train_half_1"
        elif args.half == 2:
            model_name = "snli_train_half_2"
        else:
            raise Exception("invalid argument: ", args.half)
    else:
        raise Exception("invalid argument: ", args.corpus) 
    serialization_dir = "/home/sg01/allennlp/final_logs/{}".format(model_name) 
    command = [args.gpu,
               "python -m allennlp.run train",
               config_file, 
               "--serialization-dir",
               serialization_dir,
               "--c1"]
    print(command)
    for path in execute(command):
        print(path, end="")
