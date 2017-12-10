GPU=$1
MODELNAME=$2

 
CUDA_VISIBLE_DEVICES=$GPU python -m allennlp.run train ./training_config/decomposable_attention_mnli_tampered.json --serialization-dir ./logs/$MODELNAME --c1