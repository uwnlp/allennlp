GPU=$1
MODELNAME=$3

CUDA_VISIBLE_DEVICES=$GPU python -m allennlp.run train ./training_config/decomposoable_attention_mnli_tampered.json --serialization-dir ./logs/$MODELNAME