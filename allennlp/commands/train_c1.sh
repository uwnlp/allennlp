GPU=$1
CORPUS=$2
SPLIT=$3
HALF=$4
MODELNAME=$5

python -m allennlp.commands.generate_train_config --corpus $CORPUS --split $SPLIT --half $HALF

CUDA_VISIBLE_DEVICES=$GPU \
python -m allennlp.run \
train  ./training_config/c1_train_configs/${CORPUS}_${SPLIT}_${HALF}.json \
--serialization-dir ./logs/$MODELNAME \
--c1