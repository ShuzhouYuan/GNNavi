#!/bin/bash

MODEL=${1:-'gpt2'}
EXP_TYPE=${2:-'gnn'}
MODE='do_train'
NUM_EPOCHS=50
EARLY_STOP=15
LR=0.01
OPTIMIZER='Adam'
NUM_DEMO_PER_CLASS=5
TASK_NAME=('sst2' 'emo' 'trec' 'amazon' 'agnews')
SEEDS=(0 42 421 520 1218)
PROJECT_NAME="${EXP_TYPE}_${NUM_DEMO_PER_CLASS}_GPT2"

runexp(){
python3 src/train_${MODEL}.py \
--task_name  ${1} \
--exp_type $EXP_TYPE \
--mode $MODE \
--epochs $NUM_EPOCHS \
--learning_rate $LR \
--early_stop $EARLY_STOP \
--optimizer $OPTIMIZER \
--random_seed ${2} \
--project_name $PROJECT_NAME \
--num_demo_per_class $NUM_DEMO_PER_CLASS
}

for SEED in "${SEEDS[@]}"; do
    for NAME in "${TASK_NAME[@]}"; do
        runexp $NAME $SEED
    done
done