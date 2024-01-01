#!/bin/bash
#SBATCH -o vanilla_finetune_24.%j.out
#SBATCH --gres=gpu:1
#SBATCH --partition=accelerated
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL

eval "$(conda shell.bash hook)"
conda activate g2t

nvidia-smi

EXP_TYPE='vanilla'
MODE='do_train'
NUM_EPOCHS=50
LR=5e-5
N_LAYER=24
OPTIMIZER='AdamW'
NUM_DEMO_PER_CLASS=200
TASK_NAME=('sst2' 'emo' 'agnews' 'trec')
SEEDS=(0 42 421 520 1218)
PROJECT_NAME="${EXP_TYPE}_${N_LAYER}_finetune"

runexp(){
python3 train.py \
--task_name  ${1} \
--exp_type $EXP_TYPE \
--mode $MODE \
--epochs $NUM_EPOCHS \
--learning_rate $LR \
--n_layer $N_LAYER \
--optimizer $OPTIMIZER \
--warm_up_steps 0 \
--num_gpu 1 \
--random_seed ${2} \
--project_name $PROJECT_NAME \
--num_demo_per_class $NUM_DEMO_PER_CLASS
}

for SEED in "${SEEDS[@]}"; do
    for NAME in "${TASK_NAME[@]}"; do
        runexp $NAME $SEED
    done
done