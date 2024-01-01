#!/bin/bash
#SBATCH -o sst2.%j.out
#SBATCH --gres=gpu:1
#SBATCH --partition=accelerated
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL

eval "$(conda shell.bash hook)"
conda activate g2t

nvidia-smi

TASK_NAME='sst2'
EXP_TYPE='reweight'
NUM_EPOCHS=100
LR=0.01
N_LAYER=48
OPTIMIZER='Adam'
SEED=0
NUM_DEMO_PER_CLASS=200
PROJECT_NAME="${TASK_NAME}_${EXP_TYPE}_${N_LAYER}"

python3 train.py \
--task_name $TASK_NAME \
--exp_type $EXP_TYPE \
--mode do_train \
--epochs $NUM_EPOCHS \
--learning_rate $LR \
--n_layer $N_LAYER \
--optimizer $OPTIMIZER \
--warm_up_steps 0 \
--num_gpu 1 \
--random_seed $SEED \
--project_name $PROJECT_NAME \
--num_demo_per_class $NUM_DEMO_PER_CLASS \