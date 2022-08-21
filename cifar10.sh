#!/bin/bash
export EXPR_ID=20
export DATA_DIR=/data/users/fz920/data
export CHECKPOINT_DIR=/data/users/fz920/Constrainted-NVAE/checkpoint
export CODE_DIR=/data/users/fz920/Constrainted-NVAE
cd $CODE_DIR
CUDA_VISIBLE_DEVICES=3 python3 train.py --data $DATA_DIR/cifar10 --root $CHECKPOINT_DIR --save $EXPR_ID --dataset cifar10 \
        --num_channels_enc 32 --num_channels_dec 32 --epochs 12 --num_postprocess_cells 2 --num_preprocess_cells 2 \
        --num_latent_scales 1 --num_latent_per_group 20 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
        --num_preprocess_blocks 1 --num_postprocess_blocks 1 --num_groups_per_scale 10 --batch_size 64 --learning_rate 5e-3 \
        --weight_decay_norm 5e-2 --num_nf 0 --num_process_per_node 1 --use_se --res_dist --fast_adamax
