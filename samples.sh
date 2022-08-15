export EXPR_ID=11
export DATA_DIR=/data/users/fz920/data
export CHECKPOINT_DIR=/data/users/fz920/Constrainted-NVAE/checkpoint
export CODE_DIR=/data/users/fz920/Constrainted-NVAE

python evaluate.py --checkpoint $CHECKPOINT_DIR/eval-$EXPR_ID/checkpoint.pt --eval_mode=sample --temp=0.6 --readjust_bn \
                   --save /data/users/fz920/expr