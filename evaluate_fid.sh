export EXPR_ID=6
export DATA_DIR=/data/users/fz920/data
export CHECKPOINT_DIR=/data/users/fz920/Constrainted-NVAE/checkpoint
export CODE_DIR=/data/users/fz920/Constrainted-NVAE

cd $CODE_DIR

CUDA_VISIBLE_DEVICES=0 python scripts/precompute_fid_statistics.py --data $DATA_DIR/cifar10 --dataset cifar10 --fid_dir /tmp/fid-stats-constained/
CUDA_VISIBLE_DEVICES=0 python evaluate.py --checkpoint $CHECKPOINT_DIR/eval-$EXPR_ID/checkpoint.pt --data $DATA_DIR/cifar10 --eval_mode=evaluate_fid  --fid_dir /tmp/fid-stats-constained/ --temp=0.6 --readjust_bn