# Exit immediately if a command exits with a non-zero status
set -e

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:/root/mask:/root/mask/slim

CURRENT_DIR=$(pwd)

TFRECORD_DIR="$CURRENT_DIR/tfrecords/quadrilateral_2"
DATASET_SPLIT="val"
MODEL_VARIANT="resnet_v1_50_beta"
CHECKPOINT_DIR="$CURRENT_DIR/exp/quadrilateral_2_resnet_v1_50_beta_01/train"
EVAL_DIR="$CURRENT_DIR/exp/quadrilateral_2_resnet_v1_50_beta_01/eval"

python "$CURRENT_DIR"/regression_eval.py \
  --tfrecord_dir=$TFRECORD_DIR \
  --dataset_split=$DATASET_SPLIT \
  --model_variant=$MODEL_VARIANT \
  --checkpoint_dir=$CHECKPOINT_DIR \
  --eval_dir=$EVAL_DIR \
  --batch_size=64 \
  --is_training=False \
  --eval_interval_secs=180
