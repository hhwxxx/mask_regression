# Exit immediately if a command exits with a non-zero status
set -e

export CUDA_VISIBLE_DEVICES=0

CURRENT_DIR=$(pwd)

TFRECORD_DIR="$CURRENT_DIR/tfrecords/quadrilateral_2"
DATASET_SPLIT="test"
MODEL_VARIANT="resnet_v1_50_beta"
CHECKPOINT_DIR="$CURRENT_DIR/exp/quadrilateral_2_resnet_v1_50_beta_01/train"
VIS_DIR="$CURRENT_DIR/exp/quadrilateral_2_resnet_v1_50_beta_01/vis"

python "$CURRENT_DIR"/regression_vis.py \
  --tfrecord_dir=$TFRECORD_DIR \
  --dataset_split=$DATASET_SPLIT \
  --model_variant=$MODEL_VARIANT \
  --checkpoint_dir=$CHECKPOINT_DIR \
  --vis_dir=$VIS_DIR \
  --batch_size=1 \
  --is_training=False
