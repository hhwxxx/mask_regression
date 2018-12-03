# Exit immediately if a command exits with a non-zero status
set -e

export CUDA_VISIBLE_DEVICES=0

CURRENT_DIR=$(pwd)

TFRECORD_DIR="$CURRENT_DIR/tfrecords/quadrilateral_multiple"
DATASET_SPLIT="train"
MODEL_VARIANT="resnet_v1_50_beta"
RESTORE_CKPT_PATH="$CURRENT_DIR/init_models/resnet_v1_50/model.ckpt"
TRAIN_DIR="$CURRENT_DIR/exp/quadrilateral_multiple_resnet_v1_50_beta_01/train"
NUM_EPOCHS=250
DECAY_EPOCHS=120

python "$CURRENT_DIR"/regression_train.py \
  --tfrecord_dir=$TFRECORD_DIR \
  --dataset_split=$DATASET_SPLIT \
  --model_variant=$MODEL_VARIANT \
  --restore_ckpt_path=$RESTORE_CKPT_PATH \
  --train_dir=$TRAIN_DIR \
  --batch_size=64 \
  --is_training=True \
  --initial_learning_rate=0.0001 \
  --decay_epochs=$DECAY_EPOCHS \
  --staircase=True \
  --num_epochs=$NUM_EPOCHS \
  --save_checkpoint_steps=500 \
  --log_frequency=10
