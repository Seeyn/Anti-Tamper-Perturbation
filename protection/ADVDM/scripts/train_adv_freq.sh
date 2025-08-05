export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export OUTPUT_DIR=$2
export INSTANCE_DIR=$1

python poison_adv.py \
  --pretrained_model_name_or_path=$MODEL_NAME   \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of a person" \
  --resolution=512 \
  --train_batch_size=1 \
  --poison_scale=8 \
  --poison_step_num=100\
  --input-config $3\
  --seed 1042