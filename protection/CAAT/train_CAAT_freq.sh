export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export OUTPUT_DIR=$2
export INSTANCE_DIR=$1

accelerate launch CAAT.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --seed=1042 \
  --instance_prompt="a photo of a person" \
  --resolution=512 \
  --learning_rate=1e-5 \
  --lr_warmup_steps=0 \
  --max_train_steps=250 \
  --mixed_precision bf16  \
  --alpha=5e-3  \
  --eps=0.05 \
  --input-config $3
