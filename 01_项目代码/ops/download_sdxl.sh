#!/usr/bin/env bash
set -Eeuo pipefail

source /root/r3dguard/activate.sh

mkdir -p /root/r3dguard/models/SDXL-base-1.0 /root/r3dguard/logs

echo "[$(date -Is)] Resuming SDXL FP16 minimal runtime files"
hf download stabilityai/stable-diffusion-xl-base-1.0 \
  model_index.json \
  scheduler/scheduler_config.json \
  text_encoder/config.json \
  text_encoder/model.fp16.safetensors \
  text_encoder_2/config.json \
  text_encoder_2/model.fp16.safetensors \
  tokenizer/merges.txt \
  tokenizer/special_tokens_map.json \
  tokenizer/tokenizer_config.json \
  tokenizer/vocab.json \
  tokenizer_2/merges.txt \
  tokenizer_2/special_tokens_map.json \
  tokenizer_2/tokenizer_config.json \
  tokenizer_2/vocab.json \
  unet/config.json \
  unet/diffusion_pytorch_model.fp16.safetensors \
  vae/config.json \
  vae/diffusion_pytorch_model.fp16.safetensors \
  --local-dir /root/r3dguard/models/SDXL-base-1.0

echo "[$(date -Is)] SDXL FP16 complete"
du -sh /root/r3dguard/models/SDXL-base-1.0
df -h /
