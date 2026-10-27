#!/bin/bash
# Fetch SDXL Inpainting + the fp16-fixed VAE with resumable curl (huggingface_hub's downloader stalled in testing).
set -e
cd "$(dirname "$0")"
fetch() {  # repo file
  mkdir -p "models/$1/$(dirname "$2")"
  curl -fL --retry 10 --retry-delay 5 --retry-all-errors -C - -s -S -o "models/$1/$2" "https://huggingface.co/$1/resolve/main/$2"
  echo "ok $1/$2"
}
R=diffusers/stable-diffusion-xl-1.0-inpainting-0.1
for f in model_index.json scheduler/scheduler_config.json text_encoder/config.json text_encoder_2/config.json \
         tokenizer/merges.txt tokenizer/special_tokens_map.json tokenizer/tokenizer_config.json tokenizer/vocab.json \
         tokenizer_2/merges.txt tokenizer_2/special_tokens_map.json tokenizer_2/tokenizer_config.json tokenizer_2/vocab.json \
         unet/config.json vae/config.json \
         text_encoder/model.fp16.safetensors text_encoder_2/model.fp16.safetensors unet/diffusion_pytorch_model.fp16.safetensors; do
  fetch $R $f
done
for f in config.json diffusion_pytorch_model.safetensors; do fetch madebyollin/sdxl-vae-fp16-fix $f; done
