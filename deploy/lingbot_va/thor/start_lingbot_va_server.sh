#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT="${LINGBOT_RUNTIME_ROOT:-/home/wayne/lingbot_runtime}"
MODEL_ROOT="${LINGBOT_MODEL_ROOT:-${RUNTIME_ROOT}/models}"
TOKEN_FILE="${LINGBOT_TOKEN_FILE:-/home/wayne/.config/lingbot_remote_token}"
IMAGE="${LINGBOT_IMAGE:-lingbot-va-thor:0.6.0}"

for required in \
  "${MODEL_ROOT}/checkpoint/pretrained_model/config.json" \
  "${MODEL_ROOT}/base/config.json" \
  "${MODEL_ROOT}/wan/vae/config.json" \
  "${MODEL_ROOT}/wan/text_encoder/config.json" \
  "${MODEL_ROOT}/wan/tokenizer/tokenizer_config.json" \
  "${TOKEN_FILE}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required deployment file: ${required}" >&2
    exit 1
  fi
done

docker rm -f lingbot-va-server >/dev/null 2>&1 || true
docker run --detach --name lingbot-va-server \
  --restart unless-stopped \
  --network host \
  --ipc host \
  --runtime nvidia \
  --gpus all \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --mount "type=bind,src=${MODEL_ROOT},dst=/models,readonly" \
  --mount "type=bind,src=${TOKEN_FILE},dst=/run/secrets/lingbot_token,readonly" \
  "${IMAGE}" \
  --model-dir /models/checkpoint/pretrained_model \
  --base-model /models/base \
  --wan-model /models/wan \
  --host 0.0.0.0 \
  --port 8765 \
  --token-file /run/secrets/lingbot_token \
  --device cuda \
  --text-encoder-device cpu \
  --attn-mode torch \
  --action-inference-steps 50 \
  --video-inference-steps 20 \
  --guidance-scale 5.0 \
  --offline

echo "LingBot-VA server started. Follow logs with: docker logs -f lingbot-va-server"
