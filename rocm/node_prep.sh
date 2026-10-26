#!/usr/bin/env bash
# srdc-18: create /data/zhengjli/arl (needs a root container) and download a checkpoint subfolder.
# usage: bash node_prep.sh [subfolder ...]   (default: rl-v006)
set -euo pipefail
ROOT=/data/zhengjli/arl
IMG=vllm/vllm-openai-rocm:v0.32.0
SUBS=("${@:-rl-v006}")

docker run --rm -v /data/zhengjli:/w --entrypoint bash "$IMG" -c \
  "mkdir -p /w/arl/{src,ckpt,venv,runs,logs,hf-cache} && chown -R $(id -u):$(id -g) /w/arl"

for s in "${SUBS[@]}"; do
  name="arl-dl-${s//[^a-z0-9]/-}"
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker run -d --name "$name" -v "$ROOT":/arl -e HF_HOME=/arl/hf-cache -e HF_HUB_ENABLE_HF_TRANSFER=0 \
    --entrypoint bash "$IMG" -c \
    "hf download ciciciciliu/AgenticRoboRL-Qwen3.5-9B --include '$s/*' --local-dir /arl/ckpt > /arl/logs/dl-$s.log 2>&1; echo EXIT \$? >> /arl/logs/dl-$s.log"
  echo "started $name"
done
