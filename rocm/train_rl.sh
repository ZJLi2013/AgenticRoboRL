#!/usr/bin/env bash
# Runs on the GPU node host: rl/train_rl.py in a detached `arl-train` container of the vLLM image (torch 2.13 ROCm,
# like the upstream TRAIN_ENV) with the venv /data/zhengjli/arl/venv/train (transformers 5.17.0, fla 0.5.2).
# The log ends with "EXIT <code>".
# Usage: [GPUS=0,1,2,3] bash train_rl.sh <log file> <train_rl.py args...>
set -euo pipefail
GPUS=${GPUS:-0,1,2,3}
LOG=$1; shift
ROOT=/data/zhengjli/arl
N=$(( $(tr -cd , <<<"$GPUS" | wc -c) + 1 ))
docker rm -f arl-train >/dev/null 2>&1 || true
docker run -d --name arl-train --network host --ipc host --shm-size 64g \
  --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES="$GPUS" -e HF_HOME=$ROOT/hf-cache -v /data:/data -w $ROOT/src/AgenticRoboRL \
  --entrypoint bash vllm/vllm-openai-rocm:v0.32.0 -c \
  "$ROOT/venv/train/bin/python -m torch.distributed.run --nproc_per_node $N rl/train_rl.py $* > $LOG 2>&1; echo EXIT \$? >> $LOG"
echo "arl-train started on GPUs $GPUS -> $LOG"
