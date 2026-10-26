#!/usr/bin/env bash
# Runs on the GPU node host: create `arl-rpent` (RPent + controller), and one `arl-vllm-<gpu>`
# (port 18100 + its index in VLLM_GPUS) per GPU in VLLM_GPUS. Existing containers are left alone.
# `arl-rpent` sees every GPU: the harness picks one per process with `--cuda-device {gpu}` (CUDA_VISIBLE_DEVICES).
# The paper's serve flags plus --attention-backend TRITON_ATTN: the hybrid model forces a 528-token attention block,
# which the default ROCm backend serves through a slow fallback (decode at 48k context: 17.5 tok/s vs 133 tok/s).
# Usage: VLLM_GPUS=6,7 WEIGHTS=/data/zhengjli/arl/ckpt/rl-v006 bash node_up.sh
set -euo pipefail

VLLM_GPUS=${VLLM_GPUS:-}
WEIGHTS=${WEIGHTS:-/data/zhengjli/arl/ckpt/rl-v006}
RPENT_IMAGE=rlinf/rlinf:agentic-rlinf0.3-libero-rocm6.4
VLLM_IMAGE=vllm/vllm-openai-rocm:v0.32.0
GPU_ARGS=(--device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined)
exists() { docker ps -a --format '{{.Names}}' | grep -qx "$1"; }

exists arl-rpent || docker run -d --name arl-rpent --network host --ipc host --shm-size 32g "${GPU_ARGS[@]}" \
  -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa -e LIBERO_TYPE=pro \
  -v /data:/data "$RPENT_IMAGE" sleep infinity

i=0
for g in ${VLLM_GPUS//,/ }; do
  name=arl-vllm-$g port=$((18100 + i))
  exists "$name" || docker run -d --name "$name" --network host --ipc host --shm-size 16g "${GPU_ARGS[@]}" \
    -e HIP_VISIBLE_DEVICES="$g" -e HF_HOME=/data/zhengjli/arl/hf-cache -v /data:/data \
    --entrypoint vllm "$VLLM_IMAGE" serve "$WEIGHTS" --served-model-name "$WEIGHTS" \
    --host 127.0.0.1 --port "$port" --tensor-parallel-size 1 --max-model-len 262144 --max-num-seqs 4 \
    --gpu-memory-utilization 0.62 --limit-mm-per-prompt.video 0 --limit-mm-per-prompt.image 128 \
    --enable-prefix-caching --enable-auto-tool-choice --generation-config auto \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 --attention-backend TRITON_ATTN
  echo "$name -> 127.0.0.1:$port"
  i=$((i + 1))
done
