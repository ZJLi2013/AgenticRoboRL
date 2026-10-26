#!/usr/bin/env bash
# Inside arl-rpent: one local_ctl eval controller. Usage: [TASKS=<file in harnesses/rpent_libero_mi300>] eval_in.sh <run name> <shard I/N> <gpu> <model_port> <slot>
set -euo pipefail
NAME=$1 SHARD=$2 GPU=$3 MODEL_PORT=$4 SLOT=$5
ROOT=/data/zhengjli/arl
export RPENT_DIR=$ROOT/src/RPent RPENT_ENV=$ROOT/venv/rpent
export PI05_CHECKPOINT=/data/zhengjli/rlinf/ckpt/RLinf-Pi05-LIBERO-130-fullshot-SFT SAM3_CHECKPOINT=$ROOT/ckpt/sam3/sam3.pt
export HF_HOME=$ROOT/hf-cache LIBERO_CONFIG_PATH=$ROOT/libero-config LIBERO_TYPE=pro MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
cd $ROOT/src/AgenticRoboRL
HDIR=$PWD/harnesses/rpent_libero_mi300
RUN=$ROOT/runs/t2/$NAME
mkdir -p $RUN
sed "s|^tasks: .*|tasks: $HDIR/${TASKS:-tasks_t2.jsonl}|" $HDIR/harness.yaml > $RUN/harness.yaml
exec $RPENT_ENV/bin/python rl/local_ctl.py eval --config $RUN/harness.yaml \
  --run-dir $RUN --weights $ROOT/ckpt/rl-v006 --tag rl_v006 --init-key eval_init_ids --shard "$SHARD" \
  --var gpu=$GPU --var model_port=$MODEL_PORT --var proxy_port=$((18200 + SLOT)) \
  --var vla_port=$((18300 + SLOT)) --var sam3_port=$((18400 + SLOT))
