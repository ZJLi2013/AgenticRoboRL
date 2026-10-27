#!/usr/bin/env bash
# Inside arl-rpent: rollouts + build of RL rounds (local_ctl train --no-train); the update runs in rocm/train_rl.sh,
# whose torch stack lives in the vLLM image. Resumable: rerun with the same run name.
# Usage: [TASKS=<file in harnesses/rpent_libero_mi300>] train_in.sh <run name> <weights> <groups> <gpu> <model_port> <slot>
set -euo pipefail
NAME=$1 WEIGHTS=$2 NGROUPS=$3 GPU=$4 MODEL_PORT=$5 SLOT=$6
ROOT=/data/zhengjli/arl
export RPENT_DIR=$ROOT/src/RPent RPENT_ENV=$ROOT/venv/rpent
export PI05_CHECKPOINT=/data/zhengjli/rlinf/ckpt/RLinf-Pi05-LIBERO-130-fullshot-SFT SAM3_CHECKPOINT=$ROOT/ckpt/sam3/sam3.pt
export HF_HOME=$ROOT/hf-cache LIBERO_CONFIG_PATH=$ROOT/libero-config LIBERO_TYPE=pro MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
cd $ROOT/src/AgenticRoboRL
HDIR=$PWD/harnesses/rpent_libero_mi300
RUN=$ROOT/runs/train/$NAME
mkdir -p $RUN
sed -e "s|^tasks: .*|tasks: $HDIR/${TASKS:-tasks_t9.jsonl}|" $HDIR/harness.yaml > $RUN/harness.yaml
exec $RPENT_ENV/bin/python rl/local_ctl.py train --config $RUN/harness.yaml --run-dir $RUN \
  --weights "$WEIGHTS" --rounds 1 --groups "$NGROUPS" --no-train \
  --var gpu=$GPU --var model_port=$MODEL_PORT --var proxy_port=$((18200 + SLOT)) \
  --var vla_port=$((18300 + SLOT)) --var sam3_port=$((18400 + SLOT))
