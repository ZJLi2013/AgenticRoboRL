#!/usr/bin/env bash
# Runs inside `arl-rpent`: this repo's branch, RPent f70d720 + its patch, ROCm venv, SAM3 weights. Idempotent.
set -euo pipefail

ROOT=/data/zhengjli/arl
REV=f70d720
ARL_BRANCH=${ARL_BRANCH:-rocm-mi300}
PATCH_SHA=74380a09251aa42f516457b1f410415d7d7cd1aee84d661886c164e18b0860b7
TREE=11c44fcdedab126962be13ed3f477e0212932323
export UV_CACHE_DIR=$ROOT/uv-cache UV_DEFAULT_INDEX=https://pypi.org/simple UV_HTTP_TIMEOUT=300
export HF_HOME=$ROOT/hf-cache LIBERO_CONFIG_PATH=$ROOT/libero-config
mkdir -p "$ROOT"/{logs,ckpt/sam3,libero-config,src}
cd "$ROOT/src"

[ -d AgenticRoboRL ] || git clone -b "$ARL_BRANCH" https://github.com/ZJLi2013/AgenticRoboRL.git
git -C AgenticRoboRL pull -q --ff-only origin "$ARL_BRANCH"
PATCH=$ROOT/src/AgenticRoboRL/patches/rpent/rpent_f70d720.patch
echo "$PATCH_SHA  $PATCH" | sha256sum -c -

if [ ! -d RPent ]; then
  git clone https://github.com/RLinf/RPent.git
  git -C RPent checkout -q "$REV"
  git -C RPent apply "$PATCH"
fi
# MI300 deployment changes on top of the paper's tree (OSMesa for the spawned env_server); the tree check
# runs on the paper's tree alone, so this patch is taken off first on a rerun.
MI_PATCH=$ROOT/src/AgenticRoboRL/rocm/rpent_mi300.patch
git -C RPent apply --reverse --check "$MI_PATCH" 2>/dev/null && git -C RPent apply --reverse "$MI_PATCH"
git -C RPent add -A
[ "$(git -C RPent write-tree)" = "$TREE" ] || { echo "RPent tree != $TREE"; exit 1; }
git -C RPent apply "$MI_PATCH"

# The image's ROCm build; without the override the extras resolve CUDA torch from PyPI.
# PyPI `triton` (CUDA) shares the `triton/` directory with pytorch-triton-rocm and segfaults on import.
cd "$ROOT"
printf 'torch==2.8.0+rocm6.4\ntorchvision==0.23.0+rocm6.4\ntriton ; sys_platform == "never"\n' > torch-override.txt
[ -x venv/rpent/bin/python ] || uv venv venv/rpent --python 3.11
PY=$ROOT/venv/rpent/bin/python
install() {
  uv pip install --python "$PY" -e "src/RPent[libero-pro,openpi,sam3,rlinf]" \
    "transformers==4.57.6" "pydantic-ai-slim[anthropic,openai]==2.49.0" requests modelscope \
    --override torch-override.txt \
    --extra-index-url https://download.pytorch.org/whl/rocm6.4 --index-strategy unsafe-best-match "$@"
}
has_rocm_triton() { "$PY" -c "import triton, os, sys; sys.exit('amd' not in os.listdir(os.path.dirname(triton.__file__) + '/backends'))" 2>/dev/null; }
install
if uv pip show --python "$PY" triton >/dev/null 2>&1 || ! has_rocm_triton; then
  uv pip uninstall --python "$PY" triton || true
  install --reinstall-package pytorch-triton-rocm
fi
has_rocm_triton
liberopro_req=$(uv pip freeze --python "$PY" | grep -E '^rpent-liberopro( |==|@)')
uv pip install --python "$PY" --link-mode copy --reinstall-package rpent-liberopro --no-deps "$liberopro_req"

# sam3 0.1.4 ships without the CLIP BPE vocabulary it loads from <site-packages>/assets/.
BPE=$("$PY" -c "import os, sam3; print(os.path.join(os.path.dirname(os.path.dirname(sam3.__file__)), 'assets', 'bpe_simple_vocab_16e6.txt.gz'))")
if [ ! -f "$BPE" ]; then
  mkdir -p "$(dirname "$BPE")"
  curl -fsSL -o "$BPE" https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz
fi
echo "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a  $BPE" | sha256sum -c -

venv/rpent/bin/liberopro-download-assets --skip-existing
[ -f ckpt/sam3/sam3.pt ] || venv/rpent/bin/modelscope download --model facebook/sam3 sam3.pt --local_dir ckpt/sam3

"$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "hip", torch.version.hip, "cuda_available", torch.cuda.is_available(), "devices", torch.cuda.device_count())
import mujoco, sam3, openpi, libero.libero, transformers, pydantic_ai  # noqa: F401
print("mujoco", mujoco.__version__, "transformers", transformers.__version__, "imports ok")
EOF
ls -la ckpt/sam3
echo SETUP_DONE
