#!/usr/bin/env bash
set -euo pipefail

uv venv
source .venv/bin/activate

git clone https://github.com/ShishirPatil/gorilla.git || true
cd gorilla/berkeley-function-call-leaderboard

uv pip install -e .
uv pip install -e ".[oss_eval_vllm]"

cp -n bfcl_eval/.env.example .env || true

CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve Qwen/Qwen3-8B \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto
