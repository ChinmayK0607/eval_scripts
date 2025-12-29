#!/usr/bin/env bash
set -euo pipefail

uv venv
source .venv/bin/activate

git clone https://github.com/ShishirPatil/gorilla.git || true
cd gorilla/berkeley-function-call-leaderboard

uv pip install -e .
uv pip install -e ".[oss_eval_vllm]"

cp bfcl_eval/.env.example .env || true

CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto


  bfcl generate --model Qwen/Qwen2.5-7B-Instruct-FC --test-category single_turn --skip-server-setup


bfcl generate \
--model Qwen/Qwen2.5-7B-Instruct-FC \
--test-category simple_python \
--backend vllm \
--num-gpus 4 \
--gpu-memory-utilization 0.8

bfcl evaluate --model Qwen/Qwen2.5-7B-Instruct-FC --test-category single_turn
