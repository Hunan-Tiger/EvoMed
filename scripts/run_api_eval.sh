#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen3-VL-235B-A22B}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen3-Max}"
API_BASE_URL="${API_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
API_KEY_ALIAS="${API_KEY_ALIAS:-env:DASHSCOPE_API_KEY}"
EXP_NAME="${EXP_NAME:-slake}"
KB_PATH="${KB_PATH:-knowledges/slake_tag_kb_${MODEL_NAME}.pkl}"

python evaluate.py \
  --infer_mode api \
  --en "$EXP_NAME" \
  --at closed \
  --dt 1000 \
  --ct all \
  --model "$MODEL_NAME" \
  --em "$JUDGE_MODEL" \
  --cm "$MODEL_NAME" \
  --bu "$API_BASE_URL" \
  --ak "$API_KEY_ALIAS" \
  --kb "$KB_PATH"
