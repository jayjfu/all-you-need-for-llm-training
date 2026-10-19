#!/bin/bash
set -e

MODEL="vllm/Qwen/Qwen2.5-0.5B-Instruct"
LIMIT="" #100
MAX_TOKENS=512
TEMP=0.0

echo "=== Running ARC-Challenge ==="
inspect eval inspect_evals/arc_challenge \
  --model $MODEL \
  --max-tokens 64 \
  --temperature $TEMP \
  ${LIMIT:+--limit $LIMIT}

echo "=== Running MMLU ==="
inspect eval inspect_evals/mmlu_0_shot \
  --model $MODEL \
  --max-tokens 128 \
  --temperature $TEMP \
  ${LIMIT:+--limit $LIMIT}

echo "=== Running GSM8K ==="
inspect eval inspect_evals/gsm8k \
  --model $MODEL \
  --max-tokens $MAX_TOKENS \
  --temperature $TEMP \
  ${LIMIT:+--limit $LIMIT}

echo "=== Running HumanEval ==="
inspect eval inspect_evals/humaneval \
  --model $MODEL \
  --max-tokens 512 \
  --temperature $TEMP \
  ${LIMIT:+--limit $LIMIT}

echo "=== Done. View results with: inspect view ==="