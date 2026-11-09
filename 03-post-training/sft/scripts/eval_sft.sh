# Configuration
BASE_MODEL="Qwen/Qwen3.5-0.8B-Base"
INSTRUCT_MODEL="Qwen/Qwen3.5-0.8B"
CKPTS_ROOT="../ckpts"
RESULTS_ROOT="../results"
# TASKS="leaderboard"
TASKS=(
  leaderboard_ifeval
  leaderboard_gpqa
  leaderboard_mmlu_pro
  leaderboard_musr
  leaderboard_bbh
  leaderboard_math_hard
)
BATCH_SIZE="auto"


mkdir -p "${RESULTS_ROOT}"

# Evaluation function
eval_model() {
  local model_path="$1"
  local model_name="$2"
  local apply_chat="${3:-true}"
  local task="$4"
  local out_dir="${RESULTS_ROOT}/${model_name}/${task}"

  mkdir -p "${out_dir}"

  local cmd=(
    lm-eval run \
    --model vllm \
    --model_args "pretrained=${model_path},dtype=auto" \
    --tasks "${task}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${out_dir}"
  )

  if [[ "${apply_chat}" == "true" ]]; then
    cmd+=(--apply_chat_template)
  fi

  "${cmd[@]}"
}

# Helper
eval_all_tasks() {
  local model_path="$1"
  local model_name="$2"
  local apply_chat="${3:-true}"

  for task in "${TASKS[@]}"; do
    eval_model "${model_path}" "${model_name}" "${apply_chat}" "${task}"
  done
}

# 1. Base model
eval_all_tasks "${BASE_MODEL}" "base" "false"

# 2. Official Instruct model
eval_all_tasks "${INSTRUCT_MODEL}" "instruct" "true"

# 3. Our SFT variants
if [[ -d "${CKPTS_ROOT}" ]]; then
  for sft_dir in "${CKPTS_ROOT}"/*/; do
    [[ -d "${sft_dir}" ]] || continue # skip if not a directory

    name=$(basename "${sft_dir}")
    eval_all_tasks "${sft_dir}" "sft_${name}" "true"
  done
fi
