#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/.env.remote.local" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env.remote.local"
fi

if [[ ! -d ".venv" ]]; then
  echo "Missing .venv in $ROOT_DIR"
  echo "Run:"
  echo "  python3 -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  pip install -U pip"
  echo "  pip install -e '.[serve,eval]'"
  exit 1
fi

source .venv/bin/activate

export OPENAI_API_KEY="${OPENAI_API_KEY:-${OPENROUTER_API_KEY:-}}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}"
export LLAMA2_HF_TOKEN="${LLAMA2_HF_TOKEN:-${HF_TOKEN:-}}"

export HF_HOME="${HF_HOME:-$ROOT_DIR/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

CONFIG_PATH="${CONFIG_PATH:-config.qwen_openrouter_proxy.yaml}"
PARALLEL="${PARALLEL:-8}"

STRONG_MODEL="${STRONG_MODEL:-openrouter/qwen/qwen-2.5-72b-instruct}"
WEAK_MODEL="${WEAK_MODEL:-openrouter/qwen/qwen-2.5-7b-instruct}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/qwen25_remote_all}"
GSM8K_OUTPUT="${GSM8K_OUTPUT:-$OUTPUT_ROOT/gsm8k_responses.csv}"
MMLU_OUTPUT_DIR="${MMLU_OUTPUT_DIR:-$OUTPUT_ROOT/mmlu_responses}"
GSM8K_EVAL_OUTPUT="${GSM8K_EVAL_OUTPUT:-$OUTPUT_ROOT/gsm8k_eval}"
MMLU_EVAL_OUTPUT="${MMLU_EVAL_OUTPUT:-$OUTPUT_ROOT/mmlu_eval}"

ROUTERS="${ROUTERS:-causal_llm mf sw_ranking}"
BENCHMARKS="${BENCHMARKS:-gsm8k mmlu}"

read -r -a ROUTER_ARRAY <<<"$ROUTERS"
read -r -a BENCHMARK_ARRAY <<<"$BENCHMARKS"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing config: $CONFIG_PATH"
  exit 1
fi

if [[ " $ROUTERS " == *" mf "* || " $ROUTERS " == *" sw_ranking "* ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY or OPENROUTER_API_KEY is required for mf/sw_ranking embeddings."
    exit 1
  fi
fi

mkdir -p "$GSM8K_EVAL_OUTPUT" "$MMLU_EVAL_OUTPUT"

echo "Resuming router eval with:"
echo "  ROUTERS=$ROUTERS"
echo "  BENCHMARKS=$BENCHMARKS"
echo "  ROUTELLM_DEVICE=${ROUTELLM_DEVICE:-auto}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"

for benchmark in "${BENCHMARK_ARRAY[@]}"; do
  case "$benchmark" in
    gsm8k)
      if [[ ! -f "$GSM8K_OUTPUT" ]]; then
        echo "Missing GSM8K responses: $GSM8K_OUTPUT"
        exit 1
      fi
      python -m routellm.evals.evaluate \
        --benchmark gsm8k \
        --routers "${ROUTER_ARRAY[@]}" \
        --config "$CONFIG_PATH" \
        --parallel "$PARALLEL" \
        --strong-model "$STRONG_MODEL" \
        --weak-model "$WEAK_MODEL" \
        --gsm8k-responses "$GSM8K_OUTPUT" \
        --output "$GSM8K_EVAL_OUTPUT" \
        --plot-optimal
      ;;
    mmlu)
      if [[ ! -d "$MMLU_OUTPUT_DIR" ]]; then
        echo "Missing MMLU responses dir: $MMLU_OUTPUT_DIR"
        exit 1
      fi
      if ! find "$MMLU_OUTPUT_DIR" -maxdepth 1 -name 'mmlu_*.csv' | read -r _; then
        echo "No MMLU response CSVs found in: $MMLU_OUTPUT_DIR"
        exit 1
      fi
      python -m routellm.evals.evaluate \
        --benchmark mmlu \
        --routers "${ROUTER_ARRAY[@]}" \
        --config "$CONFIG_PATH" \
        --parallel "$PARALLEL" \
        --strong-model "$STRONG_MODEL" \
        --weak-model "$WEAK_MODEL" \
        --mmlu-responses-dir "$MMLU_OUTPUT_DIR" \
        --output "$MMLU_EVAL_OUTPUT" \
        --plot-optimal
      ;;
    *)
      echo "Unsupported benchmark: $benchmark"
      exit 1
      ;;
  esac
done
