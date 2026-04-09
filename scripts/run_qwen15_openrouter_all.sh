#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/.env.remote.local" ]]; then
  # Load local machine/server secrets without committing them to git.
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
  echo "  pip install pybase64 protobuf"
  exit 1
fi

source .venv/bin/activate

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is not set."
  echo "Example:"
  echo "  export OPENROUTER_API_KEY='your_openrouter_key'"
  exit 1
fi

export OPENAI_API_KEY="${OPENAI_API_KEY:-$OPENROUTER_API_KEY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}"
export OR_SITE_URL="${OR_SITE_URL:-http://localhost}"
export OR_APP_NAME="${OR_APP_NAME:-RouteLLM-Remote}"
export LLAMA2_HF_TOKEN="${LLAMA2_HF_TOKEN:-${HF_TOKEN:-}}"

# Keep all HF artifacts writable and local to the project.
export HF_HOME="${HF_HOME:-$ROOT_DIR/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

if [[ -z "${LLAMA2_HF_TOKEN:-}" ]]; then
  echo "Warning: LLAMA2_HF_TOKEN/HF_TOKEN is not set."
  echo "The causal_llm router requires gated access to meta-llama/Meta-Llama-3-8B."
fi

HOST="${HOST:-http://127.0.0.1}"
PORT="${PORT:-6060}"
PARALLEL="${PARALLEL:-8}"
CONFIG_PATH="${CONFIG_PATH:-config.qwen15_openrouter.yaml}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-1800}"

STRONG_MODEL="${STRONG_MODEL:-openrouter/qwen/qwen-72b-chat}"
WEAK_MODEL="${WEAK_MODEL:-openrouter/qwen/qwen-7b-chat}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/qwen15_remote_all}"
GSM8K_OUTPUT="$OUTPUT_ROOT/gsm8k_responses.csv"
MMLU_OUTPUT_DIR="$OUTPUT_ROOT/mmlu_responses"
GSM8K_EVAL_OUTPUT="$OUTPUT_ROOT/gsm8k_eval"
MMLU_EVAL_OUTPUT="$OUTPUT_ROOT/mmlu_eval"

mkdir -p "$OUTPUT_ROOT" "$MMLU_OUTPUT_DIR" "$GSM8K_EVAL_OUTPUT" "$MMLU_EVAL_OUTPUT"

SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[1/5] Starting RouteLLM server on ${HOST}:${PORT}"
python -m routellm.openai_server \
  --routers random bert causal_llm mf sw_ranking \
  --config "$CONFIG_PATH" \
  --base-url https://openrouter.ai/api/v1 \
  --api-key "$OPENROUTER_API_KEY" \
  --strong-model "$STRONG_MODEL" \
  --weak-model "$WEAK_MODEL" \
  --port "$PORT" \
  >"$OUTPUT_ROOT/server.log" 2>&1 &
SERVER_PID=$!

echo "[2/5] Waiting for server health check"
for _ in $(seq 1 "$STARTUP_TIMEOUT_SECONDS"); do
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "Server process exited before health check passed. Tail of log:"
    tail -n 100 "$OUTPUT_ROOT/server.log" || true
    exit 1
  fi
  if curl -fsS "${HOST}:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -fsS "${HOST}:${PORT}/health" >/dev/null 2>&1; then
  echo "Server failed to start within ${STARTUP_TIMEOUT_SECONDS}s. Tail of log:"
  tail -n 50 "$OUTPUT_ROOT/server.log" || true
  exit 1
fi

echo "[3/5] Generating GSM8K responses"
python -m routellm.evals.gsm8k.generate_responses \
  --host "$HOST" \
  --port "$PORT" \
  --parallel "$PARALLEL" \
  --strong-model "$STRONG_MODEL" \
  --weak-model "$WEAK_MODEL" \
  --output-file "$GSM8K_OUTPUT"

echo "[4/5] Generating MMLU responses"
python -m routellm.evals.mmlu.generate_responses \
  --host "$HOST" \
  --port "$PORT" \
  --parallel "$PARALLEL" \
  --strong-model "$STRONG_MODEL" \
  --weak-model "$WEAK_MODEL" \
  --output-dir "$MMLU_OUTPUT_DIR"

echo "[5/5] Evaluating all five routers"
python -m routellm.evals.evaluate \
  --benchmark gsm8k \
  --routers random bert causal_llm mf sw_ranking \
  --config "$CONFIG_PATH" \
  --strong-model "$STRONG_MODEL" \
  --weak-model "$WEAK_MODEL" \
  --gsm8k-responses "$GSM8K_OUTPUT" \
  --output "$GSM8K_EVAL_OUTPUT" \
  --plot-optimal \
  --overwrite-cache random bert causal_llm mf sw_ranking

python -m routellm.evals.evaluate \
  --benchmark mmlu \
  --routers random bert causal_llm mf sw_ranking \
  --config "$CONFIG_PATH" \
  --strong-model "$STRONG_MODEL" \
  --weak-model "$WEAK_MODEL" \
  --mmlu-responses-dir "$MMLU_OUTPUT_DIR" \
  --output "$MMLU_EVAL_OUTPUT" \
  --plot-optimal \
  --overwrite-cache random bert causal_llm mf sw_ranking

echo
echo "Finished."
echo "Artifacts:"
echo "  Server log:       $OUTPUT_ROOT/server.log"
echo "  GSM8K responses:  $GSM8K_OUTPUT"
echo "  MMLU responses:   $MMLU_OUTPUT_DIR"
echo "  GSM8K plots:      $GSM8K_EVAL_OUTPUT"
echo "  MMLU plots:       $MMLU_EVAL_OUTPUT"
