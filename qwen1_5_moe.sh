#!/usr/bin/env bash
set -euo pipefail

# ---- Hygiene Battery (Margin-first) ----
export NOESIS_EXPERIMENT_ID="Qwen1_5_MoE"
export NOESIS_PROMPT_FILE="./prompts/sample_prompts.json"

export NOESIS_MAX_NEW_TOKENS=64

TRACE_ROOT="./traces"
METRICS_ROOT="./metrics"

export NOESIS_USE_ATTENTIONS=1

# First pass: run once everywhere
K1=1
# Second pass: if unstable prompts exist, run up to K2 total reps
K2=3

BASE_SEED=40001

# Models
NOESIS_MODEL_NAME="Qwen/Qwen1.5-MoE-A2.7B"
MODELS=(
   "Qwen/Qwen1.5-MoE-A2.7B"
)

# Token windows (fast + clean for margin sanity)
WINDOWS=(64 128)

# Conditions
COND_NAMES=("greedy" "t03" "t08" "t11")

# Thresholds for selecting "unstable" prompts from traces:
# We promote any prompt that yielded margin_bucket == boundary or indeterminate (formally 'collapsed'.)
UNSTABLE_BUCKETS=("boundary" "indeterminate")

model_tag_of () {
  echo "$1" | tr '/:' '__'
}

set_cond_env () {
  local cond="$1"

  # Defaults
  export NOESIS_DO_SAMPLE=0
  export NOESIS_TEMPERATURE=0
  export NOESIS_TOP_P=1.0
  export NOESIS_REPETITION_PENALTY=1.0  # harmless if ignored

  case "$cond" in
    greedy)
      export NOESIS_DO_SAMPLE=0
      export NOESIS_TEMPERATURE=0
      export NOESIS_TOP_P=1.0
      ;;
    t03)
      export NOESIS_DO_SAMPLE=1
      export NOESIS_TEMPERATURE=0.3
      export NOESIS_TOP_P=0.95
      ;;
    t08)
      export NOESIS_DO_SAMPLE=1
      export NOESIS_TEMPERATURE=0.8
      export NOESIS_TOP_P=0.95
      ;;
    t11)
      export NOESIS_DO_SAMPLE=1
      export NOESIS_TEMPERATURE=1.1
      export NOESIS_TOP_P=0.95
      ;;
    *)
      echo "ERROR: unknown cond=$cond" >&2
      exit 1
      ;;
  esac
}

run_one () {
  local model="$1"
  local max_new_tokens="$2"
  local rep="$3"
  local cond="$4"

  export NOESIS_SEED=$((BASE_SEED + (rep * 10000)))

  set_cond_env "$cond"

  local model_tag
  model_tag="$(model_tag_of "$model")"

  export NOESIS_TRACE_DIR="${TRACE_ROOT}/${NOESIS_EXPERIMENT_ID}/${model_tag}/w${max_new_tokens}/${cond}/rep_${rep}"
  export NOESIS_METRICS_DIR="${METRICS_ROOT}/${NOESIS_EXPERIMENT_ID}/${model_tag}/w${max_new_tokens}/${cond}/rep_${rep}"
  mkdir -p "$NOESIS_TRACE_DIR" "$NOESIS_METRICS_DIR"

  # Clean if re-running this rep
  rm -f "${NOESIS_TRACE_DIR}"/*.json 2>/dev/null || true
  rm -f "${NOESIS_METRICS_DIR}"/*.json 2>/dev/null || true

  echo "== model=${model} w=${max_new_tokens} cond=${cond} rep=${rep} seed=${NOESIS_SEED} temp=${NOESIS_TEMPERATURE} top_p=${NOESIS_TOP_P} do_sample=${NOESIS_DO_SAMPLE} =="

  # Run with per-invocation overrides (correct)
  NOESIS_MODEL_NAME="$model" \
  NOESIS_MAX_NEW_TOKENS="$max_new_tokens" \
  python noesis_current.py

  local nfiles
  nfiles=$(ls -1 "${NOESIS_TRACE_DIR}"/*.json 2>/dev/null | wc -l | tr -d " ")
  echo "Trace files: ${nfiles}"
  exit 1
  if [ "$nfiles" -eq 0 ]; then
    echo "ERROR: No traces produced in ${NOESIS_TRACE_DIR}" >&2
    exit 1
  fi
}

# Extract unstable prompt_ids (or prompt hashes) from produced trace JSON.
# We attempt multiple common fields to avoid coupling:
# - trace["run"]["prompt_id"]
# - trace["run"]["prompt"]["id"]
# - trace["prompt_id"]
# - fallback: trace["run"]["prompt_text_hash"] if present
#
# If none exist, we fallback to filtering by prompt *text* (less ideal but works).
collect_unstable_prompts () {
  local root_dir="$1"
  local out_file="$2"

  rm -f "$out_file" 2>/dev/null || true

  # Use python (stdlib only) to scan JSON; avoid jq dependency.
  python - "$root_dir" "$out_file" <<'PY'
import json, sys, glob, os

root_dir = sys.argv[1]
out_file = sys.argv[2]

unstable = set()
unstable_texts = set()

def get(d, path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

for fp in glob.glob(os.path.join(root_dir, "**", "*.json"), recursive=True):
    try:
        with open(fp, "r", encoding="utf-8") as f:
            j = json.load(f)
    except Exception:
        continue

    # stability_v0 can live under extras.noesis.cognitive_regime.stability_v0 (as in current code)
    bucket = get(j, ["extras","noesis","cognitive_regime","stability_v0","margin_bucket"])
    if bucket not in ("boundary", "indeterminate"):
        continue

    # Try prompt identifiers
    pid = (
        get(j, ["run","prompt_id"]) or
        get(j, ["run","prompt","id"]) or
        get(j, ["prompt_id"]) or
        get(j, ["run","prompt_text_hash"]) or
        get(j, ["run","prompt","hash"])
    )

    if pid:
        unstable.add(str(pid))
    else:
        # Fallback: store prompt text (works, but a bit brittle)
        ptxt = (
            get(j, ["run","prompt_text"]) or
            get(j, ["run","prompt","text"]) or
            get(j, ["prompt"]) or
            get(j, ["input_prompt"])
        )
        if isinstance(ptxt, str) and ptxt.strip():
            unstable_texts.add(ptxt.strip())

with open(out_file, "w", encoding="utf-8") as f:
    for pid in sorted(unstable):
        f.write(pid + "\n")
    # Separator for readability
    if unstable_texts:
        f.write("\n# -- fallback_prompt_texts --\n")
        for t in sorted(unstable_texts):
            f.write(t.replace("\n", "\\n") + "\n")

print(f"unstable_ids={len(unstable)} unstable_texts={len(unstable_texts)}")
PY
}

# Second pass runner:
# If we found unstable prompt IDs or texts, rerun the suite but ONLY for those prompts.
# This requires noesis_tension_current.py to support filtering prompts via env var:
# - NOESIS_PROMPT_FILTER_IDS (newline-separated or comma-separated)
# - OR NOESIS_PROMPT_FILTER_TEXTS (newline-separated)
#
# If your current runner does not support filtering, we fall back to simply re-running full suite for reps 1..K2-1
# (still acceptable because suite is small).
supports_prompt_filtering () {
  # Heuristic: if the code reads NOESIS_PROMPT_FILTER_IDS / TEXTS, it supports it.
  # We won't introspect file here; assume "no" unless updated.
  return 1
}

echo "=== PASS 1: K=${K1} across full suite ==="
for model in "${MODELS[@]}"; do
  for w in "${WINDOWS[@]}"; do
    for cond in "${COND_NAMES[@]}"; do
      for rep in $(seq 0 $((K1-1))); do
        run_one "$model" "$w" "$rep" "$cond"
      done
    done
  done
done

# Collect unstable prompts from traces
UNSTABLE_LIST="./.unstable_prompts_${NOESIS_EXPERIMENT_ID}.txt"
collect_unstable_prompts "${TRACE_ROOT}/${NOESIS_EXPERIMENT_ID}" "$UNSTABLE_LIST"

echo "=== PASS 2: if any boundary/indeterminate, run additional reps up to K=${K2} ==="
if grep -qvE '^\s*(#|$)' "$UNSTABLE_LIST"; then
  echo "Found unstable prompts (see ${UNSTABLE_LIST})."
  if supports_prompt_filtering; then
    echo "Prompt filtering supported: rerunning only unstable prompts for reps 1..$((K2-1))"
    export NOESIS_PROMPT_FILTER_IDS="$UNSTABLE_LIST"
    for model in "${MODELS[@]}"; do
      for w in "${WINDOWS[@]}"; do
        for cond in "${COND_NAMES[@]}"; do
          for rep in $(seq 1 $((K2-1))); do
            run_one "$model" "$w" "$rep" "$cond"
          done
        done
      done
    done
    unset NOESIS_PROMPT_FILTER_IDS
  else
    echo "Prompt filtering not detected/enabled in runner; rerunning full suite for reps 1..$((K2-1)) (suite is small)."
    for model in "${MODELS[@]}"; do
      for w in "${WINDOWS[@]}"; do
        for cond in "${COND_NAMES[@]}"; do
          for rep in $(seq 1 $((K2-1))); do
            run_one "$model" "$w" "$rep" "$cond"
          done
        done
      done
    done
  fi
else
  echo "No boundary/indeterminate margins detected in PASS 1; skipping PASS 2."
fi

echo "HYGIENE complete: ${NOESIS_EXPERIMENT_ID}"
