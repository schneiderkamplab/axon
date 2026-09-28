#!/usr/bin/env bash
set -euo pipefail

# Overnight ablation run: T1 on dense models + T1+T3 on MoE models.
# Usage: scripts/run_ablation_overnight.sh [REPO_ROOT]
# Output: log/ablation-overnight/<model>/ per-model, log/ablation-overnight/combined.csv merged.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${1:-$(cd "$SCRIPT_DIR/.." && pwd)}"
LOG_DIR="$REPO_ROOT/log/ablation-overnight"
BACKEND="codegen2-torch"
DEVICE="cuda"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Model definitions: axon_file|model_dir_key|tier
#   axon_file      = path under synapse/models/
#   model_dir_key  = key in MODEL_SPECS (for download) + local dir under models/
#   tier           = t1-stages or t3-per-intrinsic
# ---------------------------------------------------------------------------

DENSE_MODELS=(
    "gpt2/gpt2.axon|gpt2|t1-stages"
    "bert/bert-base-uncased.axon|bert|t1-stages"
    "smollm/SmolLM-135M.axon|smollm_135m|t1-stages"
    "smollm/SmolLM2-135M.axon|smollm2_135m|t1-stages"
    "bert/distilbert-base-uncased.axon|distilbert|t1-stages"
    "bert/albert-base-v2.axon|albert|t1-stages"
    "t5/t5-small.axon|t5_small|t1-stages"
    "qwen2/Qwen2.5-0.5B.axon|qwen2_5_0_5b|t1-stages"
    "gemma3/gemma-3-270m.axon|gemma3|t1-stages"
    "roberta/roberta-base.axon|roberta|t1-stages"
    "falcon/falcon_rw_1b.axon|falcon_rw_1b|t1-stages"
    "llama3/Llama-3.2-1B.axon|llama3_2_1b|t1-stages"
    "smollm3/SmolLM3-3B.axon|smollm3_3b|t1-stages"
    "olmo2/OLMo-2-0425-1B.axon|olmo_2_1b|t1-stages"
    "gemma3/gemma-3-1b.axon|gemma3_1b|t1-stages"
)

MOE_MODELS=(
    "olmoe/OLMoE-1B-7B-0924.axon|olmoe_1b_7b_0924|t1-stages"
    "olmoe/OLMoE-1B-7B-0924.axon|olmoe_1b_7b_0924|t3-per-intrinsic"
    "deepseekv2/DeepSeek-V2-Lite.axon|deepseek_v2_lite|t1-stages"
    "deepseekv2/DeepSeek-V2-Lite.axon|deepseek_v2_lite|t3-per-intrinsic"
    "gpt-oss/gpt-oss-20b.axon|gpt_oss_20b|t1-stages"
    "gpt-oss/gpt-oss-20b.axon|gpt_oss_20b|t3-per-intrinsic"
)

ALL_MODELS=("${DENSE_MODELS[@]}" "${MOE_MODELS[@]}")

# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

download_model() {
    local key="$1"
    python -c "
from pathlib import Path
from synapse.matrix_models import MODEL_SPECS, ensure_model_downloaded
repo = Path('$REPO_ROOT')
spec = MODEL_SPECS['$key']
model_dir = ensure_model_downloaded(repo_root=repo, spec=spec, status_cb=print)
print(f'MODEL_DIR={model_dir}')
"
}

# ---------------------------------------------------------------------------
# Run ablation for one model
# ---------------------------------------------------------------------------

run_ablation() {
    local axon_rel="$1"
    local model_key="$2"
    local tier="$3"
    local axon_path="$REPO_ROOT/synapse/models/$axon_rel"
    local model_dir="$REPO_ROOT/models/$model_key"
    local out_dir="$LOG_DIR/${axon_rel%.axon}_$tier"

    if [[ -f "$out_dir/ablation.csv" ]]; then
        echo "[skip] $axon_rel ($tier) — already done"
        return 0
    fi

    if [[ ! -d "$model_dir" ]]; then
        echo "[download] $model_key ..."
        if ! download_model "$model_key" 2>&1; then
            echo "[error] download failed for $model_key, skipping"
            return 1
        fi
    fi

    # Handle models that map to a different local dir name
    # (the download function may use a spec.local_dir different from model_key)
    if [[ ! -d "$model_dir" ]]; then
        local resolved_dir
        resolved_dir=$(python -c "
from pathlib import Path
from synapse.matrix_models import MODEL_SPECS
spec = MODEL_SPECS.get('$model_key')
if spec:
    print(Path('$REPO_ROOT') / 'models' / spec.local_dir)
else:
    print('$model_dir')
")
        model_dir="$resolved_dir"
    fi

    echo "[ablation] $axon_rel | $model_dir | tier=$tier"
    if ! synapse axon-ablation \
        "$axon_path" "$model_dir" \
        --backend "$BACKEND" \
        --device "$DEVICE" \
        --tier "$tier" \
        --output-dir "$out_dir" \
        --forward-warmup 2 \
        --forward-repeat 10 \
        2>&1; then
        echo "[error] ablation failed for $axon_rel ($tier), continuing..."
        return 1
    fi
    echo "[done] $axon_rel ($tier) -> $out_dir"
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

echo "=== Overnight Ablation Run ==="
echo "repo:   $REPO_ROOT"
echo "log:    $LOG_DIR"
echo "backend: $BACKEND"
echo "device:  $DEVICE"
echo "models:  ${#ALL_MODELS[@]} entries"
echo ""

FAILED=()
DONE=0
TOTAL=${#ALL_MODELS[@]}

for entry in "${ALL_MODELS[@]}"; do
    IFS='|' read -r axon_rel model_key tier <<< "$entry"
    DONE=$((DONE + 1))
    echo ">>> ($DONE/$TOTAL) $axon_rel | $model_key | $tier"
    if ! run_ablation "$axon_rel" "$model_key" "$tier"; then
        FAILED+=("$axon_rel|$tier")
    fi
    echo ""
done

# ---------------------------------------------------------------------------
# Merge results
# ---------------------------------------------------------------------------

echo "=== Merging results ==="
python "$SCRIPT_DIR/merge_ablation_results.py" "$LOG_DIR" 2>&1

echo ""
echo "=== Summary ==="
echo "completed: $((TOTAL - ${#FAILED[@]}))/$TOTAL"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "failed:"
    for f in "${FAILED[@]}"; do
        echo "  - $f"
    done
fi
echo "results: $LOG_DIR/combined.csv"
echo "plots:   $LOG_DIR/cross_model_nodes.svg"
echo "         $LOG_DIR/cross_model_speedup.svg"
