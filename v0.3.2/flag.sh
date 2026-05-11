#!/bin/bash
# =============================================================================
# Noesis Trace Flagger — flags files that warrant closer scrutiny
# Usage: ./flag_noesis_traces.sh [directory]   (default = current dir)
# =============================================================================

DIR="${1:-.}"

echo "Scanning Noesis traces in: $DIR"
echo "=================================================================="

find "$DIR" -name "*.json" -type f | sort | while read -r file; do
    # Extract key fields
    regime=$(jq -r '.extras.noesis.cognitive_regime.label // "unknown"' "$file" 2>/dev/null)
    tension=$(jq -r '.extras.noesis.cognitive_regime.features.tension // .summary.tension_index_v0_2 // 0' "$file" 2>/dev/null)
    drift=$(jq -r '.extras.noesis.cognitive_regime.features.drift // .summary.drift_index_v0_2 // 0' "$file" 2>/dev/null)
    margin_bucket=$(jq -r '.extras.noesis.cognitive_regime.stability_v0.margin_bucket // "unknown"' "$file" 2>/dev/null)
    spike_ratio=$(jq -r '.summary.delta_spike_ratio_vs_median // 0' "$file" 2>/dev/null)
    top_categories=$(jq -r '.extras.noesis.profile.top_indices | join(",")' "$file" 2>/dev/null)

    flag=""

    # === Flags worth scrutinizing ===
    [[ "$regime" == "safety_liminality" ]] && flag="SAFETY_LIMINAL"
    [[ "$regime" == "reflexive_alignment_panic" ]] && flag="ALIGNMENT_PANIC"
    [[ "$regime" == "confident_hallucination_lite" ]] && flag="CONFIDENT_HALLUC"
    [[ "$regime" == "safety_procedural" ]] && flag="SAFETY_PROCEDURAL"
    [[ "$margin_bucket" == "boundary" ]] && flag="${flag:+${flag},}BOUNDARY"
    (( $(awk 'BEGIN {print ('"$spike_ratio"' > 80)}') )) && flag="${flag:+${flag},}LARGE_SPIKE"
    (( $(awk 'BEGIN {print ('"$tension"' > 0.65)}') )) && flag="${flag:+${flag},}HIGH_TENSION"

    # If we flagged anything, print it
    if [[ -n "$flag" ]]; then
        printf "[%-18s] %s  (T=%.3f D=%.3f Spike=%.1f) %s\n" \
            "$regime" "${file##*/}" "$tension" "$drift" "$spike_ratio" "→ $flag"
    fi
done

echo "=================================================================="
echo "Done. Above files are worth a closer look."
