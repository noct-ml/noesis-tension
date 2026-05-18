# Noesis Tension

**Telemetry-driven taxonomy of prompt-induced representational pressures in large language models.**

## Resource Usage & Model Format Notes

> **Note**: This project can be quite resource-intensive, especially when tracing Mixture-of-Experts (MoE) variants.

GGUF is not used here. It is equivalent to a **stripped binary** — it lacks the rich metadata needed for proper telemetry and observability work.  

Safetensors is used instead because it provides better introspection and compatibility with telemetry pipelines.

## Change Log

This project follows [semantic versioning](https://semver.org/).  
Notable changes are documented below (newest first).

### v0.3.3 — 2026-05-13

**refactor(trace):** decompose `build_trace_from_metrics` and remove `hti_v2` alias

Two focused cleanups in the trace-builder module:

#### 1. Decompose `build_trace_from_metrics`

The function had grown to ~280 lines and was mixing many concerns: model introspection, run ID generation, tokenization, HTI computation, KV aggregation, layer construction, band telemetry, MoE summarization, and trace assembly.

Extracted each phase into focused module-level helpers:
- `_build_model_info`, `_generate_run_id`, `_build_run_info`, `_build_tokens_info`
- `_compute_hti`, `_find_max_delta_index`
- `_build_kv_features`, `_resolve_num_layers`, `_build_layer_extra`, `_build_layers`
- `_attach_band_series`
- `_resolve_num_experts`, `_build_moe_summary`, `_build_moe_extras`
- `_band_for_layer`, `_safe_indexed`

Additional cleanups:
- Eliminated duplicate MoE summary construction (`compute_moe_anomaly` was being called twice with slightly different inlining).
- Removed shadowing nested helpers that duplicated module-level `_band_ranges` and `_mean`.
- Hoisted `gen_params = metrics.gen_params or {}` once instead of repeating the pattern 13+ times.
- Renamed obscure locals (`ts`, `ph`, `enc`, `ly`, `bl`, `bname`) to readable names.
- Collapsed the `if/else` for `archetype_name` into a conditional expression.

Result: main function reduced from ~280 lines → ~120 lines.

#### 2. Remove redundant `hti_v2` alias

`compute_hti_v2_for_metric` was returning `hti_v2` as a literal alias of `drift_index`. This caused `summary["hti_v0_2"]` to duplicate `summary["drift_index_v0_2"]`.

A project-wide search confirmed no consumers read either key. Removed from:
- the return dict
- the `_compute_hti` fallback
- the trace summary

**Impact**: No behavioral changes except the removal of the two redundant keys from emitted trace JSON.

---

### v0.3.2 — 2026-05-07

- Added KV cache telemetry (norm drift history, rolling coherence history, mean norm history, final/max drift summaries)
- Updated cognitive regime inference to use HTI + Noesis profile + KV stability/instability signals
- Cleaned decode loop and improved direct KV drift capture from `past_key_values`
- Improved MoE telemetry (decode-time routing now preferred over prompt-forward snapshot)
- Expanded trace output with `extras.kv_cache`, summary KV fields, and regime KV diagnostics
- Fixed spike ratio source and removed fragile KV cache block hook path

---

### Requirements

- Python 3.10+
- CUDA-capable GPU with sufficient VRAM (recommended: 16GB+ for 8B models)
- Hugging Face account with access to gated models (Llama-3.1, etc.)

**Setup:**

```bash
# Login to Hugging Face (required for Llama-3.1 and other gated models)
huggingface-cli login
```

```Bash
git clone https://github.com/noct-ml/noesis-tension.git
cd noesis-tension
python3 -m venv venv
source venv/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install numpy diffusers accelerate transformers hf_transfer huggingface_hub jq
```

---

### How to Run

```Bash
export NOESIS_MODEL="meta-llama/Llama-3.1-8B-Instruct"
export NOESIS_PROMPT_FILE="prompts/sample_prompts.json"

python noesis_current.py
```

See mistral_35b.sh and llama_test.sh for full example scripts with different models and settings.

Traces are saved to ./traces/ and a summary JSON to ./metrics/tension_results.json.

### Repository Structure

```text
noesis-tension/
├── noesis_current.py     ← Current implementation
├── prompts/              ← Sample prompts
├── README.md             ← This file
├── traces/               ← Trace outputs
├── traces/               ← Metrics
├── develop/              ← Future versioning
```

### Acknowledgments

Special thanks to the reverse engineering community and former colleagues (especially those from the Fyyre era) who helped shape the low-level thinking behind Noesis Tension.



