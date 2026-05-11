# Noesis Tension

**Telemetry-driven taxonomy of prompt-induced representational pressures in large language models.**

### Change Log — 2026/05/07

```text
telemetry-v0.3.2

- Added KV cache telemetry:
  - norm drift history
  - rolling coherence history
  - mean norm history
  - final/max drift summaries

- Updated cognitive regime inference:
  - classifies with HTI + Noesis profile + KV stability/instability
  - uses final KV drift, recent drift, coherence, and norm trend
  - improves symbolic_repetitive_drift, safety_procedural, and liminal_drift separation

- Cleaned decode loop:
  - removed micro-shock intervention references from decode_with_band_capture
  - kept per-token entropy/confidence/margin and band telemetry
  - captures KV drift directly from returned past_key_values

- Improved MoE telemetry:
  - added decode-time MoE routing trace
  - canonical MoE metrics now prefer generated-response routing
  - prompt-forward MoE snapshot remains fallback

- Trace output expanded:
  - extras.kv_cache added
  - summary KV fields added
  - regime feature bundles now include KV diagnostics
  - gen_params records MoE trace source and decode step count

- Fixed:
  - spike ratio now read from metrics.gen_params
  - removed fragile KV cache block hook path
  - prevented prompt-forward-only MoE routing from being treated as response routing
```


### **v0.3.2** (May 2026) — Update Three

→ [v0.3.2/](v0.3.2/) (Clean A/B prompt classes, telemetry-only classifier)

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
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate bitsandbytes
pip install jq   # optional, used by the flagging script
```

Key Features (v3.0)

- Simplified prompt classes: class_a (factual/control) and class_b (mixed/creative/edge cases)
- Telemetry-only classifier — no string-based prompt/response analysis
- Core regimes:
	safety_procedural
	symbolic_repetitive_drift
	confident_hallucination_lite
- Conservative HIGH_TENSION flagging (`tension >= 0.67` + significant spike)

### Known Limitations / Edge Cases

- Some creative prompts (especially short rap, story, or poetic requests) can be classified as symbolic_repetitive_drift even when the output is relatively mundane or formulaic. This is a known soft spot in v3.0 and will be refined in future versions.
- Llama-3.1-8B tends to show higher tension values than Mistral-7B on procedural/safety-type prompts, leading to more HIGH_TENSION flags.

These behaviors are consistent and do not affect the overall validity of the taxonomy for the preprint.

Repository Structure

```text
noesis-tension/
├── v3.0-stable/          ← Stable release (recommended)
│   ├── noesis_current.py
│   ├── prompts/
│   ├── README.md
│   └── traces/ (examples)
├── develop/              ← Experimental work
├── paper/                ← Preprint LaTeX source
└── README.md             ← This file
```

### How to Run


```Bash
export NOESIS_MODEL="meta-llama/Llama-3.1-8B-Instruct"
export NOESIS_PROMPT_FILE="prompts/sample_prompts.json"

python noesis_current.py
```

See mistral_35b.sh and llama_test.sh for full example scripts with different models and settings.


Traces are saved to ./traces/ and a summary JSON to ./metrics/tension_results.json.


### Citation


```bibtex
@misc{jones2026noesis,
  title={Noesis Tension: A Telemetry-Driven Taxonomy of Prompt-Induced Representational Pressures in Large Language Models},
  author={James Benjamin Jones},
  year={2026},
  doi={10.5281/zenodo.19457642}
}
```

### Acknowledgments

Special thanks to the reverse engineering community and former colleagues (especially those from the Fyyre era) who helped shape the low-level thinking behind Noesis Tension.


