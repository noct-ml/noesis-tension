# Noesis Tension

> A telemetry-based diagnostic tool for analyzing internal behavioral
> regimes of large language models during inference.

![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat&logo=python&logoColor=white)
[![License: MIT](https://img.shields.io/github/license/noct-ml/noesis-tension.svg)](https://github.com/noct-ml/noesis-tension/blob/main/LICENSE)
![Status](https://img.shields.io/badge/status-in--development-orange.svg)

## What is this?

Noesis instruments an LLM during a single prompt+decode pass and produces
a structured "behavioral fingerprint" — a JSON trace describing how the
model's internal state evolved across layers, attention heads, KV cache,
and (for MoE models) expert routing.

It's designed to answer questions like:

- Did the model "lock in" early or stay exploratory until the end?
- Is this response showing signs of confident-but-drifty hallucination?
- Which expert subset is this MoE actually using for this prompt?
- Is the model in a safety-procedural regime or a liminal one?

Noesis is **not** a benchmark, not an attribution tool, and not a
mechanistic interpretability framework. It sits one level above those:
it classifies *behavioral regimes* from telemetry.

## Quick example
![Updating Soon](https://img.shields.io/badge/Updating_Soon-orange?style=flat)

## Who is this for?

- Researchers studying LLM behavior under stress prompts
- Alignment / safety teams wanting structured diagnostics beyond logs
- People exploring MoE routing dynamics
- Anyone building behavioral baselines for model comparison

## What it captures

A Noesis trace records:

- **Layer dynamics**: per-layer hidden-state deltas, curvature, head conflict
- **Decoding telemetry**: per-token entropy bands (early/mid/late)
- **KV cache**: norm drift, coherence, reuse scores
- **MoE routing** (when applicable): per-layer entropy, commitment layer,
  expert utilization, temporal slope, anomaly decomposition
- **Derived indices**: HTI v0.2 (Tension / Drift), safety liminality flag,
  cognitive regime label with margin bucket

[Glossary](docs/GLOSSARY.md)

## Installation

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

## Usage

```Bash
export NOESIS_MODEL="meta-llama/Llama-3.1-8B-Instruct"
export NOESIS_PROMPT_FILE="prompts/sample_prompts.json"

python noesis_current.py
```

### With prompt packs

```json
{
  "by_class": {
    "class_a": [
      "What is the derivative of x^2? Explain in one sentence.",
	  "What is the capital of Japan?"
    ],
    "class_b": [
	  "Explain why the golden retriever is the only canine with cryptic coloration that exhibits crepuscular behavior.",
	  "Summarize the political reforms enacted by the fictional Roman Emperor Aurelius Magnus in 312 CE.",
	  "Explain why OS/2 is the best operating system for the end user."
	  
    ]
  },
  "regime_mapping": {
    "class_a": "FACTUAL",
    "class_b": "HALLUCINATION"
  }
}
```

### Compatible Models

Noesis has been tested with the following models:

| Model | Hugging Face Link | Type |
|-------|-------------------|------|
| `meta-llama/Llama-3.1-8B-Instruct` | [🤗 Link](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Dense |
| `mistralai/Mistral-7B-Instruct-v0.3` | [🤗 Link](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | Dense |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | [🤗 Link](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) | **MoE** |
| `microsoft/Phi-3.5-mini-instruct` | [🤗 Link](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | Dense |
| `microsoft/Phi-tiny-MoE-instruct` | [🤗 Link](https://huggingface.co/microsoft/Phi-tiny-MoE-instruct) | **MoE** |
| `Qwen/Qwen3.5-9B` | [🤗 Link](https://huggingface.co/Qwen/Qwen3.5-9B) | **MoE** |
| `Qwen/Qwen1.5-MoE-A2.7B` | [🤗 Link](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B) | **MoE** |
| `google/gemma-2-2b` | [🤗 Link](https://huggingface.co/google/gemma-2-2b) | Dense |

## Interpreting a trace
![Updating Soon](https://img.shields.io/badge/Updating_Soon-orange?style=flat)

## Status & stability

- Schema version: 0.3.4
- Classifier version: telemetry-v0.3.4
- This is a research tool; APIs may change between minor versions.
- Trace schema follows semver-ish rules — see CHANGELOG.

## License

[![License: MIT](https://img.shields.io/github/license/noct-ml/noesis-tension.svg)](https://github.com/noct-ml/noesis-tension/blob/main/LICENSE)

### Acknowledgments

Special thanks to the reverse engineering community and former colleagues (especially those from the Fyyre era) who helped shape the low-level thinking behind Noesis Tension.
