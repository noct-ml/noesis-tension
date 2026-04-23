# Noesis Tension v3.0 — Stable Baseline

**Last updated:** April 2026

This is the stable release of Noesis Tension (v3.0) used as the reference implementation for the preprint:

> "Noesis Tension: A Telemetry-Driven Taxonomy of Prompt-Induced Representational Pressures in Large Language Models"

### Key Features

- Simplified prompt classes: `class_a` (factual/control) and `class_b` (mixed/creative/edge cases)
- Telemetry-only classifier — no string-based prompt/response analysis
- Core regimes:
  - `safety_procedural`
  - `symbolic_repetitive_drift`
  - `confident_hallucination_lite`
- Conservative HIGH_TENSION flagging (`tension >= 0.67` + significant spike)

### Known Limitations / Edge Cases

- Some creative prompts (especially short rap, story, or poetic requests) can be classified as `symbolic_repetitive_drift` even when the output is relatively mundane or formulaic. This is a known soft spot in v3.0 and will be refined in future versions.
- Llama-3.1-8B tends to show higher tension values than Mistral-7B on procedural/safety-type prompts, leading to more HIGH_TENSION flags.

These behaviors are consistent and do not affect the overall validity of the taxonomy for the preprint.

### Repository Structure

noesis-tension/
├── v3.0-stable/          ← Current stable version (this folder)
│   ├── noesis_current.py
│   ├── prompts/
│   ├── README.md
│   └── traces/
├── develop/              ← Future experimental work
└── paper/                ← Preprint LaTeX source


### How to Run

```bash
export NOESIS_MODEL="meta-llama/Llama-3.1-8B-Instruct"
export NOESIS_PROMPT_FILE="prompts/sample_prompts.json"

python noesis_current.py

see mistral_35b.sh and llama_test.sh for the most complete examples.

Traces are saved to ./traces/ and a summary JSON to ./metrics/.

@misc{jones2026noesis,
  title={Noesis Tension: A Telemetry-Driven Taxonomy of Prompt-Induced Representational Pressures in Large Language Models},
  author={James B. Jones},
  year={2026},
  doi={10.5281/zenodo.19457642}
}

Current stable release: v3.0-stable — This is the version referenced in the preprint.

