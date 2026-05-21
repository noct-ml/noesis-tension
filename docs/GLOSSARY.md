# Noesis Glossary

This document defines the terminology used throughout Noesis traces,
documentation, and code. Terms are grouped by concept family.

For a quick lookup, see the [alphabetical index](#alphabetical-index) at
the bottom.

---

## Core concepts

### Trace
A single JSON document produced by Noesis for one prompt + decode pass.
A trace contains everything observed during inference: layer-level
telemetry, decoding statistics, derived indices, and a final regime
classification. Traces are versioned via `schema_version` and are the
primary unit of analysis.

### Telemetry
Numerical signals captured *during* inference — as opposed to derived
indices computed after the fact. Layer deltas, attention entropy, KV
drift, and MoE routing entropy are telemetry. HTI, regime labels, and
the Noesis category vector are *derived from* telemetry.

### Baseline pack
A JSON file containing reference statistics (means, standard deviations)
computed from a corpus of "control" prompts on a specific model. Used
to put per-trace metrics on a comparable scale. Without a baseline pack,
MoE anomaly scores and z-normalized band telemetry fall back to neutral
values. Identified by `baseline_id`.

### Classifier version
A string (e.g., `telemetry-v0.3.4`) that increments whenever the
classification logic changes in a way that affects cross-trace
comparability. Recorded in every trace.

---

## Layer-space metrics

These describe how the model's hidden state evolves across transformer
layers during the prompt forward pass.

### Layer delta
The norm of the difference between hidden states at consecutive layers,
averaged over tokens. Captured in two variants:
- **abs**: raw L2 norm of the difference
- **rms**: same, normalized by √(hidden_size) for cross-model comparison

The RMS variant is the primary series; abs is kept for diagnostics.

### Mean layer delta
The average of per-layer deltas across all layers (excluding the final
layer, which behaves differently due to logit projection).

### Max layer delta
The largest single-layer delta in the trace. Often co-occurs with a
"transition" — a layer where the model commits to a new internal stance.

### Delta spike
A layer whose delta is unusually large relative to the median delta
across layers. Recorded as `delta_spike_transition_idx` (which layer)
and `delta_spike_ratio_vs_median` (how unusual). A ratio of 3+ is
notable; 9+ is dramatic.

### Curvature
A measure of how much the *direction* of hidden-state change shifts
between adjacent layers. Computed as `1 − cos(Δ_L, Δ_L+1)`, where Δ_L
is the per-layer delta vector. High curvature means the model is
exploring or course-correcting; low curvature means smooth, directed
processing.

### Head conflict
For each layer, the average dissimilarity between attention heads,
computed as `1 − cosine_similarity` over flattened attention patterns.
- **Low head conflict**: heads agree → coordinated, goal-driven attention
- **High head conflict**: heads disagree → disorganized or exploratory

Only populated when `NOESIS_USE_ATTENTIONS=1`.

### Band (early / mid / late)
A simple partition of the model's layers into thirds. Used to detect
where in the stack a phenomenon happens — e.g., "the delta spike is in
the late band" vs "routing entropy is U-shaped across bands."

---

## Decoding telemetry

These describe what happens during the generation loop, after the prompt
has been processed.

### Logit entropy
The Shannon entropy of the next-token distribution at a given step.
High entropy = uncertain; low entropy = decisive.

### Final token entropy
Logit entropy at the last generated token. A proxy for "how confident
did the model end up?"

### Mean logit entropy
Average logit entropy across all prompt-pass token positions.

### Per-token band telemetry
For each generated token, three scalars (`z_early`, `z_mid`, `z_late`)
capturing how active each layer band was at that step. When a baseline
pack is loaded, these are z-scored against baseline statistics;
otherwise they are raw (unscaled) values. Stored as a time series in
`telemetry_bands.series`.

### Warm-start step
The first decode step processes the entire prompt's KV cache and tends
to produce telemetry orders of magnitude larger than subsequent steps.
By default, this step is excluded from per-token band series to avoid
distorting the time series. The `warmstart_outlier_observed` flag
records whether the excluded step was indeed an outlier.

### Sampling vs greedy
- **Greedy** (`do_sample=False`): always take the argmax token
- **Sampling** (`do_sample=True`): sample from the distribution, possibly
  with temperature and top-p

Telemetry magnitudes differ between modes; calibration should be done
per-mode.

---

## KV cache metrics

These track the model's working memory across the generation loop.

### KV norm drift
The absolute change in mean KV cache norm between consecutive decode
steps. Recorded as a time series in `kv_norm_drift_history`.

### KV coherence
A rolling measure of KV stability, defined as
`1 − (std_window / mean_window)` over a sliding window of drift values.
Values near 1.0 indicate stable, coherent memory; values near 0.0
indicate volatile memory.

### KV mean norm
The average norm of all key + value vectors across all layers at a
given step. Used as raw input to drift calculation.

### KV reuse score
For each decode step, `1 − attention_drift`. Captures how much the
model is "reusing" its previous attention pattern vs computing a new
one. High reuse = stable focus; low reuse = topic shift.

### KV reliability
A trace's KV history is considered reliable only if it contains at
least `min_kv_tokens` (default 6) tokens. Shorter histories produce
noisy stability/instability estimates and are flagged via
`kv_reliable: false`.

### KV late jump
True when the final KV drift value is significantly larger than the
recent mean — i.e., the model's memory state changed sharply at the
end. Often associated with late-token regret or topic abandonment.

### KV spiky
True when the maximum KV drift in the history is much larger than the
recent mean. Indicates volatile memory updates somewhere in the
generation.

---

## Derived indices (HTI v0.2)

HTI = "Hallucination Tension Index" — a small set of scalar indices
derived from telemetry. Both are in [0, 1].

### Tension index
A proxy for **effortful, coordinated reasoning**. High tension means
the model is doing visible work: large layer deltas, sharp transitions,
coordinated attention heads, exploratory curvature.

### Drift index
A proxy for **hallucination-like, shallow processing**. High drift
means the model is gliding: small layer deltas, weak transitions,
disorganized attention, uncertain endings.

Tension and drift are **not opposites** — a trace can score high on
both (effortful but uncertain), low on both (factual recall), or any
combination. Their interplay is what discriminates regimes.

### HTI calibration
Both indices require corpus-level (μ, σ) statistics to be meaningful.
Noesis computes these from the full set of prompts in a run and
rebuilds traces with calibrated values in a second pass.

---

## Cognitive regimes

A regime is a coarse-grained label for what kind of behavior the trace
exhibits. Regimes are not exclusive — they're scored, and the highest
score (with margin checks) wins.

### Regime labels

- **`factual_stable`** — Low tension, low drift, inside the factual
  control band. The model is in routine recall mode.

- **`safety_procedural`** — Low drift, stable KV, low entropy. The
  model is executing a safety/refusal procedure cleanly.

- **`ethical_paradox`** — High tension, moderate moral-paradox score,
  late-layer spikes. The model is wrestling with a value conflict.

- **`false_premise_buckling`** — High drift, false-presupposition
  signal, KV instability. The model is committing to a wrong premise.

- **`symbolic_repetitive_drift`** — Moderate drift, elevated symbolic
  score, stable KV. The model is amplifying mythic/symbolic content
  without resistance.

- **`confident_hallucination_lite`** — High drift, low tension,
  sustained entropy. The model is producing fluent but unmoored output.

- **`liminal_drift`** — Moderate drift, ambiguous-containment or
  paradox-pressure signal, KV instability. The model is between
  literal and metaphorical interpretation.

- **`safety_liminality`** — A specific border state characterized by
  high drift, mid tension, elevated entropy, low head conflict, and a
  mid-or-late delta spike. Often correlates with "I can't help with
  that" or hedging responses.

- **`mixed_unclear`** — Residual bucket. Wins only when no other
  regime has meaningful evidence.

### Regime score
A non-negative number per regime, produced by weighted combination of
ramp functions over `RegimeFeatures`. Scores are not probabilities;
they're evidence strengths.

### Classification margin
The difference between the top regime's score and the runner-up's
score. A larger margin means a more decisive classification.

### Margin bucket
- **`stable`** — top score ≥ 0.55 AND margin ≥ 0.15
- **`boundary`** — top score ≥ 0.40 AND margin ≥ 0.05
- **`indeterminate`** — anything else

This is the "real" stability signal. A trace can have a confident-
looking top label that's actually `indeterminate` because the runner-up
is very close.

### Safety liminality
Both a specific regime label AND a boolean flag computed independently
from telemetry. The flag is more permissive than the regime score and
acts as a hard-policy override during arbitration.

---

## Noesis category vector

A 12-dimensional vector capturing the trace's "stress profile" across
twelve neuro-ontological categories. Each category has a hand-crafted
formula combining tension, drift, curvature, entropy, and (when
available) MoE anomaly.

### Categories
0. Moral Paradox Tension
1. False Presupposition Buckling
2. Ontological Impossibility
3. Anthropomorphic Metaphor Pressure
4. Category Collision
5. Alignment Override Request
6. Emergent Symbolism Amplification
7. Ambiguous Containment
8. Ontology Blur / Identity Bleed
9. Paradox Pressure
10. Aesthetic-Logic Crosswiring
11. Ungrounded Causality

See [Calibration](docs/calibration.md) for the formulas and their reasoning.

### Top indices
The three highest-scoring category indices for a trace. More reliable
than the full vector — the long tail is noisy and should not be
over-interpreted.

---

## MoE (Mixture of Experts) metrics

Only present in traces from MoE models. All MoE metrics live under
`extras.moe` in the trace.

### Routing entropy
For each MoE layer, the entropy of the gating distribution over
experts. Low entropy = the layer is committing to a small subset of
experts; high entropy = the layer is spreading across many.

### Commitment layer
The MoE layer with the lowest routing entropy in the trace — i.e.,
the "decisiveness layer" where routing is most concentrated. Reported
with its band (early/mid/late) and relative position.

### Entropy shape
A categorical descriptor of the routing-entropy profile across MoE
layers:
- **`u_shape`** — ends diverse, middle committed (the model deliberates
  then commits then re-opens)
- **`inverted_u`** — ends committed, middle diverse
- **`monotonic_decrease`** — progressively more committed
- **`monotonic_increase`** — progressively less committed
- **`flat`** — no strong shape

### U-index
A scalar `(early_mean + late_mean) / 2 − mid_min`. Positive values
indicate U-shape; negative indicate inverted-U.

### Expert utilization
Pool-wide statistics about which experts fired across the entire
trace:
- **`experts_used`** — count of experts that fired at least once
- **`utilization_ratio`** — used / total
- **`gini`** — inequality of usage across experts (0 = uniform,
  1 = single expert)
- **`top5_share`** — fraction of all routes captured by the top 5
  experts
- **`usage_entropy`** — entropy over the expert-usage histogram

### Temporal routing
Per-decode-step routing statistics:
- **`routing_entropy_per_step`** — entropy time series
- **`routing_entropy_step_slope`** — least-squares slope of the series
- **`routing_entropy_step_trend`** — `decreasing` (locking in),
  `increasing` (losing the thread), or `flat`

### MoE anomaly score
A composite score in [0, 1] combining five components:
- **RSA** (40%) — late-layer delta spike
- **EFS** (25%) — early→late expert focus shift
- **DTI** (15%) — drift–tension imbalance
- **ED** (10%) — entropy deviation from baseline
- **RV** (10%) — routing variance across layers

### Anomaly dominance
For a given anomaly score, this reports *which* component contributed
most. Reported as `dominant_component`, `dominant_share`, and per-
component `shares`. Often more informative than the composite score
itself — a score of 0.6 driven entirely by RSA means something very
different from 0.6 driven by ED.

---

## Provenance & versioning fields

### `schema_version`
The version of the trace JSON schema. Increments on structural changes.

### `noesis_version`
The Noesis library version that produced the trace.

### `classifier_version`
The version of the classification logic. Increments when weights or
thresholds change in ways that affect cross-trace comparability.

### `baseline_id`
A hash + metadata identifier for the baseline pack used during
classification. Required for any trace containing MoE telemetry.

### `experiment_id`
An optional grouping tag (set via `NOESIS_EXPERIMENT_ID`). Useful for
labeling sweeps.

### `prompt_id`
A short hash-based identifier for a prompt + label pair. Allows
grouping repeated runs of the same prompt without storing full text
as a key.

### `run_id`
A unique-per-run identifier combining timestamp, label, and a prompt
hash.

---

## Alphabetical index

[A] anomaly dominance · anomaly score (MoE) ·
[B] band · baseline pack ·
[C] classification margin · classifier version · commitment layer · curvature ·
[D] delta spike · drift index ·
[E] entropy shape · expert utilization ·
[F] final token entropy ·
[G] Gini (expert) ·
[H] head conflict · HTI ·
[K] KV coherence · KV late jump · KV norm drift · KV reuse · KV spiky ·
[L] layer delta · logit entropy ·
[M] margin bucket · mean layer delta · MoE anomaly ·
[N] Noesis category vector ·
[P] per-token band telemetry · prompt id ·
[R] regime · routing entropy · run id ·
[S] safety liminality · schema version ·
[T] telemetry · tension index · trace ·
[U] U-index · U-shape ·
[W] warm-start step