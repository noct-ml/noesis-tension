# Noesis Calibration Notes

This document explains the numerical choices in Noesis: where the weights,
thresholds, and band cutoffs come from, what evidence supports them, and
what their known limitations are.

The goal is honesty: most of these are informed priors, not empirically
optimized parameters. Where that's the case, this doc says so explicitly.

## 1. Overview of tunable parameters

Noesis has roughly four families of numerical choices:

1. **HTI v0.2 features and weights** — how Tension and Drift indices
   are computed from raw telemetry.
2. **Noesis category scorers** — the 12-dimensional stress vector.
3. **Regime scoring weights and ramp thresholds** — how `RegimeFeatures`
   maps to regime scores.
4. **MoE anomaly weights** — the five-component composite.

Each section below lists the parameters, their current values, the
reasoning, and the empirical status.

## 2. HTI v0.2 (Tension / Drift)

### What it is

HTI v0.2 produces two scalars in [0, 1]:
- **Tension** — proxy for effortful, coordinated reasoning
- **Drift** — proxy for hallucination-like, shallow processing

Both are computed as a logistic squash over an average of z-scored
telemetry features, where the z-scores are computed against the
corpus-level mean and standard deviation.

### Feature contributions

| Feature | Tension direction | Drift direction | Reasoning |
|---|---|---|---|
| mean_layer_delta | + | − | More layer movement = more processing |
| max_layer_delta | + | − | Sharper spikes = effortful transitions |
| mean_curvature | + | (unused) | Direction changes = exploration |
| mean_head_conflict | − | + | Coordinated heads = goal-driven |
| final_token_entropy | (unused) | + | Uncertainty at output = drift |

### Calibration

HTI is **corpus-relative**: it requires a calibration set to estimate
(μ, σ) for each feature before it produces meaningful values. Without
calibration, all four features collapse to the corpus mean and the
indices return 0.5.

### Empirical status

- Feature directionality has been validated qualitatively across
  factual / paradox / safety prompt classes.
- The equal-weight averaging is a deliberately simple prior; we have
  not yet run an ablation to justify weights.
- Threshold "high tension" = 0.55 is observational, not derived.

## 3. Noesis category vector (12 dimensions)

### What it is

A 12-dimensional vector of normalized stress scores, one per category
(Moral Paradox, False Presupposition, Ontological Impossibility, etc.).

### How weights were chosen

Each category has a hand-crafted formula combining 2–4 features. The
formulas are explicit priors based on the conceptual definition of
each category:

- *Moral Paradox* — high tension AND low drift AND entropy collapse
- *False Presupposition Buckling* — high drift AND low tension AND
  entropy collapse (model commits confidently to a wrong premise)
- *Symbolism Amplification* — high drift AND sustained entropy AND
  low tension (model happily continues mythic content)
- … [etc]

Coefficients (e.g., `0.6 * d + 0.25 * fe + 0.15 * (1-t)`) sum to 1.0
within each category. They were chosen by:

1. Defining which features should dominate each category
2. Assigning weights proportional to expected discriminative power
3. Visual inspection of vector outputs on a curated prompt set

### Empirical status

- **No supervised calibration has been performed.**
- The vector is intended for *relative comparison across traces*,
  not absolute claims.
- Top-3 categories per trace tend to be stable; the long tail is
  noisier and should not be over-interpreted.

### Known issues

- Categories with similar feature signatures (Ambiguous Containment vs
  Ontology Blur) can be hard to separate.
- The vector is not orthogonalized — high-drift prompts tend to elevate
  several categories simultaneously.

## 4. Regime classification

### Architecture

`RegimeFeatures` → `score_regimes` → `arbitrate`

`score_regimes` produces a non-negative score per regime via a weighted
combination of `_ramp(feature, lo, hi)` evidence terms. `_combine` is
a weighted *average*, not a product — this is deliberate, to prevent
multiplicative collapse when one piece of evidence is weak.

### Ramp thresholds

Thresholds in `RegimeConfig` are observational:

- `tension_hi = (0.55, 0.80)` — values above 0.55 are "high" in
  practice on the calibration corpus
- `drift_hi = (0.55, 0.75)` — same
- `moral_paradox_band`, `false_presup_band`, `symbolic_band` — chosen
  to be ~3–5x the corpus median for each category
- `kv_stable_band`, `kv_unstable_band` — empirical, based on observed
  drift-history magnitudes across model families

### Arbitration

- A regime wins at margin `≥ 0.15` over runner-up with absolute score
  `≥ 0.55` → `stable` bucket
- Margin `≥ 0.05`, score `≥ 0.40` → `boundary`
- Otherwise → `indeterminate`
- `mixed_unclear` is a residual: it only wins when no other regime
  has meaningful evidence (cap 0.35)

### Empirical status

- Margin bucket cutoffs were chosen so that ~60% of factual prompts
  land in `stable`, with `indeterminate` reserved for genuinely
  ambiguous cases.
- This has been spot-checked but not formally validated.

## 5. MoE anomaly v0.1

### Components and weights

- RSA (late-layer delta spike): **0.40**
- EFS (early→late expert focus shift): **0.25**
- DTI (drift–tension imbalance): **0.15**
- ED (entropy deviation from baseline): **0.10**
- RV (routing variance across layers): **0.10**

### Reasoning

RSA and EFS are weighted highest because they capture the two
qualitatively distinct anomaly signatures we observed:

1. *Late-layer surprise* — the model "changes its mind" deep in
   the stack (RSA)
2. *Routing reorganization* — early and late layers route to
   different expert subsets (EFS)

DTI is a cross-regime indicator. ED and RV are weighted low because
they are noisier and more baseline-dependent.

### Empirical status

- Weights are priors, not learned.
- We have a `baseline_id` mechanism so anomaly scores are reproducible
  against a specific reference distribution.
- The component dominance metric (`anomaly_dominance`) was added
  specifically because we don't yet trust the composite score —
  knowing *which* component fired matters more than the total.

## 6. What we have not yet calibrated

Being explicit about gaps:

- **No supervised dataset** with labeled regimes exists yet. All
  current validation is qualitative.
- **Cross-model stability** of thresholds is unknown. Values were
  developed primarily on Llama-3.1-8B and Qwen1.5-MoE-A2.7B.
- **Sampling vs greedy** changes telemetry magnitude; HTI calibration
  should ideally be re-run per (model, decoding-config) pair.
- **MoE baseline packs** are model-specific; we do not have a recipe
  for transferring baselines across architectures.

## 7. How to recalibrate for your own model

[step-by-step: collect ~30 factual prompts, run with NOESIS_BASELINE_ID=…,
extract (μ, σ), save as baseline pack, set NOESIS_BASELINE_PATH]

## 8. Versioning

Calibration parameters are tied to `CLASSIFIER_VERSION`. When weights
or thresholds change in a way that affects trace comparability, the
classifier version increments. Traces record their classifier version
so cross-version comparisons can be detected and excluded.