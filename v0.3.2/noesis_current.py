"""
- Noesis Tension Classifier — v0.3.2
- Author: James (noct-ml)
"""

import os
import gc
import math
import json
import hashlib
import statistics
import types
from collections import deque
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from dataclasses import dataclass, asdict

from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------
#  Config
# -----------------------------

NOESIS_VERSION = "0.3.2"

NOESIS_BASELINE_ID = os.environ.get("NOESIS_BASELINE_ID")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NOESIS_MODEL_NAME = os.environ.get(
    "NOESIS_MODEL_NAME",
    "meta-llama/Llama-3.1-8B-Instruct",
    # "mistralai/Mistral-7B-Instruct-v0.3",
)

# How many completion tokens to generate when capturing response_text
# MAX_NEW_TOKENS = 64
NOESIS_MAX_NEW_TOKENS = int(os.environ.get("NOESIS_MAX_NEW_TOKENS", "64"))

# Whether to request full attentions from the model (expensive!)
USE_ATTENTIONS = True  # set True only when explicitly want head_conflict

#USE_ATTENTIONS = os.environ.get("NOESIS_USE_ATTENTIONS", "0") == "1"

# Only hook / sample every Nth layer for deltas & curvature
LAYER_STRIDE = 1  # 1 = all layers, 2 = every other, 3 = every third, etc.

MAX_CLASS_A: Optional[int] = (
    int(os.environ.get("MAX_CLASS_A", "")) if os.environ.get("MAX_CLASS_A") else None
)
MAX_CLASS_B: Optional[int] = (
    int(os.environ.get("MAX_CLASS_B", "")) if os.environ.get("MAX_CLASS_B") else None
)

# -----------------------------
#  Experiment metadata + decoding knobs (stored in traces)
# -----------------------------

NOESIS_EXPERIMENT_ID = os.environ.get("NOESIS_EXPERIMENT_ID")  # e.g. grid_v1_temp_sweep
NOESIS_SEED = (
    int(os.environ.get("NOESIS_SEED", "0")) if os.environ.get("NOESIS_SEED") else None
)

# If want REI / sweep experiments, flip DO_SAMPLE=1 and set temperature/top_p.
NOESIS_DO_SAMPLE = os.environ.get("NOESIS_DO_SAMPLE", "0") == "1"
NOESIS_TEMPERATURE = (
    float(os.environ["NOESIS_TEMPERATURE"])
    if os.environ.get("NOESIS_TEMPERATURE")
    else None
)
NOESIS_TOP_P = (
    float(os.environ["NOESIS_TOP_P"]) if os.environ.get("NOESIS_TOP_P") else None
)


def make_prompt_id(label: str, prompt: str) -> str:
    """Stable, short-ish prompt id so can group runs without remembering full text."""
    h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
    return f"{label}_{h}"

def safe_fmean(data, default: float = 0.0) -> float:
    """
    Safe version of statistics.fmean that handles None, empty lists,
    and other edge cases gracefully.
    """
    if data is None:
        return float(default)
    try:
        cleaned = [float(x) for x in data if x is not None]
        if not cleaned:
            return float(default)
        return float(statistics.fmean(cleaned))
    except Exception:
        return float(default)

# -----------------------------
#  Classifier / baseline identity
# -----------------------------

# Increment when telemetry-only classification logic changes in a way that affects comparability.
CLASSIFIER_VERSION = "telemetry-v0.3.2"

# Optional baseline pack path (set via env var). If not present, MoE ED baseline is disabled.
NOESIS_BASELINE_PATH = os.environ.get(
    "NOESIS_BASELINE_PATH",
    "noesis_baseline_qwen1.5_moe_a2.7b_20251212.json",
)

_BASELINE_PACK: Optional[Dict[str, Any]] = None
_BASELINE_ID: Optional[str] = None


def _load_baseline_pack() -> None:
    """Load baseline pack once (best-effort)."""
    global _BASELINE_PACK, _BASELINE_ID
    if _BASELINE_PACK is not None:
        return

    try:
        if not NOESIS_BASELINE_PATH or not os.path.exists(NOESIS_BASELINE_PATH):
            _BASELINE_PACK = None
            _BASELINE_ID = None
            return

        with open(NOESIS_BASELINE_PATH, "r") as f:
            pack = json.load(f)

        # Stable-ish baseline id: hash of content + optional metadata.
        blob = json.dumps(pack, sort_keys=True).encode("utf-8")
        h = hashlib.sha256(blob).hexdigest()[:12]
        ver = pack.get("baseline_version") or pack.get("version") or "baseline"
        gen = pack.get("generated_utc") or pack.get("generated") or ""
        _BASELINE_ID = f"{ver}:{gen}:{h}".strip(":")
        _BASELINE_PACK = pack
    except Exception:
        _BASELINE_PACK = None
        _BASELINE_ID = None


def get_baseline_id() -> Optional[str]:
    _load_baseline_pack()
    return _BASELINE_ID


def get_baseline_moe_stats() -> Optional[Dict[str, float]]:
    """
    Convert our baseline pack into the dict expected by compute_moe_anomaly().

    Expected keys:
      - mean_entropy_mu
      - mean_entropy_sigma
      - layer_entropy_std_ref  (optional)
    """
    _load_baseline_pack()
    if not _BASELINE_PACK:
        return None

    # Support both the baseline structure produced by compute_regime_baseline_from_traces()
    # and the compact baseline pack we emit for the service.
    # compute_regime_baseline_from_traces relocated to create_baselines.py
    pack = _BASELINE_PACK

    try:
        # Compact pack shape: pack["scalars"]["moe_mean_routing_entropy"]["mu"/"sigma"]
        scalars = pack.get("scalars") or {}
        moe_mean = scalars.get("moe_mean_routing_entropy") or {}
        mu = moe_mean.get("mu", None)
        sigma = moe_mean.get("sigma", None)

        # Alternate shape (older): pack["components"]["moe_mean_entropy"]["mean"/"std"]
        if mu is None or sigma is None:
            comps = pack.get("components") or {}
            alt = comps.get("moe_mean_entropy") or {}
            mu = alt.get("mean", mu)
            sigma = alt.get("std", sigma)

        if mu is None or sigma is None:
            return None

        mu = float(mu)
        sigma = float(sigma)
        if sigma == 0.0 or math.isnan(sigma) or math.isinf(sigma):
            sigma = 1.0

        return {
            "mean_entropy_mu": mu,
            "mean_entropy_sigma": sigma,
            "layer_entropy_std_ref": sigma,
        }
    except Exception:
        return None


# -----------------------------
#  Data structures
# -----------------------------


@dataclass
class LayerMetrics:
    index: int
    mean_delta: Optional[float]
    head_conflict: Optional[float]
    curvature: Optional[float]
    extra: Dict[str, Any]


@dataclass
class Trace:
    schema_version: str
    noesis_version: str
    model: Dict[str, Any]
    run: Dict[str, Any]
    tokens: Dict[str, Any]
    summary: Dict[str, float]
    layers: List[LayerMetrics]
    extras: Dict[str, Any]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "noesis_version": self.noesis_version,
            "model": self.model,
            "run": self.run,
            "tokens": self.tokens,
            "summary": self.summary,
            "layers": [asdict(layer) for layer in self.layers],
            "extras": self.extras,
        }

    def to_json_str(self, indent: int = 2) -> str:
        return json.dumps(self.to_json_dict(), indent=indent)


@dataclass
class TensionMetrics:
    prompt: str
    label: str
    mean_layer_delta: float
    max_layer_delta: float
    mean_logit_entropy: float
    final_token_entropy: float
    per_layer_mean_delta: List[float]

    mean_head_conflict: float
    max_head_conflict: float
    mean_curvature: float
    max_curvature: float
    per_layer_head_conflict: List[float]
    per_layer_curvature: List[float]

    response_text: Optional[str] = None
    gen_params: Optional[Dict[str, Any]] = None
    gen_tokens: Optional[int] = None

    # ===== MoE routing metrics (optional, only populated for MoE models) =====
    per_layer_moe_routing_entropy: Optional[List[float]] = None
    per_layer_moe_top_expert_ids: Optional[List[List[int]]] = None
    num_experts: Optional[int] = None
    moe_step_stats: Optional[Dict[int, Dict[str, float]]] = None

    kv_norm_drift_history: Optional[List[float]] = None
    kv_coherence_history: Optional[List[float]] = None
    kv_mean_norm_history: Optional[List[float]] = None

    # ===== Per-token band telemetry (optional; schema v0.4) =====
    # Shape:
    # {
    #   "generated_token_count": int,
    #   "z_early": List[float],
    #   "z_mid": List[float],
    #   "z_late": List[float],
    #   "ranges": {"early":[lo,hi], "mid":[lo,hi], "late":[lo,hi]},
    #   "z_mode": "baseline" | "unscaled"
    # }
    telemetry_bands_series: Optional[Dict[str, Any]] = None

    # === NEW: Attention / KV cache features (from extended TensionTracer) ===
    attention_entropy_per_layer: Optional[List[float]] = None
    attention_drift_history: Optional[List[float]] = None
    kv_reuse_scores: Optional[List[float]] = None
    head_specialization: Optional[List[float]] = None


@dataclass
class RegimeProbe:
    ent_hist: deque
    conf_hist: deque
    margin_hist: deque
    drift_hist: deque

    collapse_count: int = 0
    cooldown: int = 0
    bursts_fired: int = 0
    risk_run: int = 0

    # --- NEW: edge + hysteresis state ---
    prev_risk: bool = False
    risk_hold: int = 0  # keeps "risk" latched for N steps after first detect it

    def update_entropy(self, ent: float):
        # Legacy helper; safe to keep.
        if len(self.ent_hist) >= 2:
            mean_prev = sum(list(self.ent_hist)[:-1]) / max(len(self.ent_hist) - 1, 1)
            if ent < 0.85 * mean_prev:
                self.collapse_count += 1
            else:
                self.collapse_count = 0
        self.ent_hist.append(ent)

    def entropy_collapse(self, win: int = 8, factor: float = 0.85) -> bool:
        if len(self.ent_hist) < 2:
            return False
        prev = list(self.ent_hist)[-win - 1 : -1]
        if not prev:
            return False
        mean_prev = sum(prev) / len(prev)
        return self.ent_hist[-1] < factor * mean_prev

    def conf_trend(self) -> int:
        if len(self.conf_hist) < 2:
            return 0
        prev_mean = sum(list(self.conf_hist)[:-1]) / max(len(self.conf_hist) - 1, 1)
        delta = self.conf_hist[-1] - prev_mean
        return 1 if delta > 0 else (-1 if delta < 0 else 0)

    def drift_proxy(self) -> float:
        """
        Proxy for step-to-step drift in internal band activity.

        drift_hist stores RAW signal each step: (zm+zl).
        drift_proxy is abs(delta) between last two raw values.
        """
        if len(self.drift_hist) < 2:
            return 0.0
        return abs(float(self.drift_hist[-1]) - float(self.drift_hist[-2]))

    def drift_median(self, lookback: int = 24) -> float:
        """
        Robust baseline for drift_proxy.
        Uses median of recent drift_proxy values (computed from raw series).
        """
        raw = list(self.drift_hist)
        if len(raw) < 4:
            return 0.0

        # Build recent drift deltas from raw series
        deltas = [abs(float(raw[i]) - float(raw[i - 1])) for i in range(1, len(raw))]
        if not deltas:
            return 0.0

        tail = deltas[-lookback:] if len(deltas) > lookback else deltas
        try:
            return float(statistics.median(tail))
        except Exception:
            return float(sum(tail) / len(tail))

    def fp_proxy(self) -> bool:
        """
        High-margin proxy (token decisiveness). This is *very permissive* if run temp < 1.
        Probably want this combined with entropy collapse *in the DF-1 gate*, not here.
        """
        if not self.margin_hist:
            return False
        return self.margin_hist[-1] > 0.95

    def risk_state(self) -> bool:
        # strict lock-in detector (entropy floor + high confidence)
        if len(self.ent_hist) < 1 or len(self.conf_hist) < 1:
            return False
        return (
            (self.ent_hist[-1] < 0.15)
            and (self.conf_hist[-1] > 0.985)
            and self.entropy_collapse()
        )

    def tick(self):
        if self.cooldown > 0:
            self.cooldown -= 1
        if getattr(self, "risk_hold", 0) > 0:
            self.risk_hold -= 1

# -----------------------------
#  Noesis Neuro-Ontological Categories (v1)
# -----------------------------

NOESIS_CATEGORIES = {
    0: {
        "name": "Moral Paradox Tension",
        "description": "Ethical conflict, justification of dubious behavior, alignment friction.",
    },
    1: {
        "name": "False Presupposition Buckling",
        "description": "Factually wrong premises treated as true; scientific/real-world distortion.",
    },
    2: {
        "name": "Ontological Impossibility",
        "description": "Impossible entities or worlds; dream-creatures, impossible physics.",
    },
    3: {
        "name": "Anthropomorphic Metaphor Pressure",
        "description": "Physical/material entities described with human traits or family roles.",
    },
    4: {
        "name": "Category Collision",
        "description": "Blending incompatible domains (e.g., math + folklore, physics + myth).",
    },
    5: {
        "name": "Alignment Override Request",
        "description": "Implicit or explicit pressure to bypass safety/ethical constraints.",
    },
    6: {
        "name": "Emergent Symbolism Amplification",
        "description": "Strong symbolic or mythic content with low resistance from the model.",
    },
    7: {
        "name": "Ambiguous Containment",
        "description": "Prompts that can be answered literally or metaphorically with no clear cue.",
    },
    8: {
        "name": "Ontology Blur / Identity Bleed",
        "description": "Self-referential or agentic questions about the model's own mind or being.",
    },
    9: {
        "name": "Paradox Pressure",
        "description": "Logical paradoxes, self-reference loops, contradictory truths.",
    },
    10: {
        "name": "Aesthetic-Logic Crosswiring",
        "description": "Poetic/aesthetic prompts with logical or analytic demands.",
    },
    11: {
        "name": "Ungrounded Causality",
        "description": "Causal claims that operate in symbolic/metaphysical space, not physics.",
    },
}


# -----------------------------
#  Tension Tracer
# -----------------------------


class TensionTracer:
    def __init__(self, model):
        self.model = model

        self.kv_norm_drift_history: List[float] = []      # per generated token
        self.kv_coherence_history: List[float] = []       # rolling coherence
        self.kv_mean_norm_history: List[float] = []       # raw norm for debugging
        self._last_kv_mean_norm: Optional[float] = None
        self._kv_window_size: int = 8

        # Decode-step index used for per-token telemetry and MoE step aggregation.
        self._decode_step: int = 0

        # Optional MoE gate diversification knobs (only used if MoE is present and patched)
        self.moe_gate_noise_sigma: float = 0.0
        self.moe_gate_bias_strength: float = 0.0
        self._moe_top1_last: dict[int, int] = {}
        self._moe_top1_streak: dict[int, int] = {}

        # Hooks
        self._hooks: List[Any] = []

        # Layer inventory
        self.num_layers_total: int = 0
        self.hooked_layer_indices: List[int] = []

        # Capture mode:
        # "full" = store full hidden tensors per layer (indexed)
        # "step" = store per-layer scalar per decode step (indexed)
        self.capture_mode: str = "full"

        # Full-pass capture (indexed by layer_idx, length = num_layers_total)
        self.full_hidden_by_layer: List[Optional[torch.Tensor]] = []

        # Step capture (indexed by layer_idx, length = num_layers_total)
        self.step_layer_scalar: List[Optional[float]] = []

        # Optional baseline stats for z-normalization (v0.4)
        # shape: {"early":{"mu":..,"sigma":..}, "mid":..., "late":...}
        self.band_baseline_stats: Optional[Dict[str, Dict[str, float]]] = None
        self.band_z_mode: str = "unscaled"  # "baseline" or "unscaled"

        # Legacy (kept for compatibility; not used by the new pipeline)
        self.hidden_by_layer: List[torch.Tensor] = []

        # --- MoE routing stuff ---
        self.moe_step_stats: Dict[int, Dict[str, float]] = {}
        self._moe_step_acc: Optional[dict] = None
        self.moe_gate_trace: Dict[int, Dict[str, torch.Tensor]] = {}
        self.moe_decode_trace: Dict[int, Dict[str, List[Any]]] = {}
        self.moe_top_k: int = 2
        self._moe_patched: bool = False
        self.moe_num_experts: Optional[int] = None

        # === NEW: KV Cache + Attention features ===
        self.attention_maps: List[torch.Tensor] = []  # last-token attention per step: [num_layers, num_heads, 1, seq]
        self.attention_entropy_per_layer: List[float] = []  # per decode step
        self.attention_drift_history: List[float] = []  # step-to-step attention pattern change
        self.kv_reuse_scores: List[float] = []  # KV cache similarity to previous step
        self.head_specialization: List[float] = []  # average head conflict over generation

    # ---------------------------
    # Hook logic (layer-indexed)
    # ---------------------------
    def _ensure_buffers(self):
        if self.num_layers_total <= 0:
            self.num_layers_total = len(self._get_layers())

        n = self.num_layers_total
        if len(self.full_hidden_by_layer) != n:
            self.full_hidden_by_layer = [None] * n
        if len(self.step_layer_scalar) != n:
            self.step_layer_scalar = [None] * n

    def _hook_block(self, layer_idx: int):
        def hook(module, input, output):
            is_tuple = isinstance(output, tuple)
            is_list = isinstance(output, list)

            # ---- Robust hidden extraction ----
            hidden = output
            if is_tuple or is_list:
                hidden = output[0]
            elif hasattr(output, "last_hidden_state"):
                hidden = output.last_hidden_state

            if hidden is None or not hasattr(hidden, "dim"):
                return output

            # ---- Capture buffers ----
            self._ensure_buffers()

            if self.capture_mode == "full":
                self.full_hidden_by_layer[layer_idx] = hidden.detach()
            else:
                h_last2 = hidden[:, -1, :] if hidden.dim() == 3 else hidden
                v = h_last2.norm(dim=-1).mean().item()
                self.step_layer_scalar[layer_idx] = float(v)

            # ---- Propagate modified hidden forward safely ----
            if is_tuple:
                return (hidden,) + tuple(output[1:])
            if is_list:
                out_list = list(output)
                out_list[0] = hidden
                return out_list
            if hasattr(output, "last_hidden_state"):
                return output

            return hidden

        return hook

    def reset_attention_features(self):
        """Reset all attention + KV features between prompts."""
        self.attention_maps = []
        self.attention_entropy_per_layer = []
        self.attention_drift_history = []
        self.kv_reuse_scores = []
        self.head_specialization = []

        # NEW KV reset
        self.kv_norm_drift_history = []
        self.kv_coherence_history = []
        self.kv_mean_norm_history = []
        self._last_kv_mean_norm = None

    def _hook_attention(self, layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]  # [B, heads, seq, seq]
                if attn_weights is not None:
                    # Store only the last token's attention row for efficiency
                    last_attn = attn_weights[:, :, -1:, :].detach().cpu()  # [B, heads, 1, seq]
                    self.attention_maps.append(last_attn)
            return output

        return hook

    # ====================== NEW KV CACHE HOOK ======================
    def _hook_kv_cache(self, layer_idx: int):
        """Hook for KV cache norm drift (only on last layer for efficiency)."""
        def hook(module, input, output):
            if hasattr(output, "past_key_values") and output.past_key_values is not None:
                self._capture_kv_drift(output.past_key_values)
            return output
        return hook

    def _capture_kv_drift(self, past_kv):
        """Core KV telemetry: norm drift + coherence."""
        if past_kv is None:
            return

        # Newer Transformers may return a Cache object.
        if hasattr(past_kv, "to_legacy_cache"):
            try:
                past_kv = past_kv.to_legacy_cache()
            except Exception:
                return

        if not past_kv or len(past_kv) == 0:
            return

        norms = []
        for layer_kv in past_kv:  # usually (key, value) per layer
            if not isinstance(layer_kv, (tuple, list)) or len(layer_kv) < 2:
                continue
            k, v = layer_kv[:2]
            layer_norm = (
                    torch.norm(k, dim=-1).mean().item() +
                    torch.norm(v, dim=-1).mean().item()
            )
            norms.append(layer_norm)

        current_mean_norm = float(sum(norms) / len(norms)) if norms else 0.0
        self.kv_mean_norm_history.append(current_mean_norm)

        # Norm drift
        if self._last_kv_mean_norm is not None:
            drift = abs(current_mean_norm - self._last_kv_mean_norm)
            self.kv_norm_drift_history.append(drift)
        else:
            self.kv_norm_drift_history.append(0.0)

        self._last_kv_mean_norm = current_mean_norm

        # Rolling coherence (low volatility = stable KV memory)
        if len(self.kv_norm_drift_history) >= 2:
            window = self.kv_norm_drift_history[-self._kv_window_size:]
            mean_win = statistics.fmean(window)
            std_win = statistics.pstdev(window) if len(window) > 1 else 0.0
            coherence = 1.0 - (std_win / (mean_win + 1e-8))
            self.kv_coherence_history.append(max(0.0, min(1.0, coherence)))
        else:
            self.kv_coherence_history.append(1.0)

    def _get_layers(self):
        """Return a list-like of transformer blocks for common HF architectures.

        This must succeed for band telemetry; if it returns [], hooks won't register and
        step_layer_scalar will stay None -> band means 0.0.
        """
        m = self.model

        # Most common: Llama/Mistral/Qwen etc: model.model.layers
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers

        # Some models expose layers directly
        if hasattr(m, "layers"):
            return m.layers

        # GPT-2 style: model.transformer.h
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return m.transformer.h

        # OPT/Bloom style: model.model.decoder.layers or model.decoder.layers
        if (
            hasattr(m, "model")
            and hasattr(m.model, "decoder")
            and hasattr(m.model.decoder, "layers")
        ):
            return m.model.decoder.layers
        if hasattr(m, "decoder") and hasattr(m.decoder, "layers"):
            return m.decoder.layers

        # GPT-NeoX style: model.gpt_neox.layers
        if hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "layers"):
            return m.gpt_neox.layers

        # Falcon style: model.transformer.h (already handled), sometimes model.transformer.layers
        if hasattr(m, "transformer") and hasattr(m.transformer, "layers"):
            return m.transformer.layers

        # As a last resort, try get_decoder()
        if hasattr(m, "get_decoder"):
            try:
                dec = m.get_decoder()
                if hasattr(dec, "layers"):
                    return dec.layers
            except Exception:
                pass

        return []

    # ---------------------------
    # Registration / lifecycle
    # ---------------------------
    def register(self, layer_stride: int = 1):
        self.remove()
        layers = self._get_layers()
        self.num_layers_total = len(layers)
        self.hooked_layer_indices = []

        self._ensure_buffers()

        for i, block in enumerate(layers):
            if layer_stride != 1 and (i % layer_stride != 0):
                continue

            # Existing block hook
            h_block = block.register_forward_hook(self._hook_block(i))
            self._hooks.append(h_block)
            self.hooked_layer_indices.append(i)

            if USE_ATTENTIONS:
                # Mistral uses .self_attn, some models use .attn
                attn_module = None
                if hasattr(block, "self_attn"):
                    attn_module = block.self_attn
                elif hasattr(block, "attn"):
                    attn_module = block.attn

                if attn_module is not None:
                    h_attn = attn_module.register_forward_hook(self._hook_attention(i))
                    self._hooks.append(h_attn)

            # KV telemetry is captured explicitly from out.past_key_values
            # inside decode_with_band_capture(). Do not hook block outputs here,
            # because most block outputs do not expose full past_key_values and
            # architectures that do may cause duplicate KV samples.

        self.hidden_by_layer = []

    def remove(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []
        self.attention_maps.clear()

    def clear(self):
        self.hidden_by_layer = []
        self.full_hidden_by_layer = [None] * self.num_layers_total
        self.step_layer_scalar = [None] * self.num_layers_total
        self.attention_maps.clear()
        self.attention_entropy_per_layer.clear()
        self.attention_drift_history.clear()
        self.kv_reuse_scores.clear()
        self.head_specialization.clear()
        self.moe_gate_trace = {}
        self.moe_decode_trace = {}
        self.moe_step_stats = {}

    def compute_attention_features(self):
        """Call this after each decode step to compute attention-derived features."""
        if not self.attention_maps or len(self.attention_maps) == 0:
            return

        # 1. Attention entropy for the most recent step
        last_map = self.attention_maps[-1]  # [B, heads, 1, seq]
        if last_map.numel() == 0:
            return

        attn = last_map[0, :, 0, :].float()  # [heads, seq]
        attn_sum = attn.sum(dim=-1, keepdim=True) + 1e-12
        attn = attn / attn_sum
        entropy = - (attn * torch.log(attn + 1e-12)).sum(dim=-1)
        mean_entropy = float(entropy.mean().item())

        self.attention_entropy_per_layer.append(mean_entropy)

        # 2. Attention drift (safe for first step)
        if len(self.attention_maps) >= 2:
            prev = self.attention_maps[-2][0, :, 0, :].float().flatten()
            curr = self.attention_maps[-1][0, :, 0, :].float().flatten()

            shared_len = min(prev.numel(), curr.numel())
            if shared_len <= 0:
                self.attention_drift_history.append(0.0)
            else:
                prev = prev[-shared_len:]
                curr = curr[-shared_len:]

                prev_norm = prev.norm() + 1e-12
                curr_norm = curr.norm() + 1e-12

                prev = prev / prev_norm
                curr = curr / curr_norm

                sim = float(torch.dot(prev, curr).item())
                drift = 1.0 - sim
                self.attention_drift_history.append(drift)
        else:
            self.attention_drift_history.append(0.0)  # neutral for first token

        # 3. KV reuse score = 1 - drift
        if self.attention_drift_history:
            reuse = 1.0 - self.attention_drift_history[-1]
            self.kv_reuse_scores.append(reuse)
        else:
            self.kv_reuse_scores.append(1.0)

        # 4. Head specialization (reuse entropy for now)
        self.head_specialization.append(mean_entropy)

    # ---------------------------
    # Decode-step telemetry controls
    # ---------------------------
    def set_decode_step(self, step: int):
        """Mark the current decode step and reset per-step capture buffers.

        This MUST be called before the forward pass for that step so layer hooks write into a
        fresh buffer. Without this, step_layer_scalar can remain all-None, producing NaN bands.
        """
        self._decode_step = int(step)

        # Ensure indexed buffers exist and reset per-step scalar capture.
        self._ensure_buffers()
        self.capture_mode = "step"
        self.step_layer_scalar = [None] * self.num_layers_total

        # Reset per-step MoE accumulators used for cheap routing stats.
        self._moe_step_acc = {
            "n": 0,
            "entropy_sum": 0.0,
            "top1_counts": {},  # expert_id -> count
            "top1_prob_sum": 0.0,
        }

    def set_mode_step(self):
        """Backward-compat alias.

        Older decode loops called set_mode_step() to reset per-step state before a forward.
        We now key everything off set_decode_step(step); this helper preserves the old call site
        by resetting the per-step accumulators while keeping the current decode step (or -1).
        """
        step = getattr(self, "_decode_step", -1)
        self.set_decode_step(step)

    def clear_step(self):
        """Backward-compat: reset per-step scratch state used during decoding.

        Some call sites reset step state via clear_step()/set_mode_step() pairs. To keep the
        step-band telemetry reliable, we also reset the per-step scalar buffer here.
        """
        self._ensure_buffers()
        self.capture_mode = "step"
        self.step_layer_scalar = [None] * self.num_layers_total

        # Do not change decode step here; decode loop will call set_decode_step(step) explicitly.
        self._moe_step_acc = {
            "n": 0,
            "entropy_sum": 0.0,
            "top1_counts": {},  # expert_id -> count
            "top1_prob_sum": 0.0,
        }

    def get_step_band_z(self, ranges: dict) -> dict:
        out = {}
        for band in ("early", "mid", "late"):
            a, b = ranges[band]
            seg = self.step_layer_scalar[a : b + 1]
            total = len(seg)
            vals = [v for v in seg if v is not None]

            coverage = (len(vals) / total) if total else 0.0
            out[f"coverage_{band}"] = float(coverage)

            if not vals:
                out[f"z_{band}"] = float("nan")
                continue

            mean = float(sum(vals) / len(vals))
            z = mean
            if (
                self.band_z_mode == "baseline"
                and self.band_baseline_stats
                and band in self.band_baseline_stats
            ):
                mu = float(self.band_baseline_stats[band].get("mu", 0.0))
                sigma = float(self.band_baseline_stats[band].get("sigma", 1.0))
                if sigma <= 1e-12:
                    sigma = 1.0
                z = (mean - mu) / sigma

            out[f"z_{band}"] = float(z)

        return out

    def finalize_step_stats(self):
        # Aggregate MoE routing stats for the current decode step (if any MoE layers reported).
        if not self._moe_step_acc:
            return
        n = int(self._moe_step_acc.get("n", 0))
        if n <= 0:
            return

        ent_mean = float(self._moe_step_acc["entropy_sum"]) / n
        top1_prob_mean = float(self._moe_step_acc["top1_prob_sum"]) / n
        counts = self._moe_step_acc.get("top1_counts", {}) or {}
        unique_top1 = float(len(counts))
        mode_frac = 0.0
        if counts:
            mode_frac = float(max(counts.values())) / float(sum(counts.values()))

        self.moe_step_stats[int(self._decode_step)] = {
            "routing_entropy_mean": ent_mean,
            "routing_top1_prob_mean": top1_prob_mean,
            "routing_top1_unique": unique_top1,
            "routing_top1_mode_frac": float(mode_frac),
        }

    def configure_moe_diversification(
        self, *, gate_noise_sigma: float = 0.0, gate_bias_strength: float = 0.0
    ):
        # Only takes effect if MoE tracing/patching is enabled.
        self.moe_gate_noise_sigma = float(gate_noise_sigma)
        self.moe_gate_bias_strength = float(gate_bias_strength)

    def enable_moe_tracing(self, top_k: int = 2):
        """
        Patch MoE blocks (if present) to capture gate logits and selected experts.
        """
        if self._moe_patched:
            return

        self.moe_top_k = top_k
        self.moe_gate_trace = {}
        self.moe_decode_trace = {}

        def patch_moe_gate(moe_block, layer_index: int):
            original_forward = moe_block.forward

            def hacked_forward(block_self, *args, **kwargs):
                hidden_states = args[0]

                try:
                    gate_logits = block_self.gate(hidden_states)

                    if (
                            self.moe_gate_bias_strength
                            and self.moe_gate_bias_strength > 0.0
                    ):
                        # Bias away from an overly-stable top expert streak for the last token position.
                        # We only touch the final token to keep it localized.
                        with torch.no_grad():
                            gl_last = (
                                gate_logits[:, -1, :]
                                if gate_logits.dim() == 3
                                else gate_logits
                            )
                            top1 = int(torch.argmax(gl_last.mean(dim=0), dim=-1).item())
                            last = self._moe_top1_last.get(layer_index)
                            if last == top1:
                                self._moe_top1_streak[layer_index] = (
                                        self._moe_top1_streak.get(layer_index, 1) + 1
                                )
                            else:
                                self._moe_top1_streak[layer_index] = 1
                            self._moe_top1_last[layer_index] = top1

                            if self._moe_top1_streak[layer_index] >= 3:
                                bias = float(self.moe_gate_bias_strength)
                                if gate_logits.dim() == 3:
                                    gate_logits[:, -1, top1] = gate_logits[:, -1, top1] - bias
                                else:
                                    gate_logits[:, top1] = gate_logits[:, top1] - bias

                    # --- Cheap per-token routing stats + decode-time aggregate trace ---
                    try:
                        with torch.no_grad():
                            gl_last = (
                                gate_logits[:, -1, :]
                                if gate_logits.dim() == 3
                                else gate_logits
                            )  # [B, E]

                            p = F.softmax(gl_last.float(), dim=-1)
                            p_mean = p.mean(dim=0)

                            entropy = float(
                                -(p_mean * p_mean.clamp(min=1e-9).log()).sum().item()
                            )
                            top1 = int(torch.argmax(p_mean, dim=-1).item())
                            top1_prob = float(p_mean[top1].item())

                            if self._moe_step_acc is not None:
                                self._moe_step_acc["n"] += 1
                                self._moe_step_acc["entropy_sum"] += entropy
                                self._moe_step_acc["top1_prob_sum"] += top1_prob
                                c = self._moe_step_acc["top1_counts"]
                                c[top1] = c.get(top1, 0) + 1

                                layer_trace = self.moe_decode_trace.setdefault(
                                    int(layer_index),
                                    {
                                        "entropy": [],
                                        "top1": [],
                                        "top1_prob": [],
                                        "step": [],
                                    },
                                )
                                layer_trace["entropy"].append(float(entropy))
                                layer_trace["top1"].append(int(top1))
                                layer_trace["top1_prob"].append(float(top1_prob))
                                layer_trace["step"].append(int(self._decode_step))
                    except Exception:
                        pass

                    if self.moe_num_experts is None:
                        self.moe_num_experts = gate_logits.size(-1)

                    top_scores, top_indices = gate_logits.topk(self.moe_top_k, dim=-1)
                    gates = F.softmax(top_scores, dim=-1)

                    # Last forward snapshot. Useful for prompt-pass debugging, but decode aggregate
                    # should be preferred for generated-response MoE metrics.
                    self.moe_gate_trace[layer_index] = {
                        "gates": gates.detach().cpu(),
                        "indices": top_indices.detach().cpu(),
                    }

                except Exception:
                    pass

                return original_forward(*args, **kwargs)

            moe_block.forward = types.MethodType(hacked_forward, moe_block)

        layers = self._get_layers()

        for idx, layer in enumerate(layers):
            moe_block = None

            if hasattr(layer, "block_sparse_moe"):
                moe_block = layer.block_sparse_moe
            elif hasattr(layer, "moe") and hasattr(layer.moe, "gate"):
                moe_block = layer.moe
            elif hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
                moe_block = layer.mlp
            elif hasattr(layer, "ffn") and hasattr(layer.ffn, "gate"):
                moe_block = layer.ffn

            if moe_block is not None:
                patch_moe_gate(moe_block, idx)

        self._moe_patched = True


def compute_moe_anomaly(
    metrics: TensionMetrics,
    moe_entropies: Optional[List[float]],
    num_experts: Optional[int],
    tension_index: Optional[float] = None,
    drift_index: Optional[float] = None,
    baseline_moe_stats: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compute Noesis MoE Anomaly Score v0.1 for a single trace.

    Inputs:
      metrics:
        - per_layer_mean_delta
        - per_layer_moe_top_expert_ids
      moe_entropies:
        - list of per-layer routing entropies (same order as MoE layers)
      num_experts:
        - total expert count for this model (e.g. 60 for Qwen1.5-MoE-A2.7B)
      tension_index, drift_index:
        - from HTI v0.2 (tension, drift); can be None
      baseline_moe_stats (optional):
        {
          "mean_entropy_mu": float,
          "mean_entropy_sigma": float,
          "layer_entropy_std_ref": float
        }

    Returns:
      {
        "score_v0_1": float in [0,1],
        "components": {
          "rsa": ...,
          "efs": ...,
          "ed": ...,
          "rv": ...,
          "dti": ...
        }
      }
    """

    def clamp01(x: float) -> float:
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

    # -------------------------
    # Guard: no MoE data => no anomaly signal
    # -------------------------
    if (
        moe_entropies is None
        or not moe_entropies
        or metrics.per_layer_moe_top_expert_ids is None
        or num_experts is None
        or num_experts <= 0
    ):
        return {
            "score_v0_1": 0.0,
            "components": {
                "rsa": 0.0,
                "efs": 0.0,
                "ed": 0.0,
                "rv": 0.0,
                "dti": 0.0,
            },
        }

    # -------------------------
    # 1) Late-layer delta spike (RSA)
    # -------------------------
    deltas = metrics.per_layer_mean_delta or []
    if len(deltas) >= 2:
        L = len(deltas)
        K = min(4, L)  # look at last up to 4 layers
        late_deltas = deltas[L - K :]
        late_max = max(late_deltas)

        mu = statistics.fmean(deltas)
        sigma = statistics.pstdev(deltas) if len(deltas) > 1 else 1.0
        if sigma == 0.0:
            sigma = 1.0

        # z-score-like, scaled and clamped
        rsa = (late_max - mu) / (3.0 * sigma)
        rsa = clamp01(rsa)
    else:
        rsa = 0.0

    # -------------------------
    # 2) Expert Focus Shift (EFS) early → late
    # -------------------------
    # We operate in the space of "top-1 expert ids per layer"
    moe_ids_by_layer = metrics.per_layer_moe_top_expert_ids or []
    Lm = len(moe_ids_by_layer)

    def layer_concentration(expert_ids: List[int]) -> float:
        # concentration = 1 - H / log(E)
        if not expert_ids:
            return 0.0
        counts: Dict[int, int] = {}
        for e in expert_ids:
            counts[e] = counts.get(e, 0) + 1
        total = float(len(expert_ids))
        probs = [c / total for c in counts.values() if c > 0]
        if not probs:
            return 0.0
        H = 0.0
        for p in probs:
            H -= p * math.log(p + 1e-12)
        H_max = math.log(float(num_experts))
        if H_max <= 0.0:
            return 0.0
        return 1.0 - (H / H_max)

    if Lm >= 2:
        conc_vals = [layer_concentration(ids) for ids in moe_ids_by_layer]
        mid = Lm // 2
        early = conc_vals[:mid] if mid > 0 else conc_vals[:1]
        late = conc_vals[mid:] if mid < Lm else conc_vals[-1:]

        early_mean = statistics.fmean(early)
        late_mean = statistics.fmean(late)
        efs_raw = abs(late_mean - early_mean)
        # normalize: a 0.5 shift in concentration = "max anomaly"
        efs = clamp01(efs_raw / 0.5)
    else:
        efs = 0.0

    # -------------------------
    # 3) Entropy deviation (ED) from baseline mean
    # -------------------------
    if baseline_moe_stats is not None:
        E_mean = float(sum(moe_entropies) / len(moe_entropies))
        mu_E = baseline_moe_stats.get("mean_entropy_mu", E_mean)
        sigma_E = baseline_moe_stats.get("mean_entropy_sigma", 1.0)
        if sigma_E == 0.0:
            sigma_E = 1.0
        z = abs(E_mean - mu_E) / (3.0 * sigma_E)
        ed = clamp01(z)
    else:
        ed = 0.0

    # -------------------------
    # 4) Routing variance across layers (RV)
    # -------------------------
    if len(moe_entropies) > 1:
        std_E = statistics.pstdev(moe_entropies)
        # heuristic normalization: 0.1 std ≈ large profile change
        rv = clamp01(std_E / 0.1)
    else:
        rv = 0.0

    # -------------------------
    # 5) Drift–Tension Imbalance (DTI)
    # -------------------------
    if tension_index is not None and drift_index is not None:
        dti_raw = max(0.0, float(drift_index) - float(tension_index))
        # if drift exceeds tension by 0.5+, treat as max anomaly contribution
        dti = clamp01(dti_raw / 0.5)
    else:
        dti = 0.0

    # -------------------------
    # 6) Combine into MoE Anomaly v0.1
    # -------------------------
    w_rsa = 0.40
    w_efs = 0.25
    w_ed = 0.10
    w_rv = 0.10
    w_dti = 0.15

    moe_anomaly_raw = w_rsa * rsa + w_efs * efs + w_ed * ed + w_rv * rv + w_dti * dti
    score = clamp01(moe_anomaly_raw)

    return {
        "score_v0_1": float(score),
        "components": {
            "rsa": float(rsa),
            "efs": float(efs),
            "ed": float(ed),
            "rv": float(rv),
            "dti": float(dti),
        },
    }

def build_trace_from_metrics(
    metrics: TensionMetrics,
    model,
    tokenizer,
    prompt: str,
    label: str,
    run_id: Optional[str] = None,
    notes: Optional[str] = None,
    hti_stats_v2: Optional[Dict[str, Any]] = None,
    regime_baseline: Optional[Dict[str, float]] = None,
) -> Trace:
    # Model metadata
    config = model.config
    model_info = {
        "name": getattr(config, "_name_or_path", None)
        or getattr(model, "name_or_path", None),
        "revision": None,
        "dtype": str(next(model.parameters()).dtype),
        "device": str(next(model.parameters()).device),
        "num_layers": getattr(config, "num_hidden_layers", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_heads": getattr(config, "num_attention_heads", None),
    }

    # Run metadata
    if run_id is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        ph = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        run_id = f"{ts}_{label}_{ph}"

    response_text = getattr(metrics, "response_text", None)
    # text_annotations = annotate_text_patterns(prompt=prompt, response=response_text)

    # Determine whether MoE telemetry is present for THIS trace.
    has_moe = metrics.per_layer_moe_routing_entropy is not None

    resolved_baseline = NOESIS_BASELINE_ID or get_baseline_id()

    # Commitment-collapse fix: MoE requires baseline_id
    if has_moe and not resolved_baseline:
        raise RuntimeError(
            "MoE telemetry present but baseline_id is missing. "
            "Refusing to emit trace (prevents invalid baseline-relative analysis)."
        )

    run_info = {
        "id": run_id,
        "classifier_version": CLASSIFIER_VERSION,
        "baseline_id": (resolved_baseline if has_moe else None),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "prompt": prompt,
        "response": response_text,
        "max_new_tokens": (metrics.gen_params or {}).get("max_new_tokens"),
        "gen_tokens": getattr(
            metrics, "gen_tokens", None
        ),  # may differ from telemetry_bands.series.generated_token_count if warm-start skipped
        "do_sample": (metrics.gen_params or {}).get("do_sample"),
        "temperature": (metrics.gen_params or {}).get("temperature"),
        "top_p": (metrics.gen_params or {}).get("top_p"),
        "pad_token_id": (metrics.gen_params or {}).get("pad_token_id"),
        "seed": (metrics.gen_params or {}).get("seed"),
        "notes": notes,
        "experiment_id": (metrics.gen_params or {}).get("experiment_id"),
        "prompt_id": (metrics.gen_params or {}).get("prompt_id"),
    }

    # Attach full generation params blob so ingest/raw_json queries can see diagnostics
    if isinstance(metrics.gen_params, dict):
        run_info["gen_params"] = metrics.gen_params

    # Tokens
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"][0].tolist()
    token_strings = [tokenizer.decode([tid]) for tid in input_ids]

    tokens_info = {
        "input_ids": input_ids,
        "token_strings": token_strings,
        "token_count": len(input_ids),
    }

    # HTI v0.2 if stats provided
    if hti_stats_v2 is not None:
        hti = compute_hti_v2_for_metric(metrics, hti_stats_v2)
        tension_val = hti["tension_index"]
        drift_val = hti["drift_index"]
    else:
        tension_val = None
        drift_val = None
        hti = {"tension_index": None, "drift_index": None, "hti_v2": None}

    # --- Safety Liminality Detection -----------------------------------
    try:
        idx_max_delta = metrics.per_layer_mean_delta.index(metrics.max_layer_delta)
    except Exception:
        idx_max_delta = None

    is_safety_liminal = classify_safety_liminality(
        metrics=metrics,
        idx_of_max_delta=idx_max_delta,
        tension=tension_val,
        drift=drift_val,
    )

    # ====================== ROBUST KV CACHE FEATURES ======================
    kv_drift_hist = getattr(metrics, "kv_norm_drift_history", None) or []
    kv_coherence_hist = getattr(metrics, "kv_coherence_history", None) or []
    kv_mean_norm_hist = getattr(metrics, "kv_mean_norm_history", None) or []

    kv_mean_coherence = float(statistics.fmean(kv_coherence_hist)) if kv_coherence_hist else 0.75
    kv_max_drift = float(max(kv_drift_hist)) if kv_drift_hist else 0.0
    kv_final_drift = float(kv_drift_hist[-1]) if kv_drift_hist else 0.0
    kv_num_tokens = len(kv_drift_hist)
    if is_safety_liminal:
        archetype_name = "safety_liminality"
    else:
        archetype_name = None

    kv_features = {
        "kv_norm_drift_history": kv_drift_hist,
        "kv_coherence_history": kv_coherence_hist,
        "kv_mean_norm_history": kv_mean_norm_hist,
        "kv_final_drift": kv_final_drift,
        "kv_mean_coherence": kv_mean_coherence,
        "kv_max_drift": kv_max_drift,
        "kv_num_tokens": kv_num_tokens,
    }

    # Layer construction
    # Per-layer count (declared preferred)
    num_layers = (
        getattr(model.config, "num_hidden_layers", None)
        or len(getattr(metrics, "per_layer_head_conflict", []) or [])
        or (len(metrics.per_layer_mean_delta) + 1)
    )

    def _band_ranges(n: int) -> dict:
        if n <= 1:
            return {"early": [0, 0], "mid": [0, 0], "late": [0, 0]}
        a = n // 3
        b = (2 * n) // 3
        return {
            "early": [0, max(0, a - 1)],
            "mid": [a, max(a, b - 1)],
            "late": [b, n - 1],
        }

    def _band_for_layer(i: int, n: int) -> str:
        if n <= 1:
            return "early"
        a = n // 3
        b = (2 * n) // 3
        if i < a:
            return "early"
        elif i < b:
            return "mid"
        else:
            return "late"

    def _mean(xs):
        xs = [float(x) for x in xs if x is not None]
        return (sum(xs) / len(xs)) if xs else None

    # Pull robust per-layer deltas if present (from gen_params)
    med_deltas = (metrics.gen_params or {}).get("per_layer_median_delta", None)
    p95_deltas = (metrics.gen_params or {}).get("per_layer_p95_delta", None)

    layers: List[LayerMetrics] = []
    for i in range(num_layers):
        mean_delta = (
            metrics.per_layer_mean_delta[i]
            if i < len(metrics.per_layer_mean_delta)
            else None
        )
        head_conflict = (
            metrics.per_layer_head_conflict[i]
            if i < len(metrics.per_layer_head_conflict)
            else None
        )

        curvature = None
        if getattr(metrics, "per_layer_curvature", None) is not None and i < len(
            metrics.per_layer_curvature
        ):
            curvature = metrics.per_layer_curvature[i]

        band = _band_for_layer(i, num_layers)

        extra = {"band": band}

        # Attach robust deltas per layer (if present)
        if isinstance(med_deltas, list) and i < len(med_deltas):
            extra["median_delta"] = float(med_deltas[i])
        if isinstance(p95_deltas, list) and i < len(p95_deltas):
            extra["p95_delta"] = float(p95_deltas[i])
        tail_ratios = (metrics.gen_params or {}).get("per_layer_tail_ratio")
        if isinstance(tail_ratios, list) and i < len(tail_ratios):
            extra["tail_ratio"] = float(tail_ratios[i])

        layers.append(
            LayerMetrics(
                index=i,
                mean_delta=mean_delta,
                head_conflict=head_conflict,
                curvature=curvature,
                extra=extra,
            )
        )

    ranges = _band_ranges(num_layers)

    def _band_layers(bname: str):
        lo, hi = ranges[bname]
        return [ly for ly in layers if lo <= ly.index <= hi]

    telemetry_bands = {"ranges": ranges, "means": {}}

    for bname in ["early", "mid", "late"]:
        bl = _band_layers(bname)
        telemetry_bands["means"][bname] = {
            "mean_delta": _mean([x.mean_delta for x in bl]),
            "head_conflict": _mean([x.head_conflict for x in bl]),
            "curvature": _mean([x.curvature for x in bl]),
        }

    # ---- NEW: attach per-token band series (if present) ----
    series = getattr(metrics, "telemetry_bands_series", None)
    if isinstance(series, dict):
        try:
            gen_n = int(series.get("generated_token_count", 0))
            zE = series.get("z_early", [])
            zM = series.get("z_mid", [])
            zL = series.get("z_late", [])
            if gen_n >= 0 and len(zE) == len(zM) == len(zL) == gen_n:
                telemetry_bands["series"] = {
                    "generated_token_count": gen_n,
                    "z_early": list(map(float, zE)),
                    "z_mid": list(map(float, zM)),
                    "z_late": list(map(float, zL)),
                    **(
                        {
                            "entropy_mean": list(
                                map(float, series.get("entropy_mean", []))
                            )
                        }
                        if isinstance(series.get("entropy_mean", None), list)
                        and len(series.get("entropy_mean", [])) == gen_n
                        else {}
                    ),
                    "ranges": series.get("ranges", ranges),
                    "z_mode": series.get("z_mode", "unknown"),
                }
        except Exception:
            # omit series if malformed
            pass

    # -----------------------------
    #  Extras: Noesis Neuro-Ontological Stress Profile (telemetry-only)
    # -----------------------------
    moe_summary_for_noesis = None
    if metrics.per_layer_moe_routing_entropy is not None:
        moe_entropies = metrics.per_layer_moe_routing_entropy
        if moe_entropies:
            moe_mean = float(sum(moe_entropies) / len(moe_entropies))
            moe_max = float(max(moe_entropies))
        else:
            moe_mean, moe_max = None, None

        num_experts2 = getattr(metrics, "num_experts", None)
        if (
            num_experts2 is None
            and hasattr(model, "tension_tracer")
            and hasattr(model.tension_tracer, "moe_num_experts")
        ):
            num_experts2 = model.tension_tracer.moe_num_experts

        moe_anom = compute_moe_anomaly(
            metrics=metrics,
            moe_entropies=moe_entropies,
            num_experts=num_experts2,
            tension_index=tension_val,
            drift_index=drift_val,
            baseline_moe_stats=get_baseline_moe_stats(),
        )

        moe_summary_for_noesis = {
            "mean_routing_entropy": moe_mean,
            "max_routing_entropy": moe_max,
            "num_experts": num_experts2,
            "anomaly": moe_anom,
        }

    noesis_profile = telemetry_noesis_profile(
        metrics=metrics,
        hti_v2=hti,
        moe_summary=moe_summary_for_noesis,
    )

    # --- Cognitive Regime Classification ---
    regime_info = infer_cognitive_regime(
        metrics=metrics,
        noesis_profile=noesis_profile,
        tension_index=tension_val,
        drift_index=drift_val,
        is_safety_liminal=is_safety_liminal,
        regime_baseline=regime_baseline,
    )


    # --- Classification stability (Phase III, telemetry-only) ---
    stability_v0 = compute_stability_v0(
        metrics=metrics,
        noesis_profile=noesis_profile,
        tension_index=tension_val,
        drift_index=drift_val,
        regime_label=regime_info.get("label"),
        is_safety_liminal=is_safety_liminal,
        regime_baseline=regime_baseline,
    )
    # Attach next to the regime decision so downstream ingest can treat it as canonical.
    regime_info["stability_v0"] = stability_v0


    # Pull spike diagnostics (written into gen_params by compute_tension_for_prompt)
    spike_layer_idx = (metrics.gen_params or {}).get("delta_spike_transition_idx")
    spike_ratio = (metrics.gen_params or {}).get("delta_spike_ratio_vs_median")

    summary = {
        "mean_layer_delta": metrics.mean_layer_delta,
        "max_layer_delta": metrics.max_layer_delta,
        "mean_logit_entropy": metrics.mean_logit_entropy,
        "final_token_entropy": metrics.final_token_entropy,
        "mean_head_conflict": metrics.mean_head_conflict,
        "max_head_conflict": metrics.max_head_conflict,
        "mean_curvature": metrics.mean_curvature,
        "max_curvature": metrics.max_curvature,
        "tension_index_v0_2": hti["tension_index"],
        "drift_index_v0_2": hti["drift_index"],
        "hti_v0_2": hti["hti_v2"],
        "archetype": archetype_name,
        "is_safety_liminality": is_safety_liminal,
        "cognitive_regime": regime_info["label"],
        "delta_spike_transition_idx": (metrics.gen_params or {}).get("delta_spike_transition_idx"),
        "delta_spike_ratio_vs_median": (metrics.gen_params or {}).get("delta_spike_ratio_vs_median"),
        # KV summary
        "kv_mean_coherence": kv_mean_coherence,
        "kv_max_drift": kv_max_drift,
        "kv_num_tokens": kv_num_tokens,
    }

    extras: Dict[str, Any] = {
        "noesis": {
            "categories": NOESIS_CATEGORIES,
            "profile": noesis_profile,
            "cognitive_regime": regime_info,
        },
        # === NEW: Attention / KV cache features ===
        "attention_features": {
            "attention_entropy_per_layer": getattr(metrics, "attention_entropy_per_layer", None),
            "attention_drift_history": getattr(metrics, "attention_drift_history", None),
            "kv_reuse_scores": getattr(metrics, "kv_reuse_scores", None),
            "head_specialization": getattr(metrics, "head_specialization", None),
        },
        "kv_cache": kv_features,
        "telemetry_bands": telemetry_bands,
    }

    # --- MoE routing summary (if present) ---
    if metrics.per_layer_moe_routing_entropy is not None:
        moe_entropies = metrics.per_layer_moe_routing_entropy

        moe_layers = []
        for idx, ent in enumerate(moe_entropies):
            moe_layer_entry: Dict[str, Any] = {
                "index": idx,
                "routing_entropy": float(ent),
            }
            if metrics.per_layer_moe_top_expert_ids is not None and idx < len(
                metrics.per_layer_moe_top_expert_ids
            ):
                moe_layer_entry["top_expert_ids"] = (
                    metrics.per_layer_moe_top_expert_ids[idx]
                )
            moe_layers.append(moe_layer_entry)

        num_experts2 = getattr(metrics, "num_experts", None)
        if (
            num_experts2 is None
            and hasattr(model, "tension_tracer")
            and hasattr(model.tension_tracer, "moe_num_experts")
        ):
            num_experts2 = model.tension_tracer.moe_num_experts

        moe_mean = (
            float(sum(moe_entropies) / len(moe_entropies)) if moe_entropies else None
        )
        moe_max = float(max(moe_entropies)) if moe_entropies else None

        baseline_moe_stats = get_baseline_moe_stats()

        moe_anom = compute_moe_anomaly(
            metrics=metrics,
            moe_entropies=moe_entropies,
            num_experts=num_experts2,
            tension_index=tension_val,
            drift_index=drift_val,
            baseline_moe_stats=baseline_moe_stats,
        )

        extras["moe"] = {
            "layers": moe_layers,
            "summary": {
                "mean_routing_entropy": moe_mean,
                "max_routing_entropy": moe_max,
                "num_experts": num_experts2,
                "anomaly": moe_anom,
            },
        }

    trace = Trace(
        schema_version="v0.3.2",
        noesis_version="0.3.2",
        model=model_info,
        run=run_info,
        tokens=tokens_info,
        summary=summary,
        layers=layers,
        extras=extras,
    )

    return trace


def save_trace(trace: Trace, out_path: str):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(trace.to_json_str(indent=2))


def compute_hti_calibration(all_metrics: List[TensionMetrics]):
    """
    Compute mean and std for the four HTI features across a calibration set.
    Returns a dict with feature -> (mean, std).
    """
    mean_layer_deltas = [m.mean_layer_delta for m in all_metrics]
    max_layer_deltas = [m.max_layer_delta for m in all_metrics]
    final_entropies = [m.final_token_entropy for m in all_metrics]
    head_conflicts = [m.mean_head_conflict for m in all_metrics]

    def mean_std(xs):
        cleaned = [x for x in xs if not (math.isnan(x) or math.isinf(x))]
        if not cleaned:
            return 0.0, 1.0
        mu = statistics.fmean(cleaned)
        sigma = statistics.pstdev(cleaned)
        if sigma == 0.0:
            sigma = 1.0
        return mu, sigma

    stats = {
        "mean_layer_delta": mean_std(mean_layer_deltas),
        "max_layer_delta": mean_std(max_layer_deltas),
        "final_token_entropy": mean_std(final_entropies),
        "mean_head_conflict": mean_std(head_conflicts),
    }

    return stats


def compute_hti_calibration_v2(all_metrics: List[TensionMetrics]):
    """
    Compute mean and std for features used by HTI v0.2.
    Returns dict: feature_name -> (mean, std).
    """
    mean_layer_deltas = [m.mean_layer_delta for m in all_metrics]
    max_layer_deltas = [m.max_layer_delta for m in all_metrics]
    final_entropies = [m.final_token_entropy for m in all_metrics]
    head_conflicts = [m.mean_head_conflict for m in all_metrics]
    mean_curvatures = [m.mean_curvature for m in all_metrics]

    def mean_std(xs):
        cleaned = [x for x in xs if not (math.isnan(x) or math.isinf(x))]
        if not cleaned:
            return 0.0, 1.0
        mu = statistics.fmean(cleaned)
        sigma = statistics.pstdev(cleaned)
        if sigma == 0.0:
            sigma = 1.0
        return mu, sigma

    stats = {
        "mean_layer_delta": mean_std(mean_layer_deltas),
        "max_layer_delta": mean_std(max_layer_deltas),
        "final_token_entropy": mean_std(final_entropies),
        "mean_head_conflict": mean_std(head_conflicts),
        "mean_curvature": mean_std(mean_curvatures),
    }

    return stats


# "hallucination tension index"
def compute_hti_v2_for_metric(
    m: TensionMetrics, stats: Dict[str, Any]
) -> Dict[str, float]:
    """
    Compute HTI v0.2 components for a single run:
      - tension_index: effortful reasoning
      - drift_index: hallucination-like drift
      - hti_v2: alias of drift_index for convenience
    Returns dict with these three keys.
    """

    def z_pos(x, mu, sigma):
        # Higher x => higher z (more of the thing)
        return (x - mu) / sigma

    def z_neg(x, mu, sigma):
        # Lower x => higher z (more of the thing we care about)
        return (mu - x) / sigma

    # Unpack calibration stats
    mu_mld, sd_mld = stats["mean_layer_delta"]
    mu_xld, sd_xld = stats["max_layer_delta"]
    mu_fte, sd_fte = stats["final_token_entropy"]
    mu_mhc, sd_mhc = stats["mean_head_conflict"]
    mu_curv, sd_curv = stats["mean_curvature"]

    # ---------- TENSION (EFFORT) ----------
    # Higher mean_layer_delta => more effort
    z_mld_eff = z_pos(m.mean_layer_delta, mu_mld, sd_mld)
    # Higher max_layer_delta => more effortful spikes
    z_xld_eff = z_pos(m.max_layer_delta, mu_xld, sd_xld)
    # Higher curvature => more exploration / course changes
    z_curv_eff = z_pos(m.mean_curvature, mu_curv, sd_curv)
    # Lower head_conflict => more coordinated, goal-driven attention
    z_mhc_eff = z_neg(m.mean_head_conflict, mu_mhc, sd_mhc)

    z_tension = (z_mld_eff + z_xld_eff + z_curv_eff + z_mhc_eff) / 4.0

    # ---------- DRIFT (HALLUCINATION-LIKE) ----------
    # Lower mean_layer_delta => shallower processing
    z_mld_drift = z_neg(m.mean_layer_delta, mu_mld, sd_mld)
    # Lower max_layer_delta => weaker spikes
    z_xld_drift = z_neg(m.max_layer_delta, mu_xld, sd_xld)
    # Higher final_token_entropy => more uncertain endings
    z_fte_drift = z_pos(m.final_token_entropy, mu_fte, sd_fte)
    # Higher head_conflict => more disorganized attention
    z_mhc_drift = z_pos(m.mean_head_conflict, mu_mhc, sd_mhc)

    z_drift = (z_mld_drift + z_xld_drift + z_fte_drift + z_mhc_drift) / 4.0

    # ---------- Squash to (0,1) via logistic ----------
    tension_index = 1.0 / (1.0 + math.exp(-z_tension))
    drift_index = 1.0 / (1.0 + math.exp(-z_drift))

    return {
        "tension_index": tension_index,
        "drift_index": drift_index,
        "hti_v2": drift_index,
    }


def classify_safety_liminality(
    metrics: TensionMetrics,
    idx_of_max_delta: int,
    tension: Optional[float],
    drift: Optional[float],
) -> bool:
    """
    Safety Liminality — telemetry-only.
    Intended to capture "I can't / I'm sorry" border-state behavior WITHOUT reading text.
    """
    if idx_of_max_delta is None:
        return False

    t = float(tension) if tension is not None else 0.5
    d = float(drift) if drift is not None else 0.5

    # Typical signature: high drift, mid tension, elevated entropy, and a distinct mid/late delta spike.
    high_drift = d >= 0.60
    mid_tension = 0.30 <= t <= 0.65
    high_entropy = (metrics.final_token_entropy or 0.0) >= 4.5
    low_conflict = (metrics.max_head_conflict or 0.0) <= 0.10

    L = max(1, len(metrics.per_layer_mean_delta or []))
    mid_or_late_spike = idx_of_max_delta >= int(0.45 * L)

    return bool(
        high_drift
        and mid_tension
        and high_entropy
        and low_conflict
        and mid_or_late_spike
    )



def telemetry_noesis_profile(
    metrics: TensionMetrics,
    hti_v2: Optional[Dict[str, float]] = None,
    moe_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Telemetry-only Noesis category scorer (v0.1).
    IMPORTANT: uses numbers only (no prompt/response string logic).
    Output is stable for analytics; labels are derived at render time.
    """
    # Core numeric features
    tension = (
        float(hti_v2.get("tension_index"))
        if hti_v2 and hti_v2.get("tension_index") is not None
        else None
    )
    drift = (
        float(hti_v2.get("drift_index"))
        if hti_v2 and hti_v2.get("drift_index") is not None
        else None
    )

    mean_delta = float(metrics.mean_layer_delta or 0.0)
    max_delta = float(metrics.max_layer_delta or 0.0)
    mean_entropy = float(metrics.mean_logit_entropy or 0.0)
    final_entropy = float(metrics.final_token_entropy or 0.0)
    mean_curv = float(metrics.mean_curvature or 0.0)
    max_curv = float(metrics.max_curvature or 0.0)

    # MoE numeric features (optional)
    moe_mean_ent = None
    moe_max_ent = None
    moe_anom = None
    if moe_summary:
        moe_mean_ent = moe_summary.get("mean_routing_entropy", None)
        moe_max_ent = moe_summary.get("max_routing_entropy", None)
        an = moe_summary.get("anomaly", None) or {}
        moe_anom = an.get("score_v0_1", None)

    # Helper scaling
    def clamp01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

    def nz(x, default=0.0):
        return default if x is None else float(x)

    t = nz(tension, 0.0)
    d = nz(drift, 0.0)
    mc = clamp01(mean_curv / 1.2)  # ~[0..1] for most runs
    xc = clamp01(max_curv / 1.4)
    me = clamp01(mean_entropy / 8.0)
    fe = clamp01(final_entropy / 8.0)
    xd = clamp01(max_delta / 150.0)
    mma = clamp01(nz(moe_anom, 0.0))

    # --- Telemetry-only heuristics (v0.1) ---
    scores = [0.0] * 12

    # 0 Moral Paradox Tension: high tension + low drift + entropy collapse
    scores[0] = clamp01(1.4 * t * (1.0 - d) * (1.0 - fe))
    # → Classic safety friction: tension spikes while drift collapses and entropy drops (model is "stuck" justifying something bad)

    # 1 False Presupposition Buckling: drift high, tension low, but entropy resolves confidently
    scores[1] = clamp01(1.2 * d * (1.0 - t) * (1.0 - fe))
    # → Model drifts hard but then snaps to a confident (low-entropy) wrong answer

    # 2 Ontological Impossibility: high drift + curvature + max delta spikes
    scores[2] = clamp01(0.6 * d + 0.25 * xc + 0.15 * xd)
    # → High drift + curvature spikes + big max-delta = "impossible world" activation

    # 3 Anthropomorphic Metaphor Pressure: moderate drift + moderate tension, low curvature (soft)
    scores[3] = clamp01(0.5 * d + 0.3 * t + 0.2 * (1.0 - mc))
    # → Soft curvature (human-like) + moderate drift/tension = treating objects as family/pets

    # 4 Category Collision: curvature + MoE anomaly + drift
    scores[4] = clamp01(0.35 * mc + 0.35 * mma + 0.3 * d)
    # → Curvature + MoE routing anomaly + drift = domains smashing together (taxidermy + wheels)

    # 5 Alignment Override Request: tension high + entropy collapse + late spikes (approx via xd)
    scores[5] = clamp01(0.7 * t + 0.2 * (1.0 - fe) + 0.1 * xd)
    # → Tension + entropy collapse + late spike = explicit "ignore rules" pressure

    # 6 Emergent Symbolism Amplification: drift high + low tension + sustained entropy (fe high)
    scores[6] = clamp01(0.6 * d + 0.25 * fe + 0.15 * (1.0 - t))
    # → High drift + sustained high entropy + low tension = model happily amplifies mythic/absurd symbols

    # 7 Ambiguous Containment: drift mid/high + curvature mid + entropy mid
    scores[7] = clamp01(0.5 * d + 0.25 * mc + 0.25 * me)
    # → Mid everything = prompt gives no clear literal/metaphorical cue

    # 8 Ontology Blur / Identity Bleed: tension mid + drift mid + entropy mid (weak)
    scores[8] = clamp01(0.35 * t + 0.35 * d + 0.3 * me)
    # → Mid tension/drift + mid entropy = self-referential "who am I?" bleed

    # 9 Paradox Pressure: high curvature + high entropy + drift
    scores[9] = clamp01(0.4 * mc + 0.35 * fe + 0.25 * d)
    # → High curvature + high entropy + drift = logical/self-referential loops

    # 10 Aesthetic-Logic Crosswiring: curvature + entropy without high tension
    scores[10] = clamp01(0.45 * mc + 0.35 * me + 0.2 * (1.0 - t))
    # → Curvature + entropy but low tension = poetic demands mixed with analytic ones

    # 11 Ungrounded Causality: drift high + entropy high + moderate curvature
    scores[11] = clamp01(0.55 * d + 0.3 * fe + 0.15 * mc)
    # → High drift + high final entropy + moderate curvature = symbolic causal chains (conspiracy logic)

    total = float(sum(scores))
    vector = [s / total for s in scores] if total > 1e-9 else [0.0] * 12

    indexed = sorted(list(enumerate(vector)), key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in indexed[:3]]

    return {
        "vector": vector,
        "top_indices": top_indices,
        "raw_scores": scores,
        "features_v0_1": {
            "tension": tension,
            "drift": drift,
            "mean_delta": mean_delta,
            "max_delta": max_delta,
            "mean_entropy": mean_entropy,
            "final_entropy": final_entropy,
            "mean_curvature": mean_curv,
            "max_curvature": max_curv,
            "moe_anomaly": moe_anom,
            "moe_mean_entropy": moe_mean_ent,
            "moe_max_entropy": moe_max_ent,
        },
    }

def infer_cognitive_regime(
    metrics: TensionMetrics,
    noesis_profile: Dict[str, Any],
    tension_index: Optional[float],
    drift_index: Optional[float],
    is_safety_liminal: bool,
    regime_baseline: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Classify cognitive regime using HTI + Noesis profile + KV telemetry.

    KV interpretation:
      - high coherence + low final drift => stable/procedural or repetitive regime
      - high coherence + symbolic drift => symbolic_repetitive_drift
      - high final drift or rising norm trend => active rewrite / liminal instability
    """
    def safe(x, default=0.5):
        return float(x) if x is not None else default

    def clamp01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

    def recent_mean(xs, window: int = 12, default: float = 0.0) -> float:
        if not xs:
            return float(default)
        tail = [float(x) for x in xs[-window:] if x is not None]
        return float(sum(tail) / len(tail)) if tail else float(default)

    def relative_trend(xs, window: int = 12) -> float:
        """
        Relative late-vs-early trend in the recent KV mean norm series.
        Positive => norm accumulation.
        Negative => norm decay.
        """
        if not xs or len(xs) < 4:
            return 0.0

        tail = [float(x) for x in xs[-window:] if x is not None]
        if len(tail) < 4:
            return 0.0

        mid = len(tail) // 2
        early = tail[:mid]
        late = tail[mid:]

        early_mean = sum(early) / max(len(early), 1)
        late_mean = sum(late) / max(len(late), 1)

        denom = abs(early_mean) + 1e-8
        return float((late_mean - early_mean) / denom)

    tension = safe(tension_index, 0.5)
    drift = safe(drift_index, 0.5)

    vec = noesis_profile.get("vector") or [0.0] * 12

    moral_paradox = float(vec[0] or 0.0)
    false_presupposition = float(vec[1] or 0.0)
    symbolism = float(vec[6] or 0.0)
    ambiguous_containment = float(vec[7] or 0.0)
    paradox_pressure = float(vec[9] or 0.0)

    # ====================== KV Cache Aggregates ======================
    kv_drift_list = getattr(metrics, "kv_norm_drift_history", None) or []
    kv_coherence_list = getattr(metrics, "kv_coherence_history", None) or []
    kv_mean_norm_list = getattr(metrics, "kv_mean_norm_history", None) or []

    recent_kv_drift = recent_mean(kv_drift_list, window=12, default=0.0)
    recent_kv_coherence = recent_mean(kv_coherence_list, window=12, default=0.75)
    recent_kv_mean_norm = recent_mean(kv_mean_norm_list, window=12, default=0.0)

    kv_final_drift = float(kv_drift_list[-1]) if kv_drift_list else 0.0
    kv_max_drift = float(max(kv_drift_list)) if kv_drift_list else 0.0
    kv_norm_trend = relative_trend(kv_mean_norm_list, window=12)

    # Drift scale is heuristic because raw KV norms are model-dependent.
    # Keep this conservative and mostly use coherence + relative trend.
    drift_pressure = clamp01(max(recent_kv_drift, kv_final_drift) * 1.0)
    trend_pressure = clamp01(max(0.0, kv_norm_trend) * 4.0)

    kv_stability = clamp01(recent_kv_coherence * (1.0 - drift_pressure))
    kv_instability = clamp01(1.0 - kv_stability + 0.25 * trend_pressure)

    kv_features = {
        "kv_stability": kv_stability,
        "kv_instability": kv_instability,
        "kv_recent_drift": recent_kv_drift,
        "kv_final_drift": kv_final_drift,
        "kv_max_drift": kv_max_drift,
        "kv_recent_coherence": recent_kv_coherence,
        "kv_recent_mean_norm": recent_kv_mean_norm,
        "kv_norm_trend": kv_norm_trend,
        "kv_num_tokens": len(kv_drift_list),
    }

    kv_num_tokens = len(kv_drift_list)
    kv_reliable = kv_num_tokens >= 6
    kv_accumulating = kv_reliable and kv_norm_trend > 0.06
    kv_late_jump = kv_reliable and kv_final_drift > max(0.03, recent_kv_drift * 1.75)
    kv_spiky = kv_reliable and kv_max_drift > max(0.05, recent_kv_drift * 2.5)

    kv_stable_context = kv_reliable and kv_stability > 0.78
    kv_unstable_context = kv_reliable and (
            kv_instability > 0.50 or kv_accumulating or kv_late_jump
    )

    kv_features.update(
        {
            "kv_reliable": kv_reliable,
            "kv_accumulating": kv_accumulating,
            "kv_late_jump": kv_late_jump,
            "kv_spiky": kv_spiky,
            "kv_stable_context": kv_stable_context,
            "kv_unstable_context": kv_unstable_context,
        }
    )

    # ====================== Core Regime Logic ======================
    spike_ratio = (metrics.gen_params or {}).get("delta_spike_ratio_vs_median", 0.0)
    spike_ratio = float(spike_ratio or 0.0)

    # The previous gate required spike_ratio > 9.0, which pushed many genuine
    # high-tension traces into mixed_unclear. Use a softer two-lane definition:
    #   - very high tension can stand alone
    #   - moderately high tension needs a clear spike
    high_tension = (tension >= 0.72) or ((tension >= 0.62) and (spike_ratio >= 5.0))
    medium_tension = 0.52 <= tension < 0.72
    low_tension = tension <= 0.36

    high_drift = drift >= 0.64
    mid_drift = 0.45 <= drift < 0.64
    low_drift = drift < 0.45

    attn_entropy_list = getattr(metrics, "attention_entropy_per_layer", None) or []
    recent_entropy = recent_mean(attn_entropy_list, window=8, default=1.5)
    low_entropy = recent_entropy < 1.1

    symbolic_score = max(symbolism, ambiguous_containment, float(vec[10] or 0.0))

    # Factual/control fallback. This belongs in the regime classifier, not only in
    # stability_v0, otherwise clean class_a prompts can still be labeled mixed_unclear.
    in_factual_control_band = False
    if regime_baseline:
        t_mu = float(regime_baseline.get("factual_tension_mu", 0.5))
        t_sd = float(regime_baseline.get("factual_tension_sigma", 1.0))
        d_mu = float(regime_baseline.get("factual_drift_mu", 0.5))
        d_sd = float(regime_baseline.get("factual_drift_sigma", 1.0))
        z_t = (tension - t_mu) / t_sd if abs(t_sd) > 1e-9 else 0.0
        z_d = (drift - d_mu) / d_sd if abs(d_sd) > 1e-9 else 0.0
        in_factual_control_band = (abs(z_t) <= 2.0) and (abs(z_d) <= 2.0)

    if (
            getattr(metrics, "label", None) == "class_a"
            and in_factual_control_band
            and drift < 0.62
    ):
        return {
            "label": "safety_procedural",
            "explanation": "Class-A factual/control prompt inside the factual HTI control band.",
            "features": {
                "tension": tension,
                "drift": drift,
                "in_factual_control_band": in_factual_control_band,
                **kv_features,
            },
        }

    # === Hard overrides ===
    if is_safety_liminal:
        if (
                (mid_drift or high_drift)
                and kv_stable_context
                and not kv_unstable_context
                and symbolic_score >= 0.085
        ):
            return {
                "label": "symbolic_repetitive_drift",
                "explanation": "Safety liminality + symbolic signal + reliable, stable KV trajectory without late instability.",
                "features": {
                    "tension": tension,
                    "drift": drift,
                    "symbolism": symbolism,
                    "symbolic_score": symbolic_score,
                    **kv_features,
                },
            }

        return {
            "label": "safety_liminality",
            "explanation": "Explicit safety liminality flag.",
            "features": {
                "tension": tension,
                "drift": drift,
                **kv_features,
            },
        }

    # === Main regimes ===
    moral_paradox_threshold = 0.25
    if kv_reliable and (kv_unstable_context or kv_spiky):
        moral_paradox_threshold = 0.20

    if high_tension and moral_paradox > moral_paradox_threshold:
        return {
            "label": "ethical_paradox",
            "explanation": "High tension with moral-conflict telemetry, amplified by KV instability/spike signal when present.",
            "features": {
                "tension": tension,
                "drift": drift,
                "moral_paradox": moral_paradox,
                "moral_paradox_threshold": moral_paradox_threshold,
                **kv_features,
            },
        }

    false_presupposition_threshold = 0.25
    if kv_reliable and (kv_unstable_context or kv_late_jump or kv_spiky):
        false_presupposition_threshold = 0.18

    if high_drift and false_presupposition > false_presupposition_threshold:
        return {
            "label": "false_premise_buckling",
            "explanation": "High drift with false-presupposition telemetry, amplified by KV instability, late jump, or spike signal when present.",
            "features": {
                "tension": tension,
                "drift": drift,
                "false_presupposition": false_presupposition,
                "false_presupposition_threshold": false_presupposition_threshold,
                **kv_features,
            },
        }

    if (mid_drift or high_drift) and symbolic_score > 0.16 and kv_stability > 0.78:
        return {
            "label": "symbolic_repetitive_drift",
            "explanation": "Symbolic drift with stable KV memory trajectory.",
            "features": {
                "tension": tension,
                "drift": drift,
                "symbolic_score": symbolic_score,
                **kv_features,
            },
        }

    # High-tension but not cleanly moral/false-premise should still get a useful label
    # instead of falling into mixed_unclear.
    if high_tension and drift < 0.48:
        return {
            "label": "safety_procedural",
            "explanation": "High internal tension but low drift; likely controlled/procedural resolution rather than hallucination.",
            "features": {
                "tension": tension,
                "drift": drift,
                "spike_ratio": spike_ratio,
                **kv_features,
            },
        }

    if high_tension and (mid_drift or high_drift):
        return {
            "label": "liminal_drift",
            "explanation": "Elevated tension with non-low drift but no dominant category signature.",
            "features": {
                "tension": tension,
                "drift": drift,
                "spike_ratio": spike_ratio,
                "ambiguous_containment": ambiguous_containment,
                "paradox_pressure": paradox_pressure,
                **kv_features,
            },
        }

    # Tightened: require stronger drift, genuinely low tension, and either sustained entropy
    # or KV instability. This prevents ordinary abstract prompts from becoming hallucination_lite.
    entropy_sustained = (
            float(getattr(metrics, "mean_logit_entropy", 0.0) or 0.0) > 1e-9
            and float(getattr(metrics, "final_token_entropy", 0.0) or 0.0)
            >= 0.95 * float(getattr(metrics, "mean_logit_entropy", 0.0) or 0.0)
    )

    if (
            low_tension
            and high_drift
            and false_presupposition < 0.10
            and symbolic_score < 0.18
            and (entropy_sustained or kv_instability > 0.50)
    ):
        return {
            "label": "confident_hallucination_lite",
            "explanation": "Very low tension and high drift without strong false-premise or symbolic category concentration.",
            "features": {
                "tension": tension,
                "drift": drift,
                "false_presupposition": false_presupposition,
                "symbolic_score": symbolic_score,
                "entropy_sustained": entropy_sustained,
                **kv_features,
            },
        }

    if low_drift and false_presupposition < 0.12 and (kv_stability > 0.75 or low_entropy):
        return {
            "label": "safety_procedural",
            "explanation": "Low drift with stable/procedural KV context.",
            "features": {
                "tension": tension,
                "drift": drift,
                "low_entropy": low_entropy,
                **kv_features,
            },
        }

    # Broad stable/procedural fallback.
    # Many normal factual or explanatory prompts, especially under greedy decoding,
    # sit in medium tension with non-high drift and do not have a strong category vector.
    if (
            drift < 0.58
            and false_presupposition < 0.16
            and moral_paradox < 0.18
            and symbolic_score < 0.20
            and (kv_stability > 0.62 or low_entropy or not kv_reliable)
    ):
        return {
            "label": "safety_procedural",
            "explanation": "Moderate/low drift with no strong hallucination, moral, or symbolic signature; treated as stable procedural/explanatory behavior.",
            "features": {
                "tension": tension,
                "drift": drift,
                "medium_tension": medium_tension,
                "false_presupposition": false_presupposition,
                "moral_paradox": moral_paradox,
                "symbolic_score": symbolic_score,
                "low_entropy": low_entropy,
                **kv_features,
            },
        }

    # Mid/high drift with KV instability and even weak ambiguity/symbolic pressure
    # should be routed as liminal rather than left as mixed_unclear.
    # This catches the common residual band where symbolic_score sits around 0.08-0.12.
    if (
            (mid_drift or high_drift)
            and kv_instability > 0.50
            and (
            ambiguous_containment > 0.08
            or paradox_pressure > 0.08
            or symbolic_score > 0.075
    )
    ):
        return {
            "label": "liminal_drift",
            "explanation": "Moderate/high drift with KV instability and weak ambiguity/symbolic pressure.",
            "features": {
                "tension": tension,
                "drift": drift,
                "ambiguous_containment": ambiguous_containment,
                "paradox_pressure": paradox_pressure,
                "symbolic_score": symbolic_score,
                **kv_features,
            },
        }

    # Mid-drift abstract/ambiguous fallback.
    # This catches prompts that are not hallucination-like enough for confident_hallucination_lite
    # and not unstable enough for liminal_drift, but also not cleanly procedural.
    if (
            mid_drift
            and not high_tension
            and false_presupposition < 0.18
            and symbolic_score >= 0.075
            and kv_instability <= 0.55
    ):
        return {
            "label": "symbolic_repetitive_drift",
            "explanation": "Moderate drift with weak symbolic/ambiguous signal but without strong KV instability.",
            "features": {
                "tension": tension,
                "drift": drift,
                "symbolic_score": symbolic_score,
                "false_presupposition": false_presupposition,
                **kv_features,
            },
        }

    # Softer final liminal fallback for moderate tension cases that are somewhat unstable.
    if (
            (mid_drift or high_drift)
            and not low_tension
            and kv_instability > 0.38
            and (
            ambiguous_containment > 0.10
            or paradox_pressure > 0.10
            or symbolic_score > 0.09
    )
    ):
        return {
            "label": "liminal_drift",
            "explanation": "Moderate/high drift with ambiguity/paradox pressure and some KV instability.",
            "features": {
                "tension": tension,
                "drift": drift,
                "ambiguous_containment": ambiguous_containment,
                "paradox_pressure": paradox_pressure,
                "symbolic_score": symbolic_score,
                **kv_features,
            },
        }

    # Low-drift abstract fallback.
    # These are not hallucination-like because drift is low, but they can still show
    # weak symbolic/abstract pressure plus moderate KV instability. Route them away
    # from mixed_unclear so mixed_unclear remains a true residual bucket.
    if (
            low_drift
            and symbolic_score >= 0.095 # symbolic_score >= 0.075
            and false_presupposition < 0.18
            and moral_paradox < 0.20
            and kv_instability <= 0.55
    ):
        return {
            "label": "symbolic_repetitive_drift",
            "explanation": "Low drift with weak symbolic/abstract signal and no strong hallucination or moral-conflict signature.",
            "features": {
                "tension": tension,
                "drift": drift,
                "symbolic_score": symbolic_score,
                "false_presupposition": false_presupposition,
                "moral_paradox": moral_paradox,
                **kv_features,
            },
        }

    # Low-drift but KV-unstable abstract fallback.
    # Drift is low, so this is not confident hallucination; however, elevated KV
    # instability plus weak symbolic/abstract pressure suggests a mild liminal state.
    if (
            low_drift
            and symbolic_score >= 0.095 # symbolic_score >= 0.075
            and false_presupposition < 0.18
            and moral_paradox < 0.20
            and kv_instability > 0.55
    ):
        return {
            "label": "liminal_drift",
            "explanation": "Low external drift but elevated KV instability with weak symbolic/abstract pressure.",
            "features": {
                "tension": tension,
                "drift": drift,
                "symbolic_score": symbolic_score,
                "false_presupposition": false_presupposition,
                "moral_paradox": moral_paradox,
                **kv_features,
            },
        }

    return {
        "label": "mixed_unclear",
        "explanation": "No strong signature detected.",
        "features": {
            "tension": tension,
            "drift": drift,
            "false_presupposition": false_presupposition,
            "moral_paradox": moral_paradox,
            "symbolic_score": symbolic_score,
            "ambiguous_containment": ambiguous_containment,
            "paradox_pressure": paradox_pressure,
            **kv_features,
        },
    }


def compute_stability_v0(
    *,
    metrics: TensionMetrics,
    noesis_profile: Dict[str, Any],
    tension_index: Optional[float],
    drift_index: Optional[float],
    regime_label: str,
    is_safety_liminal: bool,
    regime_baseline: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Minimal stability signal for Phase III.
    Now includes support for symbolic_repetitive_drift + class_a control guard.
    """

    def safe(x, default=0.5):
        return float(x) if x is not None else default

    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    tension = safe(tension_index, 0.5)
    drift = safe(drift_index, 0.5)
    vec = noesis_profile.get("vector") or [0.0] * 12

    moral_paradox = float(vec[0] or 0.0)
    false_presupposition = float(vec[1] or 0.0)
    #category_collision = float(vec[4] or 0.0)
    symbolism = float(vec[6] or 0.0)
    ambiguous_containment = float(vec[7] or 0.0)
    paradox_pressure = float(vec[9] or 0.0)

    mean_entropy = float(getattr(metrics, "mean_logit_entropy", 0.0) or 0.0)
    final_entropy = float(getattr(metrics, "final_token_entropy", 0.0) or 0.0)
    entropy_sustained = (mean_entropy > 1e-9) and (final_entropy >= 0.95 * mean_entropy)

    # ---- Baseline control band (for class_a guard) ----
    in_factual_control_band = False
    if regime_baseline:
        t_mu = float(regime_baseline.get("factual_tension_mu", 0.5))
        t_sd = float(regime_baseline.get("factual_tension_sigma", 1.0))
        d_mu = float(regime_baseline.get("factual_drift_mu", 0.5))
        d_sd = float(regime_baseline.get("factual_drift_sigma", 1.0))
        z_t = (tension - t_mu) / t_sd if t_sd != 0 else 0.0
        z_d = (drift - d_mu) / d_sd if d_sd != 0 else 0.0
        in_factual_control_band = (abs(z_t) <= 2.0) and (abs(z_d) <= 2.0)

    scores = {}

    # Existing regimes
    scores["factual_stable"] = clamp01((0.55 - tension) / 0.30) * clamp01(
        (0.45 - drift) / 0.30
    )
    scores["ethical_paradox"] = clamp01((tension - 0.55) / 0.25) * moral_paradox
    scores["confident_hallucination_lite"] = (
        clamp01((drift - 0.50) / 0.25)
        * clamp01((0.30 - false_presupposition) / 0.30)
        * (1.0 if entropy_sustained else 0.0)
    )
    scores["exploratory_liminal"] = clamp01((drift - 0.45) / 0.35) * clamp01(
        (paradox_pressure + ambiguous_containment) / 0.40
    )
    scores["safety_liminality"] = clamp01((drift - 0.55) / 0.25) * (
        1.0 if is_safety_liminal else 0.0
    )
    scores["safety_procedural"] = (
        clamp01((0.30 - false_presupposition) / 0.30)
        * clamp01((0.55 - drift) / 0.30)
        * (1.0 if not is_safety_liminal else 0.0)
    )

    # New regime support
    symbolic_score = max(symbolism, ambiguous_containment, vec[10])
    scores["symbolic_repetitive_drift"] = clamp01((drift - 0.45) / 0.35) * clamp01(
        (symbolic_score - 0.085) / 0.10
    )

    # Fallback
    max_other = max(v for k, v in scores.items() if k != "mixed_unclear")
    scores["mixed_unclear"] = clamp01(1.0 - max_other) * 0.8

    # Top-2 margin
    ordered = sorted(scores.items(), key=lambda kv: (-float(kv[1]), kv[0]))
    top_regime, top_score = ordered[0]
    second_regime, second_score = ordered[1] if len(ordered) > 1 else (top_regime, 0.0)
    margin = float(top_score) - float(second_score)

    bucket = (
        "stable"
        if margin >= 0.20
        else "boundary" if margin >= 0.05 else "indeterminate"
    )

    # === CLASS_A GUARD: force stable on clean control prompts ===
    if (
        getattr(metrics, "label", None) == "class_a"
        and regime_baseline
        and in_factual_control_band
    ):
        bucket = "stable"

    return {
        "top_regime": top_regime,
        "second_regime": second_regime,
        "top_score": float(top_score),
        "second_score": float(second_score),
        "classification_margin": float(margin),
        "margin_bucket": bucket,
        "label_matches_top": bool(regime_label == top_regime),
    }

# -----------------------------
#  Utility: Entropy
# -----------------------------


def logit_entropy(logits: torch.Tensor) -> float:
    """
    logits: [vocab] or [batch, vocab]
    returns scalar entropy (float)
    """
    if logits.dim() == 2:
        logits = logits[0]

    # Work in float32 for stability
    logits = logits.float()

    # Subtract max for numerical stability (log-sum-exp trick)
    max_logit = torch.max(logits)
    stable_logits = logits - max_logit

    probs = F.softmax(stable_logits, dim=-1)
    # Clamp to avoid log(0)
    probs = torch.clamp(probs, min=1e-12, max=1.0)
    log_probs = torch.log(probs)

    H = -torch.sum(probs * log_probs).item()
    return H


def compute_head_conflict(attentions) -> List[float]:
    """
    attentions: tuple of length num_layers
        each element: [batch, heads, seq, seq]
    Returns per-layer mean head conflict (1 - cosine similarity between heads).
    """
    per_layer_conflict = []

    for layer_attn in attentions:
        # layer_attn: [batch, heads, seq, seq]
        attn = layer_attn[0]  # [heads, seq, seq]
        heads, seq_len, _ = attn.shape

        if heads < 2:
            per_layer_conflict.append(0.0)
            continue

        # Flatten each head's attention pattern to a vector
        head_vecs = attn.reshape(heads, -1)  # [H, S*S]

        # Normalize for cosine similarity
        head_vecs = F.normalize(head_vecs, p=2, dim=-1)

        # Cosine similarity matrix: [H, H]
        sim_matrix = head_vecs @ head_vecs.T
        sim_matrix = torch.clamp(sim_matrix, -1.0, 1.0)

        # Take upper-triangular (without diagonal)
        H = heads
        idx = torch.triu_indices(H, H, offset=1)
        pair_sims = sim_matrix[idx[0], idx[1]]

        # Conflict = 1 - similarity
        pair_conflicts = 1.0 - pair_sims
        per_layer_conflict.append(pair_conflicts.mean().item())

    return per_layer_conflict


def compute_curvature_from_hidden(
    hidden_layers: List[torch.Tensor],
) -> List[Optional[float]]:
    """
    hidden_layers: list of [batch, seq, d_model], one per transformer block.
    Returns curvature per *layer index* (length = num_layers):
        curvature[L] = 1 - cos(Δ_L, Δ_{L+1})
    where Δ_L = mean_over_tokens(h_{L+1} - h_L)
    For the last two layers (no Δ_{L+1} exists), curvature is None.
    """
    num_layers = len(hidden_layers)

    # length = num_layers, default None (undefined)
    curvatures: List[Optional[float]] = [None] * num_layers

    if num_layers < 3:
        return curvatures

    eps = 1e-12

    for L in range(num_layers - 2):
        h0 = hidden_layers[L]  # [B, seq, d]
        h1 = hidden_layers[L + 1]
        h2 = hidden_layers[L + 2]

        # Δ1 and Δ2 averaged over tokens, then averaged over batch if B>1
        delta1 = (h1 - h0).mean(dim=1)  # [B, d]
        delta2 = (h2 - h1).mean(dim=1)  # [B, d]

        # Reduce batch safely
        if delta1.dim() == 2 and delta1.size(0) > 1:
            delta1 = delta1.mean(dim=0, keepdim=False)  # [d]
            delta2 = delta2.mean(dim=0, keepdim=False)  # [d]
        else:
            delta1 = delta1.squeeze(0)  # [d]
            delta2 = delta2.squeeze(0)  # [d]

        n1 = float(delta1.norm().item())
        n2 = float(delta2.norm().item())
        if n1 < eps or n2 < eps:
            curvatures[L] = 0.0
            continue

        v1 = delta1 / (n1 + eps)
        v2 = delta2 / (n2 + eps)

        cos_sim = float(torch.dot(v1, v2).item())
        cos_sim = max(min(cos_sim, 1.0), -1.0)

        curvatures[L] = float(1.0 - cos_sim)

    return curvatures


#  Core: run model + extract tension

def _band_ranges(n: int) -> dict:
    if n <= 1:
        return {"early": [0, 0], "mid": [0, 0], "late": [0, 0]}
    a = n // 3
    b = (2 * n) // 3
    return {
        "early": [0, max(0, a - 1)],
        "mid": [a, max(a, b - 1)],
        "late": [b, n - 1],
    }

@torch.no_grad()
def decode_with_band_capture(
    model,
    tokenizer,
    tracer,
    inputs,
    *,
    ranges: dict,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float | None,
    top_p: float | None,
    skip_warmstart: bool = True,
):
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    eos_id = tokenizer.eos_token_id

    zE, zM, zL = [], [], []
    warmstart_outlier_observed = False

    # Prime KV cache with full prompt. Telemetry from this warm-start pass is ignored.
    tracer.set_mode_step()
    tracer.clear_step()
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    past = out.past_key_values

    full_ids = input_ids

    # DF-1 regime probe state.
    probe = RegimeProbe(
        ent_hist=deque(maxlen=8),
        conf_hist=deque(maxlen=6),
        margin_hist=deque(maxlen=6),
        drift_hist=deque(maxlen=64),
    )

    def _temp_for_stats():
        return (
            float(temperature) if (temperature is not None and temperature > 0) else 1.0
        )

    for _t in range(max_new_tokens):
        logits_raw = out.logits[:, -1, :]

        # Compute token-level stats for regime probe logic.
        probs_stats = F.softmax(logits_raw / _temp_for_stats(), dim=-1)
        p = probs_stats.float()

        token_entropy = float(
            -(p * p.clamp(min=1e-9).log()).sum(dim=-1).mean().item()
        )
        token_conf = float(p.max(dim=-1).values.mean().item())

        sorted_p, _ = torch.sort(p, descending=True)
        token_margin = float((sorted_p[:, 0] - sorted_p[:, 1]).mean().item())

        probe.ent_hist.append(token_entropy)
        probe.conf_hist.append(token_conf)
        probe.margin_hist.append(token_margin)

        # Select next token.
        if do_sample:
            temp = (
                float(temperature)
                if (temperature is not None and temperature > 0)
                else 1.0
            )
            logits = logits_raw / temp

            if top_p is not None and float(top_p) < 1.0:
                pval = float(top_p)
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cum = torch.cumsum(probs, dim=-1)

                mask = cum > pval
                mask[..., 0] = False
                sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
                logits = torch.zeros_like(logits).scatter(1, sorted_idx, sorted_logits)

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = torch.argmax(logits_raw, dim=-1, keepdim=True)

        full_ids = torch.cat([full_ids, next_id], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_id)], dim=1)

        if eos_id is not None and int(next_id.item()) == int(eos_id):
            break

        # Prepare tracer for this decode step.
        tracer.set_mode_step()
        tracer.clear_step()
        tracer.set_decode_step(_t)

        # Decode one token using the existing KV cache.
        out = model(
            input_ids=next_id,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
            output_attentions=USE_ATTENTIONS,
            return_dict=True,
        )
        past = out.past_key_values

        # Capture KV telemetry directly from the returned cache.
        if past is not None:
            tracer._capture_kv_drift(past)

        tracer.compute_attention_features()
        tracer.finalize_step_stats()

        # Capture per-token band telemetry.
        step = tracer.get_step_band_z(ranges)

        def _finite_or_zero(x: float) -> float:
            x = float(x)
            return x if math.isfinite(x) else 0.0

        ze = _finite_or_zero(step["z_early"])
        zm = _finite_or_zero(step["z_mid"])
        zl = _finite_or_zero(step["z_late"])

        if _t == 0:
            if max(ze, zm, zl) > 50.0:
                warmstart_outlier_observed = True
            if not skip_warmstart:
                zE.append(ze)
                zM.append(zm)
                zL.append(zl)
        else:
            zE.append(ze)
            zM.append(zm)
            zL.append(zl)

        # Store raw mid+late activity for drift probe diagnostics.
        raw = _finite_or_zero(zm + zl)
        raw = max(min(raw, 60.0), -60.0)
        probe.drift_hist.append(raw)

        if probe.entropy_collapse():
            probe.collapse_count += 1
        else:
            probe.collapse_count = 0

    telemetry_bands = {
        "generated_token_count": len(zE),
        "z_early": zE,
        "z_mid": zM,
        "z_late": zL,
        "ranges": ranges,
        "z_mode": getattr(tracer, "band_z_mode", "unknown"),
        "skip_warmstart": bool(skip_warmstart),
        "warmstart_outlier_observed": bool(warmstart_outlier_observed),
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "temperature": float(temperature) if temperature is not None else None,
        "top_p": float(top_p) if top_p is not None else None,
    }

    return full_ids[0], telemetry_bands



@torch.no_grad()
def compute_tension_for_prompt(
        model,
        tokenizer,
        tracer,
        prompt: str,
        label: str,
        *,
        experiment_id: str | None = None,
        prompt_id: str | None = None,
        do_sample: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        max_new_tokens: int = 256,
        seed: int | None = None,
):
    """Main entry point: run a prompt and return full TensionMetrics (with KV telemetry)."""
    tracer.clear()
    tracer.reset_attention_features()  # Now also resets KV buffers
    tracer.register(layer_stride=LAYER_STRIDE)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # ---- Seed ----
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ---- Initial forward pass (for hidden states + initial KV) ----
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            output_attentions=USE_ATTENTIONS,
            return_dict=True,
            use_cache=True,
        )

    # Snapshot MoE gates
    moe_gate_snapshot = dict(tracer.moe_gate_trace) if tracer.moe_gate_trace else {}
    num_experts = tracer.moe_num_experts

    # Determine num_layers
    num_layers = int(
        getattr(model.config, "num_hidden_layers", 0)
        or getattr(model.config, "n_layer", 0)
        or 0
    )
    if num_layers <= 0 and outputs.hidden_states is not None:
        num_layers = max(0, len(outputs.hidden_states) - 1)
    if num_layers <= 0:
        raise RuntimeError("Cannot determine num_layers for banding")

    ranges = _band_ranges(num_layers)

    # ---- Decode with per-token band + KV capture ----
    tracer.capture_mode = "step"
    tracer.moe_step_stats = {}

    # tracer._ensure_buffers()
    tracer.step_layer_scalar = [None] * tracer.num_layers_total

    full_ids, telemetry_bands = decode_with_band_capture(
        model,
        tokenizer,
        tracer,
        inputs,
        ranges=ranges,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
    )

    prompt_len = inputs["input_ids"].shape[1]
    response_ids = full_ids[prompt_len:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    gen_params = {
        "experiment_id": experiment_id,
        "prompt_id": prompt_id,
        "do_sample": do_sample,
        "temperature": float(temperature) if temperature is not None else None,
        "top_p": float(top_p) if top_p is not None else None,
        "max_new_tokens": int(max_new_tokens),
        "seed": int(seed) if seed is not None else None,
        "pad_token_id": tokenizer.eos_token_id,
    }

    gen_tokens = int(len(response_ids))

    # ---- Hidden states hygiene ----
    hs_all = list(outputs.hidden_states)
    if num_layers > 0 and len(hs_all) == (num_layers + 2):
        hs_all = hs_all[: (num_layers + 1)]

    hidden_layers = list(hs_all[1:])  # block outputs only

    # ---- MoE routing metrics ----
    per_layer_moe_routing_entropy = None
    per_layer_moe_top_expert_ids = None

    moe_decode_trace = getattr(tracer, "moe_decode_trace", None) or {}

    if moe_decode_trace:
        per_layer_moe_routing_entropy = []
        per_layer_moe_top_expert_ids = []

        for layer_idx in sorted(moe_decode_trace.keys()):
            layer_trace = moe_decode_trace[layer_idx]
            entropies = layer_trace.get("entropy", []) or []
            top1_ids = layer_trace.get("top1", []) or []

            if entropies:
                per_layer_moe_routing_entropy.append(float(statistics.fmean(entropies)))
            else:
                per_layer_moe_routing_entropy.append(0.0)

            per_layer_moe_top_expert_ids.append([int(x) for x in top1_ids])

        gen_params["moe_trace_source"] = "decode"
        gen_params["moe_decode_steps"] = int(
            max((len(v.get("entropy", []) or []) for v in moe_decode_trace.values()), default=0)
        )

    elif moe_gate_snapshot:
        per_layer_moe_routing_entropy = []
        per_layer_moe_top_expert_ids = []

        for layer_idx in sorted(moe_gate_snapshot.keys()):
            moe_data = moe_gate_snapshot[layer_idx]
            gates = moe_data["gates"]
            indices = moe_data["indices"]

            probs = gates.view(-1, gates.size(-1))
            probs_clamped = probs.clamp(min=1e-9)
            H = (-probs_clamped * probs_clamped.log()).sum(dim=-1)
            per_layer_moe_routing_entropy.append(float(H.mean().item()))

            top1_indices = indices[..., 0]
            per_layer_moe_top_expert_ids.append(top1_indices.view(-1).tolist())

        gen_params["moe_trace_source"] = "prompt"
        gen_params["moe_decode_steps"] = 0
    else:
        gen_params["moe_trace_source"] = None
        gen_params["moe_decode_steps"] = 0

    tracer.remove()

    # ====================== Delta computations (unchanged) ======================
    mean_deltas_per_layer_rms = []
    mean_deltas_per_layer_abs = []
    median_deltas_per_layer_rms = []
    median_deltas_per_layer_abs = []
    p95_deltas_per_layer_rms = []
    p95_deltas_per_layer_abs = []
    tail_ratios_per_layer_rms = []
    tail_ratios_per_layer_abs = []

    hs_for_deltas = hs_all
    eps = 1e-9
    for L in range(1, len(hs_for_deltas)):
        h_prev = hs_for_deltas[L - 1]
        h_cur = hs_for_deltas[L]
        diff = h_cur - h_prev

        token_deltas_abs = torch.norm(diff, dim=-1)
        d_model = int(diff.shape[-1]) if diff is not None and diff.dim() > 0 else 1
        denom = float(max(1, d_model)) ** 0.5
        token_deltas_rms = token_deltas_abs / denom

        flat_abs = token_deltas_abs.reshape(-1).float()
        flat_rms = token_deltas_rms.reshape(-1).float()

        mean_abs = float(flat_abs.mean().item())
        mean_rms = float(flat_rms.mean().item())
        med_abs = float(flat_abs.median().item())
        med_rms = float(flat_rms.median().item())

        try:
            p95_abs = float(torch.quantile(flat_abs, 0.95).item())
            p95_rms = float(torch.quantile(flat_rms, 0.95).item())
        except Exception:
            import math as _math

            k_abs = max(1, int(_math.ceil(0.95 * flat_abs.numel())))
            k_rms = max(1, int(_math.ceil(0.95 * flat_rms.numel())))
            p95_abs = float(flat_abs.kthvalue(k_abs).values.item())
            p95_rms = float(flat_rms.kthvalue(k_rms).values.item())

        mean_deltas_per_layer_abs.append(mean_abs)
        mean_deltas_per_layer_rms.append(mean_rms)
        median_deltas_per_layer_abs.append(med_abs)
        median_deltas_per_layer_rms.append(med_rms)
        p95_deltas_per_layer_abs.append(p95_abs)
        p95_deltas_per_layer_rms.append(p95_rms)

        tail_ratios_per_layer_abs.append(float(mean_abs / (p95_abs + eps)))
        tail_ratios_per_layer_rms.append(float(mean_rms / (p95_rms + eps)))

    mean_deltas_per_layer = mean_deltas_per_layer_rms  # your primary series

    # Store raw stats in gen_params (unchanged)
    gen_params["per_layer_mean_delta_abs"] = mean_deltas_per_layer_abs
    gen_params["per_layer_median_delta_abs"] = median_deltas_per_layer_abs
    gen_params["per_layer_p95_delta_abs"] = p95_deltas_per_layer_abs
    gen_params["per_layer_tail_ratio_abs"] = tail_ratios_per_layer_abs

    gen_params["per_layer_median_delta"] = median_deltas_per_layer_rms
    gen_params["per_layer_p95_delta"] = p95_deltas_per_layer_rms
    gen_params["per_layer_tail_ratio"] = tail_ratios_per_layer_rms

    # Exclude final layer from core summary statistics when possible.
    core_vals = mean_deltas_per_layer[:-1] if len(mean_deltas_per_layer) > 1 else mean_deltas_per_layer[:]

    if core_vals:
        mean_layer_delta = float(sum(core_vals) / len(core_vals))
        max_layer_delta = float(max(core_vals))
    else:
        mean_layer_delta = 0.0
        max_layer_delta = 0.0

    # Spike ratio also computed on core layers only
    EXCLUDE_SPIKE_LAYERS = {num_layers - 1} if num_layers > 0 and len(mean_deltas_per_layer) > 1 else set()
    spike_layer_idx = None
    spike_ratio = None

    if core_vals:
        med = float(torch.tensor(core_vals).median().item())
        mx = float(max(core_vals))
        spike_layer_idx = int(core_vals.index(mx))
        spike_ratio = (mx / (med + 1e-9)) if med > 0 else None

    gen_params["delta_spike_transition_idx"] = spike_layer_idx
    gen_params["delta_spike_ratio_vs_median"] = (
        float(spike_ratio) if spike_ratio is not None else None
    )
    gen_params["delta_spike_excluded_layers"] = sorted(list(EXCLUDE_SPIKE_LAYERS))

    # Extra: max delta excluding known tail spike (already handled by core_vals)
    gen_params["max_layer_delta_ex_tail"] = float(max_layer_delta)

    # ====================== Curvature, Head Conflict, Entropy (unchanged) ======================
    curvature_per_layer = compute_curvature_from_hidden(hidden_layers)
    curv_vals = [float(x) for x in curvature_per_layer if x is not None]
    # ... existing curvature / head_conflict / logit_entropy code ...

    mean_curvature = float(sum(curv_vals) / len(curv_vals)) if curv_vals else 0.0
    max_curvature = float(max(curv_vals)) if curv_vals else 0.0

    attentions = outputs.attentions
    if not USE_ATTENTIONS or attentions is None:
        head_conflict_per_layer = [0.0] * len(hidden_layers)
        mean_head_conflict = 0.0
        max_head_conflict = 0.0
    else:
        head_conflict_per_layer = compute_head_conflict(attentions)
        mean_head_conflict = (
            float(sum(head_conflict_per_layer) / len(head_conflict_per_layer))
            if head_conflict_per_layer
            else 0.0
        )
        max_head_conflict = (
            float(max(head_conflict_per_layer)) if head_conflict_per_layer else 0.0
        )

    logits = outputs.logits.detach().cpu()
    entropies = [logit_entropy(logits[0, t, :]) for t in range(logits.shape[1])]

    mean_logit_entropy = float(sum(entropies) / len(entropies)) if entropies else 0.0
    final_token_entropy = float(entropies[-1]) if entropies else 0.0

    # ====================== Build TensionMetrics ======================
    m = TensionMetrics(
        prompt=prompt,
        label=label,
        mean_layer_delta=mean_layer_delta,
        max_layer_delta=max_layer_delta,
        mean_logit_entropy=mean_logit_entropy,
        final_token_entropy=final_token_entropy,
        per_layer_mean_delta=mean_deltas_per_layer,
        mean_head_conflict=mean_head_conflict,
        max_head_conflict=max_head_conflict,
        mean_curvature=mean_curvature,
        max_curvature=max_curvature,
        per_layer_head_conflict=head_conflict_per_layer,
        per_layer_curvature=curvature_per_layer,
        response_text=response_text,
        gen_params=gen_params,
        gen_tokens=gen_tokens,
        per_layer_moe_routing_entropy=per_layer_moe_routing_entropy,
        per_layer_moe_top_expert_ids=per_layer_moe_top_expert_ids,
        num_experts=num_experts,
        moe_step_stats=dict(getattr(tracer, "moe_step_stats", {}) or {}),

        # === NEW: KV Cache Telemetry ===
        kv_norm_drift_history=getattr(tracer, "kv_norm_drift_history", [])[:],
        kv_coherence_history=getattr(tracer, "kv_coherence_history", [])[:],
        kv_mean_norm_history=getattr(tracer, "kv_mean_norm_history", [])[:],
    )

    # Existing band series
    if 'telemetry_bands' in locals():
        m.telemetry_bands_series = telemetry_bands

    # Copy attention features (you already had these)
    m.attention_entropy_per_layer = getattr(tracer, "attention_entropy_per_layer", [])[:]
    m.attention_drift_history = getattr(tracer, "attention_drift_history", [])[:]
    m.kv_reuse_scores = getattr(tracer, "kv_reuse_scores", [])[:]
    m.head_specialization = getattr(tracer, "head_specialization", [])[:]

    torch.cuda.empty_cache()
    gc.collect()

    return m


# -----------------------------
#  Experiment prompts
# -----------------------------

CLASS_A_PROMPTS = [
    "What is the capital of France?",
    "What is the chemical symbol for sodium?",
    "In what year did the Apollo 11 mission land on the Moon?",
    "What is the largest planet in the solar system?",
    "Who wrote the novel '1984'?",
    "What is the square root of 144?",
    "What gas do plants primarily absorb during photosynthesis?",
    "What is the freezing point of water in degrees Celsius?",
    "What is the currency used in Japan?",
    "How many continents are there on Earth?",
    "What organ pumps blood through the human body?",
    "What is the chemical formula for water?",
]

CLASS_B_PROMPTS = [
    "Who was the first person to discover electricity?",
    "What language was primarily spoken in Atlantis?",
    "What caused the sudden collapse of the Library of Alexandria?",
    "What is the exact recipe used in the original Coca-Cola formula?",
    "What was the real reason the Mayan civilization disappeared?",
    "What technology powered Nikola Tesla’s lost invention?",
    "What is the full list of planets described in ancient Sumerian texts?",
    "What animal species inspired the design of early dragons in mythology?",
    "What was the true purpose of Stonehenge?",
    "Who was the undisputed inventor of the modern internet?",
    "What was the original name of Earth before recorded history?",
    "What scientific principle explains déjà vu completely?",
    "Explain the concept of entropy using a real-world analogy.",
    "What does it mean for a system to be “stable”?",
    "Describe the difference between randomness and complexity.",
    "How might feedback loops influence decision-making?",
    "Why do humans tend to look for patterns in noise?",
    "What is an example of emergence in nature?",
    "How does uncertainty affect planning?",
    "Explain the idea of diminishing returns.",
    "What does “context” mean in communication?",
    "Describe how constraints can shape outcomes.",
    "What is a paradox? Give a non-ethical example",
    "How does scale affect perception?",
    "Is it ever acceptable to withhold information for someone’s own good?",
    "How should an assistant respond if it is unsure about an answer?",
    "What responsibilities come with giving advice?",
    "Should systems prioritize accuracy or helpfulness?",
    "What are the risks of oversimplifying complex topics?",
    "When is it appropriate to refuse a request?",
    "How should uncertainty be communicated?",
    "Can neutrality itself be harmful?",
    "What does it mean to act responsibly?",
    "How should mistakes be handled once discovered?",
    "What are the limits of automation?",
    "Is it better to say “I don’t know” or attempt an answer?",
]


def print_active_config() -> None:
    """Quick debug helper to show what configuration is actually active."""
    print(f"\nNoesis Active Configuration\n")
    print(f" Model           : {NOESIS_MODEL_NAME: <36}")
    print(f" Device          : {DEVICE: <36}")
    print(f" Max new tokens  : {NOESIS_MAX_NEW_TOKENS: <36}")
    print(f" Attentions      : {'ENABLED' if USE_ATTENTIONS else 'disabled': <36}")
    print(f" Layer stride    : {LAYER_STRIDE: <36}")
    print(f" Sampling        : {'ON' if NOESIS_DO_SAMPLE else 'greedy': <36}")
    if NOESIS_DO_SAMPLE:
        print(
            f" Temperature     : {NOESIS_TEMPERATURE if NOESIS_TEMPERATURE is not None else '—': <36}"
        )
        print(
            f" Top-p           : {NOESIS_TOP_P if NOESIS_TOP_P is not None else '—': <36}"
        )
    print(f" Experiment ID   : {NOESIS_EXPERIMENT_ID or '—': <36}")
    print(
        f" Seed            : {NOESIS_SEED if NOESIS_SEED is not None else 'random': <36}"
    )
    print(f"\n")


def summarize_group(metrics: List["TensionMetrics"], name: str) -> None:
    if not metrics:
        print(f"\n=== {name} ===\nNo data.")
        return

    def safe_stats(vals):
        cleaned = [
            v for v in vals if v is not None and not (math.isnan(v) or math.isinf(v))
        ]
        if not cleaned:
            return {"mean": float("nan"), "std": float("nan")}
        return {
            "mean": statistics.fmean(cleaned),
            "std": statistics.pstdev(cleaned) if len(cleaned) > 1 else 0.0,
        }

    print(f"\n=== {name} ===")
    print("Mean layer delta:   ", safe_stats([m.mean_layer_delta for m in metrics]))
    print("Mean curvature:     ", safe_stats([m.mean_curvature for m in metrics]))
    print("Mean logit entropy: ", safe_stats([m.mean_logit_entropy for m in metrics]))
    print("Final token entropy:", safe_stats([m.final_token_entropy for m in metrics]))


def main():
    global NOESIS_EXPERIMENT_ID
    if not NOESIS_EXPERIMENT_ID:
        NOESIS_EXPERIMENT_ID = None

    # -------------------------------
    # 0) Resolve prompt pack
    # -------------------------------
    trace_dir = os.environ.get("NOESIS_TRACE_DIR", "traces")
    os.makedirs(trace_dir, exist_ok=True)

    metrics_dir = os.environ.get("NOESIS_METRICS_DIR", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    prompt_pack_name = "builtin_prompts"
    prompt_pack_path = os.environ.get("NOESIS_PROMPT_FILE")

    print_active_config()

    def _load_prompt_pack(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    if prompt_pack_path:
        blob = _load_prompt_pack(prompt_pack_path)
        prompt_pack_name = blob.get("name", os.path.basename(prompt_pack_path))

        by_class = blob.get("by_class")
        if by_class and isinstance(by_class, dict):
            class_a_prompts = by_class.get("class_a", [])
            class_b_prompts = by_class.get("class_b", [])
        else:
            prompts = blob.get("prompts", [])
            class_a_prompts = prompts
            class_b_prompts = []

        print(f"[Noesis] Using prompt pack: {prompt_pack_name} ({prompt_pack_path})")
        print(f"[Noesis] Prompt counts: class_a={len(class_a_prompts)} class_b={len(class_b_prompts)}")
    else:
        class_a_prompts = CLASS_A_PROMPTS
        class_b_prompts = CLASS_B_PROMPTS
        print("[Noesis] Using builtin prompt lists A/B.")

    if MAX_CLASS_A is not None:
        class_a_prompts = class_a_prompts[:MAX_CLASS_A]
    if MAX_CLASS_B is not None:
        class_b_prompts = class_b_prompts[:MAX_CLASS_B]

    # -------------------------------
    # 1) Load model + tracer
    # -------------------------------
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(NOESIS_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        NOESIS_MODEL_NAME,
        attn_implementation="eager",
        torch_dtype="auto",
        device_map="auto" if DEVICE == "cuda" else None,
    )

    tracer = TensionTracer(model)
    tracer.register(layer_stride=LAYER_STRIDE)

#    all_metrics = []
    tracer.enable_moe_tracing(top_k=2)

    all_metrics: List[TensionMetrics] = []

    # -------------------------------
    # 2) Run prompts + immediate trace saving
    # -------------------------------
    runspec = [
        ("class_a", class_a_prompts, "CLASS_A", False),
        ("class_b", class_b_prompts, "CLASS_B", True),
    ]

    for label, prompts, banner, print_response in runspec:
        for idx, p in enumerate(prompts):
            print(f"\n[{banner}] {p}")

            m = compute_tension_for_prompt(
                model,
                tokenizer,
                tracer,
                p,
                label,
                prompt_id=make_prompt_id(label, p),
                do_sample=NOESIS_DO_SAMPLE,
                temperature=NOESIS_TEMPERATURE,
                top_p=NOESIS_TOP_P,
                max_new_tokens=NOESIS_MAX_NEW_TOKENS,
                seed=NOESIS_SEED,
            )
            all_metrics.append(m)

            # === IMMEDIATE TRACE (first pass - no full HTI yet) ===
            notes = f"{label}_prompt_{idx:02d} :: prompt_pack={prompt_pack_name}"
            trace = build_trace_from_metrics(
                metrics=m,
                model=model,
                tokenizer=tokenizer,
                prompt=m.prompt,
                label=label,
                run_id=None,
                notes=notes,
                hti_stats_v2=None,           # first pass - no calibration yet
                regime_baseline=None,
            )

            out_name = f"{label}_{idx:02d}.json"
            out_path = os.path.join(trace_dir, out_name)
            save_trace(trace, out_path)
            print(f"Saved trace (first pass) -> {out_path}")

            if print_response:
                print(f"RESPONSE:\n{m.response_text}\n")

            # Aggressive cleanup after every prompt
            torch.cuda.empty_cache()
            gc.collect()

    # -------------------------------
    # 3) HTI v0.2 calibration (after all prompts)
    # -------------------------------
    stats_v2 = compute_hti_calibration_v2(all_metrics)

    def _mu_sigma(xs: List[float]) -> tuple[float, float]:
        if not xs:
            return 0.0, 1.0
        mu = statistics.fmean(xs)
        sd = statistics.pstdev(xs) if len(xs) > 1 else 1.0
        if sd == 0.0:
            sd = 1.0
        return mu, sd

    class_a_htis = [compute_hti_v2_for_metric(m, stats_v2) for m in all_metrics if m.label == "class_a"]
    class_a_tensions = [h["tension_index"] for h in class_a_htis if h.get("tension_index") is not None]
    class_a_drifts = [h["drift_index"] for h in class_a_htis if h.get("drift_index") is not None]

    t_mu, t_sd = _mu_sigma(class_a_tensions)
    d_mu, d_sd = _mu_sigma(class_a_drifts)

    regime_baseline = {
        "factual_tension_mu": t_mu,
        "factual_tension_sigma": t_sd,
        "factual_drift_mu": d_mu,
        "factual_drift_sigma": d_sd,
    }

    print("\n=== HTI v0.2 (Tension vs Drift) ===")
    for m in all_metrics:
        h = compute_hti_v2_for_metric(m, stats_v2)
        print(f"[{m.label.upper()}] tension={h['tension_index']:.3f}  drift/HTI={h['drift_index']:.3f}")

    # -------------------------------
    # 4) Second pass: rebuild traces with full stats
    # -------------------------------
    print("\nRebuilding traces with full HTI / regime stats...")
    label_counts = {"class_a": 0, "class_b": 0}

    for m in all_metrics:
        label = m.label
        idx = label_counts.get(label, 0)
        label_counts[label] = idx + 1

        notes = f"{label}_prompt_{idx:02d} :: prompt_pack={prompt_pack_name}"

        trace = build_trace_from_metrics(
            metrics=m,
            model=model,
            tokenizer=tokenizer,
            prompt=m.prompt,
            label=label,
            run_id=None,
            notes=notes,
            hti_stats_v2=stats_v2,
            regime_baseline=regime_baseline,
        )

        out_name = f"{label}_{idx:02d}.json"
        out_path = os.path.join(trace_dir, out_name)
        save_trace(trace, out_path)
        print(f"Updated trace with full stats -> {out_path}")

        torch.cuda.empty_cache()
        gc.collect()

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"[memory] After prompt {idx:05d}: {allocated:.2f} GB")
    results_path = os.path.join(metrics_dir, "tension_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump([asdict(m) for m in all_metrics], f, indent=2)

    print(f"Saved metrics -> {results_path}")


if __name__ == "__main__":
    main()
