"""Symbolic program definitions and GPT templates."""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Tuple

import numpy as np


def prog_argmax_first(state: int, timestep: int, outputs: List[Dict[int, float]]) -> int:
    if not outputs:
        return 0
    d = outputs[0]
    return max(d.items(), key=lambda kv: kv[1])[0]


def prog_even_bias(state: int, timestep: int, outputs: List[Dict[int, float]]) -> int:
    if not outputs:
        return 0
    d = outputs[0]
    evens = [(k, v) for k, v in d.items() if k % 2 == 0]
    if evens:
        return max(evens, key=lambda kv: kv[1])[0]
    return max(d.items(), key=lambda kv: kv[1])[0]


def prog_low_index(state: int, timestep: int, outputs: List[Dict[int, float]]) -> int:
    if not outputs:
        return 0
    d = outputs[0]
    best_k, best_s = 0, -1e9
    for k, v in d.items():
        s = v - 0.01 * k
        if s > best_s:
            best_s = s
            best_k = k
    return int(best_k)


def _sorted_items_desc(d: Dict[int, float]) -> List[Tuple[int, float]]:
    return sorted(d.items(), key=lambda kv: kv[1], reverse=True)


def prog_second_best(state: int, timestep: int, outputs: List[Dict[int, float]]) -> int:
    if not outputs:
        return 0
    d = outputs[0]
    items = _sorted_items_desc(d)
    if len(items) >= 2:
        return int(items[1][0])
    return int(items[0][0])


def prog_top2_margin_switch(state: int, timestep: int, outputs: List[Dict[int, float]]) -> int:
    if not outputs:
        return 0
    d = outputs[0]
    items = _sorted_items_desc(d)
    if len(items) < 2:
        return int(items[0][0])
    (k1, v1), (k2, v2) = items[0], items[1]
    margin = float(v1 - v2)
    thresh = 0.05
    return int(k2) if margin < thresh else int(k1)


def prog_confidence_threshold(state: int, timestep: int, outputs: List[Dict[int, float]]) -> int:
    if not outputs:
        return 0
    d = outputs[0]
    k1, v1 = max(d.items(), key=lambda kv: kv[1])
    tau = 0.55
    return int(k1) if float(v1) >= tau else 0


def prog_entropy_gate(state: int, timestep: int, outputs: List[Dict[int, float]]) -> int:
    if not outputs:
        return 0
    d = outputs[0]
    probs = np.array(
        [float(v) for _, v in sorted(d.items(), key=lambda kv: kv[0])], dtype=np.float64
    )
    probs = probs / (probs.sum() + 1e-12)
    ent = float(-(probs * np.log(probs + 1e-12)).sum())
    if ent > 2.1:
        return 0
    return int(max(d.items(), key=lambda kv: kv[1])[0])


def prog_odd_bias(state: int, timestep: int, outputs: List[Dict[int, float]]) -> int:
    if not outputs:
        return 0
    d = outputs[0]
    odds = [(k, v) for k, v in d.items() if k % 2 == 1]
    if odds:
        return int(max(odds, key=lambda kv: kv[1])[0])
    return int(max(d.items(), key=lambda kv: kv[1])[0])


def prog_modulo3_time(state: int, timestep: int, outputs: List[Dict[int, float]]) -> int:
    if not outputs:
        return 0
    d = outputs[0]
    target = int(timestep % 3)
    candidates = [(k, v) for k, v in d.items() if (k % 3) == target]
    if candidates:
        return int(max(candidates, key=lambda kv: kv[1])[0])
    return int(max(d.items(), key=lambda kv: kv[1])[0])


def prog_near_center(state: int, timestep: int, outputs: List[Dict[int, float]]) -> int:
    if not outputs:
        return 0
    d = outputs[0]
    keys = list(d.keys())
    if not keys:
        return 0
    num_classes = max(keys) + 1
    center = (num_classes - 1) / 2.0
    best_k, best_s = 0, -1e18
    lam = 0.015
    for k, v in d.items():
        s = float(v) - lam * abs(float(k) - center)
        if s > best_s:
            best_s = s
            best_k = int(k)
    return int(best_k)


GPT_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "parity_bias": {
        "desc": "Choose best class with parity=parity (0 even, 1 odd); else argmax.",
        "params": {"parity": "int in {0,1}"},
    },
    "mod_bias": {
        "desc": "Prefer classes where (k % mod) == target; else argmax.",
        "params": {"mod": "int in [2..6]", "target": "int in [0..mod-1]"},
    },
    "topk_margin_switch": {
        "desc": "If top-2 margin < thresh -> choose 2nd; else choose 1st.",
        "params": {"thresh": "float in [0.01..0.20]"},
    },
    "confidence_threshold": {
        "desc": "If maxprob >= tau -> argmax; else fallback_class.",
        "params": {"tau": "float in [0.40..0.85]", "fallback_class": "int in [0..C-1]"},
    },
    "entropy_gate": {
        "desc": "If entropy > tau -> fallback_class; else argmax.",
        "params": {"tau": "float in [1.0..3.0]", "fallback_class": "int in [0..C-1]"},
    },
    "low_index_bias": {
        "desc": "Score = prob - lam*k; choose max score.",
        "params": {"lam": "float in [0.001..0.05]"},
    },
    "center_bias": {
        "desc": "Score = prob - lam*|k-center|; choose max score.",
        "params": {"lam": "float in [0.001..0.05]"},
    },
}


def _safe_int(x: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(x)
        return max(lo, min(hi, v))
    except Exception:
        return default


def _safe_float(x: Any, lo: float, hi: float, default: float) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return max(lo, min(hi, v))
    except Exception:
        return default


def compile_program_from_spec(
    spec: Dict[str, Any],
    num_classes: int,
) -> Tuple[str, Callable, str, Dict[str, Any]]:
    template = str(spec.get("template", "")).strip()
    params = spec.get("params", {}) if isinstance(spec.get("params", {}), dict) else {}

    family = str(spec.get("family", "gpt")).strip() or "gpt"
    raw_name = str(spec.get("name", "")).strip()

    if template not in GPT_TEMPLATES:
        raise ValueError(f"Unknown template: {template}")

    if not raw_name:
        pitems = [f"{k}={params[k]}" for k in sorted(params.keys())]
        raw_name = f"GPT_{template}_" + "_".join(pitems) if pitems else f"GPT_{template}"

    raw_name = raw_name.replace(" ", "_")[:80]

    if template == "parity_bias":
        parity = _safe_int(params.get("parity", 0), 0, 1, 0)

        def fn(state: int, timestep: int, outputs: List[Dict[int, float]]) -> int:
            if not outputs:
                return 0
            d = outputs[0]
            candidates = [(k, v) for k, v in d.items() if (int(k) % 2) == parity]
            if candidates:
                return int(max(candidates, key=lambda kv: kv[1])[0])
            return int(max(d.items(), key=lambda kv: kv[1])[0])

        meta = {"template": template, "params": {"parity": parity}}

    elif template == "mod_bias":
        mod = _safe_int(params.get("mod", 3), 2, 6, 3)
        target = _safe_int(params.get("target", 0), 0, mod - 1, 0)

        def fn(state: int, timestep: int, outputs: List[Dict[int, float]]) -> int:
            if not outputs:
                return 0
            d = outputs[0]
            candidates = [(k, v) for k, v in d.items() if (int(k) % mod) == target]
            if candidates:
                return int(max(candidates, key=lambda kv: kv[1])[0])
            return int(max(d.items(), key=lambda kv: kv[1])[0])

        meta = {"template": template, "params": {"mod": mod, "target": target}}

    elif template == "topk_margin_switch":
        thresh = _safe_float(params.get("thresh", 0.05), 0.01, 0.20, 0.05)

        def fn(state: int, timestep: int, outputs: List[Dict[int, float]]) -> int:
            if not outputs:
                return 0
            d = outputs[0]
            items = _sorted_items_desc(d)
            if len(items) < 2:
                return int(items[0][0])
            (k1, v1), (k2, v2) = items[0], items[1]
            margin = float(v1 - v2)
            return int(k2) if margin < thresh else int(k1)

        meta = {"template": template, "params": {"thresh": thresh}}

    elif template == "confidence_threshold":
        tau = _safe_float(params.get("tau", 0.55), 0.40, 0.85, 0.55)
        fallback_class = _safe_int(params.get("fallback_class", 0), 0, num_classes - 1, 0)

        def fn(state: int, timestep: int, outputs: List[Dict[int, float]]) -> int:
            if not outputs:
                return 0
            d = outputs[0]
            k1, v1 = max(d.items(), key=lambda kv: kv[1])
            return int(k1) if float(v1) >= tau else int(fallback_class)

        meta = {"template": template, "params": {"tau": tau, "fallback_class": fallback_class}}

    elif template == "entropy_gate":
        tau = _safe_float(params.get("tau", 2.1), 1.0, 3.0, 2.1)
        fallback_class = _safe_int(params.get("fallback_class", 0), 0, num_classes - 1, 0)

        def fn(state: int, timestep: int, outputs: List[Dict[int, float]]) -> int:
            if not outputs:
                return 0
            d = outputs[0]
            probs = np.array(
                [float(v) for _, v in sorted(d.items(), key=lambda kv: kv[0])],
                dtype=np.float64,
            )
            probs = probs / (probs.sum() + 1e-12)
            ent = float(-(probs * np.log(probs + 1e-12)).sum())
            if ent > tau:
                return int(fallback_class)
            return int(max(d.items(), key=lambda kv: kv[1])[0])

        meta = {"template": template, "params": {"tau": tau, "fallback_class": fallback_class}}

    elif template == "low_index_bias":
        lam = _safe_float(params.get("lam", 0.01), 0.001, 0.05, 0.01)

        def fn(state: int, timestep: int, outputs: List[Dict[int, float]]) -> int:
            if not outputs:
                return 0
            d = outputs[0]
            best_k, best_s = 0, -1e18
            for k, v in d.items():
                s = float(v) - lam * float(int(k))
                if s > best_s:
                    best_s = s
                    best_k = int(k)
            return int(best_k)

        meta = {"template": template, "params": {"lam": lam}}

    elif template == "center_bias":
        lam = _safe_float(params.get("lam", 0.015), 0.001, 0.05, 0.015)

        def fn(state: int, timestep: int, outputs: List[Dict[int, float]]) -> int:
            if not outputs:
                return 0
            d = outputs[0]
            keys = list(d.keys())
            if not keys:
                return 0
            num_classes = max(int(k) for k in keys) + 1
            center = (num_classes - 1) / 2.0
            best_k, best_s = 0, -1e18
            for k, v in d.items():
                s = float(v) - lam * abs(float(int(k)) - center)
                if s > best_s:
                    best_s = s
                    best_k = int(k)
            return int(best_k)

        meta = {"template": template, "params": {"lam": lam}}

    else:
        raise ValueError(f"Template not implemented: {template}")

    return raw_name, fn, family, meta
