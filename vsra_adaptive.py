"""Stateful reliability gating for the VSRA contrastive teacher."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping


def compute_reliability(
    d_cs: float,
    d_vs: float,
    d_vc: float,
    temperature: float = 1.0,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Compute a bounded C-teacher reliability score from detached RKD losses."""
    d_cs = max(0.0, float(d_cs))
    d_vs = max(0.0, float(d_vs))
    d_vc = max(0.0, float(d_vc))
    conflict = d_cs / max(d_cs + d_vs, eps)
    disadvantage = max(d_vc - d_vs, 0.0) / max(d_vc + d_vs, eps)
    reliability = math.exp(-float(temperature) * (conflict + disadvantage))
    reliability = min(1.0, max(0.0, reliability))
    return {
        "conflict": conflict,
        "disadvantage": disadvantage,
        "reliability": reliability,
    }


@dataclass
class VSRAGateOutput:
    weight: float
    raw_reliability: float
    ema_reliability: float
    warmup: float
    conflict: float
    disadvantage: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "weight": self.weight,
            "raw_reliability": self.raw_reliability,
            "ema_reliability": self.ema_reliability,
            "warmup": self.warmup,
            "conflict": self.conflict,
            "disadvantage": self.disadvantage,
        }


class VSRAAdaptiveGate:
    """EMA-smoothed, warm-started controller for the C-teacher weight."""

    def __init__(
        self,
        max_weight: float,
        ema_decay: float,
        temperature: float,
        warmup_ratio: float,
        total_steps: int,
        eps: float = 1e-12,
    ) -> None:
        if max_weight < 0:
            raise ValueError("max_weight must be non-negative")
        if not 0 <= ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 <= warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be in [0, 1]")
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")

        self.max_weight = float(max_weight)
        self.ema_decay = float(ema_decay)
        self.temperature = float(temperature)
        self.warmup_ratio = float(warmup_ratio)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(math.ceil(self.total_steps * self.warmup_ratio))
        self.eps = float(eps)
        self.step = 0
        self.ema_reliability = None
        self.last_output = None

    def update(self, d_cs: float, d_vs: float, d_vc: float) -> VSRAGateOutput:
        scores = compute_reliability(d_cs, d_vs, d_vc, self.temperature, self.eps)
        raw = scores["reliability"]
        if self.ema_reliability is None:
            self.ema_reliability = raw
        else:
            self.ema_reliability = (
                self.ema_decay * self.ema_reliability + (1.0 - self.ema_decay) * raw
            )

        self.step += 1
        warmup = 1.0 if self.warmup_steps == 0 else min(1.0, self.step / self.warmup_steps)
        weight = min(self.max_weight, max(0.0, self.max_weight * warmup * self.ema_reliability))
        self.last_output = VSRAGateOutput(
            weight=weight,
            raw_reliability=raw,
            ema_reliability=self.ema_reliability,
            warmup=warmup,
            conflict=scores["conflict"],
            disadvantage=scores["disadvantage"],
        )
        return self.last_output

    def state_dict(self) -> Dict[str, object]:
        return {
            "step": self.step,
            "ema_reliability": self.ema_reliability,
            "last_output": None if self.last_output is None else self.last_output.as_dict(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.step = max(0, int(state.get("step", 0)))
        ema = state.get("ema_reliability")
        self.ema_reliability = None if ema is None else min(1.0, max(0.0, float(ema)))
        output = state.get("last_output")
        if isinstance(output, Mapping):
            self.last_output = VSRAGateOutput(**{key: float(value) for key, value in output.items()})
        else:
            self.last_output = None
