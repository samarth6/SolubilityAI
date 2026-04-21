from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
import numpy as np

CLASS_COLORS = {
    "Highly Soluble": "#1971c2",
    "Soluble":        "#2f9e44",
    "Poorly Soluble": "#e67700",
    "Insoluble":      "#c92a2a",
}

FILL_COLORS = {
    "Highly Soluble": "rgba(25,113,194,0.12)",
    "Soluble":        "rgba(47,158,68,0.12)",
    "Poorly Soluble": "rgba(230,119,0,0.12)",
    "Insoluble":      "rgba(201,42,42,0.12)",
}

def _trimf(x, a, b, c):
    if x <= a or x >= c: return 0.0
    if x <= b: return (x - a) / (b - a)
    return (c - x) / (c - b)

def _trapmf(x, a, b, c, d):
    if x <= a or x >= d: return 0.0
    if x <= b: return (x - a) / (b - a)
    if x <= c: return 1.0
    return (d - x) / (d - c)

def mu_highly_soluble(logS): return _trapmf(logS, -1.0, 0.0, 2.0, 2.0)
def mu_soluble(logS):        return _trimf(logS, -3.0, -1.0, 0.0)
def mu_poorly_soluble(logS): return _trimf(logS, -5.0, -3.5, -2.0)
def mu_insoluble(logS):      return _trapmf(logS, -8.0, -8.0, -5.0, -4.0)

@dataclass
class FuzzyResult:
    memberships: Dict[str, float] = field(default_factory=dict)

    @property
    def dominant(self):
        return max(self.memberships, key=self.memberships.get)

    @property
    def confidence(self):
        return self.memberships[self.dominant]

    @property
    def color(self):
        return CLASS_COLORS[self.dominant]

def classify(logS):
    return FuzzyResult(memberships={
        "Highly Soluble": round(mu_highly_soluble(logS), 4),
        "Soluble":        round(mu_soluble(logS),         4),
        "Poorly Soluble": round(mu_poorly_soluble(logS),  4),
        "Insoluble":      round(mu_insoluble(logS),       4),
    })

class FuzzyInferenceEngine:
    _delta_range = np.linspace(-2, 2, 300)

    @staticmethod
    def _logP_low(x):   return _trapmf(x, -5, -5, 1, 3)
    @staticmethod
    def _logP_mid(x):   return _trimf(x, 1, 3.5, 6)
    @staticmethod
    def _logP_high(x):  return _trapmf(x, 4, 6, 10, 10)
    @staticmethod
    def _mw_mid(x):     return _trimf(x, 150, 350, 550)
    @staticmethod
    def _mw_large(x):   return _trapmf(x, 450, 600, 900, 900)
    @staticmethod
    def _hbd_low(x):    return _trapmf(x, 0, 0, 1, 3)
    @staticmethod
    def _hbd_high(x):   return _trapmf(x, 2, 4, 15, 15)
    @staticmethod
    def _delta_neg(x):  return _trapmf(x, -2, -2, -1, 0)
    @staticmethod
    def _delta_zero(x): return _trimf(x, -0.5, 0, 0.5)
    @staticmethod
    def _delta_pos(x):  return _trapmf(x, 0, 1, 2, 2)

    def infer(self, logP, mol_wt, num_hbd):
        r1 = min(self._logP_high(logP), self._mw_large(mol_wt))
        r2 = min(self._logP_low(logP),  self._hbd_high(num_hbd))
        r3 = min(self._logP_mid(logP),  self._mw_mid(mol_wt))
        r4 = min(self._logP_high(logP), self._hbd_low(num_hbd))
        r5 = self._logP_low(logP)
        agg = np.array([max(
            min(r1, self._delta_neg(x)),
            min(r2, self._delta_pos(x)),
            min(r3, self._delta_zero(x)),
            min(r4, self._delta_neg(x)),
            min(r5, self._delta_pos(x)),
        ) for x in self._delta_range])
        denom = agg.sum()
        if denom < 1e-9: return 0.0
        return float(np.dot(agg, self._delta_range) / denom)

def hybrid_predict(xgb_logS, logP, mol_wt, num_hbd, alpha=0.25):
    engine = FuzzyInferenceEngine()
    delta = engine.infer(logP, mol_wt, num_hbd)
    hybrid_logS = xgb_logS + alpha * delta
    return hybrid_logS, classify(hybrid_logS)

def membership_curves(n_points=500):
    xs = np.linspace(-8, 2, n_points)
    return {
        "x":              xs,
        "Highly Soluble": np.array([mu_highly_soluble(v) for v in xs]),
        "Soluble":        np.array([mu_soluble(v)         for v in xs]),
        "Poorly Soluble": np.array([mu_poorly_soluble(v)  for v in xs]),
        "Insoluble":      np.array([mu_insoluble(v)       for v in xs]),
    }
