"""Budget-constrained customer selection (0/1 knapsack).

Objective: maximise sum(p_i * margin_i - cost_i) over selected customers,
subject to sum(cost_i) <= budget and optionally a max number of contacts.
Only customers with positive expected net value are eligible.

Costs are converted to integer cents so the budget constraint is exact.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OptimizationResult:
    selected: pd.DataFrame
    total_cost: float
    expected_net_value: float
    expected_purchases: float
    budget: float
    n_selected: int
    assumptions: dict


def expected_net_value(prob, margin, cost):
    return np.asarray(prob, float) * np.asarray(margin, float) - np.asarray(cost, float)


def optimize_campaign(scores: pd.DataFrame, budget: float, margin, cost,
                      max_contacts: int | None = None) -> OptimizationResult:
    """scores needs column 'purchase_probability'. margin/cost may be scalars
    or per-row Series/arrays aligned with scores."""
    if "purchase_probability" not in scores.columns:
        raise ValueError("scores must contain 'purchase_probability'.")
    if budget is None or not np.isfinite(budget) or budget < 0:
        raise ValueError("Budget must be a non-negative number.")
    n = len(scores)
    margin_arr = np.broadcast_to(np.asarray(margin, float), (n,)).copy() if n else np.array([])
    cost_arr = np.broadcast_to(np.asarray(cost, float), (n,)).copy() if n else np.array([])
    if n and (np.isnan(margin_arr).any() or np.isnan(cost_arr).any()):
        raise ValueError("Margin and cost inputs must not be missing.")
    if n and (cost_arr < 0).any():
        raise ValueError("Contact costs must not be negative.")
    if n and (margin_arr < 0).any():
        raise ValueError("Margins must not be negative.")
    prob = scores["purchase_probability"].to_numpy(float) if n else np.array([])
    if n and not np.isfinite(prob).all():
        raise ValueError("Probabilities must be finite.")

    assumptions = {"budget": float(budget), "max_contacts": max_contacts,
                   "formula": "expected_net_value = p * margin - cost",
                   "note": "Scenario estimate from purchase propensity; "
                           "not measured profit or causal uplift."}
    empty = scores.iloc[0:0].copy()
    if n == 0 or budget == 0:
        return OptimizationResult(_decorate(empty, [], prob, margin_arr, cost_arr),
                                  0.0, 0.0, 0.0, float(budget), 0, assumptions)

    nv = prob * margin_arr - cost_arr
    eligible = np.where(nv > 0)[0]
    # Zero-cost positive-value customers cost nothing: take them all (subject to max_contacts).
    free = [i for i in eligible if cost_arr[i] == 0]
    paid = [i for i in eligible if cost_arr[i] > 0]
    chosen: list[int] = []

    if max_contacts is None:
        chosen.extend(free)
        remaining = int(round(budget * 100))
        chosen.extend(_knapsack(paid, nv, cost_arr, remaining))
    else:
        free_sorted = sorted(free, key=lambda i: -nv[i])[:max_contacts]
        chosen.extend(free_sorted)
        slots = max_contacts - len(free_sorted)
        if slots > 0 and paid:
            chosen.extend(_knapsack_limited(paid, nv, cost_arr,
                                            int(round(budget * 100)), slots))

    chosen = sorted(set(chosen), key=lambda i: -nv[i])
    total_cost = float(cost_arr[chosen].sum()) if chosen else 0.0
    # Hard safety check (cents-exact).
    assert round(total_cost * 100) <= int(round(budget * 100)) + 0, "budget exceeded"
    sel = _decorate(scores, chosen, prob, margin_arr, cost_arr)
    return OptimizationResult(
        sel, total_cost, float(nv[chosen].sum()) if chosen else 0.0,
        float(prob[chosen].sum()) if chosen else 0.0, float(budget),
        len(chosen), assumptions)


def _decorate(scores, idx, prob, margin, cost):
    if len(idx) == 0:
        out = scores.iloc[0:0].copy()
    else:
        out = scores.iloc[idx].copy()
        out["assumed_margin"] = margin[idx]
        out["contact_cost"] = cost[idx]
        out["expected_net_value"] = prob[idx] * margin[idx] - cost[idx]
        return out.sort_values("expected_net_value", ascending=False)
    for c in ("assumed_margin", "contact_cost", "expected_net_value"):
        out[c] = pd.Series(dtype=float)
    return out


def _to_cents(cost_arr, i) -> int:
    return max(1, int(round(cost_arr[i] * 100)))


def _knapsack(items, value, cost, capacity: int) -> list[int]:
    """Exact 0/1 knapsack when tractable, else greedy by value density."""
    if not items or capacity <= 0:
        return []
    w = [_to_cents(cost, i) for i in items]
    if len(items) * capacity > 30_000_000:
        return _greedy(items, value, cost, capacity)
    best = np.zeros(capacity + 1)
    keep = np.zeros((len(items), capacity + 1), dtype=bool)
    for j, i in enumerate(items):
        wj, vj = w[j], value[i]
        if wj > capacity:
            continue
        cand = best[:capacity + 1 - wj] + vj
        better = cand > best[wj:]
        keep[j, wj:] = better
        best[wj:] = np.where(better, cand, best[wj:])
    chosen, c = [], capacity
    for j in range(len(items) - 1, -1, -1):
        if keep[j, c]:
            chosen.append(items[j])
            c -= w[j]
    return chosen


def _knapsack_limited(items, value, cost, capacity: int, max_items: int) -> list[int]:
    """Greedy by value density with hard budget and count limits."""
    return _greedy(items, value, cost, capacity, max_items)


def _greedy(items, value, cost, capacity: int, max_items: int | None = None) -> list[int]:
    order = sorted(items, key=lambda i: -(value[i] / max(cost[i], 1e-9)))
    chosen, used = [], 0
    for i in order:
        if max_items is not None and len(chosen) >= max_items:
            break
        c = _to_cents(cost, i)
        if used + c <= capacity:
            chosen.append(i)
            used += c
    return chosen
