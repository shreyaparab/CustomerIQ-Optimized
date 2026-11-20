import numpy as np
import pandas as pd
import pytest

from src.optimization.budget_optimizer import optimize_campaign


def scores(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"CustomerID": range(n), "purchase_probability": rng.random(n)})


@pytest.mark.parametrize("budget", [0, 1, 19.99, 100, 555.55, 5000])
def test_budget_never_exceeded(budget):
    r = optimize_campaign(scores(), budget, margin=300, cost=20)
    assert r.total_cost <= budget + 1e-9


def test_variable_costs_respect_budget():
    s = scores(150, 3)
    cost = np.random.default_rng(1).uniform(5, 60, len(s))
    margin = np.random.default_rng(2).uniform(100, 500, len(s))
    r = optimize_campaign(s, 400, margin, cost)
    assert r.total_cost <= 400 + 1e-9
    assert (r.selected["expected_net_value"] > 0).all()


def test_matches_bruteforce_on_small_instance():
    from itertools import combinations
    s = pd.DataFrame({"purchase_probability": [0.9, 0.5, 0.7, 0.3, 0.8, 0.6]})
    cost = np.array([30, 10, 25, 5, 40, 20.0])
    margin = 100.0
    nv = s["purchase_probability"].to_numpy() * margin - cost
    budget = 70
    best = 0.0
    for r in range(len(s) + 1):
        for combo in combinations(range(len(s)), r):
            if cost[list(combo)].sum() <= budget and all(nv[i] > 0 for i in combo):
                best = max(best, nv[list(combo)].sum())
    res = optimize_campaign(s, budget, margin, cost)
    assert res.expected_net_value == pytest.approx(best)


def test_max_contacts_limit():
    r = optimize_campaign(scores(), 10_000, 300, 10, max_contacts=7)
    assert r.n_selected <= 7


def test_empty_data():
    r = optimize_campaign(scores(0), 1000, 300, 20)
    assert r.n_selected == 0 and r.total_cost == 0


def test_zero_budget():
    assert optimize_campaign(scores(), 0, 300, 20).n_selected == 0


def test_negative_cost_rejected():
    with pytest.raises(ValueError):
        optimize_campaign(scores(), 100, 300, -5)


def test_negative_budget_rejected():
    with pytest.raises(ValueError):
        optimize_campaign(scores(), -1, 300, 20)


def test_missing_margin_rejected():
    with pytest.raises(ValueError):
        optimize_campaign(scores(), 100, np.nan, 20)


def test_unprofitable_contacts_not_selected():
    r = optimize_campaign(scores(), 10_000, margin=10, cost=50)
    assert r.n_selected == 0


def test_zero_cost_contacts_handled():
    r = optimize_campaign(scores(), 0.0, 100, 0.0)
    assert r.total_cost == 0
