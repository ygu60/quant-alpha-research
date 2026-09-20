"""Sanity checks for the purged walk-forward ML pipeline (src/ml.py).

These check the VALIDATION PLUMBING (no leakage across the train/test
purge boundary, the pipeline runs end to end), not whether any model finds
real alpha -- that's for RESEARCH_MEMO.md, backed by the real-data run.

Run: python -m tests.test_ml
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.ml import run_ml_walk_forward


def _synthetic_panel(n_days=60, n_assets=6, seed=0):
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rng = np.random.default_rng(seed)
    tickers = [f"T{i}" for i in range(n_assets)]
    prices = pd.DataFrame(
        {t: 100 * (1 + rng.normal(0.0005, 0.015, n_days)).cumprod() for t in tickers},
        index=dates,
    )
    volume = pd.DataFrame(
        {t: rng.integers(1_000_000, 5_000_000, n_days).astype(float) for t in tickers},
        index=dates,
    )
    return prices, volume


def test_purge_prevents_training_label_leakage():
    """Construct one exact walk-forward window (train_days + test_days ==
    total length), fit with a deterministic model (Ridge -- no internal
    randomness), then shock ONLY prices strictly inside the test period and
    refit. If the purge boundary is correct, none of the training labels
    (which require prices up to train_days-1+horizon) can be affected by a
    price change that starts at test_dates[0], so the fitted model must be
    IDENTICAL. A model that changes here would mean a training label reached
    past the purge boundary into the shocked region -- a leakage bug.
    """
    # train_days must comfortably exceed the longest feature warm-up (the
    # 50-day moving-average gap feature) so training rows survive dropna.
    train_days, test_days, horizon = 80, 15, 5
    prices_a, volume = _synthetic_panel(n_days=train_days + test_days, n_assets=6, seed=1)

    prices_b = prices_a.copy()
    test_start_pos = train_days
    prices_b.iloc[test_start_pos:] *= 5.0  # violent shock, test period only

    captured = {}

    class CapturingRidge(Ridge):
        def fit(self, X, y):
            captured.setdefault("coefs", []).append(y.copy())
            return super().fit(X, y)

    run_ml_walk_forward(prices_a, volume, build_model=lambda: CapturingRidge(alpha=1.0),
                         horizon=horizon, train_days=train_days, test_days=test_days,
                         top_frac=0.3, min_train_rows=50)
    y_a = captured["coefs"][0]

    captured["coefs"] = []
    run_ml_walk_forward(prices_b, volume, build_model=lambda: CapturingRidge(alpha=1.0),
                         horizon=horizon, train_days=train_days, test_days=test_days,
                         top_frac=0.3, min_train_rows=50)
    y_b = captured["coefs"][0]

    assert y_a.shape == y_b.shape, "purge should drop the exact same training rows regardless of the test shock"
    assert np.allclose(y_a, y_b), "training labels changed when only TEST-period prices were shocked -- purge leak"
    print("PASS: test_purge_prevents_training_label_leakage")


def test_pipeline_runs_end_to_end():
    prices, volume = _synthetic_panel(n_days=400, n_assets=8, seed=2)
    result = run_ml_walk_forward(
        prices, volume, build_model=lambda: Ridge(alpha=10.0),
        horizon=5, train_days=150, test_days=50, top_frac=0.25,
    )
    assert len(result.window_stats) > 0
    assert "sharpe" in result.combined_stats
    assert not result.predictions.empty
    print("PASS: test_pipeline_runs_end_to_end")


def test_shuffled_labels_run_without_crashing():
    """Negative-control smoke test: the shuffle_labels path must run
    end-to-end (used in the real study to check for leakage -- a real edge
    surviving label shuffling would indicate a bug, not alpha).
    """
    prices, volume = _synthetic_panel(n_days=300, n_assets=6, seed=3)
    result = run_ml_walk_forward(
        prices, volume, build_model=lambda: Ridge(alpha=10.0),
        horizon=5, train_days=150, test_days=50, top_frac=0.25, shuffle_labels=True,
    )
    assert len(result.window_stats) > 0
    print("PASS: test_shuffled_labels_run_without_crashing")


if __name__ == "__main__":
    test_purge_prevents_training_label_leakage()
    test_pipeline_runs_end_to_end()
    test_shuffled_labels_run_without_crashing()
    print("\nAll ML pipeline sanity checks passed.")
