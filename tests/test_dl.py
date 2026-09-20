"""Sanity checks for the PyTorch MLP wrapper (src/dl.py).

Run: python -m tests.test_dl
"""
import numpy as np

from src.dl import TorchMLP


def test_mlp_learns_a_simple_linear_relationship():
    """Positive control: on a noiseless linear target, a small MLP should
    get close. If this fails, something is wrong with the training loop
    itself (learning rate, standardization, early stopping), independent
    of whether real financial data has any signal to find.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (2000, 5)).astype(np.float32)
    true_w = np.array([1.0, -2.0, 0.5, 0.0, 0.0])
    y = X @ true_w
    model = TorchMLP(hidden_sizes=(16,), dropout=0.0, max_epochs=100, patience=15, random_state=0)
    model.fit(X, y)
    preds = model.predict(X)
    corr = np.corrcoef(preds, y)[0, 1]
    assert corr > 0.8, f"expected strong correlation on a noiseless linear target, got {corr:.3f}"
    print("PASS: test_mlp_learns_a_simple_linear_relationship")


def test_mlp_predict_shape_and_determinism():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (300, 4)).astype(np.float32)
    y = rng.normal(0, 1, 300).astype(np.float32)
    model = TorchMLP(hidden_sizes=(8,), max_epochs=5, patience=2, random_state=0)
    model.fit(X, y)
    preds1 = model.predict(X)
    preds2 = model.predict(X)
    assert preds1.shape == (300,)
    assert np.allclose(preds1, preds2), "predict() should be deterministic given a fitted model"
    print("PASS: test_mlp_predict_shape_and_determinism")


if __name__ == "__main__":
    test_mlp_learns_a_simple_linear_relationship()
    test_mlp_predict_shape_and_determinism()
    print("\nAll deep-learning wrapper sanity checks passed.")
