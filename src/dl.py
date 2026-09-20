"""
A small PyTorch MLP wrapped with an sklearn-style fit/predict interface, so
it plugs directly into src/ml.py's run_ml_walk_forward as just another
`build_model()` -- same purged walk-forward, same permutation-test negative
control, no separate validation code path for "the deep learning one".

Deliberately small and conservatively regularized: 1-2 hidden layers,
dropout, weight decay, and early stopping on a held-out slice of the
TRAINING data (never the test data) -- financial tabular data with a low
signal-to-noise ratio and only a few thousand training rows per walk-
forward window is exactly the regime where a large, deep, or under-
regularized network overfits fastest, not where more capacity helps. The
early-stopping validation split takes the chronologically LAST slice of
the training rows (not a random split), which approximately respects time
ordering since the input rows are date-major.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TorchMLP:
    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (32, 16),
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 60,
        patience: int = 8,
        batch_size: int = 256,
        val_fraction: float = 0.15,
        random_state: int = 0,
    ):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.random_state = random_state
        self._model = None
        self._mean = None
        self._std = None

    def _build_net(self, n_features: int) -> nn.Module:
        layers = []
        in_dim = n_features
        for h in self.hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(self.dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchMLP":
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self._mean = X.mean(axis=0, keepdims=True)
        self._std = X.std(axis=0, keepdims=True)
        self._std[self._std == 0] = 1.0
        X = (X - self._mean) / self._std

        n = len(X)
        n_val = max(1, int(n * self.val_fraction))
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]

        X_train_t = torch.from_numpy(X_train)
        y_train_t = torch.from_numpy(y_train)
        X_val_t = torch.from_numpy(X_val)
        y_val_t = torch.from_numpy(y_val)

        self._model = self._build_net(X.shape[1])
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        n_train = len(X_train_t)

        for epoch in range(self.max_epochs):
            self._model.train()
            perm = torch.randperm(n_train)
            for start in range(0, n_train, self.batch_size):
                idx = perm[start:start + self.batch_size]
                optimizer.zero_grad()
                pred = self._model(X_train_t[idx])
                loss = loss_fn(pred, y_train_t[idx])
                loss.backward()
                optimizer.step()

            self._model.eval()
            with torch.no_grad():
                val_loss = loss_fn(self._model(X_val_t), y_val_t).item()

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        X = (X - self._mean) / self._std
        self._model.eval()
        with torch.no_grad():
            pred = self._model(torch.from_numpy(X))
        return pred.numpy().ravel()
