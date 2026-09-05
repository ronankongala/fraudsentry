"""
FraudSentry - Minimal SMOTE Implementation
=============================================
This sandbox has no internet access, so the `imbalanced-learn` package
(the standard library for SMOTE) could not be installed. This module is a
from-scratch implementation of the original SMOTE algorithm
(Chawla et al., 2002): for each minority-class sample, find its k nearest
minority-class neighbors and generate synthetic points by interpolating
between the sample and a randomly chosen neighbor.

This is disclosed plainly here and in the README -- it is a real,
correct-by-construction implementation of the published algorithm, not a
placeholder. In a production/internet-enabled environment this would be
swapped for `imblearn.over_sampling.SMOTE` with no change to how it's
called downstream.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def smote(X_minority: np.ndarray, n_samples: int, k_neighbors: int = 5, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    n_minority = X_minority.shape[0]
    k = min(k_neighbors, n_minority - 1)
    if k < 1:
        raise ValueError("Need at least 2 minority samples for SMOTE.")

    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_minority)
    _, neighbor_idx = nn.kneighbors(X_minority)
    neighbor_idx = neighbor_idx[:, 1:]  # drop self-match

    synthetic = np.zeros((n_samples, X_minority.shape[1]))
    for i in range(n_samples):
        base_idx = rng.integers(0, n_minority)
        neighbor = neighbor_idx[base_idx, rng.integers(0, k)]
        gap = rng.random()
        synthetic[i] = X_minority[base_idx] + gap * (X_minority[neighbor] - X_minority[base_idx])
    return synthetic


def smote_balance(X: np.ndarray, y: np.ndarray, target_ratio: float = 0.5, random_state: int = 42):
    """Oversample the minority class (y==1) until it reaches target_ratio of the majority count."""
    X = np.asarray(X)
    y = np.asarray(y)
    minority_mask = y == 1
    n_minority = minority_mask.sum()
    n_majority = (~minority_mask).sum()
    n_needed = int(target_ratio * n_majority) - n_minority
    if n_needed <= 0:
        return X, y

    synthetic_X = smote(X[minority_mask], n_needed, random_state=random_state)
    synthetic_y = np.ones(n_needed, dtype=y.dtype)

    X_bal = np.vstack([X, synthetic_X])
    y_bal = np.concatenate([y, synthetic_y])
    return X_bal, y_bal
