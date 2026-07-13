"""
Feature map: standardize, then lift to a normalized random Fourier feature space.

The RFF sampler and both scalers are fit during training and pickled. Inference
and every downstream analysis loads those exact objects. Nothing is refit later.
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler

from .config import D, RFF_GAMMA, RFF_SEED

RFF_FILE      = "rff_sampler.pkl"
SCALER_X_FILE = "scaler_x.pkl"
SCALER_C_FILE = "scaler_c.pkl"


def fit_scalers(X_raw, C_raw):
    scaler_x = StandardScaler().fit(X_raw)
    scaler_c = StandardScaler().fit(C_raw)
    return scaler_x, scaler_c


def fit_rff(X_std, n_components=D, gamma=RFF_GAMMA, seed=RFF_SEED):
    rff = RBFSampler(gamma=gamma, n_components=n_components, random_state=seed)
    rff.fit(X_std)
    return rff


def phi(X_std, rff):
    """RFF map with L2 normalization, so phi^T phi = 1 and the Born rule
    p = phi^T rho phi is a probability."""
    Phi = rff.transform(X_std)
    norms = np.linalg.norm(Phi, axis=1, keepdims=True).clip(min=1e-12)
    return Phi / norms


def phi_of_raw(X_raw, rff, scaler_x):
    """Convenience: raw behavioral values -> normalized RFF features."""
    X_raw = np.asarray(X_raw, dtype=np.float64).reshape(-1, scaler_x.n_features_in_)
    return phi(scaler_x.transform(X_raw), rff)


def save_features(out_dir, rff, scaler_x, scaler_c):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in ((RFF_FILE, rff),
                      (SCALER_X_FILE, scaler_x),
                      (SCALER_C_FILE, scaler_c)):
        with open(out_dir / name, "wb") as f:
            pickle.dump(obj, f)


def load_features(model_dir):
    """Returns (rff, scaler_x, scaler_c) from the training run."""
    model_dir = Path(model_dir)
    out = []
    for name in (RFF_FILE, SCALER_X_FILE, SCALER_C_FILE):
        path = model_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run scripts/02_train/train_qdm.py first."
            )
        with open(path, "rb") as f:
            out.append(pickle.load(f))
    return tuple(out)
