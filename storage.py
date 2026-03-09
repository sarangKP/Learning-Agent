"""
Storage layer — loads/saves bandit tables.

File backend (default): tables/bandit_N.npy + bandit_Q.npy
Redis backend: swap in load_tables_redis / save_tables_redis when ready.

N_CONTEXTS is imported from state_classifier (single source of truth).
"""

from __future__ import annotations
import os
import numpy as np

# FIX: import from single source of truth instead of re-defining
from state_classifier import N_CONTEXTS

N_ACTIONS = 7

TABLE_DIR = os.environ.get("BANDIT_TABLE_DIR", "tables")
N_PATH = os.path.join(TABLE_DIR, "bandit_N.npy")
Q_PATH = os.path.join(TABLE_DIR, "bandit_Q.npy")


def load_tables() -> tuple[np.ndarray, np.ndarray]:
    """Returns (N, Q). Creates zero tables if files don't exist."""
    os.makedirs(TABLE_DIR, exist_ok=True)
    if os.path.exists(N_PATH) and os.path.exists(Q_PATH):
        N = np.load(N_PATH)
        Q = np.load(Q_PATH)
        # Guard against stale tables from before the 45-context expansion.
        # If you had existing tables/ from the 36-context version, delete them
        # and restart — this warning will tell you if that's needed.
        if N.shape != (N_CONTEXTS, N_ACTIONS) or Q.shape != (N_CONTEXTS, N_ACTIONS):
            print(
                f"[storage] WARNING: table shape {N.shape} does not match "
                f"expected ({N_CONTEXTS}, {N_ACTIONS}). Re-initialising."
            )
            N = np.zeros((N_CONTEXTS, N_ACTIONS), dtype=float)
            Q = np.zeros((N_CONTEXTS, N_ACTIONS), dtype=float)
    else:
        N = np.zeros((N_CONTEXTS, N_ACTIONS), dtype=float)
        Q = np.zeros((N_CONTEXTS, N_ACTIONS), dtype=float)
    return N, Q


def save_tables(N: np.ndarray, Q: np.ndarray) -> None:
    os.makedirs(TABLE_DIR, exist_ok=True)
    np.save(N_PATH, N)
    np.save(Q_PATH, Q)


def reset_tables() -> None:
    """Wipe and reinitialise — useful for tests."""
    N = np.zeros((N_CONTEXTS, N_ACTIONS), dtype=float)
    Q = np.zeros((N_CONTEXTS, N_ACTIONS), dtype=float)
    save_tables(N, Q)