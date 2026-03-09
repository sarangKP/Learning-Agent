"""
Storage layer — loads/saves bandit tables.

File backend (default): tables/bandit_N.npy + bandit_Q.npy
Redis backend: swap in load_tables_redis / save_tables_redis when ready.

N_CONTEXTS is imported from state_classifier (single source of truth).

── Concurrency safety ────────────────────────────────────────────────────────
FIX: the original load → mutate → save sequence was not atomic. Two concurrent
requests could both load the same stale tables, compute independent updates,
and the second save would silently overwrite the first — losing one reward
update per race.

Two-layer protection is used here:
  1. threading.Lock  — prevents races between threads inside the same uvicorn
                        worker process (covers `uvicorn --reload` / single worker).
  2. fcntl.flock     — advisory exclusive lock on the .npy file for inter-process
                        safety when multiple uvicorn workers are spawned
                        (e.g. `uvicorn main:app --workers 4`).

Both locks are acquired together inside `tables_locked()`, a context manager
that callers use to wrap the entire load → mutate → save transaction:

    with tables_locked() as (N, Q):
        bandit = UCBBandit(N, Q)
        bandit.update(ctx, action, reward)
        bandit.select_action(ctx)
        return bandit.N, bandit.Q   # caller returns updated tables

See main.py for the usage pattern.
"""

from __future__ import annotations
import fcntl
import os
import threading
from contextlib import contextmanager
from typing import Generator, Tuple

import numpy as np

from state_classifier import N_CONTEXTS

N_ACTIONS = 7

TABLE_DIR = os.environ.get("BANDIT_TABLE_DIR", "tables")
N_PATH    = os.path.join(TABLE_DIR, "bandit_N.npy")
Q_PATH    = os.path.join(TABLE_DIR, "bandit_Q.npy")
LOCK_PATH = os.path.join(TABLE_DIR, ".bandit.lock")   # advisory lock file

# In-process mutex — fast path for single-worker deployments
_thread_lock = threading.Lock()


# ── Public context manager ────────────────────────────────────────────────────

@contextmanager
def tables_locked() -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Acquire both the in-process mutex and the file-level advisory lock, then
    yield (N, Q).  The caller mutates the arrays and returns the updated
    versions; those are saved before the locks are released.

    Usage::

        with tables_locked() as (N, Q):
            bandit = UCBBandit(N, Q)
            bandit.update(ctx, action, reward)
            action_id, scores = bandit.select_action(ctx)
            # signal updated tables back via the yielded references
            N[:] = bandit.N
            Q[:] = bandit.Q
    """
    os.makedirs(TABLE_DIR, exist_ok=True)
    with _thread_lock:
        lock_file = open(LOCK_PATH, "w")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            N, Q = _load()
            yield N, Q
            _save(N, Q)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load() -> Tuple[np.ndarray, np.ndarray]:
    """Load tables from disk; returns zero tables on missing or stale files."""
    if os.path.exists(N_PATH) and os.path.exists(Q_PATH):
        N = np.load(N_PATH)
        Q = np.load(Q_PATH)
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


def _save(N: np.ndarray, Q: np.ndarray) -> None:
    os.makedirs(TABLE_DIR, exist_ok=True)
    np.save(N_PATH, N)
    np.save(Q_PATH, Q)


# ── Convenience wrappers (kept for test compatibility) ────────────────────────

def load_tables() -> Tuple[np.ndarray, np.ndarray]:
    """
    Direct load without locking — use only in tests or read-only contexts.
    For production read-modify-write use tables_locked() instead.
    """
    os.makedirs(TABLE_DIR, exist_ok=True)
    return _load()


def save_tables(N: np.ndarray, Q: np.ndarray) -> None:
    """Direct save without locking — use only in tests."""
    _save(N, Q)


def reset_tables() -> None:
    """Wipe and reinitialise — useful for tests."""
    N = np.zeros((N_CONTEXTS, N_ACTIONS), dtype=float)
    Q = np.zeros((N_CONTEXTS, N_ACTIONS), dtype=float)
    _save(N, Q)