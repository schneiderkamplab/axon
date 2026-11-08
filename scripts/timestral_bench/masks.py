"""Timestral DHF masks and positions, ported from Mistral (Apache-2.0)."""

import numpy as np


def compute_enc_key_validity(observed, attend_missing):
    return np.maximum.accumulate(observed, axis=-1) if attend_missing else observed


def enc_causal_mask(key_valid):
    length = key_valid.shape[1]
    return (np.tri(length, dtype=bool)[None] & key_valid[:, None]) | np.eye(length, dtype=bool)[
        None
    ]


def build_forecast_time_mask(length, horizon, key_valid):
    rows = key_valid.shape[0]
    top = np.concatenate(
        [enc_causal_mask(key_valid), np.zeros((rows, length, horizon), dtype=bool)], axis=2
    )
    bottom = np.concatenate(
        [
            np.broadcast_to(key_valid[:, None], (rows, horizon, length)),
            np.broadcast_to(np.tri(horizon, dtype=bool), (rows, horizon, horizon)),
        ],
        axis=2,
    )
    return np.concatenate([top, bottom], axis=1)[:, None]


def build_two_stream_time_mask(min_context, n, block_size, key_valid, known_rows):
    p = np.arange(n)[None]
    start = min_context[:, None]
    active = p >= start
    anchor = np.where(active, start + block_size * (np.maximum(p - start, 0) // block_size), 0)
    eye = np.eye(n, dtype=bool)[None]
    enc = enc_causal_mask(key_valid)
    enc = np.where(known_rows[:, None, None], key_valid[:, None] | eye, enc)
    dec_enc = active[:, :, None] & (p[:, None] < anchor[:, :, None]) & key_valid[:, None]
    dec_dec = (anchor[:, :, None] == anchor[:, None]) & np.tri(n, dtype=bool)[None]
    dec_dec &= active[:, :, None] & active[:, None]
    dec_dec |= eye
    top = np.concatenate([enc, np.zeros_like(enc)], axis=2)
    bottom = np.concatenate([dec_enc, dec_dec], axis=2)
    return np.concatenate([top, bottom], axis=1)[:, None]


def build_two_stream_variate_mask(group_ids, key_valid, n_dec, target_rows=None, known_rows=None):
    """Return (groups, positions, 1, variates, variates) attention keep masks.

    Forecast rows are flattened from (batch, variates, time), so each group
    must have the same size and occupy a contiguous, sorted run.
    """
    rows = len(group_ids)
    groups, counts = np.unique(group_ids, return_counts=True)
    if not rows or np.any(counts != counts[0]):
        raise ValueError("variate groups must be nonempty and equally sized")
    size = int(counts[0])
    if not np.array_equal(group_ids, np.repeat(groups, size)):
        raise ValueError("variate groups must be contiguous and sorted")
    batch = len(groups)
    valid = np.concatenate([key_valid, np.ones((rows, n_dec), dtype=bool)], axis=1)
    valid = valid.reshape(batch, size, -1).transpose(0, 2, 1)
    mask = np.broadcast_to(
        valid[:, :, None, None, :], (batch, valid.shape[1], 1, size, size)
    ).copy()
    if target_rows is not None:
        targets = target_rows.reshape(batch, size)
        forbid = ~targets[:, :, None] & targets[:, None, :]
        mask &= ~forbid[:, None, None]
    if known_rows is not None:
        known = known_rows.reshape(batch, size)
        forbid = known[:, :, None] & ~known[:, None, :]
        mask &= ~forbid[:, None, None]
    return mask | np.eye(size, dtype=bool)[None, None, None]


def first_observed_positions(observed, groups, n_dec, window):
    first = np.zeros(len(groups), dtype=np.int64)
    for group in np.unique(groups):
        selected = groups == group
        first[selected] = observed[selected].any(axis=0).argmax()
    n = observed.shape[1]
    positions = np.arange(n) if window else np.arange(n + n_dec)
    positions = positions[None] - first[:, None]
    return np.concatenate([positions, positions], axis=1) if window else positions
