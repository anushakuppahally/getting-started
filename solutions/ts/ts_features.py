
# ts_features.py

import numpy as np


def resample_series(series, length=128):
    """Resample each channel to a fixed number of timesteps."""
    series = np.asarray(series, dtype=np.float32)

    if series.ndim != 2:
        raise ValueError("Expected (channels, timesteps) array.")

    channels, timesteps = series.shape

    if timesteps == 0:
        return np.zeros((channels, length), dtype=np.float32)

    if timesteps == 1:
        return np.repeat(series, length, axis=1)

    old_t = np.linspace(0.0, 1.0, timesteps)
    new_t = np.linspace(0.0, 1.0, length)

    return np.stack([
        np.interp(new_t, old_t, channel)
        for channel in series
    ]).astype(np.float32)


def activation_signals(hidden_states, n_layers=8):
    """
    Convert hidden states into four time-series channels per layer.

    Input:  (layers, tokens, hidden_dimension)
    Output: (4 * selected_layers, tokens)
    """
    h = np.asarray(hidden_states, dtype=np.float32)

    if h.ndim != 3:
        raise ValueError(
            "Expected hidden states shaped "
            "(layers, tokens, hidden_dimension)."
        )

    n_available, n_tokens, hidden_dim = h.shape

    if n_tokens == 0 or hidden_dim == 0:
        raise ValueError("Hidden states cannot be empty.")

    indices = np.linspace(
        0,
        n_available - 1,
        min(n_layers, n_available),
    ).round().astype(int)

    channels = []

    for idx in indices:
        layer = h[idx]  # (tokens, hidden_dimension)

        # 1. Normalized activation magnitude.
        magnitude = np.log1p(
            np.linalg.norm(layer, axis=1) / np.sqrt(hidden_dim)
        )

        # 2. Token-to-token movement.
        movement = np.zeros(n_tokens, dtype=np.float32)

        if n_tokens > 1:
            movement[1:] = np.log1p(
                np.linalg.norm(np.diff(layer, axis=0), axis=1)
                / np.sqrt(hidden_dim)
            )

        # 3. Similarity to the preceding token.
        adjacent = np.ones(n_tokens, dtype=np.float32)

        if n_tokens > 1:
            a, b = layer[:-1], layer[1:]
            denom = (
                np.linalg.norm(a, axis=1)
                * np.linalg.norm(b, axis=1)
            )

            adjacent[1:] = np.divide(
                np.sum(a * b, axis=1),
                denom,
                out=np.ones(n_tokens - 1, dtype=np.float32),
                where=denom > 1e-12,
            )

        # 4. Similarity to the first generated token.
        first = layer[0]
        denom = np.linalg.norm(layer, axis=1) * np.linalg.norm(first)

        initial = np.divide(
            layer @ first,
            denom,
            out=np.ones(n_tokens, dtype=np.float32),
            where=denom > 1e-12,
        )

        channels.extend([magnitude, movement, adjacent, initial])

    return np.stack(channels).astype(np.float32)


def temporal_features(series, length=128):
    """Create fixed-length statistical features for each channel."""
    series = resample_series(series, length=length)
    features = []

    for channel in series:
        features.extend([
            np.mean(channel),
            np.std(channel),
            np.quantile(channel, 0.10),
            np.quantile(channel, 0.90),
            channel[-1] - channel[0],
            np.mean(np.abs(np.diff(channel))),
            np.std(np.diff(channel)),
        ])

        # Four consecutive segment means.
        for segment in np.array_split(channel, 4):
            features.append(np.mean(segment))

    return np.asarray(features, dtype=np.float32)


def trace_to_series(hidden_states, n_layers=8, length=128):
    """Return MiniRocket input: (channels, timesteps)."""
    series = activation_signals(
        hidden_states,
        n_layers=n_layers,
    )

    return resample_series(series, length=length)


def trace_to_features(hidden_states, n_layers=8, length=128):
    """Return handcrafted TRR features: (n_features,)."""
    series = trace_to_series(
        hidden_states,
        n_layers=n_layers,
        length=length,
    )

    return temporal_features(series, length=length)
