import numpy as np


def normalize(wave, max_peak, min_peak):
    max_val = np.abs(wave).max()
    if max_val > max_peak:
        return wave * (max_peak / max_val)
    if min_peak is not None and max_val < min_peak:
        return wave * (min_peak / max_val)
    return wave


def match_array_shapes(source_array, target_array):
    if source_array.shape == target_array.shape:
        return source_array

    source_len = source_array.shape[1]
    target_len = target_array.shape[1]

    if source_len > target_len:
        return source_array[:, :target_len]
    if source_len < target_len:
        padding = ((0, 0), (0, target_len - source_len))
        return np.pad(source_array, padding, mode="constant")

    return source_array
