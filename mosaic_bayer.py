import numpy as np


def get_masks(n_rows, n_cols, pattern):
    if isinstance(pattern, list):
        pattern = np.asarray(pattern).reshape((2, 2))
    filter_masks = []
    for color in range(3):
        color_mask = np.tile(pattern == color, (n_rows, n_cols))[:n_rows, :n_cols]
        filter_masks.append(color_mask)
    return np.dstack(filter_masks)


def mosaic_bayer(rgb, pattern=None):
    if pattern is None:
        pattern = [0, 1, 1, 2]

    mosaic = np.zeros(rgb.shape)
    mask = get_masks(rgb.shape[0], rgb.shape[1], pattern)

    mosaic[:, :, 0] = mask[:, :, 0] * rgb[:, :, 0]
    mosaic[:, :, 1] = mask[:, :, 1] * rgb[:, :, 1]
    mosaic[:, :, 2] = mask[:, :, 2] * rgb[:, :, 2]

    return mosaic, mask
