import numpy as np
from scipy import signal

from guidedfilter_MLRI import guidedfilter_MLRI


def interpolation(green, mosaic_color, mask_color, eps):
    # parameters for guided upsampling
    h, v = 5, 5
    # Laplacian
    F = np.array(
        [
            [0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0],
            [-1, 0, 4, 0, -1],
            [0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0],
        ]
    )

    # R interpolation
    tentative = guidedfilter_MLRI(
        green, mosaic_color, mask_color,
        signal.convolve2d(green * mosaic_color, F, mode="same"),
        signal.convolve2d(mosaic_color, F, mode="same"),
        mask_color, h, v, eps,
    )
    residual = mask_color * (mosaic_color - tentative)
    # Bilinear interpoaltion
    H = np.array([[1 / 4, 1 / 2, 1 / 4], [1 / 2, 1, 1 / 2], [1 / 4, 1 / 2, 1 / 4]])
    residual = signal.convolve2d(residual, H, mode="same")
    return residual + tentative
