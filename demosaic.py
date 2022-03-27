import numpy as np

from green import green_interpolation
from red_blue import interpolation


def demosaic(pattern, mosaic, mask, sigma=1., eps=1e-32):
    mosaic = mosaic.astype(float)
    green = green_interpolation(mosaic, mask, pattern, sigma, eps)

    green = np.clip(green, 0, 255)
    red = interpolation(green, mosaic[:, :, 0], mask[:, :, 0], eps)
    blue = interpolation(green, mosaic[:, :, 2], mask[:, :, 2], eps)

    result = np.zeros_like(mosaic)
    result[:, :, 0] = np.clip(red, 0, 255)
    result[:, :, 1] = green
    result[:, :, 2] = np.clip(blue, 0, 255)

    return result.astype(int)
