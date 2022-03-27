from scipy import signal
import numpy as np


def simple_interpolation(mosaic, masks):
    H_G =  np.array([[0, 1/4, 0], [1/4, 1, 1/4], [0, 1/4, 0]])

    H_RB = np.array([[1/4, 1/2, 1/4], [1/2, 1, 1/2], [1/4, 1/2, 1/4], ] )

    R = signal.convolve2d(mosaic[:, :, 0], H_RB, mode="same")
    G = signal.convolve2d(mosaic[:, :, 1], H_G, mode="same")
    B = signal.convolve2d(mosaic[:, :, 2], H_RB, mode="same")

    res = np.zeros(mosaic.shape)

    res[:, :, 0] = np.clip(R, 0, 255)
    res[:, :, 1] = np.clip(G, 0, 255)
    res[:, :, 2] = np.clip(B, 0, 255)

    return res
