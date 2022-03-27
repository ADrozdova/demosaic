from math import pi, sqrt, exp

import numpy as np
from scipy import signal

from guidedfilter_MLRI import guidedfilter_MLRI
from mosaic_bayer import get_masks


def green_interpolation(mosaic, mask, pattern, sigma, eps):
    rawq = mosaic.sum(axis=2)
    maskGr = np.zeros(rawq.shape)
    maskGb = np.zeros(rawq.shape)

    height, width = rawq.shape

    if pattern == [1, 0, 2, 1]:  # 'grbg')
        maskGr = get_masks(height, width, [1, 0, 2, 0])[:, :, 1]
        maskGb = get_masks(height, width, [0, 0, 2, 1])[:, :, 1]
    elif pattern == [0, 1, 1, 2]:  # 'rggb'
        maskGr = get_masks(height, width, [0, 1, 0, 2])[:, :, 1]
        maskGb = get_masks(height, width, [0, 0, 1, 2])[:, :, 1]
    elif pattern == [1, 2, 0, 1]:  # 'gbrg'
        maskGr = get_masks(height, width, [0, 2, 0, 1])[:, :, 1]
        maskGb = get_masks(height, width, [1, 2, 0, 0])[:, :, 1]
    elif pattern == [2, 1, 1, 0]:  # 'bggr'
        maskGr = get_masks(height, width, [2, 0, 1, 0])[:, :, 1]
        maskGb = get_masks(height, width, [2, 1, 0, 0])[:, :, 1]

    # guide image
    Kh = np.array([[1 / 2, 0, 1 / 2]])
    Kv = Kh.T

    rawh = signal.convolve2d(rawq, Kh, mode="same")
    rawv = signal.convolve2d(rawq, Kv, mode="same")

    Guidegh = mosaic[:, :, 1] + rawh * mask[:, :, 0] + rawh * mask[:, :, 2]
    Guiderh = mosaic[:, :, 0] + rawh * maskGr
    Guidebh = mosaic[:, :, 2] + rawh * maskGb
    Guidegv = mosaic[:, :, 1] + rawv * mask[:, :, 0] + rawv * mask[:, :, 2]
    Guiderv = mosaic[:, :, 0] + rawv * maskGb
    Guidebv = mosaic[:, :, 2] + rawv * maskGr

    # tentative image
    h, v = 3, 3

    F = np.array([[-1, 0, 2, 0, -1]])

    # horizontal

    tentativeRh = guidedfilter_MLRI(
        Guidegh, mosaic[:, :, 0], mask[:, :, 0],
        signal.convolve2d(Guidegh * mask[:, :, 0], F, mode="same"),
        signal.convolve2d(mosaic[:, :, 0], F, mode="same"),
        mask[:, :, 0], h, v, eps,
    )

    tentativeGrh = guidedfilter_MLRI(
        Guiderh, mosaic[:, :, 1] * maskGr, maskGr,
        signal.convolve2d(Guiderh * maskGr, F, mode="same"),
        signal.convolve2d(mosaic[:, :, 1] * maskGr, F, mode="same")
        , maskGr, h, v, eps
    )

    tentativeBh = guidedfilter_MLRI(
        Guidegh, mosaic[:, :, 2], mask[:, :, 2],
        signal.convolve2d(Guidegh * mask[:, :, 2], F, mode="same"),
        signal.convolve2d(mosaic[:, :, 2], F, mode="same"),
        mask[:, :, 2],
        h,
        v,
        eps,
    )

    tentativeGbh = guidedfilter_MLRI(
        Guidebh, mosaic[:, :, 1] * maskGb, maskGb,
        signal.convolve2d(Guidebh * maskGb, F, mode="same"),
        signal.convolve2d(mosaic[:, :, 1] * maskGb, F, mode="same"),
        maskGb, h, v, eps
    )

    # vertical
    F = F.T
    tentativeRv = guidedfilter_MLRI(
        Guidegv, mosaic[:, :, 0], mask[:, :, 0],
        signal.convolve2d(Guidegv * mask[:, :, 0], F, mode="same"),
        signal.convolve2d(mosaic[:, :, 0], F, mode="same"),
        mask[:, :, 0], v, h, eps,
    )

    tentativeGrv = guidedfilter_MLRI(
        Guiderv, mosaic[:, :, 1] * maskGb, maskGb,
        signal.convolve2d(Guiderv * maskGb, F, mode="same"),
        signal.convolve2d(mosaic[:, :, 1] * maskGb, F, mode="same"),
        maskGb, v, h, eps
    )

    tentativeBv = guidedfilter_MLRI(
        Guidegv, mosaic[:, :, 2], mask[:, :, 2],
        signal.convolve2d(Guidegv * mask[:, :, 2], F, mode="same"),
        signal.convolve2d(mosaic[:, :, 2], F, mode="same"),
        mask[:, :, 2], v, h, eps,
    )

    tentativeGbv = guidedfilter_MLRI(
        Guidebv, mosaic[:, :, 1] * maskGr, maskGr,
        signal.convolve2d(Guidebv * maskGr, F, mode="same"),
        signal.convolve2d(mosaic[:, :, 1] * maskGr, F, mode="same"),
        maskGr, v, h, eps
    )

    # residual
    residualGrh = (mosaic[:, :, 1] - tentativeGrh) * maskGr
    residualGbh = (mosaic[:, :, 1] - tentativeGbh) * maskGb
    residualRh = (mosaic[:, :, 0] - tentativeRh) * mask[:, :, 0]
    residualBh = (mosaic[:, :, 2] - tentativeBh) * mask[:, :, 2]
    residualGrv = (mosaic[:, :, 1] - tentativeGrv) * maskGb
    residualGbv = (mosaic[:, :, 1] - tentativeGbv) * maskGr
    residualRv = (mosaic[:, :, 0] - tentativeRv) * mask[:, :, 0]
    residualBv = (mosaic[:, :, 2] - tentativeBv) * mask[:, :, 2]

    # residual linear interpolation
    Kh = np.array([[1 / 2, 0, 1 / 2]])
    residualGrh = signal.convolve2d(residualGrh, Kh, mode="same")
    residualGbh = signal.convolve2d(residualGbh, Kh, mode="same")
    residualRh = signal.convolve2d(residualRh, Kh, mode="same")
    residualBh = signal.convolve2d(residualBh, Kh, mode="same")

    Kv = Kh.T

    residualGrv = signal.convolve2d(residualGrv, Kv, mode="same")
    residualGbv = signal.convolve2d(residualGbv, Kv, mode="same")
    residualRv = signal.convolve2d(residualRv, Kv, mode="same")
    residualBv = signal.convolve2d(residualBv, Kv, mode="same")

    # add tentative image
    Grh = (tentativeGrh + residualGrh) * mask[:, :, 0]
    Gbh = (tentativeGbh + residualGbh) * mask[:, :, 2]
    Rh = (tentativeRh + residualRh) * maskGr
    Bh = (tentativeBh + residualBh) * maskGb
    Grv = (tentativeGrv + residualGrv) * mask[:, :, 0]
    Gbv = (tentativeGbv + residualGbv) * mask[:, :, 2]
    Rv = (tentativeRv + residualRv) * maskGb
    Bv = (tentativeBv + residualBv) * maskGr

    # vertical and horizontal color difference
    difh = mosaic[:, :, 1] + Grh + Gbh - mosaic[:, :, 0] - mosaic[:, :, 2] - Rh - Bh
    difv = mosaic[:, :, 1] + Grv + Gbv - mosaic[:, :, 0] - mosaic[:, :, 2] - Rv - Bv

    ### Combine Vertical and Horizontal Color Differences ###
    # color difference gradient
    Kh = np.array([[1, 0, -1]])
    Kv = Kh.T
    difh2 = np.abs(signal.convolve2d(difh, Kh, mode="same"))
    difv2 = np.abs(signal.convolve2d(difv, Kv, mode="same"))

    # directional weight
    K = np.ones((5, 5))

    convh = signal.convolve2d(difh2, K, mode="full")
    convv = signal.convolve2d(difv2, K, mode="full")

    Ww = convh[2:-2, :-4]
    We = convh[2:-2, 4:]
    Wn = convv[:-4, 2:-2]
    Ws = convv[4:, 2:-2]

    Ww = 1.0 / (np.square(Ww) + 1e-32)
    We = 1.0 / (np.square(We) + 1e-32)
    Ws = 1.0 / (np.square(Ws) + 1e-32)
    Wn = 1.0 / (np.square(Wn) + 1e-32)

    # gaussian filter
    r = range(-int(9 / 2), int(9 / 2) + 1)
    h = np.array(
        [1 / (sigma * sqrt(2 * pi)) * exp(-float(x) ** 2 / (2 * sigma ** 2)) for x in r]
    )

    Kw = np.array([[0, 0, 0, 0, 1, 1, 1, 1, 1]]) * h
    Ke = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0]]) * h

    Ke = Ke / Ke.sum()
    Kw = Kw / Kw.sum()

    Ks = Ke.T
    Kn = Kw.T

    difn = signal.convolve2d(difv, Kn, mode="same")
    difs = signal.convolve2d(difv, Ks, mode="same")
    difw = signal.convolve2d(difh, Kw, mode="same")
    dife = signal.convolve2d(difh, Ke, mode="same")

    Wt = Ww + We + Wn + Ws
    dif = (Wn * difn + Ws * difs + Ww * difw + We * dife) / Wt

    return (dif + rawq) * (1 - mask[:, :, 1]) + rawq * mask[:, :, 1]
