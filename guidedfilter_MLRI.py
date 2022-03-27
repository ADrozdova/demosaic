import numpy as np

from boxfilter import boxfilter


def guidedfilter_MLRI(G, R, mask, img, p, M, h, v, eps):
    N = boxfilter(M, h, v)
    N[np.nonzero(N == 0)] = 1

    mean_Ip = boxfilter(img * p * M, h, v) / N
    mean_II = boxfilter(img * img * M, h, v) / N

    a = mean_Ip / (mean_II + eps)
    N2 = boxfilter(mask, h, v)
    N2[np.nonzero(N2 == 0)] = 1
    mean_G = boxfilter(G * mask, h, v) / N2
    mean_R = boxfilter(R * mask, h, v) / N2
    b = mean_R - a * mean_G

    dif = (
        boxfilter(G * G * mask, h, v) * a * a
        + b * b * N2
        + boxfilter(R * R * mask, h, v)
        + 2 * a * b * boxfilter(G * mask, h, v)
        - 2 * b * boxfilter(R * mask, h, v)
        - 2 * a * boxfilter(R * G * mask, h, v)
    )

    dif = dif / N2
    dif[np.nonzero(dif < 0.01)] = 0.01

    dif = 1.0 / dif
    wdif = boxfilter(dif, h, v)
    wdif[np.nonzero(wdif < 0.01)] = 0.01
    mean_a = boxfilter(a * dif, h, v) / wdif
    mean_b = boxfilter(b * dif, h, v) / wdif
    return mean_a * G + mean_b
