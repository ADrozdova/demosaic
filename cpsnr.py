import numpy as np

def cpsnr(X, Y, peak=255):
    dif = np.square(X[5:-5, 5:-5] - Y[5:-5, 5:-5])
    mse = dif.sum() / dif.size

    return 10 * np.log10(peak * peak / mse)
