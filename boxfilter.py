import cv2
import numpy as np


def boxfilter(img, h, v):
    res = np.zeros(img.shape)
    cv2.boxFilter(np.float32(img), 0, (h, v), res, (-1, -1), False, cv2.BORDER_DEFAULT)
    return res
