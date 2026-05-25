"""
Reference: https://github.com/TongkunGuan/CCD/blob/543109a1e1d9acd15080abb3e4e72d68588ba493/Dino/utils/kmeans.py#L7
"""

import numpy as np
from scipy.cluster.vq import *
from pylab import *


def clusterpixels(im, k):
    im = np.array(im)
    h, w = im.shape
    im = im.astype(float).reshape(-1)
    # 聚类， k是聚类数目
    centroids, variance = kmeans(im, k)
    code, distance = vq(im, centroids)
    code = code.reshape(h, w)
    fc = sum(code[:, 0])
    lc = sum(code[:, -1])
    fr = sum(code[0, :])
    lr = sum(code[-1, :])
    num = int(fr > w // 2) + int(lr > w // 2) + int(fc > h // 2) + int(lc > h // 2)
    if num >= 3:
        return 1 - code
    else:
        return code
