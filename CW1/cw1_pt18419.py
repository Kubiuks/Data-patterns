import os, sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from utilities import load_points_from_file, view_data_segments


def setify20(xs, ys):
    xSets = []
    ySets = []
    for i in range(0, len(xs)//20):
        xSets.append(xs[i*20:i*20+20])
        ySets.append(ys[i * 20:i * 20 + 20])
    return np.array(xSets), np.array(ySets)


def least_squares(xs, ys):
    ones = np.ones(20)
    xtrans = np.array((ones, xs))
    x = xtrans.T
    xxt = np.matmul(xtrans, x)
    a = np.linalg.inv(xxt) @ xtrans @ ys.T
    return a


xs, ys = load_points_from_file("train_data/noise_3.csv")
xSets, ySets = setify20(xs, ys)

print(len(xSets))
for i in range(0, len(xSets)):
    a = least_squares(xSets[i], ySets[i])
    plt.scatter(xSets[i], ySets[i])
    ds = a[0] + a[1]*xSets[i]
    plt.plot(xSets[i], ds, c='r')
plt.show()
