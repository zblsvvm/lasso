"""
数据可视化，包含一些绘图方法
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_in_order(X_train, y_train, y_predict):
    xt = X_train.T[0]
    yt = y_train
    xp = np.sort(xt)
    yp = y_predict[np.argsort(xt)]
    plt.scatter(xt, yt)
    plt.plot(xp, yp, color="r")
    plt.show()
