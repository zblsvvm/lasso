"""
提供一些虚拟数据
"""
import numpy as np
from model_selection import train_test_split


class Data:
    def poly_data_1(self, seed=666):
        np.random.seed(seed)
        size = 200
        x = np.random.uniform(-3, 3, size=size)
        X = x.reshape(-1, 1)
        y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=size)
        return train_test_split(X, y)
