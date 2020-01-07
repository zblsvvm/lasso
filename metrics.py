"""
衡量机器学习算法的指标 Metrics for measuring machine learning algorithms
"""
import numpy as np
def mean_squared_error(y_true, y_predict):
    """计算y_true和y_predict之间的RMSE，均方根误差"""
    assert len(y_true) == len(y_predict)
    return np.sum((y_true - y_predict) ** 2)


def sum_of_squares(y_true):
    """计算y_true和y_predict之间的MAE，平均绝对误差"""
    return  np.sum((y_true - np.mean(y_true)) ** 2)


def r2_score(y_true, y_predict):
    return 1 - mean_squared_error(y_true, y_predict) / sum_of_squares(y_true)

