"""
衡量机器学习算法的指标 Metrics for measuring machine learning algorithms
"""
import numpy as np
from math import sqrt


def mean_squared_error(y_true, y_predict):
    """计算y_true和y_predict之间的MSE，均方误差"""
    """Calculate the MSE between y_true and y_predict, mean square error"""
    assert len(y_true) == len(y_predict), \
        "the size of the y_true must be equal to the size of y_predict"
    return np.sum((y_true - y_predict) ** 2) / len(y_true)


def root_mean_squared_error(y_true, y_predict):
    """计算y_true和y_predict之间的RMSE，均方根误差"""
    return sqrt(mean_squared_error(y_true, y_predict))


def mean_absolute_error(y_true, y_predict):
    """计算y_true和y_predict之间的MAE，平均绝对误差"""
    assert y_true.shape[0] == y_predict.shape[0], \
        "the size of y_true must be equal to the size of y_predict"
    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)


def r2_score(y_true, y_predict):
    """计算y_true和y_predict之间的R Square"""
    """Calculate R Square between y_true and y_predict"""
    # 分子是使用我们的模型预测产生的错误 Molecules are errors that are predicted using our model
    # 分子是使用y=y_mean(Baseline Model)预测产生的错误
    # Molecules are errors predicted using y=y_mean(Baseline Model)
    return 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)

