"""
衡量机器学习算法的指标 Metrics for measuring machine learning algorithms
# @Author   : Tian Xiao
"""
import numpy as np


def mean_squared_error(y_true, y_predict):
    """计算y_true和y_predict之间的MSE，均方误差"""
    assert len(y_true) == len(y_predict), \
        "the size of the y_true must be equal to the size of y_predict"
    return np.sum((y_true - y_predict) ** 2) / len(y_true)


def r2_score(y_true, y_predict):
    """计算y_true和y_predict之间的R Square"""
    # 分子是使用我们的模型预测产生的错误
    # 分子是使用y=y_mean(Baseline Model)预测产生的错误
    return 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)
