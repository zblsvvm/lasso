"""
将数据进行归一化处理。使用线性回归解决多项式回归问题的数据预处理。
Standardization the data
Data Preprocessing for Solving Polynomial Regression Problems Using Linear Regression
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class StandardScaler:
    """使用方法：先fit，后transform"""
    """How to use: first fit, then transform"""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """根据训练数据集X获得数据的均值和方差。对测试数据集进行归一化要使用训练数据集的均值和方差。"""
        """The mean and variance of the data are obtained from the training data set X. 
        Standardize the test data set to use the mean and variance of the training data set."""
        assert X.ndim == 2, "the dimension of X must be 2"
        self.mean_ = np.array([np.mean(X[:, i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:, i]) for i in range(X.shape[1])])
        return self

    def transform(self, X):
        """将X根据StandardScaler进行均值和方差归一化处理"""
        """Normalize the training data set or test data set X according to the StandardScaler"""
        assert X.ndim == 2, "the dimension of X must be 2"
        assert self.mean_ is not None and self.scale_ is not None, \
            "must fit before transform"
        assert X.shape[1] == len(self.mean_), \
            "the feature number of X must be equal to mean_ and std_"
        resX = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            resX[:, col] = (X[:, col] - self.mean_[col]) / self.scale_[col]
        return resX

def polynomialFeatures(X, degree):
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    c = polynomial_features.fit_transform(X)
    return c
