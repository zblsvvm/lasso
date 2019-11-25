"""
将数据进行归一化处理。使用线性回归解决多项式回归问题的数据预处理。
Standardization the data
Data Preprocessing for Solving Polynomial Regression Problems Using Linear Regression
"""
import numpy as np


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


class PolynomialFeatures:
    def __init__(self, degree=2):
        """
        :param degree: 转化后的阶数，而数据的维度存在于X矩阵本身。
        The transformed order, and the dimension of the data exists in the X matrix itself
        """
        self._degree = degree

    def fit(self, X):
        """fit()函数应该计算输出特征的个数。此处无用但保留类似于sklearn包中的接口"""
        """The fit() function should calculate the number of output features. 
        Not useful here but retains interfaces similar to those in the sklearn package"""
        pass

    def transform(self, X):
        """将X矩阵转化为适应多项式特诊的新矩阵"""
        """Transforming the X matrix into a new matrix that accommodates polynomial special diagnosis"""
        # TODO cross-multiplication
        # 注意：由于实现过程的原因，在此处没有添加第0列的全1向量，而是在fit的过程中添加
        # Note: Due to the implementation process, the all 1 vector of column 0 is not added here,
        # but is added during the fit process.
        features = X.shape[1]
        degree = self._degree
        Xp = np.ones((len(X), 1))
        for i in range(features):
            for j in range(degree):
                X_1 = X[:, i] ** (j + 1)
                # print(X.shape, X_1.shape, len(X_1))
                X_1 = X_1.reshape(len(X_1), 1)
                Xp = np.hstack([Xp, X_1])
        Xp = Xp[:, 1:]
        return Xp
