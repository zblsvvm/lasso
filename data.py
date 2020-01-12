
import numpy as np
from sklearn import datasets
from read_lasso_file import readfile
from model_selection import train_test_split

"""
获取数据集 Get the dataset
# @Author   : Tian Xiao
"""

def readfile(location):

    # number of features
    nof=8


    n = tuple(np.arange(nof))

    # output column
    y = 8

    # Extracted matrix
    m = np.loadtxt(location, skiprows=1, usecols=n)

    # Extracted expFit vector
    v = np.loadtxt(location, skiprows=1, usecols=y)
    return m, v




class Data:
    def poly_1(self, seed=666):
        """随机生成的1维2阶含有高斯噪音的数据"""
        np.random.seed(seed)
        size = 200
        x = np.random.uniform(-3, 3, size=size)
        X = x.reshape(-1, 1)
        y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=size)
        return X, y

    def boston(self):
        """从datasets中加载波士顿房价数据，为13维数据"""
        boston = datasets.load_boston()
        X = boston.data
        y = boston.target
        X = X[y < 50.0]
        y = y[y < 50.0]


        return X, y

    def Residuals_Match_DMhydro_Less_z(self):

        #SPECIFY FILEPATH HERE:
        # location='D:\Residuals_Match_DMhydro_Less_z.txt'
        location = 'testdata_rgb.txt'

        X, y = readfile(location)
        return X, y
