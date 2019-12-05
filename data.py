"""
提供一些虚拟数据
"""
import numpy as np
from model_selection import train_test_split
from sklearn import datasets


class Data:
    def poly_data_1(self, seed=666):
        """随机生成的1维2阶含有高斯噪音的数据"""
        """Randomly generated 1D and 2nd order data with Gaussian noise"""
        np.random.seed(seed)
        size = 200
        x = np.random.uniform(-3, 3, size=size)
        X = x.reshape(-1, 1)
        y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=size)
        
        
        
        return train_test_split(X, y)

    def multi_data_boston(self):
        """从datasets中加载波士顿房价数据，为13维数据"""
        """Load Boston house price data from datasets for 13D data"""
        boston = datasets.load_boston()
        X = boston.data
        y = boston.target
        return X, y
 
    def rgb_testdata(self):
        dataset=np.loadtxt('testdata_rgb.txt', skiprows=1)
        dataset=np.loadtxt('testdata_rgb.txt', skiprows=1)
        X=dataset[:, 0:8]
        Y=dataset[:,8]
        return X,Y 
        

