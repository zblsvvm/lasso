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
    
    def dataset_rgb(self):
        dataset = np.loadtxt(fname = "testdata_rgb.txt", skiprows=1, unpack=False)
        X=dataset[:, 0:8]
        y=dataset[:,8]
        return X,y
    
    def random_dataset_known(self, nd):
        X=np.asarray(np.arange(nd))
        X=X.reshape(-1,1)
        
        for i in range(4):
            X=np.hstack((X, np.asarray(np.arange(nd).reshape(-1,1))))
        
        Y=[]
        for i in range(nd):
            y=3+2*X[i][0] +1.5*X[i][1]-3*X[i][2] + 4.5*X[i][3] +0.1*X[i][4] +0.2*X[i][0]**2 +0.5*X[i][1]**2-3*X[i][2]**2 - 4*X[i][3]**2 +X[i][4]**2
            Y.append(y)
        
        return X,np.asarray(Y).reshape(-1,1)
        
            
        
        
        
