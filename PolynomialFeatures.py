#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:42:00 2019

@author: michaelyoung
"""

class PolynomialFeatures:
    def _init_(self, degree=2):
        self._degree = degree

    def fit(self, X):
        """fit()函数计算输出特征的个数。此处无用但保留接口"""
        pass

    def transform(self, X):
        # 假设没有交叉项 Assume that there is no cross-multiplication
        features = X.shape[1]
        degree = self._degree
        Xp = np.ones((len(X), 1))
        for i in range(features):
            for j in range(degree):
                X_1 = X[:, i] ** (j + 1)
                X_1 = X_1.reshape(len(X_1), 1)
                Xp = np.hstack([Xp, X_1])
        Xp = Xp[:, 1:]
        return Xp
    
    def LSsolution(self, X, Y):

        xtx=np.dot(np.transpose(X), X)
        xty=np.dot(np.transpose(X), Y)
    
    
        # xtx=np.transpose(xtx)
        # xty=np.transpose(xty)
    
        # print(xtx)
        # print(xty)
    
        try:
                beta=np.linalg.solve(xtx, xty)
        except:
                xtxinv=np.linalg.inv(xtx)
                beta=np.dot(xtxinv, np.dot(np.transpose(X), Y))
    
        return beta        