"""
存放了一些测试中产生的代码 Stored some code generated in the test
"""
from LassoRegression import LassoRegression
from data import Data
import numpy as np
from model_selection import train_test_split

X, y = Data().multi_data_boston()


def poly_test(X, y, degree=1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)
    poly_reg = LassoRegression(degree=degree)
    poly_reg.fit(X_train, y_train, lasso=False, method="normal")
    print(poly_reg.score(X_test, y_test))
    X_test = X_test[:5]
    y_predict = poly_reg.predict(X_test)
    y_true = y_test[:5]
    for i in range(len(y_true)):
        print(y_true[i], y_predict[i])
    print()


def lasso_test(X, y, degree=1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)
    lasso_reg = LassoRegression(degree=degree)
    lasso_reg.fit(X_train, y_train, lasso=True, method="bgd")
    print(lasso_reg.score(X_test, y_test))
    X_test = X_test[:5]
    y_predict = lasso_reg.predict(X_test)
    y_true = y_test[:5]
    for i in range(len(y_true)):
        print(y_true[i], y_predict[i])
    print()
