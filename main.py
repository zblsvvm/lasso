import matplotlib.pyplot as plt
from LassoRegression import LassoRegression
import numpy as np
from model_selection import train_test_split, cross_val_score
from read_lasso_file import readfile
from visualisation import plt_coefs_lambs
from visualisation import plt_scores_lambs
from visualisation import plt_pred_obser
from visualisation import plt_residu_lambs
from visualisation import plt_scores_datasize
from visualisation import plt_square_lambs
from visualisation import plt_coefs_coefs
from sklearn.decomposition import PCA
from model_selection import _k_split

# Depending on your actual file location

X, y = readfile('testdata_rgb.txt')

X_train, X_test, y_train, y_test = train_test_split(X, y, seed=466)

lasso_reg = LassoRegression(degree=2, method="bgd", lamb=0.2)
lasso_reg.fit(X_train, y_train)
coefficients = lasso_reg.theta

method = None
while method == None:
    l = input('What method do you want to use ?\n a  Batch Gradient Descent\n b  Stochastic Gradient descent\n c  '
              'Mini-batch '
              'Gradient Descent\n d  Normal Equation Solution\n e  Coordinate Descent\n')
    if l == "a":
        method = "bgd"
    elif l == "b":
        method = "sgd"
    elif l == "c":
        method = "mgd"
    elif l == "d":
        method = "normal"
    elif l == "e":
        method = "cd"
    else:
        print('incorrect value')

# method = "bgd"
#lasso_reg = LassoRegression(degree=2, method="bgd", lamb=0.2)
#lasso_reg.fit(X_train, y_train)
#coefficients = lasso_reg.theta
#print(coefficients)
"""If you want to use the drawing function, just remove the #"""
"""If you want to use the drawing function, just remove the #"""
"""If you want to use the drawing function, just remove the #"""

# plt_coefs_lambs(X_train, y_train, method)

plt_scores_lambs(X_train, y_train, X_test, y_test, method)

# plt_pred_obser(X_train, y_train, X_test, y_test, method, lamb=0.2)

# plt_residu_lambs(X_train, y_train, X_test, y_test, method)

# plt_scores_datasize(X_train, y_train, X_test, y_test, method)

# plt_square_lambs(X, y, 5, method)

# plt_coefs_coefs(X_train, y_train, method)
