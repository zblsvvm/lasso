import matplotlib.pyplot as plt
from LassoRegression import LassoRegression
from data import Data
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

# Depending on your actual file location

X, y = readfile('C:/Users/82569/Desktop/Residuals_Match_DMhydro_Less_z.txt')

X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)

method = "bgd"

"""If you want to use the drawing function, just remove the #"""
"""If you want to use the drawing function, just remove the #"""
"""If you want to use the drawing function, just remove the #"""

#plt_coefs_lambs(X_train, y_train, method)

#plt_scores_lambs(X_train, y_train, X_test, y_test, method)

#plt_pred_obser(X_train, y_train, X_test, y_test, method, lamb=0)

#plt_residu_lambs(X_train, y_train, X_test, y_test, method)

#plt_scores_datasize(X_train, y_train, X_test, y_test, method)

#plt_square_lambs(X_train, y_train, X_test, y_test, method)

#plt_coefs_coefs(X_train, y_train, method)