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
from visualisation import create_plots
import time
from data import Data

# Paste the dataset location below
location='testdata_rgb.txt'



d=Data()
# X, y = d.boston()
X, y = d.Residuals_Match_DMhydro_Less_z()

X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)

method = None
while method == None:
    l = input('What method do you want to use ?\n a  Batch Gradient Descent\n b  Stochastic Gradient descent\n c  '
              'Mini-batch '
              'Gradient Descent\n d  Ordinary Least Squares Solution\n e  Coordinate Descent\n f Accelerated Proximal Gradient Descent\n g  Alternating Direction Method of Multipliers ')
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
    elif l == "f":
        method = "pgd_acc"
    elif l == "g":
        method = "admm"

    else:
        print('Invalid Input')

# method = "bgd"

"""If you want to use the drawing function, just remove the #"""
"""If you want to use the drawing function, just remove the #"""
"""If you want to use the drawing function, just remove the #"""


choice= None
while choice == None:
    c = input('What would you like to do?\n a Optimise \n b  Residuals vs Lambda \n c Evolution of coefficients \n d Dataset Size vs Scores \n e Coefficients vs Coefficients ')

    if c == 'a':
        create_plots(X, y, X_train, y_train, X_test, y_test, method)

    elif c == 'b':
        plt_residu_lambs(X_train, y_train, X_test, y_test, method)

    elif c =='c':
        plt_coefs_lambs(X_train, y_train, method)

    elif c == 'd':
        plt_scores_datasize(X_train, y_train, X_test, y_test, method)

    elif c == 'e':
        plt_coefs_coefs(X_train, y_train, method)

    else:
        print('Invalid Input')










