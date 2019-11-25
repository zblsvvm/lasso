from LassoRegression import LassoRegression
from data import Data
import numpy as np
from model_selection import train_test_split, cross_val_score

X, y = Data().multi_data_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)
lasso_reg = LassoRegression(degree=1)




