import matplotlib.pyplot as plt
from LassoRegression import LassoRegression
from data import Data
import numpy as np
from model_selection import train_test_split, cross_val_score
from read_lasso_file import readfile


X, y = readfile('C:/Users/82569/Desktop/Residuals_Match_DMhydro_Less_z.txt')

X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)

alphas = np.logspace(-10, 1, 12, base=2)
lasso = False
method = "bgd"

fig, axes = plt.subplots(2, 3, sharex=False, sharey=False, figsize=(15, 10))
d = 0
for i in range(2):
    for j in range(3):
        scores = []
        lasR = LassoRegression(d)
        for a in alphas:
            lasR.fit(X_train, y_train, lasso=lasso, method=method, alpha=a)
            scores.append(lasR.score(X_test, y_test))
        axes[i][j].plot(alphas, scores)
        d += 1
fig.savefig("file_name")




