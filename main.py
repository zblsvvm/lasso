import matplotlib.pyplot as plt
from LassoRegression import LassoRegression
from data import Data
import numpy as np
from model_selection import train_test_split, cross_val_score
from read_lasso_file import readfile
import time

t = time.time()

X, y = readfile('C:/Users/82569/Desktop/Residuals_Match_DMhydro_Less_z.txt')

nd = X.shape[0]
nf = X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)
lasR = LassoRegression(4)
alpha = np.logspace(-8, 1, 12, base=2)
final_coefs = {}
graph = {}
scores = []

lasso = None
while (lasso == None):

    l = input('should we do lasso? (True/False)')
    if l == 'True':
        lasso = True
    elif l == 'False':
        lasso = False
    else:
        print('incorrect value')
method = str(input('what method? (bgd, sgd, normal)'))
file_name = str(input('name of the file'))

for a in alpha:
    lasR.fit(X_train, y_train, lasso=lasso, method=method, alpha=a)
    scores.append(lasR.score(X_test, y_test))
    final_coefs[a] = lasR.coef_
    graph[a] = lasR.graph

fig, axes = plt.subplots(2, 3, sharex=False, sharey=False, figsize=(15, 10))
for j in range(len(final_coefs.keys())):
    axes[0][0].plot(alpha, [final_coefs[i][j] for i in alpha])
axes[0][0].set_xlabel('Lambda')
axes[0][0].set_ylabel('Coefficients')

axes[0][1].plot(alpha, scores)
axes[0][1].set_xlabel('Lambda')
axes[0][1].set_ylabel('Scores')

lasR.fit(X_train, y_train, lasso=True, method='bgd', alpha=1)
y_pred = lasR.predict(X_test)
axes[0][2].scatter(y_pred, y_test)
axes[0][2].set_xlabel('Predicted y')
axes[0][2].set_ylabel('Observed y')

for i in range(X_train.shape[1]):
    axes[1][0].scatter(X_test[:, i], y_test - y_pred)
axes[1][0].set_xlabel('Coefficients\' values')
axes[1][0].set_ylabel('Residue')

for i in range(X_train.shape[1]):
    for j in range(X_train.shape[1]):
        axes[1][1].plot([final_coefs[a][i] for a in alpha], [final_coefs[a][j] for a in alpha])
axes[1][1].set_xlabel('C[i]')
axes[1][1].set_ylabel('C[j]')

fig.savefig(file_name)
