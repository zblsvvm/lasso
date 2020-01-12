"""
Data visualization, including some drawing methods
"""
import matplotlib.pyplot as plt
import numpy as np
from LassoRegression import LassoRegression
from sklearn.decomposition import PCA
from model_selection import _k_split


def plot_in_order(X_train, y_train, y_predict):
    xt = X_train.T[0]
    yt = y_train
    xp = np.sort(xt)
    yp = y_predict[np.argsort(xt)]
    plt.scatter(xt, yt)
    plt.plot(xp, yp, color="r")
    plt.show()


def plt_coefs_lambs(X_train, y_train, method):
    """绘制参数与lambda的关系"""
    coefs = []
    lambs = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    lasso_reg = None
    for l in lambs:
        lasso_reg = LassoRegression(degree=2, method=method, lamb=l)
        lasso_reg.fit(X_train, y_train)
        coefs.append(lasso_reg.theta)
    coefs = np.array(coefs)
    fea_len = len(lasso_reg.theta)
    plt.xlabel("lambdas")
    plt.ylabel("coefs")
    plt.xscale("log")
    plt.hlines(y=0, xmin=lambs[0], xmax=lambs[-1])
    for i in range(fea_len):
        plt.plot(lambs, coefs[:, i], '.-')
    plt.show()
    print('Polynomial coefficients: \n', lasso_reg.theta, '\nScore: ', lasso_reg.score(X_test, y_test) )

def plt_scores_lambs(X_train, y_train, X_test, y_test, method):
    scores = []
    lambs = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    for l in lambs:
        lasso_reg = LassoRegression(degree=2, method=method, lamb=l)
        lasso_reg.fit(X_train, y_train)
        score = lasso_reg.score(X_test, y_test)
        scores.append(score)
    plt.xlabel("lambdas")
    plt.ylabel("scores")
    plt.xscale("log")
    plt.plot(lambs, scores, '.-')
    plt.show()


def plt_pred_obser(X_train, y_train, X_test, y_test, method, lamb=0):
    lasso_reg = LassoRegression(degree=2, method=method, lamb=lamb)
    lasso_reg.fit(X_train, y_train)
    y_pred = lasso_reg.predict(X_test)
    plt.xlabel("Predicted y")
    plt.ylabel("Observed y")
    plt.scatter(y_pred, y_test)
    plt.show()


def plt_residu_lambs(X_train, y_train, X_test, y_test, method):
    lambs = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    for l in lambs:
        lasso_reg = LassoRegression(degree=2, method=method, lamb=l)
        lasso_reg.fit(X_train, y_train)
        y_pred = lasso_reg.predict(X_test)
        alpha_p = np.array(list([l]) * len(y_test))
        for i in range(len(y_test)):
            plt.scatter(alpha_p, y_test - y_pred)
    plt.xlabel("lambdas")
    plt.ylabel("Residue'")
    plt.xscale("log")
    plt.show()


def plt_scores_datasize(X_train, y_train, X_test, y_test, method):
    scores = []
    splits = np.linspace(0.05, 1, 20) * len(X_train)
    splits = [int(i) for i in splits]
    lasso_reg = LassoRegression(degree=1, method=method, lamb=0)
    for s in splits:
        lasso_reg.fit(X_train[:s, :], y_train[:s])
        score = lasso_reg.score(X_test, y_test)
        scores.append(score)
    plt.xlabel("datasize")
    plt.ylabel("scores")
    plt.plot(splits, scores)
    plt.show()


def plt_square_lambs(X, y, k, method):
    k = k + 1
    Xs_train, ys_train, Xs_val, ys_val = _k_split(X, y, k)
    lambs = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    min_lamb_list = np.array([])
    for i in range(1, k):
        x_2_list = np.array([])
        for l in lambs:
            lasso_reg = LassoRegression(degree=2, method=method, lamb=l)
            lasso_reg.fit(Xs_train[i], ys_train[i])
            y_pred = lasso_reg.predict(Xs_val[i])
            x_2 = np.sum((ys_val[i] - y_pred) ** 2)
            x_2_list = np.append(x_2_list, x_2)
        min_pos = np.argmin(x_2_list)
        plt.plot(lambs, x_2_list)
        min_lamb_list = np.append(min_lamb_list, lambs[min_pos])
    best_lamb = np.mean(min_lamb_list)
    print('Best result for alpha is', best_lamb)
    plt.xlabel("lambdas")
    plt.ylabel("log(X_2)")
    plt.xscale("log")
    plt.show()


def plt_coefs_coefs(X_train, y_train, method):
    coefs = []
    num = [1, 4, 5, 7, 8, 9]
    lambs = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    pca = PCA(n_components=3)
    pca.fit(X_train, y_train)
    X_train = pca.transform(X_train)
    for l in lambs:
        lasso_reg = LassoRegression(degree=2, method=method, lamb=l)
        lasso_reg.fit(X_train, y_train)
        coefs.append(lasso_reg.theta)
    coefs = np.array(coefs)
    k = 0
    plt.figure(figsize=(15, 15))
    coefs_num = [1, 2, 3, 4]
    for i in coefs_num[:-1]:
        for j in coefs_num[i:]:
            plt.subplot(3, 3, num[k])
            k += 1
            x = coefs[:, i]
            y = coefs[:, j]
            plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, label='${c_%d}$ versus ${c_%d}$' % (i, j))
            plt.legend()
            j += 1

    plt.show()


def create_plots(X, y, X_train, y_train, X_test, y_test, method):
    k = 6
    Xs_train, ys_train, Xs_val, ys_val = _k_split(X, y, k)
    lambs = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    min_lamb_list = np.array([])

    fig, axs = plt.subplots(1, 2)

    for i in range(1, k):
        x_2_list = np.array([])
        for l in lambs:
            lasso_reg = LassoRegression(degree=2, method=method, lamb=l)
            lasso_reg.fit(Xs_train[i], ys_train[i])
            y_pred = lasso_reg.predict(Xs_val[i])
            x_2 = np.sum((ys_val[i] - y_pred) ** 2)
            x_2_list = np.append(x_2_list, x_2)
        min_pos = np.argmin(x_2_list)
        axs[0].plot(lambs, x_2_list)
        min_lamb_list = np.append(min_lamb_list, lambs[min_pos])
    best_lamb = np.mean(min_lamb_list)




    axs[0].set_xlabel("lambdas")
    axs[0].set_ylabel("log(X_2)")
    axs[0].set_xscale("log")

    lasso_reg = LassoRegression(degree=2, method=method, lamb=0)
    lasso_reg.fit(X_train, y_train)
    y_pred = lasso_reg.predict(X_test)

    axs[1].set_xlabel("Predicted y")
    axs[1].set_ylabel("Observed y")
    axs[1].scatter(y_pred, y_test)

    lmax=max([np.max(y_pred), np.max(y_test)])
    lmin=min([np.min(y_pred), np.min(y_test)])
    axs[1].set_xlim([lmin, lmax])
    axs[1].set_ylim([lmin, lmax])


    print('Best result for alpha is', best_lamb, '\nScore= ', lasso_reg.score(X_test, y_test))
    print('Coefficients =', lasso_reg.theta)













    plt.show()

