import numpy as np
from metrics import r2_score
import itertools


class LinearRegression:

    def __init__(self):
        """初始化Linear Regression模型"""
        self.theta = None

    def fit_normal(self, X_train, y_train):
        """根据训练数据集X_train, y_train训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        return self

    def fit_bgd(self, X_train, y_train, initial_theta=None, lamb=0, eta=0.01, n_iters=20000, epsilon=1e-8):
        """根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        def J(theta, X_b, y, lamb):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y) + lamb * sum(np.abs(theta))
            except:
                return float('inf')


        def uaf_derivative(x):
            """the derivative of the uniform approximation function"""
            u = 0.01
            a = (np.exp(x / u) - np.exp(- x / u)) / (np.exp(x / u) + np.exp(- x / u))
            return a


        def dJ(theta, X_b, y, lamb):
            penalty = lamb * uaf_derivative(theta)
            vec = X_b.T.dot(X_b.dot(theta) - y) * 2. / len(y)
            return vec + penalty

        def gradient_descent(X_b, y, initial_theta, lamb=lamb, eta=eta, n_iters=n_iters, epsilon=epsilon):

            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y, lamb)
                last_theta = theta
                theta = theta - eta * gradient
                if abs(J(theta, X_b, y, lamb) - J(last_theta, X_b, y, lamb)) < epsilon:
                    break
                cur_iter += 1

            print(cur_iter)
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        if initial_theta is None:
            initial_theta = np.zeros(X_b.shape[1])
        self.theta = gradient_descent(X_b, y_train, initial_theta, lamb, eta, n_iters)
        return self

    def fit_cd(self, X, y, lamb=0.2, threshold=0.1):
        """ coordinate descent Method to get Lasso Regression Coefficient"""
        # Calculate RSS(residual sum of square)
        # initialize w.
        m, n = X.shape
        x0 = np.ones((m, 1))
        X = np.hstack((x0, X))
        m, n = X.shape
        w = np.zeros((n, 1))
        rss = lambda X, y, w: np.sum((y - X.dot(w))**2)
        # initialize w.
        r = rss(X, y, w)
        # CD method
        niter = itertools.count(1)
        for it in niter:
            for k in range(n):
                # z_k and p_k calculation
                z_k = np.dot(X[:, k].T , X[:, k])
                p_k = 0
                for i in range(m):
                    p_k += X[i, k] * (y[i] - sum([X[i, j] * w[j] for j in range(n) if j != k]))
                if p_k < -lamb / 2:
                    w_k = (p_k + lamb / 2) / z_k
                elif p_k > lamb / 2:
                    w_k = (p_k - lamb / 2) / z_k
                else:
                    w_k = 0
                w[k, 0] = w_k
            r_prime = rss(X, y, w)
            delta = abs(r_prime - r)
            r = r_prime
            print('Iteration: {}, delta = {}'.format(it, delta))
            if delta < threshold:
                break
        self.theta = w
        return self

    def fit_sgd(self, X_train, y_train, lamb=0, n_iters=10, t0=5, t1=5000):
        """根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert n_iters >= 1

        def uaf_derivative(x):
            """the derivative of the uniform approximation function"""
            u = 0.01
            a = (np.exp(x / u) - np.exp(- x / u)) / (np.exp(x / u) + np.exp(- x / u))
            return a


        def dJ_sgd(theta, X_b_i, y_i, lamb):

            return X_b_i * (X_b_i.dot(theta) - y_i) * 2. + lamb * uaf_derivative(theta)

        def sgd(X_b, y, initial_theta, lamb=lamb, n_iters=n_iters, t0=t0, t1=t1):

            def learning_rate(t):
                return t0 / (t + t1)

            theta = initial_theta
            m = len(X_b)
            for i_iter in range(n_iters):
                indexes = np.random.permutation(m)
                X_b_new = X_b[indexes, :]
                y_new = y[indexes]
                for i in range(m):
                    gradient = dJ_sgd(theta, X_b_new[i], y_new[i], lamb)
                    theta = theta - learning_rate(i_iter * m + i) * gradient

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.random.randn(X_b.shape[1])
        self.theta = sgd(X_b, y_train, initial_theta, lamb, n_iters, t0, t1)
        return self

    def predict(self, X_train):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.theta is not None, \
            "must fit before predict!"
        assert X_train.shape[1] + 1 == len(self.theta), \
            "the feature number of X_predict must be equal to X_train"
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        return X_b.dot(self.theta)

    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"

    def fit_mgd(self, X_train, y_train, lamb=0):
        theta = self.fit_sgd(X_train, y_train, lamb=lamb).theta
        self.theta = self.fit_bgd(X_train, y_train, initial_theta=theta, lamb=lamb).theta
        return self
    def fit_pgd(self, X, y):
        def obj(X, w, y, lambd):
            return f(X, w, y) + lambd * np.sum(np.abs(w))

        def f(X, w, y):
            Xw_b = X.dot(w) - y
            return 0.5 * (Xw_b.T.dot(Xw_b))

        def grf(X, w, y):
            return X.T.dot(X.dot(w) - y)

        # Model function
        def m(w, wk, X, y, GammaK):
            innerProd = grf(X, wk, y).T.dot(w - wk)
            wDiff = w - wk
            return f(X, wk, y) + innerProd + (1.0 / (2.0 * GammaK)) * wDiff.T.dot(wDiff)

        # Shrinkage
        def proxNorm1(v, lambd):
            return np.sign(v) * np.maximum(np.zeros(np.shape(v)), np.abs(v) - lambd)

        def acc(X, y, niter=500):
            # initialize w.
            m, n = X.shape
            w = np.zeros((n, 1))
            lambd = np.sqrt(2 * n * np.log(n)).tolist()
            beta = 0.75
            wk = w
            yk = wk
            t = 1
            Diff = 0
            for k in range(niter):
                Gammak = 0.01

                # Back Tracking Line Search
                while True:
                    wk_acc = yk - Gammak * grf(X, yk, y)
                    if f(X, wk_acc, y) <= m(wk_acc, wk, X, y, Gammak):
                        break
                    else:
                        Gammak = beta * Gammak
                wk_acc = proxNorm1(wk, Gammak * lambd)
                tk = 0.5 + 0.5 * np.sqrt(1 + 4 * (t ** 2.))
                yk_acc = wk_acc + ((tk - 1) / (tk + 1)) * (wk_acc - wk)

                Diff = np.linalg.norm(obj(X, wk_acc, y, lambd) - obj(X, wk, y, lambd))

                wk = wk_acc
                t = tk
                yk = yk_acc

                if Diff < 1:
                    break

            return wk

        self.theta = acc(X, y)
        return self