import numpy as np
from metrics import r2_score
import itertools
from numpy.linalg import cholesky, norm
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.linear_model import Lasso


class LinearRegression:

    def __init__(self):
        # @Author   : Tian Xiao
        """
        Initialize the Linear Regression model
        """
        self.theta = None

    def fit_normal(self, X_train, y_train):
        # @Author   : Tian Xiao
        """
        Train Linear Regression model based on training data set X_train, y_train
        """
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        return self

    def fit_bgd(self, X_train, y_train, initial_theta=None, lamb=0, eta=0.01, n_iters=20000, epsilon=1e-8):
        # @Author   : Tian Xiao
        """
        Train the Linear Regression model using the gradient descent method based on the training data set X_train, y_train
        """
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

    def fit_cd(self, X, y, lamb=0.5, threshold=0.01):
        X = np.hstack([np.ones((len(X), 1)), X])
        self.theta = Lasso(alpha=lamb, tol=threshold, fit_intercept=False).fit(X, y).coef_
        return self

    def fit_cd_pure(self, X, y, lamb=0.2, threshold=10):
        """ coordinate descent Method to get Lasso Regression Coefficient"""
        # initialize w.
        X = np.hstack((np.ones((len(X), 1)), X))
        m, n = X.shape
        w = np.zeros((n, 1))
        # Calculate RSS(residual sum of square)
        rss = lambda X, y, w: np.sum((y - X.dot(w)) ** 2)
        r = rss(X, y, w)
        # CD method
        niter = itertools.count(1)
        for it in niter:
            for k in range(n):
                # z_k and p_k calculation
                z_k = np.sum(X[:, k] ** 2)
                p_k = 0
                for i in range(m):
                    p_k += X[i, k] * (y[i] - sum([X[i, j] * w[j] for j in range(n) if j != k]))
                # update w
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
        self.theta = w.ravel()
        return self

    def fit_sgd(self, X_train, y_train, lamb=0, n_iters=10, t0=5, t1=5000):
        # @Author   : Tian Xiao
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert n_iters >= 1

        def uaf_derivative(x):
            """the derivative of the uniform approximation function"""
            u = 0.07
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
        # @Author   : Tian Xiao
        assert self.theta is not None, \
            "must fit before predict!"
        assert X_train.shape[1] + 1 == len(self.theta), \
            "the feature number of X_predict must be equal to X_train"
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        return X_b.dot(self.theta)

    def score(self, X_test, y_test):
        # @Author   : Tian Xiao
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"

    def fit_mgd(self, X_train, y_train, lamb=0):
        theta = self.fit_sgd(X_train, y_train, lamb=lamb).theta
        self.theta = self.fit_bgd(X_train, y_train, initial_theta=theta, lamb=lamb).theta
        return self

    def fit_pgd(self, X, y, lamb=1, threshold=0.1, niter=5000, acc=False):
        # initialize functions
        err = lambda X, w, y: X.dot(w) - y
        Xsq = lambda X: (X.T.dot(X))
        grad = lambda X, w, y: X.T.dot(err(X, w, y))
        obj = lambda X, w, y, lamb: Xsq(err(X, w, y)) + lamb * np.sum(np.abs(w))
        model = lambda w, wk, X, y, GammaK: Xsq(err(X, wk, y)) + \
                                            X.T.dot(X.dot(wk) - y).T.dot(w - wk) + \
                                            (1.0 / (2.0 * GammaK)) * (w - wk).T.dot(w - wk)
        # proximal operation
        prox = lambda x, kappa: np.maximum(0., x - kappa) - np.maximum(0., -x - kappa)
        # stack a column of 1 to X
        X = np.hstack([np.ones((len(X), 1)), X])
        y = [[i] for i in y]
        # initialize w and some variables
        m, n = X.shape
        w = np.zeros((n, 1))
        beta = 0.75
        wk = w.copy()
        vk = wk
        tk = 1
        Gammak = 0.01
        for k in range(niter):
            if not acc:
                while True:
                    wk_acc = wk - Gammak * grad(X, wk, y)
                    if (0.5 * Xsq(err(X, wk_acc, y))) <= model(wk_acc, wk, X, y, Gammak):
                        break
                    else:
                        Gammak = beta * Gammak
                wk_acc = prox(wk_acc, Gammak * lamb)
                diff = np.linalg.norm(obj(X, wk_acc, y, lamb) - obj(X, wk, y, lamb))
                wk = wk_acc
            # Accelerated Gradient Descent (GD) Step
            elif acc:
                while True:
                    wk_acc = vk - Gammak * grad(X, vk, y)
                    if (0.5 * Xsq(err(X, wk_acc, y))) <= model(wk_acc, wk, X, y, Gammak):
                        break
                    else:
                        Gammak = beta * Gammak
                wk_acc = prox(wk_acc, Gammak * lamb)
                tk_acc = 0.5 + 0.5 * np.sqrt(1 + 4 * (tk ** 2.))
                vk_acc = wk_acc + ((tk - 1) / (tk + 1)) * (wk_acc - wk)
                diff = np.linalg.norm(obj(X, wk_acc, y, lamb) - obj(X, wk, y, lamb))
                wk = wk_acc
                tk = tk_acc
                vk = vk_acc

            if diff < threshold:
                break

        self.theta = wk.ravel()
        return self

    def objective(self, X, y, alpha, x, z):
        return .5 * np.square(X.dot(x) - y).sum().sum() + alpha * norm(z, 1)

    def shrinkage(self, x, kappa):
        return np.maximum(0., x - kappa) - np.maximum(0., -x - kappa)

    def factor(self, X):
        m, n = X.shape
        if m >= n:
            L = cholesky(X.T.dot(X) + sparse.eye(n))
        else:
            L = cholesky(sparse.eye(m) + (X.dot(X.T)))
        L = sparse.csc_matrix(L)
        U = sparse.csc_matrix(L.T)
        return L, U

    def fit_admm(self, X, y, alpha=5, rel_par=1., QUIET=True,
                 MAX_ITER=100, ABSTOL=1e-3, RELTOL=1e-2):
        # Data preprocessing
        X = np.hstack([np.ones((len(X), 1)), X])
        m, n = X.shape
        # save a matrix-vector multiply
        Xty = X.T.dot(y)
        # ADMM solver
        x = np.zeros((n, 1))
        z = np.zeros((n, 1))
        u = np.zeros((n, 1))

        # cache the (Cholesky) factorization
        L, U = self.factor(X)

        # Saving state
        h = {}
        h['objval'] = np.zeros(MAX_ITER)
        h['r_norm'] = np.zeros(MAX_ITER)
        h['s_norm'] = np.zeros(MAX_ITER)
        h['eps_pri'] = np.zeros(MAX_ITER)
        h['eps_dual'] = np.zeros(MAX_ITER)

        for k in range(MAX_ITER):
            tmp_variable = np.array(Xty) + (z - u)[0]  # (temporary value)
            if m >= n:
                x = spsolve(U, spsolve(L, tmp_variable))[..., np.newaxis]
            else:
                ULXq = spsolve(U, spsolve(L, X.dot(tmp_variable)))[..., np.newaxis]
                x = tmp_variable - (X.T.dot(ULXq))

            # z-update with relaxation
            zold = np.copy(z)
            x_hat = rel_par * x + (1. - rel_par) * zold
            z = self.shrinkage(x_hat + u, alpha * 1.)

            # u-update
            u += (x_hat - z)

            # diagnostics, reporting, termination checks
            h['objval'][k] = self.objective(X, y, alpha, x, z)
            h['r_norm'][k] = norm(x - z)
            h['s_norm'][k] = norm((z - zold))
            h['eps_pri'][k] = np.sqrt(n) * ABSTOL + \
                              RELTOL * np.maximum(norm(x), norm(-z))
            h['eps_dual'][k] = np.sqrt(n) * ABSTOL + \
                               RELTOL * norm(u)
            if (h['r_norm'][k] < h['eps_pri'][k]) and (h['s_norm'][k] < h['eps_dual'][k]):
                break
        self.theta = z.ravel()
        return self
