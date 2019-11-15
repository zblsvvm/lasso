import numpy as np
from metrics import r2_score


class LinearRegression:
    def __init__(self):
        """初始化Linear Regression模型"""
        self.coef_ = None  # 系数
        self.interception_ = None  # 截距
        self._theta = None  # 参数向量

    def fit_normal(self, X_train, y_train):
        """根据训练数据集X_train, y_train训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])  # 在X_train前边叠加一列单位向量
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)  # 正规方程解
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.interception_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        """根据测试数据集x_test和y_test确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"

    def fit_bgd(self, X_train, y_train, eta=0.01, n_iters=1e4, lasso=False, alpha=1):
        """使用梯度下降法训练模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        def J(theta, X_b, y):
            """
            求在theta为特定值时候的损失函数的值
            :param theta: 参数列表
            :param X_b: X_train添加了第一列1的数据矩阵，每一行是一个元素，每一列是一个特征
            :param y: y_train
            :return: 损失函数值 loss function value
            """
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)
            except:
                return float('inf')

        def J_lasso(theta, X_b, y, alpha):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b) + alpha * sum(theta ** 2)
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            """
            求theta在特定值时候的梯度下降算法中的梯度。使用向量化的方法
            :param theta: 参数列表 parameters list
            :param X_b: X_train添加了第一列1的数据矩阵，每一行是一个元素，每一列是一个特征
            :param y: y_train
            :return: 梯度 gradient
            """
            return 2. / len(X_b) * X_b.T.dot(X_b.dot(theta) - y)

        def dJ_lasso(theta, X_b, y, alpha):
            res = np.empty(len(theta))
            res[0] = np.sum(X_b.dot(theta) - y) + alpha * np.sum(theta)
            for i in range(1, len(theta)):
                res[i] = (X_b.dot(theta) - y).dot(X_b[:, i]) + alpha * np.sum(theta)
            return res * 2 / len(X_b)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e-4, epsilon=1e-8):
            """
            梯度下降算法
            :param X_b: X_train添加了第一列1的数据矩阵，每一行是一个元素，每一列是一个特征
            :param y: y_train
            :param initial_theta: 参数列表theta的初始值。如要避免局部最优解，可使用不同初值多次计算
            :param eta: 学习率：eta*dJ为梯度下降中每一步的步长
            :param n_iters: 为避免eta选择过大而导致距离最小值越来越远而无限循环，限制最高迭代次数
            :param epsilon: 为最小值设置精度：当两步迭代所得最小值小于这一精度时，迭代停止
            :return: 最后最佳适应的theta列表
            """
            theta = initial_theta
            cur_iter = 0
            if not lasso:
                while cur_iter < n_iters:
                    gradient = dJ(theta, X_b, y)
                    last_theta = theta
                    theta = theta - eta * gradient
                    if abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon:
                        break
                    cur_iter += 1
            else:
                while cur_iter < n_iters:
                    gradient = dJ_lasso(theta, X_b, y, alpha)
                    last_theta = theta
                    theta = theta - eta * gradient
                    if abs(J_lasso(theta, X_b, y, alpha) - J_lasso(last_theta, X_b, y, alpha)) < epsilon:
                        break
                    cur_iter += 1
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def fit_sgd(self, X_train, y_train, n_iters=5, t0=5, t1=50, lasso=False, alpha=1):
        """
        根据训练数据集X_train, y_train，使用随机梯度下降法训练线性回归模型
        n_iters为迭代次数，意为需要将样本数量看几圈(后面n*样本数量)。t0, t1为使学习率eta逐渐减小的参数
        """
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert n_iters >= 1

        def dJ_sgd(theta, X_b_i, y_i):
            """对数据集中任意选取的一个数据计算随机梯度，使用向量化的方法"""
            return X_b_i * (X_b_i.dot(theta) - y_i) * 2.

        def dJ_sgd_lasso(theta, X_b_i, y_i, alpha):
            return X_b_i * (X_b_i.dot(theta) - y_i) * 2. + alpha * np.sum(theta)

        def sgd(X_b, y, initial_theta, n_iters, t0=5, t1=50):
            def learning_rate(t):
                return t0 / (t + t1)

            theta = initial_theta
            m = len(X_b)
            if not lasso:
                for cur_iter in range(n_iters):
                    # 需要使得每一圈中所有的样本都被看一遍，但仍要保证其顺序是随机的，所以对数据进行了乱序操作
                    indexes = np.random.permutation(m)
                    X_b_new = X_b[indexes]
                    y_new = y[indexes]
                    for i in range(m):
                        gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
                        theta = theta - learning_rate(cur_iter * m + i) * gradient
                # 由于是随机梯度，所以不使用两次迭代损失函数差距小于一定的范围来结束循环，而是迭代固定的次数
            else:
                for cur_iter in range(n_iters):
                    indexes = np.random.permutation(m)
                    X_b_new = X_b[indexes]
                    y_new = y[indexes]
                    for i in range(m):
                        gradient = dJ_sgd_lasso(theta, X_b_new[i], y_new[i], alpha)
                        theta = theta - learning_rate(cur_iter * m + i) * gradient
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.random.randn(X_b.shape[1])
        self._theta = sgd(X_b, y_train, initial_theta, n_iters, t0, t1)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self