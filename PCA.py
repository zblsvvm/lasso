import numpy as np


class PCA:
    """用户传来样本数据集。经过初始化算出主成分之后，用户可以使用这些主成分对其他的数据进行降维处理"""
    """The user sent a sample data set. After initializing the principal components, 
    the user can use these principal components to reduce the dimensionality of other data."""

    def __init__(self, n_components):
        """initialize PCA"""

        assert n_components >= 1, "n_components must be valid"
        self.n_components = n_components  # 有多少个主成分 How many principal components
        self.components_ = None

    def fit(self, X, eta=0.01, n_iters=1e4):
        """获得数据集的前n个主成分"""
        """Get the first n principal components of the data set"""
        assert self.n_components <= X.shape[1], \
            "n_components must not be greater than the feature number of X"

        def demean(X):
            """将样本的均值归为0"""
            """Classify the mean of the sample as 0"""
            # 在行这个方向求均值，也就是说求得的结果是每一列的均值，是一1*n的向量：将每一个样本中的每一个维度的求均值放入一个向量中
            # 得到的结果仍然是一个m*n的矩阵
            return X - np.mean(X, axis=0)

        def f(w, X):
            """效用函数 Utility Function"""
            return np.sum((X.dot(w) ** 2)) / len(X)

        def df(w, X):
            """效用函数的梯度，向量化方法"""
            """Gradient of utility function, vectorization method"""
            return X.T.dot(X.dot(w)) * 2. / len(X)

        def direction(w):
            """将w转化为只表示方向的单位向量，即模为1。使用单位向量可以提高搜索的效率"""
            """Convert w to a unit vector that only represents the direction, ie the modulus is 1. 
            Use unit vectors to improve search efficiency"""
            return w / np.linalg.norm(w)

        def first_component(X, initial_w, eta=0.01, n_iters=1e4, epsilon=1e-8):
            """求第一主成分的过程The process of finding the first principal component"""
            w = direction(initial_w)
            cur_iter = 0
            while cur_iter < n_iters:
                gradient = df(w, X)
                last_w = w
                w = w + eta * gradient
                w = direction(w)
                if abs(f(w, X) - f(last_w, X)) < epsilon:
                    break
                cur_iter += 1
            return w

        X_pca = demean(X)
        # 为W_k矩阵创建空间。k行意味着前k个主成分，n列对应着n个特征。
        self.components_ = np.empty(shape=(self.n_components, X.shape[1]))
        for i in range(self.n_components):
            initial_w = np.random.random(X_pca.shape[1])  # 随机生成第一个梯度上升法的初始搜索点
            w = first_component(X_pca, initial_w, eta, n_iters)  # 求第一主成分对应的方向w
            self.components_[i, :] = w
            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w  # 将数据在第一个主成分上的分量去掉
        return self

    def transform(self, X):
        """将给定的X，映射到各个主成分分量中"""
        """Map a given X to each principal component"""
        assert X.shape[1] == self.components_.shape[1]
        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        """将给定的X，反向映射到原来的特征空间"""
        """Map a given X back to the original feature space"""
        assert X.shape[1] == self.components_.shape[0]
        return X.dot(self.components_)

    def __repr__(self):
        return "PCA(n_components=%d)" % self.n_components
