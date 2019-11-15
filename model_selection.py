"""
模型选择：将数据集分割为训练数据集和测试数据集
"""
import numpy as np


def train_test_split(X, y, test_radio=0.2, seed=None):
    """将数据X和y按照test_radio分割成X_train, X_test, y_train, y_test"""
    assert X.shape[0] == y.shape[0], \
        "the size of X must be equal to the size of y"
    assert 0.0 <= test_radio <= 1.0, \
        "test_ratio must be valid"

    if seed:  # 期望在测试程序时，可以得到相同的伪随机数序列（这时只需要指定相同的seed）
        np.random.seed(seed)

    # 先将数据集打乱顺序。由于x和y一一对应，所以不打乱X和y本身，而是将len(X)个索引打乱
    shuffled_indexes = np.random.permutation(len(X))

    test_size = int(len(X) * test_radio)
    test_indexes = shuffled_indexes[:test_size]  # 分别获取训练集和测试集对应的索引
    train_indexes = shuffled_indexes[test_size:]

    X_train = X[train_indexes]  # 根据索引确定训练集和测试集，其中X和y的索引应该是对应的
    y_train = y[train_indexes]
    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test
