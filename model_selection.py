"""
模型选择：数据集分割，交叉验证，网格搜索
Model selection: data set segmentation, cross validation, grid search
"""
import numpy as np


def train_test_split(X, y, test_ratio=0.2, seed=None):
    """将数据 X 和 y 按照test_ratio分割成X_train, X_test, y_train, y_test"""
    assert X.shape[0] == y.shape[0], \
        "the size of X must be equal to the size of y"
    assert 0.0 <= test_ratio <= 1.0, \
        "test_ration must be valid"

    if seed:
        np.random.seed(seed)

    shuffled_indexes = np.random.permutation(len(X))

    test_size = int(len(X) * test_ratio)
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test


"""交叉验证。将训练数据集分为k份，将其中的每一份作为验证数据集，而剩下的部分作为训练数据集"""


def _k_split(X, y, k):
    """将数据分割为k份 Split the data into k shares"""
    length = len(X)
    indexes = []
    cur_index = 0
    if length % k == 0:
        steps = length // k
    else:
        steps = length // k + 1
    for i in range(k):
        indexes.append(cur_index)
        cur_index += steps
    indexes.append(len(X))
    Xs_val = []
    ys_val = []
    Xs_train = []
    ys_train = []
    for i in range(k):
        Xs_val.append(X[indexes[i]:indexes[i + 1], :])
        ys_val.append(y[indexes[i]:indexes[i + 1]])
        Xs_train.append(np.vstack([X[:indexes[i], :], X[indexes[i + 1]:, :]]))
        ys_train.append(np.hstack([y[:indexes[i]], y[indexes[i + 1]:]]))
    return np.array(Xs_train), np.array(ys_train), np.array(Xs_val), np.array(ys_val)


def cross_val_score(model, X_train, y_train, k=3):
    """传入一个算法以及相应的X_train, y_train，返回生成的k个模型中每个模型对应的准确率"""
    Xs_train, ys_train, Xs_val, ys_val = _k_split(X_train, y_train, k=k)
    scores = []
    for i in range(k):
        model.fit_bgd(Xs_train[i], ys_train[i], method="bgd", lamb=model.lamb)
        score = model.score(Xs_val[i], ys_val[i])
        scores.append(score)
    return scores


def grid_search(model, X_train, y_train, lamb_range, degree_range):
    """传入一个算法以及相应的X_train, y_train，以及想要搜索的超参数的范围，返回最高score对应的超参数组合"""
    best_score, best_alpha, best_degree = 0, 0, 0
    history = []
    for lamb in lamb_range:
        for degree in degree_range:
            model._degree = degree
            model.lamb = lamb
            scores = cross_val_score(model, X_train, y_train, k=3)
            score = np.mean(scores)
            history.append([lamb, degree, score])
            print(score, lamb, degree)  # 打印测试
            if score > best_score:
                best_score, best_alpha, best_degree = score, lamb, degree
    return best_alpha, best_degree, best_score, history
