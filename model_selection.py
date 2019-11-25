"""
模型选择：数据集分割，交叉验证，网格搜索
Model selection: data set segmentation, cross validation, grid search
"""
import numpy as np


def train_test_split(X, y, test_radio=0.2, seed=None):
    """将数据X和y按照test_radio分割成X_train, X_test, y_train, y_test"""
    """Divide the data X and y into X_train, X_test, y_train, y_test according to test_radio"""
    assert X.shape[0] == y.shape[0], \
        "the size of X must be equal to the size of y"
    assert 0.0 <= test_radio <= 1.0, \
        "test_ratio must be valid"

    if seed:
        # 期望在测试程序时，可以得到相同的伪随机数序列（这时只需要指定相同的seed）
        # It is expected that the same pseudo-random number sequence can be obtained when testing the program
        # (in this case, only the same seed needs to be specified)
        np.random.seed(seed)

    # 先将数据集打乱顺序。由于x和y一一对应，所以不打乱X和y本身，而是将len(X)个索引打乱
    # The data set is first shuffled. Since x and y correspond one-to-one,
    # it does not disturb X and y itself, but lining up len(X) indexes.
    shuffled_indexes = np.random.permutation(len(X))

    test_size = int(len(X) * test_radio)
    # 分别获取训练集和测试集对应的索引
    # Obtain the index corresponding to the training set and test set respectively
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]
    # 根据索引确定训练集和测试集，其中X和y的索引应该是对应的
    # Determine the training set and the test set according to the index,
    # where the indexes of X and y should be corresponding
    X_train = X[train_indexes]
    y_train = y[train_indexes]
    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test


"""交叉验证。将训练数据集分为k份，将其中的每一份作为验证数据集，而剩下的部分作为训练数据集"""
"""Cross-validation. Divide the training data set into k shares, each of which is used as a validation data set, 
and the remaining part as a training data set"""


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
    return np.array(Xs_val), np.array(ys_val), np.array(Xs_train), np.array(ys_train)


def cross_val_score(model, X_train, y_train, k=3, alpha=1):
    """传入一个算法以及相应的X_train, y_train，返回生成的k个模型中每个模型对应的准确率"""
    """Pass in an algorithm and the corresponding X_train, y_train, 
    return the accuracy of each model in the generated k models"""
    Xs_val, ys_val, Xs_train, ys_train = _k_split(X_train, y_train, k=k)
    scores = []
    for i in range(k):
        model.fit(Xs_train[i], ys_train[i], lasso=True, method="bgd", alpha=alpha)
        score = model.score(Xs_val[i], ys_val[i])
        scores.append(score)
    return scores


def grid_search(model, X_train, y_train, alpha_range, degree_range):
    """传入一个算法以及相应的X_train, y_train，以及想要搜索的超参数的范围，返回最高score对应的超参数组合"""
    """Pass in an algorithm and the corresponding X_train, y_train, and the range of hyperparams that you 
    want to search, return the hyperparameter combination corresponding to the highest score"""
    best_score, best_alpha, best_degree = 0, 0, 0
    history = []
    for alpha in alpha_range:
        for degree in degree_range:
            model._degree = degree
            scores = cross_val_score(model, X_train, y_train, k=3, alpha=alpha)
            score = np.mean(scores)
            history.append([alpha, degree, score])
            if score > best_score:
                best_score, best_alpha, best_degree = score, alpha, degree
    return best_alpha, best_degree, best_score, history
