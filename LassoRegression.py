from sklearn.preprocessing import StandardScaler
from LinearRegression import LinearRegression
from metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

"""
对lasso回归的过程进行封装
# @Author   : Tian Xiao
"""


class LassoRegression:
    def __init__(self, degree=1, method="bgd", lamb=0):
        assert method == "bgd" or method == "sgd" or method == "normal" or method == "cd" or method == "mgd"
        assert lamb >= 0
        assert degree >= 1
        self.lamb = lamb  # lambda超参数，lambda hyperparameters
        self.degree = degree  # 多项式回归的阶数,Order of polynomial regression
        self.method = method  # 最小化方法，Minimization method
        self._lin_reg = LinearRegression()  # 调用一个线性回归器,Calling a Linear Regressor
        self._poly_fea = None  # PolynomialFeatures
        self._std_scaler = None  # StandardScaler
        self.theta = None  # 系数向量,Coefficient vector
        self.pca = None

    def _train_data_preprocess(self, X_train, y_train):
        """
        在拟合前对数据进行预处理，包括PolynomialFeatures，StandardScaler等
        Preprocess the data before fitting, including PolynomialFeatures, StandardScaler, etc.
        """
        self._poly_fea = PolynomialFeatures(degree=self.degree)
        self._poly_fea.fit(X_train, y_train)
        X_p_train = self._poly_fea.transform(X_train)[:, 1:]
        # self.pca = PCA()
        # self.pca.fit(X_p_train, y_train)
        # X_p_train = self.pca.transform(X_p_train)
        # print(X_p_train.shape[1])
        # print(self.pca.explained_variance_ratio_)
        self._std_scaler = StandardScaler()
        self._std_scaler.fit(X_p_train)
        X_p_s_train = self._std_scaler.transform(X_p_train)
        return X_p_s_train

    def fit(self, X_train, y_train):
        X_p_s_train = self._train_data_preprocess(X_train, y_train)
        if self.method == "normal":
            self._lin_reg.fit_normal(X_p_s_train, y_train)
        elif self.method == "bgd":
            self._lin_reg.fit_bgd(X_p_s_train, y_train, lamb=self.lamb)
        elif self.method == "sgd":
            self._lin_reg.fit_sgd(X_p_s_train, y_train, lamb=self.lamb)
        elif self.method == "mgd":
            self._lin_reg.fit_mgd(X_p_s_train, y_train, lamb=self.lamb)
        elif self.method == "cd":
            self._lin_reg.fit_cd(X_p_s_train, y_train, lamb=self.lamb)
        self.theta = self._lin_reg.theta
        return self

    def _test_data_preprocess(self, X_test):
        """
        在测试前对数据进行预处理，处理规则与训练数据集一致
        Preprocess the data before testing, and the processing rules are consistent with the training data set
        """
        X_p_test = self._poly_fea.transform(X_test)[:, 1:]
        # X_p_test = self.pca.transform(X_p_test)
        X_p_s_test = self._std_scaler.transform(X_p_test)
        return X_p_s_test

    def predict(self, X_test):
        """
        给定待预测数据集X_predict，返回表示X_predict的结果向量
        Given the data set X_predict to be predicted, return the result vector representing X_predict
        """
        assert self.theta is not None, \
            "must fit before predict!"
        X_p_s_test = self._test_data_preprocess(X_test)
        return self._lin_reg.predict(X_p_s_test)

    def score(self, X_test, y_test):
        """
        根据测试数据集x_test和y_test确定当前模型的准确度
        Determine the accuracy of the current model based on the test data sets x_test and y_test
        """
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LassoRegression()\n" \
               "degree=" + self.degree + "\n"
