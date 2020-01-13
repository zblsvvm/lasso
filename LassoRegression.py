from sklearn.preprocessing import StandardScaler
from LinearRegression import LinearRegression
from metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

"""
Encapsulation the Lasso regression process
# @Author   : Tian Xiao
"""


class LassoRegression:
    def __init__(self, degree=1, method="bgd", lamb=0):
        assert method == "bgd" or method == "sgd" or method == "normal" or method == "cd" or method == "mgd" or \
            method == "pgd" or method == "pgd_acc" or method == "admm" or method == "cd_pure", \
            "No such method, please select from bgd, sgd, normal, cd, cd_pure, mgd, pgd, pgd_acc and admm"
        assert lamb >= 0
        assert degree >= 1
        self.lamb = lamb  # lambda hyperparameters
        self.degree = degree  # Order of polynomial regression
        self.method = method  # Minimization method
        self._lin_reg = LinearRegression()  # Calling a Linear Regressor
        self._poly_fea = None  # PolynomialFeatures
        self._std_scaler = None  # StandardScaler
        self.theta = None  # Coefficient vector
        self.pca = None

    def _train_data_preprocess(self, X_train, y_train):
        """
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
        elif self.method == "pgd":
            self._lin_reg.fit_pgd(X_p_s_train, y_train, lamb=self.lamb)
        elif self.method == "pgd_acc":
            self._lin_reg.fit_pgd(X_p_s_train, y_train, lamb=self.lamb, acc=True)
        elif self.method == "admm":
            self._lin_reg.fit_admm(X_p_s_train, y_train, alpha=self.lamb)
        elif self.method == "cd_pure":
            self._lin_reg.fit_cd_pure(X_p_s_train, y_train, lamb=self.lamb)
        self.theta = self._lin_reg.theta
        return self

    def _test_data_preprocess(self, X_test):
        """
        Preprocess the data before testing, and the processing rules are consistent with the training data set
        """
        X_p_test = self._poly_fea.transform(X_test)[:, 1:]
        # X_p_test = self.pca.transform(X_p_test)
        X_p_s_test = self._std_scaler.transform(X_p_test)
        return X_p_s_test

    def predict(self, X_test):
        """
        Given the data set X_predict to be predicted, return the result vector representing X_predict
        """
        assert self.theta is not None, \
            "must fit before predict!"
        X_p_s_test = self._test_data_preprocess(X_test)
        return self._lin_reg.predict(X_p_s_test)

    def score(self, X_test, y_test):
        """
        Determine the accuracy of the current model based on the test data sets x_test and y_test
        """
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LassoRegression()\n" \
               "degree=" + self.degree + "\n"
