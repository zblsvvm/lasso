"""
对lasso回归的过程进行封装
"""
from preprocessing import PolynomialFeatures
from preprocessing import StandardScaler
from LinearRegression import LinearRegression
from metrics import r2_score


class LassoRegression:
    def __init__(self, degree=2):
        """初始化Lasso Regression模型"""
        self._degree = degree  # 多项式回归的阶数
        self.coef_ = None  # 系数
        self.interception_ = None  # 截距
        self.lin_reg = None  # 调用一个线性回归器
        self.std_scaler = StandardScaler()  # 调用一个数据归一化类
        self.alpha = None  # lasso回归的alpha超参数

    def _preprocess_data(self, X, first_fit=False):
        """对数据进行预处理，包括扩展数据维度以适应多项式，以及数据归一化"""
        poly_fea = PolynomialFeatures(self._degree)
        X_poly = poly_fea.transform(X)
        if first_fit:
            self.std_scaler.fit(X_poly)
        X_std = self.std_scaler.transform(X_poly)
        return X_std

    def fit(self, X_train, y_train, lasso=False, method="bgd", alpha=1):
        """根据标准化训练数据集X_train_std, y_train_std训练Linear Regression模型"""
        """调用PolynomialFeatures数据预处理和LinearRegression线性回归解决多项式回归问题"""
        assert X_train.shape[1] == 1, \
            "multiple features are not supported at this time"
        X_train_std = self._preprocess_data(X_train, first_fit=True)
        self.lin_reg = LinearRegression()
        if not lasso and method == "normal":
            self.lin_reg.fit_normal(X_train_std, y_train)
        elif not lasso and method == "bgd":
            self.lin_reg.fit_bgd(X_train_std, y_train)
        elif not lasso and method == "sgd":
            self.lin_reg.fit_sgd(X_train_std, y_train)
        elif lasso and method == "bgd":
            self.lin_reg.fit_bgd(X_train_std, y_train, lasso=True, alpha=alpha)
        elif lasso and method == "sgd":
            self.lin_reg.fit_sgd(X_train_std, y_train, lasso=True, alpha=alpha)
        else:
            raise RuntimeError("Unsupported method")
        self.coef_ = self.lin_reg.coef_
        self.interception_ = self.lin_reg.interception_

    def predict(self, X_test):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.interception_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        X_test_std = self._preprocess_data(X_test)
        return self.lin_reg.predict(X_test_std)

    def score(self, X_test, y_test):
        """根据测试数据集x_test和y_test确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "PolynomialRegression()"
