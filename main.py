from LassoRegression import LassoRegression
from data import Data
from visualisation import plot_in_order

data = Data()
X_train, X_test, y_train, y_test = data.poly_data_1()
degree = 20
lasso_reg = LassoRegression(degree=degree)
lasso_reg.fit(X_train, y_train, lasso=False, method="normal")
print(lasso_reg.score(X_test, y_test))
y_predict = lasso_reg.predict(X_train)
plot_in_order(X_train, y_train, y_predict)

lasso_reg2 = LassoRegression(degree=degree)
lasso_reg2.fit(X_train, y_train, lasso=True, method="bgd")
print(lasso_reg2.score(X_test, y_test))
y_predict = lasso_reg2.predict(X_train)
plot_in_order(X_train, y_train, y_predict)
