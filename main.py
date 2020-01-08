from LassoRegression import LassoRegression
from data import Data
from model_selection import train_test_split


X, y = Data().boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)

lasso = LassoRegression(method="bgd", degree=2, lamb=0)
lasso.fit(X_train, y_train)
score = lasso.score(X_test, y_test)
print(score)
print(lasso.theta)
