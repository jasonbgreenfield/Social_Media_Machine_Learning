import dill as pickle
import numpy as np
from math import sqrt
from sklearn import metrics, svm, linear_model, utils
from sklearn import ensemble
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.tree import DecisionTreeRegressor

a = open("Data/data_final_standardized.pkl", "rb")
data,desc = pickle.load(a)
X,y = data

Xlist = []
ylist = []
for i in X:
    Xlist.append(i.tolist())
for i in y:
    ylist.append(i.tolist())

X = np.asarray(Xlist)
y = np.asarray(ylist)

X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=0.1)


num_iterations = 50
total_rmse = 0
for i in range(num_iterations):
    # clf = linear_model.LinearRegression()
    # for r in [0,1,10,50,100]:
    clf = linear_model.RidgeCV()
    # for nn in range(1,51):
    #     clf = KNeighborsRegressor(n_neighbors=nn)
    # clf = svm.SVR(kernel='linear', C=1000)
    # clf = ensemble.AdaBoostRegressor()
    # clf = ensemble.RandomForestRegressor()
    # clf = MLPRegressor()
    # clf = DecisionTreeRegressor()

    # parameters = [
    # {'C': [1, 10, 100, 1000], 'degree': [2,3,4,5], 'kernel': ['poly']},
    # {'C': [1, 10, 100, 1000], 'gamma': [1.0,0.1,0.01,0.001], 'kernel': ['rbf']},
    # {'C': [1, 10, 100, 1000], 'gamma': [.1,1,10,100], 'kernel': ['sigmoid']},
    # {'C': [1], 'kernel': ['linear']},
    # ]
    # svr = sklearn.svm.SVR()
    # clf = GridSearchCV(estimator=svr, param_grid=parameters, scoring='f1', cv=5)


    clf.fit(X_tr, y_tr)
    # importance = clf.feature_importances_
    # print(importance)
    preds = clf.predict(X_te)
    total_rmse += sqrt(mse(y_te, preds))
print(f"Avg RMSE: {total_rmse/num_iterations}")
# print(f"Best parameters: {clf.best_params_}")
