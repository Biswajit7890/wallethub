import pandas as  pd
import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

def train(train_path,target_path):
    train_error=[]
    test_error=[]
    folds=[]
    for (root, dirs, files) in os.walk(train_path):
        for file in files:
            if(file=='train_pca.csv'):
                file_1 = file
    for (root, dirs, files) in os.walk(target_path):
        for file in files:
            if(file=='train.csv'):
                file_2 = file
    df_train=pd.read_csv(train_path+'/'+file_1)
    df_target=pd.read_csv(target_path+'/'+file_2)
    df_target=df_target[:79997]
    X=np.array(df_train)
    y=df_target['y'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    space = {'max_depth': hp.quniform("max_depth", 3, 18, 1),
             'learning_rate': hp.choice('learning_rate', np.arange(0.005, 0.31, 0.05)),
             'gamma': hp.uniform('gamma', 1, 9),
             'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
             'reg_lambda': hp.uniform('reg_lambda', 0, 1),
             'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
             'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
             'n_estimators': 1600,
             'seed': 0
             }
    def objective(space):
        clf = xgb.XGBRegressor(
            n_estimators=space['n_estimators'], max_depth=int(space['max_depth']), gamma=space['gamma'],
            reg_alpha=int(space['reg_alpha']), min_child_weight=int(space['min_child_weight']),
            colsample_bytree=int(space['colsample_bytree']),eval_metric='rmse')

        evaluation = [(X_train, y_train), (X_test, y_test)]

        clf.fit(X_train, y_train,eval_set=evaluation,early_stopping_rounds=10, verbose=False)
        pred = clf.predict(X_test)
        test_error =np.sqrt(mean_squared_error(y_test,pred))
        print("Test Error",test_error)
        return {'loss':test_error , 'status': STATUS_OK}

    trials = Trials()
    best_hyperparams = fmin(fn=objective,space=space,algo=tpe.suggest,max_evals=100,trials=trials)
    print("The best hyperparameters are : ", "\n")
    print(best_hyperparams)
    kf = KFold(n_splits=5)
    j = 1
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        nb4 = xgb.XGBRegressor(n_estimators=1600, max_depth=8, gamma=8.838817028111649,reg_alpha=66.0, min_child_weight=5.0,
            colsample_bytree=0.8264029302118475,learning_rate=0.025,reg_lambda=0.4359729407833042)
        search = nb4.fit(X_train, y_train)
        pred_test = search.predict(X_test)
        pred_train = search.predict(X_train)
        print("The rmse score of Train", np.sqrt(mean_squared_error(y_train, pred_train)))
        print("The rmse score of Test", np.sqrt(mean_squared_error(y_test, pred_test)))
        print('*' * 50)
        train_error.append(np.sqrt(mean_squared_error(y_train, pred_train)))
        test_error.append(np.sqrt(mean_squared_error(y_test, pred_test)))
        folds.append(j)
        j=j+1
    train_metrics=pd.DataFrame({'Folds':j,'Train Error':train_error,'Test Error':test_error})
    return train_metrics,search



