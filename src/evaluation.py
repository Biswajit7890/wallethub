import pandas as pd
import numpy as np
import pickle
from sklearn import metrics
import os
from memory_profiler import profile

@profile
def evaluation(data_path,artifact_path,target_path):
    for (root, dirs, files) in os.walk(data_path):
        for file in files:
            if (file == 'test_pca.csv'):
                file_1 = file
    for (root, dirs, files) in os.walk(target_path):
        for file in files:
            if (file == 'test.csv'):
                file_2 = file
    for (root, dirs, files) in os.walk(artifact_path):
        for file in files:
            if (file == 'xgb_model_1.pkl'):
                file_3 = file
    df_test=pd.read_csv(data_path+'/'+file_1)
    df_target=pd.read_csv(target_path+'/'+file_2)
    X=np.array(df_test)
    y_true=df_target['y'].values
    model=pickle.load(open(artifact_path+'/'+file_3, 'rb'))
    pred_test=model.predict(X)
    evaldf=pd.DataFrame(y_true,columns=['True_target'])
    evaldf['pred_target']=pred_test
    evaldf['rmse']=np.sqrt(metrics.mean_squared_error(pred_test,y_true))
    evaldf['absolute difference']=np.abs(evaldf['True_target']-evaldf['pred_target'])
    evaldf['Accuracy prediction']=np.where(evaldf['absolute difference']>3,'wrong','correct')
    return evaldf







