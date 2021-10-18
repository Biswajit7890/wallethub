import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
from  sklearn.preprocessing import StandardScaler,MinMaxScaler
from  sklearn.decomposition import PCA
import os
import sys
import csv
from memory_profiler import profile


def preprocess(df):
    missing_cols = []
    for i in df.columns:
        if (df[i].isna().any()):
            missing_cols.append(i)
    categorical_cols=[]
    continuous_cols=[]
    for j in df.columns:
        if(df[j].nunique()>30):
            continuous_cols.append(j)
        else:
            categorical_cols.append(j)
    mode_fill=[]
    bfill=[]
    for k in missing_cols:
        if(df[k].nunique()<10):
           mode_fill.append(k)
        else:
           bfill.append(k)
    for j in  mode_fill:
        var=df[j].mode()[0]
        df[j]=df[j].fillna(var)
    for k in bfill:
        df[k]=df[k].bfill()
    return df


def transform(df_process):
    df_process = df_process.dropna()
    df_process= df_process.drop(labels=['y'], axis=1)
    X=np.array(df_process)
    sc_scaler = StandardScaler()
    X = sc_scaler.fit_transform(X)
    pca = PCA(n_components=150)
    pca_fit = pca.fit(X)
    reduced_X = pca_fit.transform(X)
    pca_df = pd.DataFrame(reduced_X, columns=['pc_' + str(i) for i in range(1, 151)])
    return pca_df

@profile
def inference():
    data_path=(sys.argv[1])
    print(data_path)
    parent_dir='wallethub'
    df=pd.read_csv(data_path)
    df_process=preprocess(df)
    df_transform=transform(df_process)
    X_test=np.array(df_transform)
    model_pkl=pickle.load(open(os.getcwd()+'/artifacts'+'/xgb_model_1.pkl', 'rb'))
    prediction=model_pkl.predict(X_test)
    pred_df=pd.DataFrame(prediction, columns=['prediction'])
    pred_df.to_csv('hold_out_prediction.csv', index=False)


if __name__ == '__main__':
      inference()

