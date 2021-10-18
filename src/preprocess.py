import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt

def preprocees(data_path):
    df=pd.read_csv(data_path)
    columns=df.columns
    for j in columns:
        print("The unique value of"+j+"is",df[j].nunique())
    print("The datatype info of all columns in the dataframe",df.info())
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






