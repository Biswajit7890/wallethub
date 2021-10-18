import pandas as pd
import numpy as np
from scipy.stats import iqr
import statsmodels.api as sm
from scipy.stats import norm
from scipy.stats import kstest, norm
from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import shapiro
import scipy.stats as stats
from scipy.stats import normaltest
from scipy.stats import jarque_bera
import os

def test_statistics(data_path,statistics_path):
    cols=[]
    std_list=[]
    skew_list=[]
    min_list=[]
    iqr_list=[]
    max_list=[]
    df = pd.read_csv(data_path)
    df=df.dropna()
    print(df.describe(include=np.number))
    for col in df.columns:
        print("The standard deviation of", format(col), df[col].std())
        print("The skewness of", format(col), df[col].skew())
        print("The Minimum of", format(col), df[col].min())
        print("The IQR of", format(col), iqr(df[col]))
        print("The Maximun of", format(col), df[col].max())
        cols.append(col)
        std_list.append(df[col].std())
        skew_list.append(df[col].skew())
        min_list.append(df[col].min())
        iqr_list.append(iqr(df[col]))
        max_list.append(df[col].max())
    #Kolmogorov Smirnov test
    df = np.reshape(np.array(df), (1, -1))
    ks_statistic, p_value = kstest(df, 'norm')
    if (p_value > 0.05):
        print("Data is Gaussian Distributed on Kolmogorov Smirnov test ")
    else:
        print("Data is Not Gaussian distributed on Kolmogorov Smirnov test")
    ##Shapiro Wilk test
    stat, p_value = shapiro(df)
    try:
       if (p_value > 0.05):
           print("Data is Gaussian Distributed on Shapiro Wilk test ")
    except:
        raise("There is an Data Limit warning")
    else:
        print("Data is Not Gaussian distributed on Shapiro Wilk test")
    ##jarque_bera test
    stat, p_value = jarque_bera(df)
    try:
       if (p_value > 0.05):
          print("Data is Gaussian Distributed on jarque_bera test")
    except:
        raise("There is an Data Limit warning")
    else:
        print("Data is Not Gaussian distributed on jarque_bera test")
    stat_df=pd.DataFrame({'col':cols,'Std':std_list,'Skew':skew_list,'Min':min_list,'IQR':iqr_list,'Max':max_list})
    stat_df.to_csv(os.curdir+statistics_path+'/stats.csv', index=False)