import pandas as pd
import numpy as np
from  sklearn.preprocessing import StandardScaler,MinMaxScaler
from  sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def transformation(data_path):
    X=pd.read_csv(data_path)
    X=X.dropna()
    X=X.drop(labels=['y'],axis=1)
    sc_scaler=StandardScaler()
    X=sc_scaler.fit_transform(X)
    pca=PCA(n_components=150)
    pca_fit = pca.fit(X)
    reduced_X = pca_fit.transform(X)
    print(np.round(reduced_X[0:150], 2))
    var_explained = pca.explained_variance_ratio_75
    print("The Explained variance of PCA on 150 columns", np.round(var_explained, 2))
    var_explained_cumulative = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100)
    print("The Cummalative Explained variance of PCA on 150 columns", var_explained_cumulative)
    plt.plot(range(1, 151), var_explained_cumulative)
    plt.xlabel('Number of components')
    plt.ylabel('% Variance explained')
    plt.show()
    np.set_printoptions(suppress=True)
    pca_df = pd.DataFrame(reduced_X, columns=['pc_' + str(i) for i in range(1, 151)])
    return pca_df

