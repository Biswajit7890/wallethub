import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def data_split(data_path):
    data=pd.read_csv(data_path)
    train , test = train_test_split(data, test_size=0.20, random_state=562)
    return train ,test
