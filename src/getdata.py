import pandas as pd
import numpy as np
import os
import requests

def getdata(data_path):
    data = pd.read_csv(data_path)
    print(data.head)
    print(data.shape)
    return data

