import pandas as pd
import numpy as np
import os
import csv
from src.getdata import getdata
from src.preprocess import preprocees
from src.dataset_split import data_split
from src.transformation import transformation
from src.stats_assumption import test_statistics
from src.train import train
from src.evaluation import evaluation
import pickle


parent_dir='wallethub'
currentpath=os.path.join(os.curdir ,parent_dir)
rawdataset='rawdataset'
os.mkdir(rawdataset)
rawdataset_path=os.path.abspath(rawdataset)
file='dataset_00_with_header.csv'
data_path=os.path.join(rawdataset_path,file)
df=getdata(data_path)
preprocess_dir='preprocess'
os.mkdir(preprocess_dir)
preprocess_path=os.path.join(currentpath,preprocess_dir)
proc_df=preprocees(data_path)
filename_1='process.csv'
proc_df.to_csv(os.curdir+preprocess_path+'/'+filename_1, index=False)
split_path=os.curdir+preprocess_path+'/'+filename_1
datasplit_dir='datasplit'
os.mkdir(datasplit_dir)
datasplit_path=os.path.join(currentpath,datasplit_dir)
traindf ,testdf =data_split(split_path)
traindf.to_csv(os.curdir+datasplit_path+'/train.csv', index=False)
testdf.to_csv(os.curdir+datasplit_path+'/test.csv', index=False)
transformation_dir='transformation'
os.mkdir(transformation_dir)
transformation_path=os.path.join(currentpath,transformation_dir)
for (root,dirs,files) in os.walk(os.curdir+datasplit_path):
    for file in files:
        if (file == 'train.csv'):
            train_pca = transformation(os.curdir + datasplit_path + '/' + file)
            train_pca.to_csv(os.curdir + transformation_path + '/train_pca.csv', index=False)
        else:
            test_pca = transformation(os.curdir + datasplit_path + '/' + file)
            test_pca.to_csv(os.curdir + transformation_path + '/test_pca.csv', index=False)
statistics_dir='statistics'
os.mkdir(statistics_dir)
statistics_path=os.path.join(currentpath,statistics_dir)
test_statistics(os.curdir+preprocess_path+'/'+filename_1,statistics_path)
train_error_dir='train_error_logs'
os.mkdir(train_error_dir)
train_error_path=os.path.join(currentpath,train_error_dir)
trainerr_df,model= train(os.curdir+transformation_path,os.curdir+datasplit_path)
artifacts_dir='artifacts'
os.mkdir(artifacts_dir)
artifacts_path=os.path.join(currentpath,artifacts_dir)
trainerr_df.to_csv(os.curdir+train_error_path+'/train_error.csv', index=False)
file_model = 'xgb_model_1.pkl'
pickle.dump(model, open(os.curdir+artifacts_path+'/'+file_model, 'wb'))
eval_df=evaluation(os.curdir+transformation_path,os.curdir+artifacts_path,os.curdir+datasplit_path)
eval_dir='evaluation'
os.mkdir(eval_dir)
eval_path=os.path.join(currentpath,eval_dir)
eval_df.to_csv(os.curdir + eval_path + '/eval.csv', index=False)











