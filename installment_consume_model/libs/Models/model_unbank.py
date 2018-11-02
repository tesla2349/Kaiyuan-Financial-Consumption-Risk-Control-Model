# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 10:16:36 2017

@author: chenbin
"""

import pandas as pd
import os
from sklearn.externals import joblib
import warnings

warnings.filterwarnings('ignore')

dir_path = os.path.dirname(os.path.realpath(__file__))


def data_pro(data):
    #education
    #data['education'].value_counts()
    education_map = {'小学':1, '初中':2,'专科':2,'高中':3, '技校':3,'本科以上':4}
    data['education'] = data['education'].map(education_map).fillna(0)

#==============================================================================
#     #data['netStatus'].value_counts()
#     netStatus_map = {'正常':1, '不正常':2}
#     data['netStatus'] = data['netStatus'].map(netStatus_map).fillna(-1)
# 
#==============================================================================
    #threeverify
    threeVerify_map = {'一致':1, '不一致':0}
    data['threeVerify'] = data['threeVerify'].map(threeVerify_map).fillna(-1)
    
    maritalStatus_map = {'未婚':1, '已婚':2}
    data['maritalStatus'] = data['maritalStatus'].map(maritalStatus_map).fillna(0)
                            
    #idverify
    idVerify_map = {'一致':1, '不一致':0}
    data['idVerify'] = data['idVerify'].map(idVerify_map).fillna(-1)
    
    #netLength
    netLen_map = {'12个月以内':1, '12-24个月':2, '24个月以上':3}
    data['netLength'] = data['netLength'].map(netLen_map).fillna(0)

def forp(data):
    columns =['CityId', 'sex']
    p_unbank = pd.read_excel(os.path.join(dir_path, 'p_unbank.xlsx'),encoding = 'utf-8', sheetname=None)
    for k in p_unbank.keys():
        p_map = p_unbank[k].set_index('码值').to_dict()['概率']
        data[k] = data[k].map(p_map).fillna(0)
    return None

def pro_transfer_func(seri, y):
    """将一个categorical字段转化成对应的条件概率特征
    
    Param
    -----
    seri: pd.Series
    y: pd.Series
    """
    cat_values = seri.unique()
    pro_map = {}
    
    for v in cat_values:
        pro_map[v] = y[seri == v].mean()
    return pro_map

def norm(data):
    l_list =['education','maritalStatus','netLength','age']
    l_unbank = pd.read_csv(os.path.join(dir_path, 'l_unbank.csv'),encoding = 'gbk')
    for i,name in enumerate(l_list):
        data[name] = (data[name]-l_unbank['min'][i])/(l_unbank['max'][i]-l_unbank['min'][i])       
    return None

def load_model():
    rf_clf_unbank = joblib.load(os.path.join(dir_path, "rf_clf_unbank_model.m"))
    lr_clf_unbank = joblib.load(os.path.join(dir_path, "lr_clf_unbank_model.m"))
    return rf_clf_unbank, lr_clf_unbank

data = pd.read_csv(os.path.join(dir_path, 'banknot.csv'))

data_pro(data)
data = data.fillna(0)
forp(data)
norm(data)
x= data.ix[:,1:]
columns = list(x.columns)

# rf_clf_unbank=joblib.load("/Users/chenbin/Desktop/开元/rf_clf_unbank_model.m")
# lf_clf_unbank=joblib.load("/Users/chenbin/Desktop/开元/lr_clf_unbank_model.m")

# data_columns = x.columns
# rf_clf_unbank.predict_proba(x)[:,1]