# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 10:16:36 2017

@author: chenbin
"""
import os
import pandas as pd
from sklearn.externals import joblib
import warnings

warnings.filterwarnings('ignore')

dir_path = os.path.dirname(os.path.realpath(__file__))

def data_pro(data):
    #education
    education_map = {'小学':1, '初中':2,'专科':2,'高中':3, '技校':3,'本科以上':4}
    data['education'] = data['education'].map(education_map).fillna(0)

    #threeverify
    threeVerify_map = {'一致':1, '不一致':0}
    data['threeVerify'] = data['threeVerify'].map(threeVerify_map).fillna(-1)

    maritalStatus_map = {'未婚':1, '已婚':2}
    data['maritalStatus'] = data['maritalStatus'].map(maritalStatus_map).fillna(0)
                            
    #idverify
    idVerify_map = {'一致':1, '不一致':0}
    data['idVerify'] = data['idVerify'].map(idVerify_map).fillna(-1)
    
    #netLength
    netLen_map = {'无效':-1, '0-6个月':0, '6-12个月':1, '12-24个月':2, '24 个月以上':3}
    data['netLength'] = data['netLength'].map(netLen_map).fillna(-1)
            
    #卡龄转化
    data['card_age'][data['card_age']< 0] = 0

def forp(data):
    columns =['CityId', 'sex']
    p_bank = pd.read_excel(os.path.join(dir_path, 'p_bank.xlsx'),encoding = 'utf-8', sheetname=None)
    for k in p_bank.keys():
        p_map = p_bank[k].set_index('码值').to_dict()['概率']
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
    l_list =['education','maritalStatus','netLength','age','cashTotalAmt','cashTotalCnt','monthCardLargeAmt','onlineTransAmt','onlineTransCnt','publicPayAmt','publicPayCnt','transTotalAmt'
    ,'transTotalCnt','transCnt_non_null_months','transAmt_mean','transAmt_non_null_months'
    ,'cashCnt_mean','cashCnt_non_null_months','cashAmt_mean','cashAmt_non_null_months','card_age']
    l_bank = pd.read_csv(os.path.join(dir_path, 'l_bank.csv'),encoding = 'gbk')
    for i,name in enumerate(l_list):
        data[name] = (data[name]-l_bank['min'][i])/(l_bank['max'][i]-l_bank['min'][i])       
    return None

def load_model():
    rf_clf_bank = joblib.load(os.path.join(dir_path, "rf_clf_bank_model.m"))
    lr_clf_bank = joblib.load(os.path.join(dir_path, "lr_clf_bank_model.m"))
    return rf_clf_bank, lr_clf_bank



data = pd.read_csv(os.path.join(dir_path, 'bank.csv'))

data_pro(data)
data = data.fillna(0)
forp(data)
norm(data)
x= data.ix[:,1:]
columns = list(x.columns)

# rf_clf_bank=joblib.load("rf_clf_bank_model.m")
# lf_clf_bank=joblib.load(os.path.join(dir_path, "lr_clf_bank_model.m"))

# data_columns = x.columns
# rf_clf_bank.predict_proba(x)[:,1]