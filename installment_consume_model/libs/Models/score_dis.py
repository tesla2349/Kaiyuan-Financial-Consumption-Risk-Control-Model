# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 15:56:40 2017

@author: chenbin
"""
import os
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))
sco = pd.read_csv(os.path.join(dir_path, 'score.csv'), encoding='gbk')
pp = pd.read_csv(os.path.join(dir_path, 'pp.csv'), encoding='utf-8')
COL_NAME_DICT = pd.read_csv(os.path.join(dir_path, 'col_name_mapping.csv'), encoding='utf-8').set_index(['字段代码']).to_dict()['字段中文名']

def score_dis(p):
    for i in range(len(sco)):
        if (p*100>sco['名称'][i] and p*100 <= sco['名称2'][i]):
        	s = p*100
        	d_rate = sco['坏账率'][i]
        	b_rate = (pp['rf'] < p).mean()
        	return s, d_rate, b_rate