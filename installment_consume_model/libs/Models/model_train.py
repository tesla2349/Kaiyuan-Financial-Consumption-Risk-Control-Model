import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn import preprocessing
from sklearn import grid_search
from sklearn.cross_validation import train_test_split
from sklearn import grid_search
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.externals import joblib

dir_path = os.path.dirname(os.path.realpath(__file__))
if not dir_path in sys.path:
	sys.path.append(dir_path)

import model_bank as bk
import model_unbank as ubk

bank_data = pd.read_csv(os.path.join(dir_path, 'bank.csv'))
bk.data_pro(bank_data)
bank_data = bank_data.fillna(0)
bk.forp(bank_data)
bk.norm(bank_data)
unbank_data = pd.read_csv(os.path.join(dir_path, 'banknot.csv'))
ubk.data_pro(unbank_data)
unbank_data = unbank_data.fillna(0)
ubk.forp(unbank_data)
ubk.norm(unbank_data)

data_dict = {'bank':bank_data, 'unbank':unbank_data}


for k in data_dict.keys():
	data = data_dict[k]
	y = data['isBad_30']
	x= data.ix[:,1:]

	x_train, x_test,y_train,y_test = train_test_split(x, y, test_size=0.2,random_state = 33)
	#逻辑回归分类
	lr = LogisticRegression()
	tuned_parameters = [{'penalty':('l1','l2'),'C':[0.03, 0.1, 0.3, 1,100,10000],'class_weight':[None,'balanced']}]
	lr_clf = grid_search.GridSearchCV(lr,tuned_parameters,cv=5,scoring='roc_auc')
	#dir(lr_clf)
	#lr_clf.best_estimator_
	lr_clf.fit(x_train, y_train)


	rf = RandomForestClassifier()
	rf.get_params()
	tuned_parameters = [{'max_features':['log2'],'n_estimators':[500,1000],'max_depth':[12,15]}]
	                     
	rf_clf = grid_search.GridSearchCV(rf,tuned_parameters,cv=5,scoring='roc_auc')
	rf_clf.fit(x_train, y_train)
	joblib.dump(lr_clf, os.path.join(dir_path, 'lr_clf_%s_model.m' % k))
	joblib.dump(rf_clf, os.path.join(dir_path, 'rf_clf_%s_model.m' % k))
