# -*- coding:utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn import preprocessing
from sklearn import grid_search
from sklearn.cross_validation import train_test_split
from sklearn import grid_search
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost.sklearn import XGBClassifier
from sklearn.utils import shuffle
from sklearn.externals import joblib
from scipy.stats import ks_2samp
from sklearn.utils import check_random_state
from scipy.sparse import hstack,vstack
from discretize import QuantileDiscretizer


dir_path = os.path.dirname(os.path.realpath(__file__))
if not dir_path in sys.path:
	sys.path.append(dir_path)

import json
from libs.data2model import data_before_model
import libs.Models.model_bank as bk
import libs.Models.model_unbank as ubk
from libs.Models.score_dis import score_dis, COL_NAME_DICT
from libs.Models.CalCeoflmp import GetTopImp
import warnings

warnings.filterwarnings('ignore')

class Model():

    def get_dataframe(self, json_str):
        data, isbank = self.preprocess_data(json.loads(json_str))
        model_data, model_cols = self.process_data(data, isbank)
        return model_data


    def fit_sample(self,X, y):
        """Resample the dataset.
        """
        label = np.unique(y)
        stats_c_ = {}
        maj_n = 0
        for i in label:
            nk = sum(y == i)
            stats_c_[i] = nk
            if nk > maj_n:
                maj_n = nk
                maj_c_ = i

        # Keep the samples from the majority class
        X_resampled = X[y == maj_c_]
        y_resampled = y[y == maj_c_]
        # Loop over the other classes over picking at random
        for key in stats_c_.keys():

            # If this is the majority class, skip it
            if key == maj_c_:
                continue

            # Define the number of sample to create
            num_samples = int(0.5*(stats_c_[maj_c_] - stats_c_[key]))

            # Pick some elements at random
            random_state = check_random_state(42)
            indx = random_state.randint(low=0, high=stats_c_[key], size=num_samples)

            # Concatenate to the majority class
            X_resampled = vstack([X_resampled, X[y == key], X[y == key][indx]])
            y_resampled = np.array(list(y_resampled) + list(y[y == key]) + list(y[y == key][indx]))
        return X_resampled, y_resampled


    def train_dataframe(self, df):
        bk.data_pro(df)
        bk.forp(df)

        df = shuffle(df)

        y = df['Identification'].values
        x = df.drop(['Identification'], axis=1).values
        for i in range(len(y)):
            y[i] = np.round(y[i])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state = 33)
        x_train, y_train = self.fit_sample(x_train, y_train)

        # 逻辑回归分类
        lr = LogisticRegression()
        tuned_parameters = [{'penalty': ('l1', 'l2'), 'C': [0.001], 'class_weight': [None]}]
        lr_clf = grid_search.GridSearchCV(lr, tuned_parameters, cv=5, scoring='roc_auc')
        lr_clf.fit(x_train, y_train)

        print('逻辑回归模型train AUC:')
        print(lr_clf.score(x_train,y_train))

        print('逻辑回归模型test AUC:')
        print(lr_clf.score(x_test,y_test))

        get_ks = lambda y_pred, y_true: ks_2samp(y_pred[y_true == 1], y_pred[y_true != 1]).statistic

        lr_ks = get_ks(lr_clf.predict(x_train), y_train)
        print('逻辑回归模型train KS值：')
        print(lr_ks)

        lr_ks = get_ks(lr_clf.predict(x_test), y_test)
        print('逻辑回归模型test KS值：')
        print(lr_ks)

        rf = RandomForestClassifier()
        tuned_parameters = [{'n_estimators': [200], 'criterion':['entropy'] ,'max_depth': [15], 'min_samples_split': [3]}]

        rf_clf = grid_search.GridSearchCV(rf, tuned_parameters, cv=5, scoring='roc_auc')
        rf_clf.fit(x_train, y_train)

        print ('随机森林模型train AUC:')
        print(rf_clf.score(x_train, y_train))

        print ('随机森林模型test AUC:')
        print(rf_clf.score(x_test,y_test))

        rf_ks = get_ks(rf_clf.predict(x_train), y_train)
        print('随机森林模型train KS值：')
        print(rf_ks)

        rf_ks = get_ks(rf_clf.predict(x_test), y_test)
        print('随机森林模型test KS值：')
        print(rf_ks)


        joblib.dump(lr_clf, os.path.join(dir_path, 'lr_clf_bank_model.m'))
        joblib.dump(rf_clf, os.path.join(dir_path, 'rf_clf_bank_model.m'))



    def predict_dataframe(self,df):
        lr = joblib.load(os.path.join(dir_path, "lr_clf_bank_model.m"))
        rf = joblib.load(os.path.join(dir_path, "rf_clf_bank_model.m"))

        bk.data_pro(df)
        bk.forp(df)

        df = shuffle(df)
        df_x = df.drop(['Identification'], axis=1)
        train_x = df_x.values
        model_cols = df_x.columns
        results = []
        for x in train_x:
            score, d_rate, b_rate = score_dis(rf.predict_proba(x.reshape(1,-1))[0, 1])
            d_rate = "%.2f%%" % (d_rate * 100)
            b_rate = '%.2f%%' % (b_rate * 100)
            score = "%.2f" % score
            topFive = GetTopImp(pd.Series(x.flatten(), index=model_cols), (lr.best_estimator_.coef_.flatten() * (-1)))
            fiveReduceItem = [c[0] for c in topFive]
            fiveReduceItemValue = [c[1] for c in topFive]
            fiveReduceItem = [COL_NAME_DICT[item] for item in fiveReduceItem]
            fiveReduceItemValue = [("%.2f" % v) for v in fiveReduceItemValue]
            fiveReduceItem.extend([""]*(5 - len(fiveReduceItem)))
            fiveReduceItemValue.extend([""] * (5 - len(fiveReduceItem)))
            result = {'grade': score,
                      'fiveReduceItem': fiveReduceItem,
                      'fiveReduceItemValue': fiveReduceItemValue,
                      'referenceDefaultsRate': d_rate,
                      'beatUserRate': b_rate}
            results.append(result)
        return pd.DataFrame(results)


    def get_result(self, json_str):
        """得到PRO需要的输出结果

        :param json_str: str
            Json字符串
        :return: str
            模型返回结果的json字符串
        """
        data, isbank = self.preprocess_data(json.loads(json_str))
        model_data, _ = self.process_data(data, isbank)
        model_cols = ['CityId', 'cashTotalAmt', 'cashTotalCnt', 'education', 'idVerify', 'inCourt', 'maritalStatus',
                      'monthCardLargeAmt', 'netLength', 'noTransWeekPre', 'onlineTransAmt', 'onlineTransCnt',
                      'publicPayAmt', 'publicPayCnt', 'threeVerify', 'transTotalAmt', 'transTotalCnt', 'transCnt_mean',
                      'transCnt_non_null_months', 'transAmt_mean', 'transAmt_non_null_months', 'cashCnt_mean',
                      'cashCnt_non_null_months', 'cashAmt_mean', 'cashAmt_non_null_months', 'isCrime', 'isBlackList',
                      'age', 'sex', 'card_age']
        try:
            model_data.drop('Han', axis=1, inplace=True)
        except:
            pass
        try:
            model_data.drop('isDue', axis=1, inplace=True)
        except:
            pass
        rf, lr = self.model_load(1)
        final_x = model_data.ix[:, model_cols].fillna(0).values
        score, d_rate, b_rate = score_dis(rf.predict_proba(final_x)[0, 1])
        d_rate = "%.2f%%" % (d_rate * 100)
        b_rate = '%.2f%%' % (b_rate * 100)
        score = "%.2f" % score
        topFive = GetTopImp(pd.Series(final_x.flatten(), index=model_cols), (lr.best_estimator_.coef_.flatten() * (-1)))
        fiveReduceItem = [c[0] for c in topFive]
        fiveReduceItemValue = [c[1] for c in topFive]
        fiveReduceItem = [COL_NAME_DICT[item] for item in fiveReduceItem]
        fiveReduceItemValue = [("%.2f" % v) for v in fiveReduceItemValue]
        fiveReduceItem.extend([""]*(5 - len(fiveReduceItem)))
        fiveReduceItemValue.extend([""] * (5 - len(fiveReduceItem)))
        result = {'grade':score,
                  'fiveReduceItem':fiveReduceItem,
                  'fiveReduceItemValue':fiveReduceItemValue,
                  'referenceDefaultsRate':d_rate,
                  'beatUserRate':b_rate}
        return json.dumps(result)


    @staticmethod
    def preprocess_data(json_dict):
        return data_before_model(json_dict)

    @staticmethod
    def process_data(data, isbank):
        if isbank:
            bk.data_pro(data)
            data = data.fillna(0)
            bk.forp(data)
            bk.norm(data)
            model_cols = bk.columns
        else:
            ubk.data_pro(data)
            data = data.fillna(0)
            ubk.forp(data)
            ubk.norm(data)
            model_cols = ubk.columns
        return data, model_cols

    @staticmethod
    def model_load(isbank):
        """导入模型

        :param isbank: bool
            是否有银行数据
        :return: tuple
            (RandomForest, LogisticRegression)
        """
        if isbank:
            rf, lr = bk.load_model()
        else:
            rf, lr = ubk.load_model()
        return rf, lr

