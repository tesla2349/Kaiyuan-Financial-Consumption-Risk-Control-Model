#-*- coding:utf-8 -*-
from database_connecter import connecter,data_parser_train
import pandas as pd
from model import Model
import json

if __name__ == '__main__':
    # 从数据库中获取数据
    all_data = connecter(host='47.104.244.32', port=3306, user='yss', password='123456', db='pro')
    # 结构化处理数据
    df_pro = data_parser_train(all_data)
    # 保存入模数据
    df_pro.to_csv('modeling_data.csv', index=False)
	
	# 读取入模数据
    df_pro = pd.read_csv('modeling_data.csv')
    df_pro.drop('Han', axis=1, inplace=True)
    df_pro.drop('isDue', axis=1, inplace=True)
    df_pro.fillna(0, inplace=True)
	# 训练风控模型
    Model().train_dataframe(df_pro)
	# 预测结果
    df_prediction = Model().predict_dataframe(df_pro)
	df_prediction.to_csv('predictions.csv', index=False)