# -*- coding:utf-8 -*-
from libs.EDP.external_data_parsing import edp
from libs.Preprocess.preprocess import preprocess_data, data_gen

def data_before_model(json_str):
    """将数据处理成进入模型前的格式

    :param json_str: str
        json字符串
    :return: tuple (pd.DataFrame, bool)
        (data, isBank)
    """
    data = edp.individual_data_process(json_str)
    data, isbank = preprocess_data(data)
    return data_gen(data, isbank)

