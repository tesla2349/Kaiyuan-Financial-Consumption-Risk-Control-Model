# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os

preprocess_dir = os.path.dirname(os.path.realpath(__file__))
bank_file = os.path.join(preprocess_dir, 'preprocess_cols_bank')
notbank_file = os.path.join(preprocess_dir, 'preprocess_cols_notbank')

def process_maritalStatus(s):
    """处理婚姻状况字段

    :param s:
    :return:
    """
    maritalCodes = {'未说明婚姻状况':'未说明', '离婚':'离婚', '丧偶':'未婚',
                    '复婚':'已婚', '再婚':'已婚', '初婚':'已婚', '已婚':'已婚', '未婚':'未婚'}
    return maritalCodes.get(s, '未说明')

def process_education(s):
    """处理文化程度字段

    :param s: str
        文化程度
    :return:
    """
    if '初中' in s:
        return '初中'
    elif '高中' in s:
        return '高中'
    elif '专' in s:
        return '专科'
    elif '小学' in s:
        return '小学'
    elif '大学' in s:
        return '本科以上'
    elif '研究生' in s:
        return '本科以上'
    elif '技' in s:
        return '技校'
    elif '文盲' in s:
        return '小学'
    else:
        return 'Missing'

def process_court_info(data):
    """处理法院失信被执行人信息

    :param data:
    :return:
    """
    court_data = [k for k in list(data.keys()) if k.startswith('performance')]
    if len(court_data) > 0:
        return {'inCourt': 1}
    else:
        return {'inCourt': 0}

def process_due_info(s):
    """处理逾期信息

    :param s:
    :return:
    """
    if s is np.nan:
        return s

    due_codes = {'成功查得': 0, '未查得': 1}
    return due_codes.get(s, np.nan)

def process_crime_info(data):
    """处理不良信息

    :param s:
    :return:
    """
    crime_data = [k for k in list(data.keys()) if k.startswith('checkCode')]
    if len(crime_data) > 0:
        return {'isCrime': 1}
    else:
        return {'isCrime': 0}

def process_black_list(data):
    """处理黑名单信息

    :param data:
    :return:
    """
    bl_data = [k for k in list(data.keys()) if k.startswith('reason')]
    if len(bl_data) > 0:
        return {'isBlackList': 1}
    else:
        return {'isBlackList': 0}

def process_nation(s):
    """处理民族信息

    :param s:
    :return:
    """
    if s is np.nan:
        return s
    return 1 if s == '汉族' else 0

def process_age(data):
    """计算申请人年龄

    :param data:
    :return:
    """
    sendDate = pd.to_datetime(data.get('sendTime', np.nan), format='%Y-%m-%d', errors='coerce')
    birthDate = pd.to_datetime(data['idCard'][6:14], format='%Y%m%d', errors='coerce')
    age = np.NaN
    if str(sendDate)!='NaT':
        age = int((sendDate - birthDate).days/365)
    return age

def process_netLength(data):
    """处理申请人在网时长数据

    :param data:
    :return:
    """
    netL_cols = [k for k in list(data.keys()) if k.endswith('netLength')]
    if len(netL_cols) == 0:
        return np.nan
    elif len(netL_cols) == 1:
        return data[netL_cols[0]]

    netLen = [data[col] for col in netL_cols]
    netLen = [d for d in netLen if (d != '无效 ') or (d is not np.nan)]
    result = np.nan if len(netLen) == 0 else netLen[0]
    return '12个月以内' if (result == '0-6个月') or (result == '6-12个月') else result

def card_age_calculate(data):
    """计算银行卡距离申请授信时的开卡时长(月数)

    :param data:
    :return:
    """
    sendDate = pd.to_datetime(data['sendTime'], format='%Y-%m-%d', errors='coerce')
    firstTransDate = pd.to_datetime(data['firstTransDate'], format='%Y%m%d', errors='coerce')

    return np.around((sendDate - firstTransDate).days/30)

def detail_combine(data, head):
    """处理银行卡明细数据

    :param data: dict
    :param head: str
    :return: dict
    """
    detail_data = [data[k] for k in list(data.keys()) if k.startswith(head)]
    if len(detail_data) == 0:
        return {head+'_mean':0, head+'_non_null_months':0}
    non_null = [d for d in detail_data if (d is not np.nan) and ((type(d) == float) or (type(d) == int))]
    return {head + '_mean': np.mean(non_null), head + '_non_null_months': len(non_null)}

def preprocess_data(data):
    """对数据进行预处理

    :param data: dict
    :return: dict
    """
    isBank = 'firstTransDate' in data.keys()

    # 有银行卡数据和没有银行卡数据的客户都需要预处理的字段
    data['sex'] = data.pop('sexId', np.nan)
    # data['CityId'] = data.pop('cityName', np.nan)
    data['maritalStatus'] = process_maritalStatus(data.get('maritalStatus', '未说明婚姻状况'))
    data['education'] = process_education(data.get('education', 'Missing'))
    data['isDue'] = process_due_info(data.get('isDue', np.nan))
    data['Han'] = process_nation(data.get('nation', np.nan))
    data['age'] = process_age(data)
    data['netLength'] = process_netLength(data)
    data.update(process_court_info(data))
    data.update(process_crime_info(data))
    data.update(process_black_list(data))

    if isBank:
        # 有银行卡数据客户需要处理的字段
        data['card_age'] = card_age_calculate(data)
        detail_heads = ['transCnt', 'transAmt', 'cashCnt', 'cashAmt']
        for head in detail_heads:
            data.update(detail_combine(data, head))

    return data, isBank

def data_gen(data, isBank):
    """根据数据将要进入的模型决定最终的数据所包含的字段

    :param data: dict
    :param isBank: bool
    :return: pd.DataFrame
    """
    file_path = bank_file if isBank else notbank_file
    with open(file_path, 'r') as f:
        col_fetch = f.read().splitlines()
    return pd.DataFrame([{col:data.get(col, np.nan) for col in col_fetch}]), isBank

