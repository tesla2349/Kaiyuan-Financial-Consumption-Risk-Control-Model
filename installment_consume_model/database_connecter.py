import pymysql.cursors
import json
import pandas as pd
import numpy as np
from model import Model

def connecter(host, port, user, password, db):
    """从数据库中获取数据
    :param
        host: ip地址
        port：端口,
        user：用户名,
        password：密码,
        db：数据库名字
    :return: json
        数据库中所有的数据，json格式
    """
    # 连接MySQL数据库
    connection = pymysql.connect(host=host, port=port, user=user, password=password, db=db, charset='utf8mb4',cursorclass=pymysql.cursors.DictCursor)
    # 通过cursor创建游标
    cursor = connection.cursor()
    # 获取所有数据
    sql = "SELECT * FROM ModelData"
    cursor.execute(sql)
    all_data = cursor.fetchall()
    return all_data


def data_parser_train(all_data):
    """训练模型
    :param all_data json
    从数据库中获取的数据，json格式
    :return: Dataframe
        训练数据，pandas Dataframe格式
    """
    individual_keys = ['PLCS1116FX001', 'PLINTER', 'PLCS1116YY003', 'PLCS1116YY004', 'PLCS1116YY001', 'PLCS1116FX007',
                       'PLCS1116YY002', 'PLCS1116FX008', 'PLCS1116FX005', 'PLCS1116FX006', 'PLCS1116YY005',
                       'PLCS1116FX003', 'PLCS1116YY006', 'PLCS1116FX004', 'PLCS1116SH004', 'PLCS1116SF003',
                       'PLCS1116YY009', 'PLCS1116YH001']
    df_pro = None
    for record in all_data:
        s = {}
        for record_key in record:
            try:
                data = json.loads(record[record_key])
                s[data['resNum']] = data
            except:
                pass

        for key in individual_keys:
            if key not in s.keys():
                s[key] = None

        s['idCard'] = record['idCard']
        pro = Model().get_dataframe(json.dumps(s))
        pro['Identification'] = record['Identification']
        try:
            pro['CityId'] = s['PLCS1116YH001']['data']['indexProperty']['fromCity']
        except:
            pro['CityId'] = np.NaN

        if df_pro is None:
            df_pro = pro
        else:
            df_pro = pd.concat([df_pro,pro])

    df_pro.Identification.replace(0, np.NaN, inplace = True)
    df_pro.Identification.replace(2, 0, inplace=True)
    city_1 = ['北京市', '上海市', '广州市', '深圳市']
    city_n1 = ['成都市', '杭州市', '武汉市', '重庆市', '南京市', '天津市', '苏州市', '西安市', '长沙市', '沈阳市', '青岛市', '郑州市 ', '大连市', '东莞市', '宁波市']
    city_2 = ['厦门市', '福州市', '无锡市', '合肥市', '昆明市', '哈尔滨市', '济南市', '佛山市', '长春市', '温州市', '石家庄市', '南宁市', '常州市', '泉州市', '南昌市', '贵阳市', '太原市', '烟台市', '嘉兴市', '南通市', '金华市', '珠海市', '惠州市', '徐州市', '海口市', '乌鲁木齐市', '绍兴市', '中山市', '台州市', '兰州市']
    df_pro.CityId = df_pro.CityId.apply(lambda x: '一线城市' if x in city_1 else '新一线城市' if x in city_n1 else '二线城市' if x in city_2 else '其它')
    return df_pro