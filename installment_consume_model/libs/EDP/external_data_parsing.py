# -*- coding: utf-8 -*-
import os
import copy
import json
import numpy as np
import pandas as pd

conf_file_dir = os.path.dirname(os.path.realpath(__file__))
class ExternalDataParsing():
    """用于解析各种外部数据,需要跟配置文件配合使用

    参数
    ----
    conf_file_path: str
        规定格式的配置文件所在路径
    """
    def __init__(self, conf_file_path):
        """构建解析工具

        :param conf_file_path: str
            配置文件路径
        """
        self.conf_info = pd.read_excel(conf_file_path, encoding='utf-8', sheetname=None,
                                       converters={'码值': str})
        self.BASE_INFO_ = self.__base_conf_info__(self.conf_info['字段数据'])
        self.CODES_ = self.__codes_name_mapping__(self.conf_info['码表'])
        self.STRU_KEYS_ = self.__keys_info_parsing__(self.conf_info['结构性键名'])
        self.CONT_KEYS_ = self.__keys_info_parsing__(self.conf_info['内容性键名'])

    def is_interface_na(self, individual, interface_index):
        """接口返回数据是否为空

        :param individual: dict
            字典键为接口编号
        :param interface_index: str
            外部数据的接口编号
        :return: True,如果接口interface_index没有返回数据;否则,False
        """
        individual = individual[interface_index]
        return (individual is None) or (int(individual['resStatus']) == 2)

    def is_interface_an(self, individual, interface_index):
        """接口返回数据是否异常

        :param individual: dict
            字典键为接口编号
        :param interface_index:
            外部数据的接口编号
        :return: True,如果接口interface_index返回数据异常;否则,False
        """
        individual = individual[interface_index]
        return (individual is None) or (int(individual['resStatus']) == 3)

    def data_fetch(self, individual, interface_index, fea_name):
        """返回一个客户在接口interface_index中的数据

        :param individual: dict
            字典的键为接口编号
        :param interface_index: str
            需要返回的数据的接口编号
        :param structural_loc: list
            需要提取的指标在individual中的具体位置,是一些格式化的key名称。
        :return: (list/dict, bool)
            客户在接口interface_index中的数据,返回类型视具体情况而定
        """
        if self.is_interface_na(individual, interface_index) or self.is_interface_an(individual, interface_index):
            return {}

        result = individual[interface_index]['data']
        structural_loc = self.STRU_KEYS_.get(interface_index, None)
        if structural_loc is not None:
            structural_loc = structural_loc.get(fea_name, [])
            for k in structural_loc:
                result = result.get(k, {})

        return result

    def get_codes(self, key_str, interface_index):
        """返回字段代码key_str对应的码值映射字典

        :param key_str: str
            字段代码
        :param interface_index: str
            接口编号

        :return: dict/None
            当没有对应的码值映射字典时返回None,否则返回该字典
        """
        result = self.CODES_.get(interface_index, None)
        if result is not None:
            result = result.get(key_str, None)
        return result

    def value_mapping(self, value, key_str, interface_index):
        """根据码表进行映射。
        如果该函数运行报错大致是两个原因造成:1、原始数据类型不是字符型,或原始Series中的数据
        不是字符型;2、码表在读入的时候不是字符型(less likely)

        :param value: str/pd.Series
            需要转化的数据
        :param key_str: str
            数据在外部数据中的字段代码
        :param interface_index: str
            数据在外部数据中的所属接口代码名称

        :return: str/pd.Series
            完成映射后的数据,具体视传入参数而定。
        """
        if value is np.nan:
            return value

        codes_tmp = self.get_codes(key_str, interface_index)
        if codes_tmp is None:
            return value

        if type(value) == str:
            return codes_tmp.get(value, np.nan)
        if type(value) == pd.Series:
            return value.map(codes_tmp)
        else:
            return value

    def type_mapping_helper(self, value, before_transform, after_transform):
        """将数值映射成需要的类型

        :param value: str/float/int
        :param before_transform: str
        :param after_transform: str
        :return: str/float/int
            转化后的数据
        """
        if before_transform == after_transform:
            return value

        if before_transform == 'percentile':
            value = value[:-1]
            return float(value) / 100 if self.__is_float(value) else np.nan

        if after_transform == 'float':
            return float(value) if self.__is_float(value) else np.nan
        elif after_transform == 'int':
            return int(value) if self.__is_int(value) else np.nan
        else:
            return str(value)

    def type_mapping(self, value, key_str, interface_index):
        """将数值映射成需要的类型

        :param value: str/float/int
            需要进行转换的数据
        :param key_str: str
            数据在外部数据中对应的字段代码
        :param interface_index: str
            数据在外部数据中所属接口的代码名称
        :return: str/float/int
            转化后的数据,具体类型视配置文件中的信息而定
        """
        if value is np.nan:
            return value

        before_transform = self.BASE_INFO_[interface_index][key_str]['原始数据类型']
        after_transform = self.BASE_INFO_[interface_index][key_str]['转化数据类型']
        return self.type_mapping_helper(value, before_transform, after_transform)

    def __content_fetch_final(self, data, fea_name, interface_index, acc_keys):
        """处理最内层的json

        :param data: list/dict
            最内层json,该json可能包含在一个list当中
        :param fea_name: str
            所需数据字段的代码名
        :param interface_index: str
            所需数据在外部数据中的接口代码名称
        :param acc_keys: list
            需要在最终生成的字段名称中出现的字符的集合
        :return: dict
            处理完的json:{} 或 {最终字段代码:数据}
        """
        processed_fea_name = self.BASE_INFO_[interface_index][fea_name]['加工后字段代码']
        if type(data) == dict:
            k, v = ('_'.join(acc_keys + [processed_fea_name]), data.get(fea_name, np.nan))
            if v is np.nan:
                return {k: v}
            return {k: self.value_mapping(self.type_mapping(v, fea_name, interface_index), fea_name, interface_index)}
        elif type(data) == list:
            data_copy = copy.deepcopy(data)
            for r in data_copy:
                if fea_name in r.keys():
                    r.update({fea_name: self.value_mapping(self.type_mapping(r.get(fea_name, np.nan), fea_name, interface_index),
                                                           fea_name,
                                                           interface_index)})
            data_copy = pd.DataFrame(data_copy)
            process_method = self.BASE_INFO_[interface_index][fea_name]['数组处理方式']
            if fea_name in data_copy.columns:
                return self.__process_array__(data_copy, process_method, fea_name, processed_fea_name, acc_keys)
            else:
                return {}
        return {}

    def __content_fetch_helper(self, data, fea_name, interface_index, content_loc=[], acc_keys=[]):
        """协助content_fetch提取数据

        :param data: json/list
            经过data_fetch后的数据
        :param fea_name: str
            需要提取的最终的字段代码名
        :param interface_index: str
            fea_name所在外部数据接口的代码名
        :param content_loc: list
            字段内容性键的名称集合,详见配置文件
        :param acc_keys: list
            累计处理过的内容性键的名称集合
        :return: dict
            提取完成的字典
        """
        if len(data) == 0:
            return {}

        if len(content_loc) == 0:
            return self.__content_fetch_final(data, fea_name, interface_index, acc_keys)

        if type(data) == list:
            return self.__content_fetch_final([self.__content_fetch_helper(row, fea_name, content_loc, []) for row in data],
                                            fea_name,
                                            interface_index,
                                            acc_keys)
        elif type(data) == dict:
            data = data.get(content_loc[0], {})
            return self.__content_fetch_helper(data,
                                               fea_name,
                                               interface_index,
                                               content_loc[1:],
                                               acc_keys + [content_loc[0]])
        else:
            return {}

    def content_fetch(self, data, fea_name, interface_index):
        """提取外部数据接口interface_index中fea_name字段中的数据

        :param data: list/dict
            经过data_fetch后的数据
        :param fea_name: str
            需要提取的最终的字段代码名
        :param interface_index: str
            fea_name所在外部数据接口的代码名
        :return: dict
            完成数据提取的字典:{fea_name: 数据}
        """
        content_loc = self.CONT_KEYS_[interface_index][fea_name]
        if len(data) == 0:
            return {}
        return self.__content_fetch_helper(data, fea_name, interface_index, content_loc)

    def bank_data_deep_fetch(self,
                             data_str,
                             col_str,
                             is_mtc = False,
                             transform_info=('str', 'float'),
                             is_time_cols=False):
        """银行数据深度提取

        :param data_str:
        :param is_mtc:
        :param transform_info:
        :param is_time_cols:
        :return:
        """
        splited_data_str = [tuple(s.split('_')) for s in data_str.split(';')]
        rec_json = {(self.value_mapping(k, 'MTC', 'PLCS1116YH001') if is_mtc else k):
                        self.type_mapping_helper(v, transform_info[0], transform_info[1]) if v != 'NA' else np.nan
                    for k, v in splited_data_str}

        if is_time_cols:
            rec_json = {int(k): v for k, v in rec_json.items()}
            rec_keys = np.sort(list(rec_json.keys()))
            time_codes = ['last_%s' % i for i in range(12, 0, -1)]
            key_map = {k: v for k, v in zip(rec_keys, time_codes)}
            rec_json = {(col_str + '_' + key_map[k]): rec_json[k] for k in key_map.keys()}
        return rec_json

    def special_post_process(self, processed_dict):
        """在完成后对银联个人数据报告接口中的字段做单独处理。
        该函数不具有普适性,希望以后的更新能够使用更好的机制将该函数替换。

        :param processed_dict: dict
            处理过后的数据字典
        :return: dict
            经过后续处理的字典
        """
        yh_keys_process_needed = ['transCnt', 'transAmt', 'cashCnt', 'cashAmt',
                                  'transCntPre', 'transAmtPre', 'cntRank', 'amtRank']
        processed_info = {'transCnt': (False, ('str', 'float'), True),
                          'transAmt': (False, ('str', 'float'), True),
                          'cashCnt': (False, ('str', 'float'), True),
                          'cashAmt': (False, ('str', 'float'), True),
                          'transCntPre': (True, ('percentile', 'float'), False),
                          'transAmtPre': (True, ('percentile', 'float'), False),
                          'cntRank': (False, ('percentile', 'float'), True),
                          'amtRank': (False, ('percentile', 'float'), True)}

        for kk in yh_keys_process_needed:
            if kk in processed_dict.keys():
                vv = processed_dict[kk]
                if (vv is not None) and (vv != 'NA') and (vv is not np.nan):
                    is_mtc = processed_info[kk][0]
                    transform_info = processed_info[kk][1]
                    is_time_cols = processed_info[kk][2]
                    processed_dict.update(self.bank_data_deep_fetch(vv, kk, is_mtc, transform_info, is_time_cols))
                processed_dict.pop(kk)
        return processed_dict

    def individual_data_process(self, data):
        """提取个体信息数据

        :param data: dict
            单个客户各项信息数据
        :return: dict
            解析后的数据
        """
        indivi_result = {}

        for inter_id in self.BASE_INFO_.keys():
            inter_info = self.BASE_INFO_[inter_id]
            for fea_n in inter_info.keys():
                data_tmp = self.data_fetch(data, inter_id, fea_n)
                indivi_result.update(self.content_fetch(data_tmp, fea_n, inter_id))
        result = self.special_post_process(indivi_result)

        # result['idCard'] = data['idCard']
        # for json_value in list(data.values()):
        #     if type(json_value)==dict and json_value['resDate'] is not None:
        #         result['sendTime'] = json_value['resDate']
        return result

    @staticmethod
    def __process_array__(df, process_method, fea_name, processed_fea_name, acc_keys):
        """处理json中的数组

        各项参数信息略
        :return: dict
            处理完的数组
        """
        k_series = df[fea_name]
        if process_method == 'categorical':
            k_series = k_series.value_counts()
            return {'_'.join(acc_keys + [processed_fea_name, i]): k_series[i] for i in k_series.index}
        elif process_method == 'mean':
            return {'_'.join(acc_keys + [processed_fea_name, 'mean']): k_series.mean()}
        elif process_method == 'sum':
            return {'_'.join(acc_keys + [processed_fea_name, 'sum']): k_series.sum()}
        print("%s未制定处理方式" % processed_fea_name)
        return {'_'.join(acc_keys + [processed_fea_name]): np.nan}

    @staticmethod
    def __base_conf_info__(info_df):
        """ 配置文件基础信息部分解析

        """
        info = {}
        info_cols = list(info_df.columns)
        info_cols.remove('外部接口代码')
        info_cols.remove('外部字段代码')
        interfaces = info_df['外部接口代码'].unique()

        for inter in interfaces:
            info[inter] = {}
            info_tmp = info_df[info_df['外部接口代码'] == inter]
            for c in info_tmp['外部字段代码'].unique():
                info[inter].update({c: json.loads(info_tmp[info_tmp['外部字段代码'] == c][info_cols].to_json(orient='records'))[0]})
        return info

    @staticmethod
    def __keys_info_parsing__(keys_df):
        """配置文件json结构信息解析

        """
        result = {}

        for inter_id in keys_df['外部接口代码'].unique():
            result[inter_id] = {}
            inter_df = keys_df[keys_df['外部接口代码'] == inter_id]
            for col in inter_df['外部字段代码'].unique():
                keys = inter_df[inter_df['外部字段代码'] == col].iloc[0, 2]
                result[inter_id].update({col: keys.split(',') if type(keys) == str else []})
        return result

        for col in keys_df.columns:
            keys = keys_df[col].dropna().tolist()
            inter_id, fea_name = col.split(',')
            result[inter_id] = {fea_name: keys}
        return result

    @staticmethod
    def __codes_name_mapping__(codes_df):
        """配置文件码表解析

        """
        maps = {}
        codes_df['码值'] = codes_df['码值'].astype(str)
        interfaces = codes_df['外部接口代码'].unique()

        for inter in interfaces:
            maps[inter] = {}
            maps_tmp = codes_df[codes_df['外部接口代码'] == inter]
            codes_tmp = maps_tmp['外部字段代码'].unique()
            for c in codes_tmp:
                maps[inter].update(
                    {c: maps_tmp[maps_tmp['外部字段代码'] == c][['码值', '含义']].set_index(keys=['码值']).to_dict()['含义']})
        return maps

    @staticmethod
    def __is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False
    @staticmethod
    def __is_int(value):
        try:
            int(value)
            return True
        except ValueError:
            return False

edp = ExternalDataParsing(os.path.join(conf_file_dir, 'external_data_configuration_info.xlsx'))