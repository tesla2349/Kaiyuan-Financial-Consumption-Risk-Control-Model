import pandas as pd
import numpy as np

def GetTopImp(X,coefs,top=5,normalize=(0,5)):
    '''
    获取前top个最大减分项，并归一化到normalize区间内。目前的方法是将变量取值与变量系数相乘，然后获取得分为负的，abs后归一化，截取前top个。
    注意：变量取值最好是标准化后的，变量系数最好也是变量标准化后回归模型的系数，否则可能出现没有负分的情况。
         如果没有负分项，目前的处理方式是返回空列表；如果负分项不足top个，则返回的列表也会不足top项。
    :param X: series，其index为变量名，值为变量取值，代表一个客户。
    :param coefs: 列表或一维数组或series，表示各变量对应的系数，
                  若为列表或一维数组，顺序应与X的顺序一致；若为series，其index应为变量名。
    :param top: int，获取前top个最大减分项。
    :param normalize: 二元tuple，表示归一化区间上下限。
    :return: 列表，每个元素为二元tuple（分别为变量名和归一化取值），且按顺序从高到低排列（减分作用）。
    '''
    index=X.index.tolist()
    if isinstance(coefs,pd.Series):
        coefs=coefs.reindex(index=index)
    else:
        coefs=pd.Series(coefs)
        coefs.index=index
    result=X*coefs
    result=result.loc[result<0]
    if result.shape[0]==0:
        return []
    result=result.abs().sort_values(ascending=False)
    r_min=result.min()
    r_max=result.max()
    if r_min==r_max:
        return [(xx[0],normalize[1]) for xx in result.head(top).items()]
    result=(result-r_min)/float(r_max-r_min)
    result=normalize[0]+result*(normalize[1]-normalize[0])
    return list(result.head(top).items())

def GetRadar(X,coefs_radar,maps_from,maps_to,score_min=None,score_max=None,normalize=(0,5)):
    '''
    雷达图数据计算。
    根据线性模型计算出每个维度的得分，再根据对应的分割点获取每个维度的得分映射，并归一化到normalize区间内。
    :param X: series，其index为变量名，值为变量取值，代表一个客户。
    :param coefs_radar: dataframe，其index为变量名，columns为['coefs','radar']，分别表示变量系数及其所属维度。
    :param maps_from: dict of list，键为雷达图维度编号（或名称），值为原始得分分割点list，升序排列。
    :param maps_to: dict of list，键为雷达图维度编号（或名称），值为转化得分分割点list，与maps_from对应，须为0到100之间，升序排列。
    :param score_min: int，原始得分空间最小值，用于处理端点情况，None则使用指数衰减。
    :param score_max: int，原始得分空间最大值，用于处理端点情况，None则使用指数衰减。
    :param normalize: 二元tuple，表示归一化区间上下限。
    :return: 字典，键为雷达图维度编号（或名称），值为维度得分。
    '''
    coefs_radar['score']=X*coefs_radar['coefs']
    scores=coefs_radar.groupby('radar')['score'].sum().to_dict()
    results={key:map_util(scores[key],maps_from[key],maps_to[key],score_min=score_min,score_max=score_max,normalize=normalize) for key in scores}
    return results


def map_util(score,maps_from,maps_to,score_min=None,score_max=None,normalize=None):
    '''
    根据原始得分分割点（maps_from）和转化得分分割点（maps_to）的映射关系转化得分（score）。
    作用：可以用于辅助概率分转化为信用得分、人群占比排名计算。

    例子：
    maps_from=[8,12,45,67,157]  #原始得分分割点
    maps_to=[5,10,20,50,100]    #转化得分分割点，每一项与maps_from对应，如原始得分8对应的转化得分为5，原始得分157对应的转化得分为100
    实际操作是将score从maps_from空间映射到maps_to空间，中间取值采取线性插值的方式。
    目前需要限制maps_to空间为0到100，方便处理端点取值的情况，然后根据normalize映射到对应区间。

    :param score: int，原始得分（或概率值或字段取值均可）。
    :param maps_from: list，原始得分分割点，升序排列。
    :param maps_to: list，转化得分分割点，与maps_from对应，须为0到100之间，升序排列。
    :param score_min: int，原始得分空间最小值，用于处理端点情况，None则使用指数衰减。
    :param score_max: int，原始得分空间最大值，用于处理端点情况，None则使用指数衰减。
    :param normalize: 二元tuple，表示归一化区间上下限，默认为(0,100)。
    :return: int，转化得分。
    '''
    if normalize is None:
        normalize=(0,100)
    low=np.searchsorted(maps_from,score)
    diff_max=np.diff(maps_from).max()
    if low==0:
        if maps_to[0]==0:
            result=0
        elif (score_min is None) or (score_min>=maps_from[0]):
            result=maps_to[0]*np.exp((score-maps_from[0])/float(diff_max))
        elif score<score_min:
            result=0
        else:
            result=(score-score_min)*maps_to[0]/float(maps_from[0]-score_min)
    elif low==len(maps_from):
        if maps_to[-1]==100:
            result=100
        elif (score_max is None) or (score_max<=maps_from[-1]):
            result = 100-(100-maps_to[-1]) * np.exp((maps_from[0]-score) / float(diff_max))
        elif score>score_max:
            result=100
        else:
            result=maps_to[-1]+(score-maps_from[-1])*(100-maps_to[-1])/float(score_max-maps_from[-1])
    else:
        result=maps_to[low-1]+(score-maps_from[low-1])*(maps_to[low]-maps_to[low-1])/float(maps_from[low]-maps_from[low-1])
    return normalize[0]+(normalize[1]-normalize[0])*result/float(100)



def test():
    np.random.seed(13)
    X=pd.Series(np.random.randn(20))
    X.index=['x%d'%i for i in range(X.shape[0])]

    coefs_radar=pd.DataFrame()
    coefs_radar['coefs']=2*np.random.random(X.shape[0])-1
    coefs_radar['radar']=np.random.choice(['A','B','C'],X.shape[0])
    coefs_radar.index=X.index


    # 提取减分最多的几项，并归一化
    GetTopImp(X,coefs_radar['coefs'],top=5,normalize=(0,5))

    # map_util测试（通过设置参数可直接进行分数映射、人群占比计算）
    map_util(score=34, maps_from=[8, 12, 45, 67, 157], maps_to=[5, 10, 20, 50, 100], score_min=None, score_max=None)
    map_util(score=67, maps_from=[8, 12, 45, 67, 157], maps_to=[5, 10, 20, 50, 100], score_min=None, score_max=None)
    map_util(score=157, maps_from=[8, 12, 45, 67, 157], maps_to=[5, 10, 20, 50, 100], score_min=None, score_max=None)
    map_util(score=160, maps_from=[8, 12, 45, 67, 157], maps_to=[5, 10, 20, 50, 100], score_min=None, score_max=200)
    map_util(score=-100, maps_from=[8, 12, 45, 67, 157], maps_to=[5, 10, 20, 50, 100], score_min=None, score_max=None)
    map_util(score=-100, maps_from=[8, 12, 45, 67, 157], maps_to=[5, 10, 20, 50, 100], score_min=0, score_max=None)

    # # 提取雷达图维度得分
    # GetRadar(X, coefs_radar, maps_from={'A':[-2,-1,0,1,2],'B':[-1,0,1],'C':[-1,1]},
    #          maps_to, score_min=None, score_max=None, normalize=(0, 5))