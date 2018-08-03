import pandas as pd
import numpy as np


def convert_series_data_2_supervised(data, in_node, out_node, dropna=True):
    """
    将输入的序列数据通过pandas转换为有监督格式的数据.
    参数列表:
    data: 待转换数据,可以是list表示单变量(或者单特征)序列数据,
        也可以是numpy数组,表示多变量(多特征)数据
    in_node: 希望转换的输入序列长度(比如希望t-预测t,那么取1,
        希望t-2,t-1,预测t那么取2,..etc.)
    out_node: 希望预测的序列长度(为1,表示预测t,为2表示预测t,t+1,...etc)
    dropna: 是否丢弃na数据,因为使用pandas的shift方法来进行序列移动,
        显然会导致初始几行或者末尾几行出现na数据,默认丢弃
    """
    # 特征或者变量的数量
    num_ftrs = 1 if isinstance(data, (list)) else data.shape[1]
    # 首先将输入数据转换为pandas的DataFrame
    df = pd.DataFrame(data)
    # 准备收集不同的时间序列数据以及相应的名称
    series_cols = []
    series_var_names = []

    for i in range(in_node, 0, -1):
        series_cols.append(df.shift(i))
        series_var_names.extend(["ftr{}[t-{}]".format(j, i)
                                 for j in range(num_ftrs)])

    for i in range(0, out_node):
        series_cols.append(df.shift(-i))
        if i == 0:
            series_var_names.extend(["ftr{}[t]".format(j)
                                     for j in range(num_ftrs)])
        else:
            series_var_names.extend(["ftr{}[t+{}]".format(j, i)
                                     for j in range(num_ftrs)])

    series_all = pd.concat(series_cols, axis=1)
    series_all.columns = series_var_names
    if dropna:
        series_all.dropna(inplace=True)
    return series_all


if __name__ == "__main__":
    # dumy_data = [i for i in range(-5, 5)]
    # dumy_series = convert_series_data_2_supervised(dumy_data, 1, 1)
    # print(dumy_series)
    data=pd.DataFrame({"x":[i for i in range(-5,5)],"y":[i for i in range(5,15)]})

    data_series=convert_series_data_2_supervised(data.values,1,2)
    print(data_series)
