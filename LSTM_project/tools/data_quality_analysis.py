import pandas as pd
import numpy as np


"""
检查是否存在重复特征
缺失值检查
train、test特征一致性检查
特征数据类型一致性检查
"""


# 获取参数的名称
def namestr(obj, namespace=None):
    if not namespace:
        namespace = globals()
    names = [name for name in namespace if namespace[name] is obj]
    return names[0]


# 检查是否包含重复特征
def repetitive_features(data):
    return len(data.columns) == len(set(data.columns))


# train、test特征是否一致
def consistency_features(train, test):
    return set(train.columns) == set(test.columns)


# train、test检查分类是否一致
def consistency_classification_feature(train, test, col):
    return set(train[col].unique()) == set(test[col].unique())


# train、test分类特征分类一致性
def consistency_classification_features(train, test, cols):
    not_consistency_features = []
    for col in cols:
        if not consistency_classification_feature(train, test, col):
            not_consistency_features.append(col)
    return not_consistency_features


# 不一致的特征，检查哪些分类不是在所有数据都存在
def get_anomaly_classification(train, test, col):
    anomaly_classification = []
    train_classifications = train[col].unique()
    test_classifications = test[col].unique()
    all_classifications = set(train_classifications + test_classifications)
    anomaly_classification.extend(all_classifications - set(train_classifications))
    anomaly_classification.extend(all_classifications - set(test_classifications))
    return anomaly_classification


# 获取数据基础信息
def data_info(data):
    str_name = namestr(data)
    data_inf = (
        data.nunique()
        .to_frame()
        .rename(columns={0: "值数量"})
        .assign(值类型=data.dtypes)
        .assign(空值=data.isnull().sum())
        .assign(空值比例=lambda x: x.空值 / data.shape[0])
        .assign(
            空值比例=lambda x: x.空值比例.apply(lambda y: "{:.2f}%".format(100 * y))
        )
    )
    data_inf.rename(columns=lambda x: "_".join([str_name, x]), inplace=True)
    return data_inf


# train、test数据info
def all_data_info(train, test):
    """_summary_

    Args:
        train (_type_): DataFrame
        test (_type_): DataFrame

    Returns:
        _type_: _description_
    """
    all_data_inf =(
        pd.concat([data_info(train), data_info(test)], axis=1)
        .assign(type一致性=lambda x: x.iloc[:, 1] == x.iloc[:, 5])
    )

    return all_data_inf


if __name__ == "__main__":
    df1 = pd.read_csv(
        r"C:\Users\AIO\Desktop\Walmart.csv", parse_dates=["Date"], index_col="Date"
    )
    df2 = pd.read_csv(
        r"C:\Users\AIO\Desktop\Walmart.csv", parse_dates=["Date"], index_col="Date"
    )

    print(df1["Weekly_Sales"].unique())

    print(all_data_info(df1, df2))
