import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

"""
异常数据检测
1.GRUBBS TEST
2.Z分数法(Z-score method)
3.稳健z分数(Robust Z-score)
4.四分位距法(IQR METHOD)
5.截尾处理(Winsorization method(Percentile Capping))
6.DBSCAN聚类(DBSCAN Clustering)
7.孤立森林(Isolation Forest)
8.画图
"""


# GRUBBS TEST是一个假设检验方法
def grubbs_test(x):
    n = len(x)
    mean_x = np.mean(x)
    sd_x = np.std(x)
    numerator = max(abs(x - mean_x))
    g_calculated = numerator / sd_x
    print("Grubbs Calculated Value:", g_calculated)
    t_value = stats.t.ppf(1 - 0.05 / (2 * n), n - 2)
    g_critical = ((n - 1) * np.sqrt(np.square(t_value))) / (
        np.sqrt(n) * np.sqrt(n - 2 + np.square(t_value))
    )
    print("Grubbs Critical Value:", g_critical)
    if g_critical > g_calculated:
        print(
            "从Grubbs_test中我们观察到计算值小于临界值,接受零假设,得出结论：不存在异常值"
        )
    else:
        print(
            "从Grubbs_test中我们观察到计算值大于临界值,拒绝零假设,得出结论：存在一个异常值"
        )


# Z分数法（Z-score method）
def Zscore_outlier(x, n=3):
    """
    para:series
    return:list index
    """
    m = np.mean(x)
    sd = np.std(x)
    rule = abs((x - m) / sd) > n
    outliers_index = x[rule].index.to_list()
    return outliers_index


# 稳健z分数（Robust Z-score）
def ZRscore_outlier(x, n=3):
    """
    para:series
    return:list index
    """
    med = np.median(x)
    ma = np.median(stats.median_abs_deviation(x))
    rule = abs(0.6745 * (x - med) / ma) > n
    outliers_index = x[rule].index.to_list()
    return outliers_index


# 四分位距法（IQR METHOD）
def iqr_outliers(x):
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    Lower_tail = q1 - 1.5 * iqr
    Upper_tail = q3 + 1.5 * iqr
    rule = x > Upper_tail | x < Lower_tail
    outliers_index = x[rule].index.to_list()
    return outliers_index


# 截尾处理（Winsorization method(Percentile Capping)）
def winsorization_outliers(x):
    q1 = np.percentile(x, 1)
    q3 = np.percentile(x, 99)
    rule = x > q3 | x < q1
    outliers_index = x[rule].index.to_list()
    return outliers_index


# x = pd.DataFrame([[1, 1], [1, 2], [2, 2], [1, 100000000]])
x = pd.DataFrame([-12222222222,1,2,3,4,5,10000,1])

# DBSCAN聚类（DBSCAN Clustering)
def DB_outliers(x,min_samples=3):
    outlier_detection = DBSCAN(eps=2, metric="euclidean", min_samples=min_samples)
    clusters = outlier_detection.fit_predict(x.values.reshape(-1, 1))
    data = pd.DataFrame()
    data["cluster"] = clusters
    print(data["cluster"].value_counts().sort_values(ascending=False))


# 使用DBSCAN进行异常值检测 TODO
def detect_outliers_with_dbscan(X, eps=4, min_samples=2):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # 异常值标记为-1
    labels[~core_samples_mask] = -1
    return labels

# 随机森林异常值检测 TODO
def Iso_outliers(df):
    iso = IsolationForest( random_state=1, contamination="auto")
    preds = iso.fit_predict(df.values.reshape(-1, 1))
    data = pd.DataFrame()
    data["cluster"] = preds
    print(data["cluster"].value_counts().sort_values(ascending=False))


if __name__=='__main__':
    # print(detect_outliers_with_dbscan(x))
    DB_outliers(x)
    Iso_outliers(x)



import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.graphics.gofplots import qqplot

# 通过图查看异常值
# Box and whisker plot (box plot). 箱线图
# Scatter plot. 散点图
# Histogram. 直方图
# Distribution Plot. 分布图
# QQ plot. Q-Q图

# # Box and whisker plot (box plot). 箱线图
# def Box_plots(df):
#     plt.figure(figsize=(10, 4))
#     plt.title("Box Plot")
#     sns.boxplot(df)
#     plt.show()


# # Scatter plot. 散点图
# def hist_plots(df):
#     plt.figure(figsize=(10, 4))
#     plt.hist(df)
#     plt.title("Histogram Plot")
#     plt.show()


# # Histogram. 直方图
# def scatter_plots(df1, df2):
#     fig, ax = plt.subplots(figsize=(10, 4))
#     ax.scatter(df1, df2)
#     ax.set_xlabel("Age")
#     ax.set_ylabel("Fare")
#     plt.title("Scatter Plot")
#     plt.show()


# # Distribution Plot. 分布图
# def dist_plots(df):
#     plt.figure(figsize=(10, 4))
#     sns.distplot(df)
#     plt.title("Distribution plot")
#     sns.despine()
#     plt.show()


# # QQ plot. Q-Q图
# def qq_plots(df):
#     plt.figure(figsize=(10, 4))
#     qqplot(df, line="s")
#     plt.title("Normal QQPlot")
#     plt.show()
