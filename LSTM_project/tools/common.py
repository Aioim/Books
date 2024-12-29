import matplotlib.pyplot as plt
import warnings
import pandas as pd

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号


warnings.filterwarnings("ignore")

from sqlalchemy import create_engine

# 建立数据库连接
# "mysql+pymysql://{用户名}:{密码}@{域名}:{端口号}/{数据库名}"
engine = create_engine("mysql+pymysql://root:123456@localhost:3306/data")
# 定义SQL查询
sql_query = "select * from sc"
con = engine.connect()
# 执行查询操作:把sql查询结果读取为dataframe
df = pd.read_sql(sql_query, con)
df.to_sql("table_name", con, if_exists="append", index=False)
"""
if_exists:如果表存在怎么办？
fail:抛出ValueError异常
replace:在插入数据之前删除表。注意不是仅删除数据，是删除原来的表，重新建表哦。
append:插入新数据。
"""


def namestr(obj, namespace=None):
    if not namespace:
        namespace = globals()
    names = [name for name in namespace if namespace[name] is obj]
    return names


if __name__ == "__main__":
    name_space = 1
    print(namestr(name_space))
