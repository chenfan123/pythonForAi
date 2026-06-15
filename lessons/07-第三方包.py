import pandas as pd  # 操作数据
import matplotlib

data = pd.read_csv("lessons/07-data.csv")
print(data)
print(type(data))  # <class 'pandas.core.frame.DataFrame'>
print(data[data["name"] == "张三"])
