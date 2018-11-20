import pandas as pd
import numpy as np

"""
vehicle_id STRING 车辆唯一标志码
charge_start_time INT 充电开始时间
charge_end_time INT 充电结束时间
mileage FLOAT 充电开始时刻车辆仪表里程（km）
charge_start_soc INT 充电开始时刻动力电池 SOC
charge_end_soc INT 充电结束时刻动力电池 SOC
charge_start_U FLOAT 充电开始时刻动力电池总电压（V）
charge_end_U FLOAT 充电结束时刻动力电池总电压（V）
charge_start_I FLOAT 充电开始时刻动力电池总电流（A）
charge_end_I FLOAT 充电结束时刻动力电池总电流（A）
charge_max_temp FLOAT 充电过程中电池系统温度探针最大值（℃）
charge_min_temp FLOAT 充电过程中电池系统温度探针最小值（℃）
charge_energy FLOAT 此充电过程的充电能量（kWh）
"""

data = pd.read_csv('data/train.csv')
data.head()
# 找出缺失值的列
data.isnull().sum()
# 计算中位数
charge_start_soc_mean = data.groupby(['vehicle_id']).charge_start_soc.mean()
charge_end_soc_mean = data.groupby(['vehicle_id']).charge_end_soc.mean()
charge_end_U_mean = data.groupby(['vehicle_id']).charge_end_U.mean()
charge_end_I_mean = data.groupby(['vehicle_id']).charge_end_I.mean()
charge_max_temp_mean = data.groupby(['vehicle_id']).charge_max_temp.mean()
charge_min_temp_mean = data.groupby(['vehicle_id']).charge_min_temp.mean()

# 设置索引
data.set_index(['vehicle_id'], inplace=True)

# 填充缺失值
data.charge_start_soc.fillna(charge_start_soc_mean, inplace=True)
data.charge_end_soc.fillna(charge_end_soc_mean, inplace=True)
data.charge_end_U.fillna(charge_end_U_mean, inplace=True)
data.charge_end_I.fillna(charge_end_I_mean, inplace=True)
data.charge_max_temp.fillna(charge_max_temp_mean, inplace=True)
data.charge_min_temp.fillna(charge_min_temp_mean, inplace=True)

# 重置索引
data.reset_index(inplace=True)
num = len(data)

train = data[:int(num*0.7)]
test = data[int(num*0.7):]
print(num, len(train), len(test))
selected_features = ['vehicle_id',
                     'charge_start_time', 'charge_end_time',
                     'mileage',
                     'charge_start_soc', 'charge_end_soc',
                     'charge_start_U','charge_end_U',
                     'charge_start_I', 'charge_end_I',
                     'charge_max_temp', 'charge_min_temp']

X_train = train[selected_features]
X_test = test[selected_features]
y_train = train['charge_energy']
y_test = test['charge_energy']
# 特征向量化
from sklearn.feature_extraction import DictVectorizer
dict_vec = DictVectorizer(sparse=False)

X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
X_test = dict_vec.transform(X_test.to_dict(orient='record'))
# 预测
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor() # 极端随机森林回归模型

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 评估
e = 0
for a, r in zip(y_test, y_pred):
    e += ((r- a) / a) ** 2

e ** 0.5
test = pd.read_csv('data/testA.csv')
test.head()
