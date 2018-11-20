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
data.isnull().any()
# 计算平均值
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
# 两列相减
data['time'] = data.apply(lambda row: row.charge_end_time - row.charge_start_time, axis=1)
data['soc'] = data.apply(lambda row: row.charge_end_soc - row.charge_start_soc, axis=1)
data['U'] = data.apply(lambda row: row.charge_end_U - row.charge_start_U, axis=1)
data['I'] = data.apply(lambda row: row.charge_end_I - row.charge_start_I, axis=1)
data['temp'] = data.apply(lambda row: row.charge_max_temp - row.charge_min_temp, axis=1)
num = len(data)

train = data[:int(num*0.7)]
test = data[int(num*0.7):]
print(num, len(train), len(test))
selected_features = ['vehicle_id',
                     'time',
                     'mileage',
                     'soc',
                     'U',
                     'I',
                     'temp']

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
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor() # 随机森林回归模型

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 评估
e = 0
for a, r in zip(y_test, y_pred):
    e += ((r- a) / a) ** 2

e ** 0.5
test = pd.read_csv('data/testA.csv')
test.head()
# 找出缺失值的列
test.isnull().any()
# 填充缺失值的列
test['charge_end_soc'].fillna(test['charge_end_soc'].mean(), inplace=True)
# 两列相减
test['time'] = test.apply(lambda row: row.charge_end_time - row.charge_start_time, axis=1)
test['soc'] = test.apply(lambda row: row.charge_end_soc - row.charge_start_soc, axis=1)
test['U'] = test.apply(lambda row: row.charge_end_U - row.charge_start_U, axis=1)
test['I'] = test.apply(lambda row: row.charge_end_I - row.charge_start_I, axis=1)
test['temp'] = test.apply(lambda row: row.charge_max_temp - row.charge_min_temp, axis=1)
x = dict_vec.transform(test[selected_features].to_dict(orient='record'))
y = model.predict(x)
ids = list(test['vehicle_id'])
len(ids)
with open('submit-A_张彬城_暨南大学_15521106350_20181011.csv', 'w') as f:
    f.write('vehicle_id,charge_energy\n')
    for _id, _y in zip(ids, y):
        line = '%d,%d\n' % (_id, _y)
        f.write(line)