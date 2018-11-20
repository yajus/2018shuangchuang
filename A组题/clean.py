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

data = pd.read_csv('data/predict_data_e_train.csv')
data = data.sample(frac=1) # 打乱顺序
data[data == 0] = np.nan
data.head()
data.isnull().sum()
data = data.dropna(subset=['charge_energy'])
data.head()
data.isnull().sum()
data.to_csv('data/train.csv')