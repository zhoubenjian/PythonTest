from datetime import datetime

import numpy as np
import pandas as pd



# 对于.xlsx文件，指定引擎:openpyxl
# 对于.xls，指定引擎:xlrd
data = pd.read_excel('./data/934cff6b03644603afc2be1db7acfef2.xlsx', engine='openpyxl') # 相对路径
print(type(data))
print(data)

# 所有总统姓名（所有行，presidentName列）
president_name = data.loc[:, 'presidentName']
print(president_name)
print(type(president_name))     # <class 'pandas.core.series.Series'>

print('====================================')

# 所有总统出生地
president_birthPlace = data.loc[:, 'birthPlace']
print(president_birthPlace)

print('====================================')

# 筛选（1900-01-01后出生的总统）
p_birthday_after_1900s = data[data['birthday'] >= '1900-01-01']['presidentName']
print(p_birthday_after_1900s)
print('====================================')
p_birthday_after_1900s = data.loc[data['birthday'] > '1900-01-01', 'presidentName']
print(p_birthday_after_1900s)

print('=====================================')

# data => numpy.ndarray
data_array = np.array(data)
print(type(data_array))
print(data_array)

print('=====================================')

# # 计算年龄（当前年份 - 出生年份）
data['age'] = datetime.now().year - pd.to_datetime(data['birthday']).dt.year
data_new = data[['presidentName', 'birthday', 'age']]
print(data_new)
# 保存为新文件
data_new.to_csv('./data/data_new.csv')
