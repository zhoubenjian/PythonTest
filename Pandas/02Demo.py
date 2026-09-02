import pandas as pd


# 读取相对路径Excel文件，指定引擎:openpyxl
data = pd.read_excel('./data/934cff6b03644603afc2be1db7acfef2.xlsx', engine='openpyxl')
print(type(data))       # <class 'pandas.core.frame.DataFrame'>

print('行数:', data.shape[0], '，列数:', data.shape[1], sep='')     # 行数:45，列数:16
# 列名
print(list(data.columns[:-1]))      # ['presidentName', 'gender', 'birthday', 'birthPlace', 'deathday', 'locationOfDeath', 'isAlive', 'termOfOffice', 'termStartDate', 'termEndDate', 'type', 'partyId', 'stateId', 'status']