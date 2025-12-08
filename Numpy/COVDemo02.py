import numpy as np
import pandas as pd


# 数据
height = [170, 180, 165, 175, 185]  # 身高
weight = [65, 75, 55, 70, 80]       # 体重
age = [25, 30, 22, 27, 35]          # 年龄

# 身高和体重的协方差
height_weight_matrix = np.cov(height, weight)

# Pandas美化输出
hw_variables = ['Height', 'Weight']
hw_cov = pd.DataFrame(height_weight_matrix, index=hw_variables, columns=hw_variables)
print('身高和体重的协方差矩阵（带标签）：')
'''
        Height  Weight
Height    62.5    75.0
Weight    75.0    92.5
'''
print(hw_cov)

print('=' * 30)

# 身高和年龄的协方差（同理）
height_age_matrix = np.cov(height, age)
ha_variables = ['Height', 'Age']
ha_cov = pd.DataFrame(height_age_matrix, index=ha_variables, columns=ha_variables)
print('身高和年龄的协方差（带标签）：')
'''
        Height    Age
Height   62.50  38.75
Age      38.75  24.70
'''
print(ha_cov)

print('=' * 30)

# 体重和年龄的协方差（同理）
weight_age_matrix = np.cov(weight, age)
wa_variables = ['Weight', 'Age']
wa_cov = pd.DataFrame(weight_age_matrix, index=wa_variables, columns=wa_variables)
print('体重和年龄的协方差（带标签）：')
'''
        Weight   Age
Weight    92.5  46.0
Age       46.0  24.7
'''
print(wa_cov)
