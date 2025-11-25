import numpy as np
import pandas as pd


# 数据
height = [170, 180, 165, 175, 185]  # 身高
weight = [65, 75, 55, 70, 80]       # 体重
age = [25, 30, 22, 27, 35]          # 年龄

# 身高和体重的协方差
height_weight_matrix = np.cov(height, weight)

# Pandas美化输出
variables = ['Height', 'Age']
hw_cov = pd.DataFrame(height_weight_matrix, index=variables, columns=variables)
print('身高和体重的协方差矩阵（带标签）：')
'''
        Height   Age
Height    62.5  75.0
Age       75.0  92.5
'''
print(hw_cov)

# 身高和年龄的协方差（同理）

# 体重和年龄的协方差（同理）

