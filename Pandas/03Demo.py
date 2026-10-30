import pandas as pd


df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "score": [90, 85, 88]
})


# 原始数据
'''
      name  age  score
0    Alice   25     90
1      Bob   30     85
2  Charlie   35     88
'''
print(df)


print('=' * 30)


# 第一行数据
'''
name     Alice
age         25
score       90
Name: 0, dtype: object
'''
print(df.iloc[0])


print('*' * 30)


# 第二列数据
'''
0        25
1        30
2        35
Name: age, dtype: int64
'''
print(df.iloc[:, 1])


print('&' * 30)


# 第一，三列数据
'''
      name  score
0    Alice     90
1      Bob     85
2  Charlie     88
'''
print(df.iloc[:, [0, 2]])


print('$' * 30)


# 第二行，第三列数据
'''
      name  age  score
1      Bob   30     85
2  Charlie   35     88
'''
print(df.iloc[[1, 2]])

