import pandas as pd


df_oof = pd.DataFrame({
    "id": [1, 2, 3, 4],
    "pred": [0, 1, 1, 0],
    "correct": [True, False, True, False]
})


# 过滤错误样本
'''
   id  pred  correct
1   2     1    False
3   4     0    False
'''
print(df_oof[df_oof["correct"] == False])


# df_oof[df_oof["correct"] == False] => df_oof[~df_oof['correct']]


'''
   id  pred  correct
1   2     1    False
3   4     0    False
'''
print(df_oof[~df_oof['correct']])
