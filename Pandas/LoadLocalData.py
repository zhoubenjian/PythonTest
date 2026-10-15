import pandas as pd
from sklearn.datasets import load_files


# 1.加载数据集 打乱样本顺序
local_data = load_files(
        container_path="../local_datas/20_newsgroups",
        encoding='ISO-8859-1',
        shuffle=True,
        random_state=42,
        decode_error='ignore'
)
print(local_data.data[0])


print('=' * 77)


# 2.将数据转换为DataFrame
df = pd.DataFrame({
        'text': local_data.data[0:5],
        'category_id': local_data.target[0:5]
})
print(df)

print('-' * 77)

df['category_name'] = df['category_id'].map(lambda x: local_data.target_names[x])
print(df)

