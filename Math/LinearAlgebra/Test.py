import numpy as np


# '''
# 3D张量(层数, 行数, 列数)
# '''
# a = np.array([
#     [
#         [1, 2, 3, 4, 5, 6]
#     ]
# ])
# print(f'a形状：{a.shape}，a维度：{a.ndim}')        # a形状：(1, 1, 6)，a维度：3
#
# b = np.array([
#     [
#         [1, 2, 3]
#     ],
#     [
#         [4, 5, 6]
#     ],
#     [
#         [7, 8, 9]
#     ]
# ])
# print(f'b形状：{b.shape}，b维度：{b.ndim}')        # b形状：(3, 1, 3)，b维度：3
#
# c = np.array([
#     [
#         [1, 2, 3],
#         [4, 5, 6]
#     ],
#     [
#         [7, 8, 9],
#         [10, 11, 12]
#     ]
# ])
# print(f'c形状：{c.shape}，c维度：{c.ndim}')        # c形状：(2, 2, 3)，c维度：3
#
# d = np.array([
#     [
#         [1, 2],
#         [3, 4]
#     ],
#     [
#         [5, 6],
#         [7, 8]
#     ],
#     [
#         [9, 10],
#         [11, 12]
#     ]
# ])
# print(f'd形状：{d.shape}，d维度：{d.ndim}')        # d形状：(3, 2, 2)，d维度：3
#
#
# print('\n' + '*' * 50 + '\n')
#
#
# array = np.array([
#     [2, 3, 5],
#     [7, 11, 13],
#     [17, 23, 29]
# ])
# print(array / 3)



# a = np.array([
#     [.1, 1.2, 2.3, 3.4],
#     [4.5, 5.6, 6.7, 7.8],
#     [8.9, 9.0, 10.1, 11.2]
# ])
#
# # 对a的每一行向量的模长（L2范数）
# norm_a = np.linalg.norm(a, axis=1, keepdims=True)
# print('a每行向量的模长：\n', norm_a, sep='')
#
# # 对a进行归一化
# a_normalized = a / norm_a
# print('\na每行向量归一化后：\n', a_normalized, sep='')
#
# # 余弦相似度矩阵
# cos_sim_matrix = a_normalized @ a_normalized.T
# print(f'\n余弦相似度矩阵：\n{np.round(cos_sim_matrix, 3)}')


a = np.diag([2, 3, 5, 7, 9])
'''
[[2 0 0 0 0]
 [0 3 0 0 0]
 [0 0 5 0 0]
 [0 0 0 7 0]
 [0 0 0 0 9]]
'''
print(a)
print(np.diag(a[:2]))    # [2 3]


'''
对称阵的特征值和特征向量
'''
a = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5]
])

eigenvals, eigenvecs = np.linalg.eig(a)
print('特征值λ：', eigenvals, sep='')
print('\n特征向量A：\n', eigenvecs, sep='')

print('-' * 50)

'''
手动校验 Av = λv
注意：np.linalg.eig 返回的特征向量矩阵中，每一列对应一个特征值
'''
for i in range(len(eigenvals)):
    # 提取第 i 个特征值 (标量)
    lambda_val = eigenvals[i]
    # 提取第 i 个特征向量 (向量)
    eigenvec = eigenvecs[:, i]

    # 计算等式左边：矩阵 A 乘以 特征向量 v
    left_side = a @ eigenvec
    # 计算等式右边：特征值 λ 乘以 特征向量 v
    right_side = lambda_val * eigenvec

    # 使用 np.allclose 进行浮点数容差比较
    is_equal = np.allclose(left_side, right_side)
    print(f'特征值λ={lambda_val:.2f}，特征向量v={np.array2string(eigenvec, precision=2, separator=", ")}，是否相等：{is_equal}')





