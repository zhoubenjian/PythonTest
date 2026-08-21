'''
特征值和特征向量(Eigenvalue and Eigenvector)
    特征值：λ
    特征向量：v
    Av = λv

    只有方阵才有特征值和特征向量！！！特征值和特征向量是对应关系的
'''
import numpy as np


A = np.array([
    [2, 0],
    [0, 1]
])

eigvals, eigvecs = np.linalg.eig(A)
print('A特征值:\n', eigvals, sep='')
print('A特征向量:\n', eigvecs, sep='')


print('\n' + '-' * 30 + '\n')


# 手动校验 Av = λv
# 特征矩阵的第i列对应特征值的第i个索引
for i in range(len(eigvals)):
    # 特征值
    eigval = eigvals[i]
    # 特征向量
    eigvec = eigvecs[:, i]

    # 左侧
    left_side = A @ eigvec
    print(f'Av:\n{left_side}')
    # 右侧
    right_side = eigval * eigvec
    print(f'λv:\n{right_side}')

    print(f'第{i + 1}个特征值和特征向量相等：{np.allclose(left_side, right_side)}')

