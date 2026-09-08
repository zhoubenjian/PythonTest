'''
Softmax 函数
    1.用于将一个向量转换为一个概率分布
    2.保序性
    3.温度系数 T
        当 T=1：就是标准的 Softmax（你现在掌握的）。
        当 T>1（比如 5、10）：概率分布会变得更平滑（小概率的类别概率会提升，模型输出更“柔和”）。
        当 T<1（比如 0.5）：概率分布会变得更尖锐（最大的概率会接近 1，模型输出更“自信”）。
'''
import numpy as np
import torch
import torch.nn.functional as F


def manual_softmax(score, axis=-1):
    '''
    手动实现Softmax函数
    :param score: 输入向量
    :param axis: 指定轴进行计算，默认最后一个轴
    :param keepdims: 是否保留输入的维度，默认False
    :return: 输出概率分布
    '''
    # 减去最大值防止溢出，keepdims保证广播正确
    shifted = score - np.max(score, axis, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis, keepdims=True)


if __name__ == '__main__':

    # torch封装实现Softmax函数
    score = torch.tensor([0.0, 1.0, 2.0])
    weights = F.softmax(score, dim=-1)
    print(f"torch softmax: {weights}")      # torch softmax: tensor([0.0900, 0.2447, 0.6652])


    print('-' * 55)


    # 手动实现Softmax函数
    weights = manual_softmax(np.array([0.0, 1.0, 2.0]))
    print(f'manual softmax: {np.round(weights, 4)}')  # manual softmax: [0.0900 0.2447 0.6652]


    print("\n" + "-" * 55 + "\n")


    # 矩阵按列（axis=0）进行Softmax计算
    weights = manual_softmax(np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [4.0, 5.0, 6.0]
    ]), 0)
    '''
    [[0.042  0.042  0.042 ]
     [0.1142 0.1142 0.1142]
     [0.8438 0.8438 0.8438]]
    '''
    print(f'matrix manual softmax（axis=0）: \n{np.round(weights, 4)}')

    print('=' * 20)

    # 矩阵按行（axis=1）进行Softmax计算
    weights = manual_softmax(
        np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [4.0, 5.0, 6.0]]), 1
    )
    '''
    [[0.09   0.2447 0.6652]
     [0.09   0.2447 0.6652]
     [0.09   0.2447 0.6652]]
    '''
    print(f"matrix manual softmax（axis=1）: \n{np.round(weights, 4)}")
