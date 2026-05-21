import torch
import torch.nn as nn
import math


'''
Transformer 位置编码（论文原版）
'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个足够长的位置矩阵 [max_len, d_model]
        position = torch.arange(max_len).unsqueeze(1)

        # 公式中的分母项：10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # 位置编码矩阵 PE
        pe = torch.zeros(max_len, d_model)

        # 偶数维度用 sin
        pe[:, 0::2] = torch.sin(position * div_term)

        # 奇数维度用 cos
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加 batch 维度 [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # 注册为缓冲区（不参与训练，但会随模型保存）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: 词嵌入向量 [batch_size, seq_len, d_model]
        """
        # 取出对应序列长度的位置编码，直接加到词嵌入上
        x = x + self.pe[:, :x.size(1)]

        # 过 dropout（可选）
        return self.dropout(x)


# 1. 超参数
d_model = 128    # 词向量维度
batch_size = 2
seq_len = 10

# 2. 模拟词嵌入
x = torch.randn(batch_size, seq_len, d_model)

# 3. 初始化位置编码
pos_encoder = PositionalEncoding(d_model=d_model)

# 4. 前向传播（自动加上位置信息）
x_with_pos = pos_encoder(x)

print("输入 shape:", x.shape)
print("输出 shape:", x_with_pos.shape)