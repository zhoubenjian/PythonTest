'''
注意力机制：
Attention(Q, K, V) = softmax(QK^T / √d_k)V

Q (Query)：当前需要计算输出的查询项
K (Key)：用于与查询项匹配的键
V (Value)：与键对应的实际值
d_k：键的维度，用于缩放点积结果
'''
import torch
import torch.nn.functional as F


# 简化的自注意力实现示例
def self_attention(query, key, value):
    scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, value)