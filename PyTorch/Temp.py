import torch


x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
print(f'x原始形状: {x.shape}')      # x原始形状:torch.Size([2, 3])
'''
tensor([[1, 2, 3],
        [4, 5, 6]])
'''
print(x)

print('-' * 50)

# 在dim=0位置插入新维度 (最外层加括号)
y1 = torch.unsqueeze(x, 0)
print(f"unsqueeze(x, 0)，形状: {y1.shape}")        # unsqueeze(x, 0)，形状: torch.Size([1, 2, 3])
'''
tensor([[[1, 2, 3],
         [4, 5, 6]]])
'''
print(y1)

print('*' * 60)

# 在dim=1位置插入新维度 (行之间加括号)
y2 = torch.unsqueeze(x, 1)
print(f"unsqueeze(x, 1)，形状: {y2.shape}")        # unsqueeze(x, 1)，形状: torch.Size([2, 1, 3])
'''
tensor([[[1, 2, 3]],

        [[4, 5, 6]]])
'''
print(y2)

print('*' * 60)

# 在dim=2位置插入新维度 (每个元素加括号)
y3 = torch.unsqueeze(x, 2)
print(f"unsqueeze(x, 2)，形状: {y3.shape}")        # unsqueeze(x, 2)，形状: torch.Size([2, 3, 1])
'''
tensor([[[1],
         [2],
         [3]],

        [[4],
         [5],
         [6]]])
'''
print(y3)
