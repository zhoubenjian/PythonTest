import torch
import torch.nn.functional as F


result = torch.arange(10)
print(result)

print(result.unsqueeze(1))


print('\n' + '*' * 20 + '\n')


a = torch.tensor([
    [1, 2],
    [3, 4]
])

b = a.T

'''
tensor([[1, 2],
        [3, 4]])
'''
print(f'a = {a}')
'''
tensor([[1, 3],
        [2, 4]])
'''
print(f'b = {b}')
'''
tensor([[ 5, 11],
        [11, 25]])
'''
print(f'a @ b = {a @ b}')

c = a @ b