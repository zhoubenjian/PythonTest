import torch
import torch.nn.functional as F



score = torch.tensor([0.0, 1.0, 2.0])
weights = F.softmax(score, dim=-1)
print(weights)


# import torch
# import torch.nn.functional as F
#
# score = torch.tensor([2.0, 1.0, 0.0])
# weight = F.softmax(score, dim=-1)
# print(weight)