import torch
import torch.nn.functional as F


score = torch.tensor([0.0, 1.0, 2.0])
weights = F.softmax(score, dim=-1)
print(weights)      # tensor([0.0900, 0.2447, 0.6652])