'''
给定一组数字，把任意实数，转换成 0~1 之间、总和 = 1 的概率分布。
数值越大，输出概率越高
数值越小，输出概率越低
完美用来做「权重分配」—— 这就是注意力要用它的原因。

Softmax函数:
1.单调递增：
    原数越大，输出越大，保留大小关系；

2.指数拉大差距：
    轻微的分数差，强关联更强、弱关联更弱，注意力更集中；
'''
import torch
import torch.nn.functional as F


score = torch.tensor([1.414, 1.732, 2.236, 3.141])
weight = F.softmax(score, dim=-1)
print(weight)

sum = 0.0
for i in weight:
    sum += i.item()
print('%.2f' % sum)






