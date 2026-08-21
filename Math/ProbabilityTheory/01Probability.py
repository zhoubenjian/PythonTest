'''
概率论
'''
import numpy as np

'''
大数定律
'''
# 验证投硬币实验
for i in [10, 100, 1000, 10000]:
    tosses = np.random.choice(['heads', 'tails'], i)
    freq = np.mean(tosses == 'heads')
    print(f'样本数: {i:4d}, heads的概率: {freq:.4f}')
    print('-' * 50)


'''
蒙提霍尔三门问题
换门胜率2/3
'''
def monty_hall(switch=True, n_simulations=10000):
    win = 0
    for _ in range(n_simulations):
        # 车所在的门
        car = np.random.randint(0, 3)
        # 玩家选择的门
        choice = np.random.randint(0, 3)
        # 主持人打开的门（不是选择的门也不是车的门）
        revealed = np.random.choice([d for d in range(3) if d != choice and d != car])

        if switch:
            # 玩家选择剩余的门（既不是最初选择的门也不是主持人打开的门）
            choice = [d for d in range(3) if d != choice and d != revealed][0]

        # 统计玩家获胜次数
        win += int(choice == car)

    return win / n_simulations

print(f'\n蒙提霍尔三门问题：换门胜率{monty_hall(switch=True):.4f}，不换门胜率{monty_hall(switch=False):.4f}')
