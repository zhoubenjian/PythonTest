import math
import random


# 百鸡百钱问题
# 小鸡：三只1元；母鸡：3元一只；公鸡：5元一只     100元买一百只鸡(小鸡、母鸡、公鸡至少各买1只)，有多少购买种方案？
# for i in range(0, 101, 3):
#     for j in range(0, 34):
#         for k in range(0, 21):
#             if i + j + k == 100 and (i / 3) + (j * 3) + (k * 5) == 100:
#                 print(f'小鸡{i}只，母鸡{j}只，公鸡{k}只')


# # 九九乘法表
# for i in range(1, 10):
#     for j in range(1, i+1):
#         print(f'{j}x{i}={i * j}', end='\t')
#     print()


# print(math.pi)
# print(f'{math.pi:.2f}')         # 四舍五入保留两位小数
# print('{:.3f}'.format(math.pi)) # 四舍五入保留三位小数
# print('%.4f' % math.pi)         # 四舍五入保留四位小数

# print(29 / 10)
# print(29 // 10)



# list = [2, 3, 5, 7, 11, 13, 17, 19]
# for i in range(10):
#     print(random.choices(list), end = ', ')



# 模拟暴击几率
for i in range(0, 20):
    random_num = random.randint(0, 100)
    print(f'暴击几率：{random_num % 34}%')

