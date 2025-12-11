'''
模拟CNN卷积操作
'''
import numpy as np
from scipy.signal import convolve2d


# 输入图片
input_image = np.random.randint(1, 10, 25).reshape(5, 5)
'''
[[9 4 3 3 5]
 [2 3 8 4 2]
 [5 5 4 6 8]
 [3 4 8 9 5]
 [5 1 5 9 6]]
'''
print(input_image)

# 卷积核
kernel = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 0]
])

# 卷积操作
output = convolve2d(input_image, kernel, mode="valid")
print('\n卷积结果：')
'''
[[15 22 21]
 [24 21 21]
 [18 24 28]]
'''
print(output)









