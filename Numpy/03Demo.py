import numpy as np


a = np.arange(9).reshape(3, 3)
'''
[[0 1 2]
 [3 4 5]
 [6 7 8]]
'''
print(a)

'''
[0 3 6]
'''
print(f'\na所有行，第0列:\n{a[:, 0]}')        # 所有行，第0列

'''
[3 4 5]
'''
print('\na所有列，第1行:\n%s' % str(a[1, :])) # 第1行，所有列