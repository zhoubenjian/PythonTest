import numpy as np


# -.0 == +.0
print('-.0 == +.0: ', np.allclose(-.0, +.0), sep='')        # True
print('-.0 == +.0: ', np.isclose(-.0, +.0), sep='')         # True


print([.1] * 10)     # [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


print(np.log(np.e))
