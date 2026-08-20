'''
偏导数（Partial Derivative）
    对x求偏导：∂f/∂x，除x外的变量看成常量
    对y求偏导：∂f/∂y，除y外的变量看成常量
    ...
'''
import sympy as sp          # 求导工具


x, y = sp.symbols('x y')
f = x ** 2 + y ** 2

print('原函数：f(x, y) =', f)

# 一阶偏导数
print('\n一阶偏导数：')
print('对x求偏导：∂f/∂x =', sp.diff(f, x))       # ∂f/∂x = 2*x
print('对y求偏导：∂f/∂y =', sp.diff(f, y))       # ∂f/∂y = 2*y

# 二阶偏导数
print('\n二阶偏导数：')
print('对x求偏导：∂²f/∂²x =', sp.diff(f, x, 2))  # ∂²f/∂²x = 2
print('对y求偏导：∂²f/∂²y =', sp.diff(f, y, 2))  # ∂²f/∂²y = 2


# 线性回归模型损失对参数的偏导数
w, b, xi, yi = sp.symbols('w b xi yi')
loss = (w * xi + b - yi) ** 2
print('\n原损失函数：loss =', loss)
print('对w求偏导：∂loss/∂w =', sp.diff(loss, w))  # ∂loss/∂w = 2*xi*(b + w*xi - yi)
print('对b求偏导：∂loss/∂b =', sp.diff(loss, b))  # ∂loss/∂b = 2*b + 2*w*xi - 2*yi



