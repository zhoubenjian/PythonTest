'''
常见函数求导
'''
import sympy as sp      # 求导工具


x = sp.Symbol('x')

# 初等函数
funcs = {'x^3': x**3, 'e^x': sp.exp(x), 'ln(x)': sp.log(x), 'x^2 + 3x + 5': x**2 + 3*x + 5}

print('------- 符号求导验证 -------')
print('初等函数的导数：')
for name, func in funcs.items():
    print(f"f(x) = {name:15s} => f'(x) = {sp.diff(func, x)}")


# 复合函数
f = x**2*sp.exp(x)
print('\n复合函数的导数：')
print('f(x) = x**2*sp.exp(x) => ', "f'(x) = ", sp.diff(f, x), sep='')


# 链式法则  (2x+1)^3
g = (2*x+1)**3
print('\n链式法则：')
print('g(x) = (2x+1)^3 => ', "g'(x) = ", sp.diff(g, x), sep='')


