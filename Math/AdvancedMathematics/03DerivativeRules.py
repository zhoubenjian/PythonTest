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


# 链式法则 g(x) = (2x+1)^3
g = (2*x+1)**3
print('\n链式法则：')
print('g(x) = (2x+1)^3 => ', "g'(x) = ", sp.diff(g, x), sep='')

# f(g(x)) = sin(x^2)
# 直接求导
d_direct = sp.diff(sp.sin(x**2), x)
d_chain = sp.cos(x**2) * 2*x
print(f"\n直接求导: {d_direct}")
print(f"链式法则: cos(x^2)·2x = {d_chain}")
print(f"一致: {sp.simplify(d_direct - d_chain) == 0}")


'''
两层网络反向传播
'''
w, xi, yi = sp.symbols('w xi yi')
# 中间变量单独定义为符号
a = sp.symbols('a')

# a 关于 w 的具体表达式
a_expr = w * xi

# loss 定义为关于符号 a 的函数
loss = (a - yi) ** 2

# ∂L/∂a
dL_da = sp.diff(loss, a)
# ∂a/∂w
da_dw = sp.diff(a_expr, w)

# 链式求导法则
dL_dw_chain = dL_da.subs(a, a_expr) * da_dw

# 把 loss 完全展开成 w 的函数
loss_direct = loss.subs(a, a_expr)
dL_dw_direct = sp.diff(loss_direct, w)
print(f"\n链式 ∂L/∂w: {sp.simplify(dL_dw_chain)}")
print(f"直接 ∂L/∂w: {sp.simplify(dL_dw_direct)}")
print(f"一致: {sp.simplify(dL_dw_chain - dL_dw_direct) == 0}")




