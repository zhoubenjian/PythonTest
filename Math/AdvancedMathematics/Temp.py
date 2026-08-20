import sympy as sp


w, xi, yi = sp.symbols('w xi yi')
a_expr = w * xi

# 定义中间变量 a 为符号 a
a = sp.symbols('a')
loss = (a - yi) ** 2

# ∂L/∂a
dL_da = sp.diff(loss, a)
# ∂a/∂w
da_dw = sp.diff(a_expr, w)

# chain rule
dL_dw_chain = dL_da.subs(a, a_expr) * da_dw

# direct rule
dL_dw_direct = sp.diff(loss.subs(a, a_expr), w)

print(f'chain rule: {sp.signsimp(dL_dw_chain)}')
print(f'direct rule: {sp.signsimp(dL_dw_direct)}')
print(f'same: {sp.simplify(dL_dw_chain - dL_dw_direct) == 0}')


print('=' * 50)


# 复合函数求导
x = sp.symbols('x')

f = sp.log(x, 2)
g = sp.cos(f)

dg_dx = sp.diff(g, x)
# sympy会对数表达式进行常用对数换底！！！
print(f'g′(x) = {dg_dx}')       # -sin(log(x)/log(2))/(x*log(2))
