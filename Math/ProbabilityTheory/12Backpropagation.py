'''
反向传播
    链式求导
'''
import torch


x = torch.tensor([2.0])
y = torch.tensor([4.0])
w1 = torch.tensor([0.5], requires_grad=True)
b1 = torch.tensor([0.0], requires_grad=True)
w2 = torch.tensor([1.0], requires_grad=True)
b2 = torch.tensor([0.0], requires_grad=True)


# 前向传播
z1 = w1 * x + b1
a1 = torch.relu(z1)
y_pred = w2 * a1 + b2
loss = (y_pred - y) ** 2

# 反向传播
loss.backward()


print('------- Pytorch验证 -------')
'''
∂L/∂b2 = ∂L/∂y_pred * ∂y_pred/∂b2
       = 2*(y_pred - y) * (1)
       = 2*(1 - 4)
       = -6
'''
print(f"∂L/∂b2 = {b2.grad.item():.1f}  (手算: -6)")

'''
∂L/∂w2 = ∂L/∂y_pred * ∂y_pred/∂w2
       = 2*(y_pred - y) * (a1)
       = 2*(1 - 4) * (1)
       = -6
'''
print(f"∂L/∂w2 = {w2.grad.item():.1f}  (手算: -6)")

'''
∂L/∂b1 = ∂L/∂y_pred * ∂y_pred/∂a1 * ∂a1/∂z1 * ∂z1/∂b1
       = 2*(y_pred - y) * (w2) * (1) * (1)
       = 2*(1 - 4) * (1) * (1) * (1)
       = -6
'''
print(f"∂L/∂b1 = {b1.grad.item():.1f}  (手算: -6)")

'''
∂L/∂w1 = ∂L/∂y_pred * ∂y_pred/∂a1 * ∂a1/∂z1 * ∂z1/∂w1
       = 2*(y_pred - y) * (w2) * (1) * (x)
       = 2*(1 - 4) * (1) * (1) * (2)
       = -12
'''
print(f"∂L/∂w1 = {w1.grad.item():.1f} (手算: -12)")


# 梯度下降
lr = 0.01
with torch.no_grad():
    w1 -= lr * w1.grad
    w2 -= lr * w2.grad
    new_loss = ((w2 * torch.relu(w1 * x + b1) + b2 - y) ** 2).item()

print(f"\n原损失:{loss.item():.1f} => 新损失:{new_loss:.2f}")
print(f"损失减小了！")



