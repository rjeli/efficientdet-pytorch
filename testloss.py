import torch
import torch.nn.functional as F

x = torch.ones(2, 4, 4) * -100
x[0,0,1] = 100

print(x)
x = torch.sigmoid(x)
print(x)

y = torch.zeros(2, 4, 4)
y[0,0,0] = 1
y[0,1,2] = 1
y[1,3,2] = 1
print(y)

loss = F.binary_cross_entropy(
    x, y, reduction='none')

print(loss)
