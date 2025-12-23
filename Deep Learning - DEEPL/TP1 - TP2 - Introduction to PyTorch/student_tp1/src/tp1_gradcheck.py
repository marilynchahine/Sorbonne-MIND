import torch
from tp1 import MSE, Linear, Context



# Test du gradient de MSE

yhat = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
y = torch.randn(10,5, requires_grad=True, dtype=torch.float64)

## Utile dans ce TP que pour le script tp1_gradcheck
mse = MSE.apply

print(torch.autograd.gradcheck(mse, (yhat, y)))


#  TODO:  Test du gradient de Linear (sur le même modèle que MSE)
X = torch.randn(10, 3, requires_grad=True, dtype=torch.float64)
W = torch.randn(3, 5, requires_grad=True, dtype=torch.float64)
b = torch.randn(5, requires_grad=True, dtype=torch.float64)

linear = Linear.apply

print(torch.autograd.gradcheck(linear, (X, W, b)))