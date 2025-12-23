import torch
from torch.utils.tensorboard import SummaryWriter
from tp1 import MSE, Linear, Context


# Les données supervisées
x = torch.randn(50, 13)
y = torch.randn(50, 3)

# Les paramètres du modèle à optimiser
w = torch.randn(13, 3)
b = torch.randn(3)

epsilon = 0.05


writer = SummaryWriter()


for n_iter in range(100):
    ##  TODO:  Calcul du forward (loss)

    ctx_lin = Context()
    ctx_mse = Context()

    yhat = Linear.forward(ctx_lin, x, w, b)
    loss = MSE.forward(ctx_mse, yhat, y)


    # `loss` doit correspondre au coût MSE calculé à cette itération
    # on peut visualiser avec
    # tensorboard --logdir runs/
    writer.add_scalar('Loss/train', loss, n_iter)

    # Sortie directe
    print(f"Itérations {n_iter}: loss {loss}")

    ##  TODO:  Calcul du backward (grad_w, grad_b)
    grad_mse_yhat, grad_mse_y = MSE.backward(ctx_mse, torch.tensor(1.0))
    grad_x, grad_w, grad_b = Linear.backward(ctx_lin, grad_mse_yhat)

    ##  TODO:  Mise à jour des paramètres du modèle
    w -= epsilon * grad_w
    b -= epsilon * grad_b

print(w)
print(b)