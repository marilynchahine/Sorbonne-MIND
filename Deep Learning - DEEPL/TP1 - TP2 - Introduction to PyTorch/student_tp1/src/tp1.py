
import torch
from torch.autograd import Function
from torch.autograd import gradcheck


class Context:
    """Un objet contexte très simplifié pour simuler PyTorch

    Un contexte différent doit être utilisé à chaque forward
    """
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors


class MSE(Function):
    """Début d'implementation de la fonction MSE"""
    @staticmethod
    def forward(ctx, yhat, y):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(yhat, y)

        mse = torch.mean(torch.square(yhat - y))

        return mse
        #  TODO:  Renvoyer la valeur de la fonction

    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        yhat, y = ctx.saved_tensors

        grad_mse_yhat = 2 * (yhat - y) / yhat.numel()
        grad_mse_y = - grad_mse_yhat

        grad_mse_yhat = grad_output * grad_mse_yhat
        grad_mse_y = grad_output * grad_mse_y

        return grad_mse_yhat, grad_mse_y
        #  TODO:  Renvoyer par les deux dérivées partielles (par rapport à yhat et à y)


#  TODO:  Implémenter la fonction Linear(X, W, b)sur le même modèle que MSE
class Linear(Function):
    """Début d'implementation de la fonction MSE"""
    @staticmethod
    def forward(ctx, X, W, b):
        ## Garde les valeurs nécessaires pour le backwards

        yhat = torch.mm(X, W) + b

        ctx.save_for_backward(X, W, b)

        return yhat
        #  TODO:  Renvoyer la valeur de la fonction

    @staticmethod
    def backward(ctx, grad_mse_yhat):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        X, W, b = ctx.saved_tensors

        grad_f_X = torch.mm(grad_mse_yhat, W.T)
        grad_f_w = torch.mm(torch.transpose(X, 0, 1), grad_mse_yhat)
        grad_f_b = grad_mse_yhat.sum(0)


        return grad_f_X, grad_f_w, grad_f_b


