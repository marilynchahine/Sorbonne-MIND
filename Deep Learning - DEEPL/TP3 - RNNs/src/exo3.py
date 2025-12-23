from utils import RNN, device,  ForecastMetroDataset

from torch.utils.data import  DataLoader
import torch
import torch.nn as nn
import os

# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 32

PATH = os.getcwd()


matrix_train, matrix_test = torch.load(open(PATH+"/hzdataset.pch", "rb"))
ds_train = ForecastMetroDataset(
    matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = ForecastMetroDataset(
    matrix_test[:, :, :CLASSES, :DIM_INPUT], length=LENGTH, stations_max=ds_train.stations_max)
data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

#  TODO:  Question 3 : Prédiction de séries temporelles

nh = 64
ny = DIM_INPUT * CLASSES  
lr = 1e-3
epochs = 10

model = RNN(nx=DIM_INPUT * CLASSES, nh=nh, ny=ny).to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

def run_epoch(loader, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total = 0

    for x, y in loader:
        B, L, S, D = x.shape

        # flatten input
        x = x.reshape(B, L, -1).to(device)
        y = y.reshape(B, L, -1).to(device)

        if train:
            optim.zero_grad()

        hs = model(x)                   
        logits = model.decode(hs)     

        # reshape for MSE
        logits = logits.transpose(0, 1)     
        loss = criterion(logits, y)

        if train:
            loss.backward()
            optim.step()

        total_loss += loss.item() * B
        total += B

    return total_loss / total

print("Training")
for ep in range(epochs):
    train_loss = run_epoch(data_train, train=True)
    test_loss  = run_epoch(data_test, train=False)
    print(f"[{ep}] Train loss={train_loss:.4f} | Test loss={test_loss:.4f}")