from utils import RNN, device, SampleMetroDataset
import torch
from torch.utils.data import DataLoader
import os
import torch.nn as nn

# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 32

PATH = os.getcwd()

matrix_train, matrix_test = torch.load(open(PATH+"/hzdataset.pch","rb"))
ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length = LENGTH, stations_max = ds_train.stations_max)
data_train = DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE,shuffle=False)


#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence

nh = 64
ny = CLASSES 
lr = 1e-3
epochs = 10

model = RNN(nx=DIM_INPUT, nh=nh, ny=ny).to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

def run_epoch(loader, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    correct = 0
    total = 0

    for x, station in loader:
        x = x.to(device)               
        station = station.to(device)    

        if train:
            optim.zero_grad()

        hs = model(x)                  
        h_last = hs[-1]               
        logits = model.decode(h_last)  

        loss = criterion(logits, station)

        if train:
            loss.backward()
            optim.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == station).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total

print("Training...")
for epoch in range(epochs):
    train_loss, train_acc = run_epoch(data_train, train=True)
    test_loss, test_acc = run_epoch(data_test, train=False)

    print(f"[{epoch}] Train loss={train_loss:.4f}  acc={train_acc:.3f} | "
          f"Test loss={test_loss:.4f}  acc={test_acc:.3f}")