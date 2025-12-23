from pathlib import Path
import os
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime


# GPU
device = torch.device( 'cuda' if torch.cuda. is_available() else 'cpu')
print(device)


# Téléchargement des données

from datamaestro import prepare_dataset
ds = prepare_dataset("com.lecun.mnist");
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels =  ds.test.images.data(), ds.test.labels.data()

train_images = torch.tensor(train_images, dtype=torch.float32) / 255
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_images = torch.tensor(test_images, dtype=torch.float32) / 255
test_labels = torch.tensor(test_labels, dtype=torch.long)

train_images = train_images.to(device)
train_labels = train_labels.to(device)
test_images = test_images.to(device)
test_labels = test_labels.to(device)


# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Pour visualiser
# Les images doivent etre en format Channel (3) x Hauteur x Largeur
images = torch.tensor(train_images[0:8]).unsqueeze(1).repeat(1,3,1,1).double()
# Permet de fabriquer une grille d'images
images = make_grid(images)
# Affichage avec tensorboard
writer.add_image(f'samples', images, 0)


savepath = Path("model.pch")

#  TODO: 

BATCH_SIZE = 254
NB_EPOCHS = 100


# Data Preparation
class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        return (self.X[index], self.Y[index])
    
    def __len__(self):
        return self.X.size()[0]

batches = DataLoader(MyDataset(train_images, train_labels) , shuffle=True , batch_size=BATCH_SIZE)
    

# Checkpointing
class State:
    def __init__(self, model, optim):
        self.model = model
        self.optim = optim
        self.epoch = 0
        self.iteration = 0


# Autoencodeur
class moduleLinAutoenc(torch.nn.Module):

    def __init__(self):
        super(moduleLinAutoenc, self).__init__()
        self.encode_lin = torch.nn.Linear(784, 50)
        self.encode_relu = torch.nn.ReLU()
        self.lin1 = torch.nn.Linear(50, 100)
        self.tanh = torch.nn.Tanh()
        self.lin2 = torch.nn.Linear(100,50)
        self.decode_lin = torch.nn.Linear(50,784)
        self.decode_sig = torch.nn.Sigmoid()

    def forward(self, x):
        encode = self.encode_relu(self.encode_lin(x))
        m1 = self.lin1(self.tanh(encode))
        m2 = self.lin2(m1)
        decode = self.decode_sig(self.decode_lin(m2))
        return decode


if savepath.is_file():
    with savepath.open("rb") as fp:
        state = torch.load(fp,  weights_only=False)
else:
    m_autoenc = moduleLinAutoenc().to(device)
    optim = torch.optim.Adam(params=m_autoenc.parameters(), lr=0.001)
    optim.zero_grad()
    state = State(m_autoenc, optim)
    

lossfunc = nn.MSELoss().to(device)

for epoch in range(state.epoch, NB_EPOCHS):
    print("epoch: ", epoch)
    for x, y in batches:
        x = x.view(x.size(0), -1)
        yhat = state.model.forward(x)
        loss = lossfunc(yhat, x)
        loss.backward()
        state.optim.step()
        state.optim.zero_grad()
        state.iteration+=1

        # tensorboard logging loss:
        if state.iteration % 10 == 0:
            writer.add_scalar("Loss/train", loss.item(), state.iteration)
    
    with savepath.open("wb") as fp:
        state.epoch = epoch + 1
        torch.save(state,fp)

    # tensorboard reconstruct image
    with torch.no_grad():
        samples = train_images[0:8].to(device).view(8, -1)
        recon = state.model(samples).view(-1, 1, 28, 28)
        recon = recon.repeat(1, 3, 1, 1)
        grid = make_grid(recon, nrow=4)
        print(recon.min().item(), recon.max().item())
        writer.add_image(f"Reconstructions/epoch_{epoch}", grid, epoch)


writer.close()

# run in terminal to get link for tensorboard tracking:
# tensorboard --logdir=runs

# Code for previous questions:
"""
# Linear Module
class moduleLin(torch.nn.Module):

    def __init__(self):
        super(moduleLin, self).__init__()
        self.lin1 = torch.nn.Linear(784, 100)
        self.tanh = torch.nn.Tanh()
        self.lin2 = torch.nn.Linear(100,10)

    def forward(self, x):
        m1 = self.lin1(self.tanh(x))
        m2 = self.lin2(m1)
        return m2


# Linear Module using Sequential()
class moduleLinSeq(torch.nn.Module):

    def __init__(self):
        super(moduleLinSeq, self).__init__()
        self.m = nn.Sequential(
            nn.Linear(10,100), 
            nn.Tanh(), 
            nn.Linear(100,1)
            )
        self.loss = nn.MSELoss()

    def forward(self, x, y):
        return self.loss(self.m(x), y)

        
if savepath.is_file():
    with savepath.open("rb") as fp:
        state = torch.load(fp,  weights_only=False)
else:
    m_SGD = moduleLin().to(device)
    optimSGD = torch.optim.SGD(params=m_SGD.parameters(), lr=0.001)
    optimSGD.zero_grad()
    state = State(m_SGD, optimSGD)
    

crossEnt = nn.CrossEntropyLoss().to(device)

for epoch in range(state.epoch, NB_EPOCHS):
    print("epoch: ", epoch)
    for x, y in batches:
        x = x.view(x.size(0), -1)
        yhat = state.model.forward(x)
        loss = crossEnt(yhat, y)
        loss.backward()
        state.optim.step()
        state.optim.zero_grad()
        state.iteration+=1
    with savepath.open("wb") as fp:
        state.epoch = epoch + 1
        torch.save(state,fp)


# Optimizer Adam

m_ADAM = moduleLin()

# ADAM
optimADAM = torch.optim.Adam(params=m_ADAM.parameters(), lr=0.001)
optimADAM.zero_grad()

for i in range(NB_EPOCHS):
    loss = m_ADAM.forward(x, y)
    loss.backward()
    if ( (i%100) == 0):
        optimADAM.step()
        optimADAM.zero_grad()

    
# Optimizer SGD, Sequential model

m_seq_SGD = moduleLinSeq()

# SGD
optimSGD = torch.optim.SGD(params=m_seq_SGD.parameters(), lr=0.001)
optimSGD.zero_grad()

for i in range(NB_EPOCHS):
    loss = m_seq_SGD.forward(x, y)
    loss.backward()
    if ( (i%100) == 0):
        optimSGD.step()
        optimSGD.zero_grad()


# Optimizer Adam, Sequential model

m_seq_ADAM = moduleLinSeq()

# ADAM
optimADAM = torch.optim.Adam(params=m_seq_ADAM.parameters(), lr=0.001)
optimADAM.zero_grad()

for i in range(NB_EPOCHS):
    loss = m_seq_ADAM.forward(x, y)
    loss.backward()
    if ( (i%100) == 0):
        optimADAM.step()
        optimADAM.zero_grad()
"""