import string
import unicodedata
import torch
import sys
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,DataLoader
import os

from utils import RNN, device

## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

def normalize(s):
    """ Nettoyage d'une chaîne de caractères. """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """ Transformation d'une chaîne de caractère en tenseur d'indexes """
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ Transformation d'une liste d'indexes en chaîne de caractères """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

class TrumpDataset(Dataset):
    def __init__(self,text,maxsent=None,maxlen=None):
        """  Dataset pour les tweets de Trump
            * text : texte brut
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        full_text = normalize(text)
        self.phrases = [p[:maxlen].strip()+"." for p in full_text.split(".") if len(p)>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)
    def __getitem__(self,i):
        t = string2code(self.phrases[i])
        t = torch.cat([torch.zeros(self.MAX_LEN-t.size(0),dtype=torch.long),t])
        return t[:-1],t[1:]



#  TODO: 

batch_size = 32
embed_dim = len(lettre2id)
nh = 256
ny = embed_dim
lr = 1e-3
epochs = 5

PATH = os.getcwd()
data_trump = DataLoader(TrumpDataset(open(PATH+"/trump_full_speech.txt","rb").read().decode(),maxlen=1000), batch_size= batch_size, shuffle=True)

model = RNN(nx=embed_dim, nh=nh, ny=ny).to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

def one_hot(batch_ids):
    B, L = batch_ids.shape
    oh = torch.zeros(B, L, embed_dim, device=device)
    oh.scatter_(2, batch_ids.unsqueeze(2), 1.0)
    return oh

def train_epoch():
    model.train()
    total_loss = 0
    count = 0

    for x,y in data_trump:
        x = x.to(device)
        y = y.to(device)

        x_oh = one_hot(x)  
        optim.zero_grad()

        hs = model(x_oh)            
        logits = model.decode(hs)     
        logits = logits.permute(1,0,2).reshape(-1, embed_dim)
        y = y.reshape(-1)

        loss = criterion(logits, y)
        loss.backward()
        optim.step()

        total_loss += loss.item()
        count += 1

    return total_loss / count

def generate(prompt, max_len=200):
    model.eval()
    with torch.no_grad():
        s = string2code(prompt).to(device)
        h = model.geth0(1)
        # prompt
        for ch in s:
            x = torch.zeros(1, 1, embed_dim, device=device)
            x[0,0,ch] = 1
            h = model.one_step(x.squeeze(1), h)

        # generate
        out = []
        ch = s[-1]
        for _ in range(max_len):
            x = torch.zeros(1, 1, embed_dim, device=device)
            x[0,0,ch] = 1
            h = model.one_step(x.squeeze(1), h)
            logits = model.decode(h)
            ch = logits.argmax(dim=1).item()
            out.append(ch)
        return code2string(out)

print("Training")
for ep in range(epochs):
    loss = train_epoch()
    print(f"[{ep}] loss={loss:.4f}")

print("\nGenerated text:")
print(generate("America "))
