import numpy as np
import torch
from model import Model
from utils import *


train_path = r"new data"
val_path = r"D:\Desktop\todo\DHE\valiset"

trainset = Dataset(
    np.load(f"{train_path}\\train_emb.npy"),
    np.load(f"{train_path}\\train_gse.npy"),
    np.load(f"{train_path}\\train_sub.npy"),
    np.load(f"{train_path}\\train_label.npy"),
)

valiset = Dataset(
    np.load(f"{val_path}\\vali_emb.npy"),
    np.load(f"{val_path}\\vali_gse.npy"),
    np.load(f"{val_path}\\vali_sub.npy"),
    np.load(f"{val_path}\\vali_label.npy"),
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
valiloader = torch.utils.data.DataLoader(valiset, batch_size=32, shuffle=True)

model = Model()
criterion = torch.nn.BCELoss(weight=torch.Tensor([0.82, 0.18]))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)

epochs = 100

for epoch in range(epochs):
    train_loss, train_acc = train(model, trainloader, criterion, optimizer)
    vali_loss, vali_acc = eval(model, valiloader, criterion)

    print(f"{epoch+1}/{epochs} Train Loss: {train_loss} Val Loss: {vali_loss} ")

    if epoch == 0:
        torch.save(model.state_dict(), "new data/model.pth")
        prev_vali_loss = vali_loss
    else:
        if vali_loss < prev_vali_loss:
            torch.save(model.state_dict(), "new data/model.pth")
            prev_vali_loss = vali_loss
            print("Model Saved.")
