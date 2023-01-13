import numpy as np
from utils import *


train_path = r"D:\Desktop\todo\DHE\trainset"
val_path = r"D:\Desktop\todo\DHE\valiset"

emb = np.load(f"{train_path}\\train_emb.npy")
gse = np.load(f"{train_path}\\train_gse.npy")
sub = np.load(f"{train_path}\\train_sub.npy")
label = np.load(f"{train_path}\\train_label.npy")


# get only 682 samples with label = 0 and get the correspond emb, gse, sub values
# and concatenate them with the original samples with label = 1
index = np.where(label == 0)[0]
index1 = np.where(label == 1)[0]

emb0 = emb[index]
gse0 = gse[index]
sub0 = sub[index]
label0 = label[index]

emb1 = emb[index1]
gse1 = gse[index1]
sub1 = sub[index1]
label1 = label[index1]

emb = np.concatenate((emb0, emb1), axis=0)
gse = np.concatenate((gse0, gse1), axis=0)
sub = np.concatenate((sub0, sub1), axis=0)
label = np.concatenate((label0, label1), axis=0)

np.save(f"new data\\train_emb.npy", emb)
np.save(f"new data\\train_gse.npy", gse)
np.save(f"new data\\train_sub.npy", sub)
np.save(f"new data\\train_label.npy", label)