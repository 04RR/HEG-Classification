
# Data used for this project is uploaded on on drive at the link - https://drive.google.com/drive/folders/1k9VyxIBDcZi9-lVf794N65SnGqSk1E6A?usp=sharing

train_emb.npy - features from PPI network (https://github.com/aditya-grover/node2vec) 
train_gse.npy - file represents features extracted from gene expression profiles. According to the time step, the control group and the replication samples, it is reshaped as (5988, 8, 3, 2).
train_label.npy - represents the label of the protein, where 0 represents the non-essential protein and 1 represents the essential protein.
train_sub.npy - represents features extracted from subcellular localization.
