import torch


# _emb.npy features from PPI network (https://github.com/aditya-grover/node2vec) 
# # _gse.npy file represents features extracted from gene expression profiles. According to the time step, the control group and the replication samples, we reshape it as (5988, 8, 3, 2).
# _sub.npy file represents features extracted from subcellular localization.
# _label.npy file represents the label of the protein, where 0 represents the non-essential protein and 1 represents the essential protein.


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc)

    return acc


def train(model, trainloader, criterion, optimizer):
    model.train()
    train_loss = []
    acc_list = []

    for emb, gse, sub, label in trainloader:
        optimizer.zero_grad()

        output = model((emb, gse, sub))
        loss = criterion(output.to(torch.float), label.to(torch.float))
        acc = binary_acc(output, label)

        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        acc_list.append(acc.item())

    return sum(train_loss) / len(train_loss), sum(acc_list) / len(acc_list)


def eval(model, valiloader, criterion):
    model.eval()
    val_loss = []
    acc_list = []

    for emb, gse, sub, label in valiloader:
        output = model((emb, gse, sub))
        loss = criterion(output.to(torch.float), label.to(torch.float))
        acc = binary_acc(output, label)

        val_loss.append(loss.item())
        acc_list.append(acc.item())

    return sum(val_loss) / len(val_loss), sum(acc_list) / len(acc_list)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, emb, gse, sub, label):
        self.emb = torch.tensor(emb, dtype=torch.float32)
        self.gse = torch.tensor(gse, dtype=torch.float32)
        self.sub = torch.tensor(sub, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        emb, gse, sub, label = (
            self.emb[idx],
            self.gse[idx],
            self.sub[idx],
            self.label[idx],
        )

        gse = gse.reshape(48)

        if label == 1:
            label = torch.tensor([1, 0])
        else:
            label = torch.tensor([0, 1])

        return emb, gse, sub, label
