import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from dataset import *
from tqdm import tqdm
from models.mlp import MLP
from models.conv import CNN
from utils import *


# parse configure file
with open('./config.yaml') as f:
    config = yaml.safe_load(f)
# set the default paramters
train_batch_size = config['train_batch_size']
test_batch_size = config['test_batch_size']
lr = config['lr']
weight_decay = config['weight_decay']
epochs = config['epochs']
dataset_num = config['dataset_num']
load_old_model = config['load_old_model']
seed = config['seed']
split_method = config['split_method']
input_dim = 20

# initial
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
criterion = nn.CrossEntropyLoss()
model = MLP(20)
# model = CNN(1, hidden_dim=128)
model = model.to(device)


def get_dataloaders():
    # dataset = CSIDatasetV2(dataset_num, 'csi')
    dataset = FeatureDatasetV1(dataset_num, 'features-20')
    percent = 0.01
    train_size = int(len(dataset)*percent)
    test_size = len(dataset)-train_size
    if split_method == 'random':
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    elif split_method == 'sequence':
        indices = list(range(len(dataset)))
        train_dataset = torch.utils.data.Subset(dataset, indices[:train_size])
        test_dataset = torch.utils.data.Subset(dataset, indices[train_size:])
        # assert max([item[2] for item in train_dataset]) < min([item[2] for item in test_dataset])
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader


def main():
    save_path = f'./checkpoints/MLP-{dataset_num}.pt'
    best_prec = torch.load(save_path)['best_prec'] if load_old_model and os.path.exists(save_path) else 0.
    model.load_state_dict(torch.load(save_path)['model_state'])
    
    train_loader, test_loader = get_dataloaders()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(1, epochs+1):
        # train for one epoch
        # train(train_loader, optimizer, epoch)
        # evaluate on validation set
        prec = evaluate(test_loader, epoch)
        return
        # remember best Accuracy and save checkpoint
        if prec > best_prec:
            best_prec = prec
            save_model(save_path, model, best_prec, epoch)


def train(train_loader, optimizer, epoch):
    losses = AverageMeter()
    acc = AverageMeter()
    # switch to train mode
    model.train()
    print('Training')
    for feature, label in tqdm(train_loader):
        feature = feature.to(device)
        label = label.view(label.size(0))
        # feature = feature.unsqueeze(1).unsqueeze(1)
        # compute y_pred
        pred = model(feature)
        loss = criterion(pred, label)
        # measure accuracy and record loss
        prec = accuracy(pred.detach(), label.detach())
        losses.update(loss.item(), label.size(0))
        acc.update(prec, label.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print('Train * EPOCH {epoch} | Loss: {losses.avg:.5f} | Accurucy: {acc.avg:.5f}'.format(epoch=epoch, losses=losses, acc=acc))


def evaluate(val_loader, epoch):
    losses = AverageMeter()
    y_preds = []
    y_trues = []
    y_scores = []
    # switch to evaluate mode
    model.eval()
    print('Evaluating')

    for feature, label in tqdm(val_loader):
        feature = feature.to(device)
        label = label.view(label.size(0))
        # feature = feature.unsqueeze(1).unsqueeze(1)
        # compute y_pred
        pred = model(feature)
        loss = criterion(pred, label)
        # measure accuracy and record loss
        losses.update(loss.item(), label.size(0))
        y_preds.append(torch.max(pred.detach(), dim=1)[1].numpy())
        y_trues.append(label.detach().numpy())
        y_scores.append(pred.detach()[:,1].numpy())
    
    y_preds = np.array(y_preds)
    y_trues = np.array(y_trues)
    y_scores = np.array(y_scores)
    from sklearn import metrics
    auc = metrics.roc_auc_score(y_trues, y_scores)
    print(f'AUC={auc}')
    acc, FAR, MDR = accuracy2(y_preds, y_trues)
    print('Evaluating * EPOCH {epoch} | Loss: {losses.avg:.5f} | Accuracy: {acc:.5f} | False alarm: {FAR:.5f} | Miss detection: {MDR:.5f}'.format(epoch=epoch, losses=losses, acc=acc, FAR=FAR, MDR=MDR))
    return acc


if __name__ == '__main__':
    main()