import os
import sys
import torch
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from models.mlp import *
# from models.siamese import *
from models.siamese import SiameseNet
from dataset import *
from tqdm import tqdm
from torch.optim import lr_scheduler
from utils import *

# https://github.com/tqdm/tqdm#installation

# parse configure file
with open('./config.yaml') as f:
    config = yaml.safe_load(f)
# set the default paramters
train_batch_size = config['train_batch_size']
test_batch_size = config['test_batch_size']
lr = config['lr']
weight_decay = config['weight_decay']
epochs = config['epochs']
model_name = config['model_name']
dataset_num = config['dataset_num']
load_old_model = config['load_old_model']
seed = config['seed']
dataset_cls_name = config['dataset_cls_name']
dataset_filename = config['dataset_filename']
input_dim = config['input_dim']
split_method = config['split_method']
train_percent = config['train_percent']


# set random seed
def setup_seed(seed):
    if not seed:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f'set the random seed to {seed}')

setup_seed(seed)

# initial
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
criterion = nn.CrossEntropyLoss()
# model_cls = getattr(sys.modules[__name__], model_name)
model_cls = getattr(sys.modules[__name__], 'SiameseNet')
model = model_cls(input_dim)
model = model.to(device)


def get_dataloaders():
    cls = getattr(sys.modules[__name__], dataset_cls_name)
    dataset = cls(dataset_num, dataset_filename)
    train_size = int(len(dataset)*train_percent)
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
    save_path = f'./checkpoints/{model_name}-{dataset_num}.pt'
    best_prec = torch.load(save_path)['best_prec'] if load_old_model and os.path.exists(save_path) else 0.
    
    train_loader, test_loader = get_dataloaders()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in range(1, epochs+1):
        # train for one epoch
        train(train_loader, optimizer, epoch)
        # evaluate on validation set
        prec = evaluate(test_loader, epoch)
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
    for feature1, feature2, label in tqdm(train_loader):
        feature1 = feature1.to(device)
        feature2 = feature2.to(device)
        label = label.view(label.size(0))
        # compute y_pred
        pred = model(feature1, feature2)
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
    acc = AverageMeter()
    # switch to evaluate mode
    model.eval()
    print('Evaluating')

    for feature1, feature2, label in tqdm(val_loader):
        feature1 = feature1.to(device)
        feature2 = feature2.to(device)
        label = label.view(label.size(0))
        # compute y_pred
        pred = model(feature1, feature2)
        loss = criterion(pred, label)

        # measure accuracy and record loss
        prec = accuracy(pred.detach(), label.detach())
        losses.update(loss.item(), label.size(0))
        acc.update(prec, label.size(0))
    print('Evaluate * EPOCH {epoch} | Loss: {losses.avg:.5f} | Accurucy: {acc.avg:.5f}'.format(epoch=epoch, losses=losses, acc=acc))
    return acc.avg


if __name__ == '__main__':
    main()