import torch
import numpy as np
from sklearn import metrics


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model(save_path, model, best_prec, epoch):
    data = {'epoch': epoch, 'best_prec': best_prec,
    'model_state': model.state_dict()}
    torch.save(data, save_path)
    print(f'saved best model {save_path}')

def accuracy(y_pred, y_true):
    y_pred = torch.max(y_pred, dim=1)[1]
    return metrics.accuracy_score(y_true, y_pred)

def accuracy2(y_pred, y_true):
    CM = metrics.confusion_matrix(y_true, y_pred)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    # FAR = FP/(TN+FP)
    FAR = FN/(TP+FN)
    MDR = FP/(TN+FP)
    return metrics.accuracy_score(y_true, y_pred), FAR, MDR
