from dataset import *
import torch
import numpy as np
from tqdm import tqdm
from main import config
import pickle
import os
from models.siamese import SiameseNet


# setup_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = CSIDatasetV2(2, 'csi')
window_size = config['window_size']


# for l2 distance and person coeff
def process_by_window(data_num, save_name):
    filename = f'./data/{data_num}/{save_name}.pkl'
    i = 0
    valid_csi = []
    while len(valid_csi) < window_size:
        if dataset[i][1] == 1:
            valid_csi.append(dataset[i][0])
        i += 1
    valid_csi = torch.stack(valid_csi, dim=0).to(device)

    if os.path.exists(filename):
        os.remove(filename)
    f = open(filename, 'ab')
    for idx in tqdm(range(i, len(dataset))):
        csi, label = dataset[idx]
        csi = csi.to(device)
        distances, coeffs = extract_similarity_feature(csi, valid_csi)
        # feature = np.array([distances.cpu().numpy(), coeffs.cpu().numpy()])
        # feature = distances.cpu().numpy()
        feature = coeffs.cpu().numpy()
        pickle.dump(feature, f)
        if label == 1:
            valid_csi = torch.cat((valid_csi, csi.unsqueeze(0)), dim=0)[1:]
    
    f.close()


# for siamese net
def process_by_window2(data_num, save_name):
    filename = f'./data/{data_num}/{save_name}.pkl'
    i = 0
    valid_csi = []
    while len(valid_csi) < window_size:
        if dataset[i][1] == 1:
            valid_csi.append(dataset[i][0])
        i += 1
    valid_csi = torch.stack(valid_csi, dim=0).to(device)

    model = SiameseNet(360)
    model.load_state_dict(torch.load(f'./checkpoints/SiameseNet-{data_num}.pt')['model_state'])
    model = model.to(device)
    model.eval()
    if os.path.exists(filename):
        os.remove(filename)
    f = open(filename, 'ab')
    for idx in tqdm(range(i, len(dataset))):
        csi, label = dataset[idx]
        csi = csi.to(device)
        feature = []
        for i in range(window_size):
            pred = model(csi.unsqueeze(0), valid_csi[i].unsqueeze(0))
            distance = pred[0][1].item()
            feature.append(distance)
        feature = np.array(feature)
        pickle.dump(feature, f)
        if label == 1:
            valid_csi = torch.cat((valid_csi, csi.unsqueeze(0)), dim=0)[1:]
    
    f.close()


# for cosine and L1
def process_by_window3(data_num, save_name):
    filename = f'./data/{data_num}/{save_name}.pkl'
    i = 0
    valid_csi = []
    while len(valid_csi) < window_size:
        if dataset[i][1] == 1:
            valid_csi.append(dataset[i][0])
        i += 1
    valid_csi = torch.stack(valid_csi, dim=0).to(device)

    if os.path.exists(filename):
        os.remove(filename)
    f = open(filename, 'ab')
    cosine = torch.nn.CosineSimilarity(dim=0)
    L1 = torch.nn.L1Loss()
    for idx in tqdm(range(i, len(dataset))):
        csi, label = dataset[idx]
        csi = csi.to(device)
        feature = []
        for i in range(window_size):
            # distance = cosine(csi, valid_csi[i])
            distance = L1(csi, valid_csi[i])
            feature.append(distance.item())
        feature = np.array(feature)
        pickle.dump(feature, f)
        if label == 1:
            valid_csi = torch.cat((valid_csi, csi.unsqueeze(0)), dim=0)[1:]
    
    f.close()



def extract_similarity_feature(vector, valid_data):
    # euclidean distance
    diff = vector - valid_data
    distances = torch.sqrt(torch.sum(diff ** 2, dim=1))
    # person coeff
    vx = vector - torch.mean(vector)
    vy = valid_data - torch.mean(valid_data, dim=1, keepdim=True)
    coeffs = torch.sum(vx * vy, dim=1) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2, dim=1)))

    return distances, coeffs


def euclidean_distance(x, y):
    diff = x - y
    res = torch.sqrt(torch.sum(diff ** 2))
    return res.item()


def person_coeff(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    res = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return res.item()


if __name__ == '__main__':
    # process_by_window(2, f'features-coeff-{window_size}')
    # process_by_window2(2, f'features-{window_size}')
    process_by_window3(2, f'features-l1-{window_size}')