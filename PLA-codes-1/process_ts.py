import numpy as np
import os
import pickle
from numpy.ma.core import minimum
from tqdm import tqdm
from CSIKit.reader import IWLBeamformReader
from CSIKit.util import csitools
from scipy import signal

# https://github.com/Gi-z/CSIKit

# for dataset1
cond1 = lambda timestamp_now, timestamp_next: 10 <= timestamp_next-timestamp_now <= 30
# for dataset2
cond2 = lambda timestamp_now, timestamp_next: 5 <= timestamp_next-timestamp_now <= 20


def save(res, name):
    print(f'save {name}, size: {len(res)}')
    with open(name, 'wb') as f:
        pickle.dump(res, f, protocol=4)


def get_labels(data_num, cond, file='ts.txt'):
    file = f'./data/{data_num}/{file}'
    with open(file) as f:
        lines = f.readlines()
    m = {}
    i = 0
    while i < len(lines):
        timestamp_now = int(lines[i].strip())
        if i+1 < len(lines):
            timestamp_next = int(lines[i+1].strip())
        else:
            break
        if cond(timestamp_now, timestamp_next):
            m[i] = 1
            m[i+1] = 0
            i += 2
        else:
            i += 1
    return m, lines


def process_ts(data_num):
    data_file = f'./data/{data_num}/CSI.dat'
    reader = IWLBeamformReader()
    data = reader.read_file(data_file, scaled=True)
    print(f'sample numbers is {len(data.frames)}')
    # using higer order function
    m, lines = get_labels(data_num, cond2)
    assert len(m) <= len(data.frames)
    datas = []
    labels = []
    timestamps = []
    for idx, frame in enumerate(data.frames):
        if not idx in m:
            continue
        frame = data.frames[idx]
        label = m[idx]
        timestamp = int(lines[idx])
        real = np.real(frame.csi_matrix).flatten()
        imag = np.imag(frame.csi_matrix).flatten()
        datas.append(np.concatenate((real, imag)))
        labels.append(label)
        timestamps.append(timestamp)
    
    datas = np.array(datas)
    labels = np.array(labels)
    save(datas, f'./data/{data_num}/csi.pkl')
    save(labels, f'./data/{data_num}/labels.pkl')
    save(timestamps, f'./data/{data_num}/timestamps.pkl')


def process_ts_amp(data_num):
    data_file = f'./data/{data_num}/CSI.dat'
    reader = IWLBeamformReader()
    data = reader.read_file(data_file, scaled=True)
    print(f'smaple numbers is {len(data.frames)}')
    csi_matrix, no_frames, no_subcarriers = csitools.get_CSI(data, metric="amplitude", squeeze_output=True, extract_as_dBm=False)
    # using higer order function
    m, lines = get_labels(data_num, cond2)
    assert len(m) <= len(data.frames)
    indies = list(m.keys())
    datas = csi_matrix[indies]
    save(datas, f'./data/{data_num}/amp.pkl')


def process_ts_rss(data_num):
    data_file = f'./data/{data_num}/CSI.dat'
    reader = IWLBeamformReader()
    data = reader.read_file(data_file, scaled=True)
    print(f'smaple numbers is {len(data.frames)}')
    # using higer order function
    m, lines = get_labels(data_num, cond2)
    assert len(m) <= len(data.frames)
    datas = []
    for idx, frame in enumerate(data.frames):
        if not idx in m:
            continue
        frame = data.frames[idx]
        datas.append(np.array([frame.rssi_a, frame.rssi_b, frame.rssi_c]))
    
    datas = np.array(datas)
    save(datas, f'./data/{data_num}/rssi.pkl')
        

if __name__ == '__main__':
    process_ts(2)