import numpy as np
import pickle
from process_raw_data import save


def awgn(x, snr):
    '''
    加入高斯白噪声 Additive White Gaussian Noise
    :param x: 原始信号
    :param snr: 信噪比
    :return: 加入噪声后的信号
    '''
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    noise = np.random.randn(len(x)) * np.sqrt(npower)
    return noise


def add_noise2dataset(n, snr):
    with open(f'./data/{n}/csi.pkl', 'rb') as f:
        data = pickle.load(f)
    for i in range(360):
        noise = awgn(data[:,i], 2)
        data[:,i] += noise
    save(data, f'./data/{n}/csi-noised{snr}.pkl')


if __name__ == '__main__':
    add_noise2dataset(1, -10)