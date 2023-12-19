from CSIKit.reader import IWLBeamformReader
from CSIKit.util import csitools
import numpy as np
import pickle

def save(res, name):
    print(f'''save {name}, size: {len(res)}''')
    with open(name, 'wb') as f:
        pickle.dump(res, f, 4, **('protocol',))
        None(None, None, None)
# WARNING: Decompyle incomplete


def process(data_num, filename):
    data_file = f'''./data/{data_num}/{filename}.dat'''
    reader = IWLBeamformReader()
    # data = reader.read_file(data_file, True, **('scaled',))
    data = reader.read_file(data_file, True)
    print(f'''smaple numbers is {len(data.frames)}''')
    datas = []
    for frame in data.frames:
        real = np.real(frame.csi_matrix).flatten()
        imag = np.imag(frame.csi_matrix).flatten()
        datas.append(np.concatenate((real, imag)))
    datas = np.array(datas)
    save(datas, f'''./data/{data_num}/{filename}.pkl''')

if __name__ == '__main__':
    process(1, 'Bob')
    process(1, 'Eve')
