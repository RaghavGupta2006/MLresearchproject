import torch
import numpy as np
import os
import random


def generate_o_d(origin, str, data):
    list = []
    for i in range(data.shape[0]):
        list.append(origin[data.iloc[i][str]])
    data[str + '_'] = list
    data.drop(str, axis=1, inplace=True)
    return data


def make_index(data):
    index = []
    for i in range(data.shape[0]):
        index.append(i)
    data.index = index
    return data


def mix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
