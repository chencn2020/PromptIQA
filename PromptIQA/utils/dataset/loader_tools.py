import torch.utils.data as data
import torch

import os
import scipy.io
import numpy as np
import csv
from openpyxl import load_workbook
import cv2
import random

class prompt_data(data.Dataset):
    def __init__(self, sample, gt, transform, div=1):
        self.samples, self.gt = [sample], [gt]
        self.transform = transform
        self.div = div

    def __getitem__(self, index):
        img_tensor, gt_tensor = get_item(self.samples, self.gt, index, self.transform, self.div)

        return img_tensor, gt_tensor

    def __len__(self):
        length = len(self.samples)
        return length

def get_prompt(samples_p, gt_p, transform, n, length, sample_type='fix'):
    combined_data = list(zip(samples_p, gt_p))

    if sample_type == 'fix':
        combined_data.sort(key=lambda x: x[1])
    elif sample_type == 'random':
        random.seed()
        random.shuffle(combined_data)
    else:
        raise NotImplementedError('Only Support fix | random')

    length = len(samples_p)
    sample, gt = [], []
    if n == 2:
        sample.append(combined_data[0][0])
        gt.append(combined_data[0][1])
        return prompt_data(sample, gt, transform)
    data_len = (length - 2) // (n - 2)
    sample.append(combined_data[0][0])
    gt.append(combined_data[0][1])
    for i in range(data_len, length, data_len):
        sample.append(combined_data[i][0])
        gt.append(combined_data[i][1])
        if len(sample) == n - 1:
            break
    sample.append(combined_data[-1][0])
    gt.append(combined_data[-1][1])

    assert len(sample) == n
    return prompt_data(sample, gt, transform)

def reshuffle(sample:list, gt:list):
    combine = list(zip(sample.copy(), gt.copy()))
    random.shuffle(combine)

    sample_new, gt_new = [], []
    for i, j in combine:
        sample_new.append(i)
        gt_new.append(j)
    
    return sample_new, gt_new


def split_array(arr, m):
    return [arr[i:i + m] for i in range(0, len(arr), m)]


def get_item(samples, gt, index, transform, div=1):
    div = 1
    path_list, target_list = samples[index], gt[index]
    img_tensor, gt_tensor = None, None
    for path, target in zip(path_list, target_list):
        target = [target / div]

        values_to_insert = np.array([0.0, 1.0])
        position_to_insert = 0
        target = np.insert(target, position_to_insert, values_to_insert)

        sample = load_image(path)
        samples = {'img': sample, 'gt': target}
        samples = transform(samples)

        if img_tensor is None:
            img_tensor = samples['img'].unsqueeze(0)
            gt_tensor = samples['gt'].type(torch.FloatTensor).unsqueeze(0)
        else:
            img_tensor = torch.cat((img_tensor, samples['img'].unsqueeze(0)), dim=0)
            gt_tensor = torch.cat((gt_tensor, samples['gt'].type(torch.FloatTensor).unsqueeze(0)), dim=0)

    return img_tensor, gt_tensor

def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename

def load_image(img_path, size=224):
    try:
        d_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (size, size), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))
    except:
        print(img_path)

    return d_img

def normalization(data):
    data = np.array(data)
    range = np.max(data) - np.min(data)
    data = (data - np.min(data)) / range
    data = list(data.astype('float').reshape(-1, 1))
    return data

def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename
