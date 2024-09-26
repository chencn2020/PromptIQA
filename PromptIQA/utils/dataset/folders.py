import torch.utils.data as data
import torch

import os
import scipy.io
import numpy as np
import csv
from openpyxl import load_workbook
import cv2
import random
import torch
import torchvision

from utils.dataset import folders
from utils.dataset.process import ToTensor, Normalize, RandHorizontalFlip

def get_prompt(samples_p, gt_p, transform, n, length, sample_type='fix'):
    # transform = torchvision.transforms.Compose([Normalize(0.5, 0.5), ToTensor()])
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

def reshuffle(sample:list, gt:list):
    combine = list(zip(sample.copy(), gt.copy()))
    random.shuffle(combine)

    sample_new, gt_new = [], []
    for i, j in combine:
        sample_new.append(i)
        gt_new.append(j)
    
    return sample_new, gt_new

class LIVEC(data.Dataset):
    def __init__(self, root, index, transform, batch_size=11, istrain=True):
        imgpath = scipy.io.loadmat(os.path.join(root, 'Data', 'AllImages_release.mat'))
        imgpath = imgpath['AllImages_release']
        imgpath = imgpath[7:1169]
        mos = scipy.io.loadmat(os.path.join(root, 'Data', 'AllMOS_release.mat'))
        labels = mos['AllMOS_release'].astype(np.float32)
        labels = labels[0][7:1169]

        sample, gt = [], []
        for i, item in enumerate(index):
            sample.append(os.path.join(root, 'Images', imgpath[item][0][0]))
            gt.append(labels[item])
        # gt = normalization(gt)
        gt = list((np.array(gt) - 1) / 100)
        
        self.samples_p, self.gt_p = sample, gt

        self.samples, self.gt = split_array(sample, batch_size), split_array(gt, batch_size)
        if len(self.samples[-1]) != batch_size and istrain:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]
        self.transform = transform
        self.batch_size = batch_size
    
    def reshuffle(self):
        shuffle_sample, shuffle_gt = reshuffle(self.samples_p, self.gt_p)
        self.samples, self.gt = split_array(shuffle_sample, self.batch_size), split_array(shuffle_gt, self.batch_size)
        if len(self.samples[-1]) != self.batch_size:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]

    def __getitem__(self, index):
        img_tensor, gt_tensor = get_item(self.samples, self.gt, index, self.transform, div=5)
        return img_tensor, gt_tensor, self.samples[index], 'livec'

    def __len__(self):
        length = len(self.samples)
        return length

    def get_promt(self, n=10, sample_type='fix'):
        return get_prompt(self.samples_p, self.gt_p, self.transform, n, self.__len__(), sample_type=sample_type)

class AIGCIQA3W(data.Dataset):
    def __init__(self, root, index, transform, batch_size=11, istrain=True):

        imgname = []
        mos_all = []

        xls_file = os.path.join(root, 'info_train.xlsx')
        workbook = load_workbook(xls_file)
        booksheet = workbook.active
        rows = booksheet.rows
        count = 1
        prompt = []
        for row in rows:
            count += 1
            img_name = booksheet.cell(row=count, column=1).value
            imgname.append(img_name)
            mos = booksheet.cell(row=count, column=3).value
            mos = np.array(mos)
            mos = mos.astype(np.float32)
            mos_all.append(mos)
            prompt.append(str(booksheet.cell(row=count, column=2).value))
            if count == 14002:
                break

        sample, gt = [], []
        for i, item in enumerate(index):
            sample.append(os.path.join(root, 'train', imgname[item]))
            gt.append(mos_all[item])
        # gt = normalization(gt)
        gt = list(np.array(gt) / 1)
        self.samples_p, self.gt_p = sample, gt

        self.samples, self.gt = split_array(sample, batch_size), split_array(gt, batch_size)
        if len(self.samples[-1]) != batch_size and istrain:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]
        self.transform = transform
        self.batch_size = batch_size
    
    def reshuffle(self):
        shuffle_sample, shuffle_gt = reshuffle(self.samples_p, self.gt_p)
        self.samples, self.gt = split_array(shuffle_sample, self.batch_size), split_array(shuffle_gt, self.batch_size)
        if len(self.samples[-1]) != self.batch_size:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]

    def __getitem__(self, index):
        img_tensor, gt_tensor = get_item(self.samples, self.gt, index, self.transform, div=5)
        return img_tensor, gt_tensor, "", 'AIGCIQA3W'

    def __len__(self):
        length = len(self.samples)
        return length

    def get_promt(self, n=10, sample_type='fix'):
        return get_prompt(self.samples_p, self.gt_p, self.transform, n, self.__len__(), sample_type=sample_type)

class AIGCIQA2023(data.Dataset):
    def __init__(self, root, index, transform, batch_size=11, istrain=True):
        mos = scipy.io.loadmat(os.path.join(root, 'DATA', 'MOS', 'mosz1.mat'))
        labels = mos['MOSz'].astype(np.float32)

        sample, gt = [], []
        for i, item in enumerate(index):
            sample.append(os.path.join(root, 'Image', 'allimg', f'{item}.png'))
            gt.append(labels[item][0])
        # gt = normalization(gt)
        gt = list(np.array(gt) / 100)
        self.samples_p, self.gt_p = sample, gt

        self.samples, self.gt = split_array(sample, batch_size), split_array(gt, batch_size)
        if len(self.samples[-1]) != batch_size and istrain:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]
        self.transform = transform
        self.batch_size = batch_size
    
    def reshuffle(self):
        shuffle_sample, shuffle_gt = reshuffle(self.samples_p, self.gt_p)
        self.samples, self.gt = split_array(shuffle_sample, self.batch_size), split_array(shuffle_gt, self.batch_size)
        if len(self.samples[-1]) != self.batch_size:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]

    def __getitem__(self, index):
        img_tensor, gt_tensor = get_item(self.samples, self.gt, index, self.transform, div=5)

        return img_tensor, gt_tensor, "", 'AIGCIQA2023'

    def __len__(self):
        length = len(self.samples)
        return length

    def get_promt(self, n=10, sample_type='fix'):
        return get_prompt(self.samples_p, self.gt_p, self.transform, n, self.__len__(), sample_type=sample_type)


class Koniq10k(data.Dataset):
    def __init__(self, root, index, transform, batch_size=11, istrain=True):
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, 'koniq10k_distributions_sets.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['image_name'])
                mos = np.array(float(row['MOS'])).astype(np.float32)
                mos_all.append(mos)

        sample, gt = [], []
        for i, item in enumerate(index):
            sample.append(os.path.join(root, '1024x768', imgname[item]))
            gt.append(mos_all[item])
        # gt = normalization(gt)
        gt = list(np.array(gt) / 100)

        self.samples_p, self.gt_p = sample, gt

        self.samples, self.gt = split_array(sample, batch_size), split_array(gt, batch_size)
        if len(self.samples[-1]) != batch_size and istrain:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]
        self.transform = transform
        self.batch_size = batch_size
    
    def reshuffle(self):
        shuffle_sample, shuffle_gt = reshuffle(self.samples_p, self.gt_p)
        self.samples, self.gt = split_array(shuffle_sample, self.batch_size), split_array(shuffle_gt, self.batch_size)
        if len(self.samples[-1]) != self.batch_size:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]

    def __getitem__(self, index):
        img_tensor, gt_tensor = get_item(self.samples, self.gt, index, self.transform, div=5)

        return img_tensor, gt_tensor, "", 'koniq10k'

    def __len__(self):
        length = len(self.samples)
        return length

    def get_promt(self, n=10, sample_type='fix'):
        return get_prompt(self.samples_p, self.gt_p, self.transform, n, self.__len__(), sample_type=sample_type)


class uhdiqa(data.Dataset):
    def __init__(self, root, index, transform, batch_size=11, istrain=True):
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, 'uhd-iqa-training-metadata.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['image_name'])
                mos = np.array(float(row['quality_mos'])).astype(np.float32)
                mos_all.append(mos)

        sample, gt = [], []
        for i, item in enumerate(index):
            sample.append(os.path.join(root, 'challenge/training', imgname[item]))
            gt.append(mos_all[item])
        # gt = normalization(gt)

        self.samples_p, self.gt_p = sample, gt

        self.samples, self.gt = split_array(sample, batch_size), split_array(gt, batch_size)
        if len(self.samples[-1]) != batch_size and istrain:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]
        self.transform = transform
        self.batch_size = batch_size
    
    def reshuffle(self):
        shuffle_sample, shuffle_gt = reshuffle(self.samples_p, self.gt_p)
        self.samples, self.gt = split_array(shuffle_sample, self.batch_size), split_array(shuffle_gt, self.batch_size)
        if len(self.samples[-1]) != self.batch_size:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]

    def __getitem__(self, index):
        img_tensor, gt_tensor = get_item(self.samples, self.gt, index, self.transform, div=5)

        return img_tensor, gt_tensor, "", 'uhdiqa'

    def __len__(self):
        length = len(self.samples)
        return length

    def get_promt(self, n=10, sample_type='fix'):
        return get_prompt(self.samples_p, self.gt_p, self.transform, n, self.__len__(), sample_type=sample_type)


class CGIQA6k(data.Dataset):
    def __init__(self, root, index, transform, batch_size=11, istrain=True):
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, 'mos.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['Image'])
                mos = np.array(float(row['MOS'])).astype(np.float32)
                mos_all.append(mos)

        sample, gt = [], []
        for i, item in enumerate(index):
            sample.append(os.path.join(root, 'database', imgname[item]))
            gt.append(mos_all[item])
        gt = normalization(gt)

        self.samples_p, self.gt_p = sample, gt

        self.samples, self.gt = split_array(sample, batch_size), split_array(gt, batch_size)
        if len(self.samples[-1]) != batch_size and istrain:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]
        self.transform = transform
        self.batch_size = batch_size
    
    def reshuffle(self):
        shuffle_sample, shuffle_gt = reshuffle(self.samples_p, self.gt_p)
        self.samples, self.gt = split_array(shuffle_sample, self.batch_size), split_array(shuffle_gt, self.batch_size)
        if len(self.samples[-1]) != self.batch_size:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]

    def __getitem__(self, index):
        img_tensor, gt_tensor = get_item(self.samples, self.gt, index, self.transform, div=5)

        return img_tensor, gt_tensor, "", 'CGIQA6k'

    def __len__(self):
        length = len(self.samples)
        return length

    def get_promt(self, n=10, sample_type='fix'):
        return get_prompt(self.samples_p, self.gt_p, self.transform, n, self.__len__(), sample_type=sample_type)
    
class CSIQ(data.Dataset):
    def __init__(self, root, index, transform, patch_num=1, batch_size=11, istrain=True, dist_type=None):

        refpath = os.path.join(root, 'src_imgs')
        refname = getFileName(refpath, '.png')
        txtpath = os.path.join(root, 'csiq_label.txt')
        fh = open(txtpath, 'r')
        imgnames = []
        target = []
        refnames_all = []
        for line in fh:
            line = line.split('\n')
            words = line[0].split()
            if dist_type is None:
                imgnames.append((words[0]))
                target.append(words[1])
                ref_temp = words[0].split(".")
                refnames_all.append(ref_temp[0] + '.' + ref_temp[-1])
            else:
                if words[0].split('.')[1] == dist_type:
                    imgnames.append((words[0]))
                    target.append(words[1])
                    ref_temp = words[0].split(".")
                    refnames_all.append(ref_temp[0] + '.' + ref_temp[-1])

        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)

        sample = []
        gt = []

        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append(os.path.join(root, 'dst_imgs_all', imgnames[item]))
                    gt.append(labels[item])
        # gt = normalization(gt)
        gt = list(np.array(gt) / 1)
        self.samples_p, self.gt_p = sample, gt

        self.samples, self.gt = split_array(sample, batch_size), split_array(gt, batch_size)
        if len(self.samples[-1]) != batch_size and istrain:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]
        self.transform = transform
        self.batch_size = batch_size
        
        self.dist_type = dist_type
    
    def reshuffle(self):
        shuffle_sample, shuffle_gt = reshuffle(self.samples_p, self.gt_p)
        self.samples, self.gt = split_array(shuffle_sample, self.batch_size), split_array(shuffle_gt, self.batch_size)
        if len(self.samples[-1]) != self.batch_size:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]

    def __getitem__(self, index):
        img_tensor, gt_tensor = get_item(self.samples, self.gt, index, self.transform, div=5)

        if self.dist_type is None:
            return img_tensor, gt_tensor, self.samples[index], 'csiq'
        else:
            return img_tensor, gt_tensor, "", 'csiq_' + self.dist_type

    def __len__(self):
        length = len(self.samples)
        return length

    def get_promt(self, n=10, sample_type='fix'):
        return get_prompt(self.samples_p, self.gt_p, self.transform, n, self.__len__(), sample_type=sample_type)
    
class csiq_other(data.Dataset):
    def __init__(self, root, index, transform, patch_num=1, batch_size=11, istrain=True, dist_type=None, types='SSIM'):

        refpath = os.path.join(root, 'src_imgs')
        refname = getFileName(refpath, '.png')
        txtpath = os.path.join(root, 'csiq_label.txt')
        fh = open(txtpath, 'r')
        imgnames = []
        target = []
        refnames_all = []
        
        idx = {
            'SSIM': 2,
            'FSIM': 3,
            'LPIPS': 4
        } 
        print('Get type ', types)
        
        for line in fh:
            line = line.split('\n')
            words = line[0].split()
            if dist_type is None:
                imgnames.append((words[0]))
                target.append(words[idx[types]])
                ref_temp = words[0].split(".")
                refnames_all.append(ref_temp[0] + '.' + ref_temp[-1])
            else:
                if words[0].split('.')[1] == dist_type:
                    imgnames.append((words[0]))
                    target.append(words[idx[types]])
                    ref_temp = words[0].split(".")
                    refnames_all.append(ref_temp[0] + '.' + ref_temp[-1])

        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)

        sample = []
        gt = []

        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append(os.path.join(root, 'dst_imgs_all', imgnames[item]))
                    gt.append(labels[item])
        # gt = normalization(gt)
        gt = list(np.array(gt) / 1)
        self.samples_p, self.gt_p = sample, gt

        self.samples, self.gt = split_array(sample, batch_size), split_array(gt, batch_size)
        if len(self.samples[-1]) != batch_size and istrain:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]
        self.transform = transform
        self.batch_size = batch_size
        
        self.dist_type = dist_type
    
    def reshuffle(self):
        shuffle_sample, shuffle_gt = reshuffle(self.samples_p, self.gt_p)
        self.samples, self.gt = split_array(shuffle_sample, self.batch_size), split_array(shuffle_gt, self.batch_size)
        if len(self.samples[-1]) != self.batch_size:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]

    def __getitem__(self, index):
        img_tensor, gt_tensor = get_item(self.samples, self.gt, index, self.transform, div=5)

        if self.dist_type is None:
            return img_tensor, gt_tensor, "", 'csiq_other'
        else:
            return img_tensor, gt_tensor, "", 'csiq_' + self.dist_type

    def __len__(self):
        length = len(self.samples)
        return length

    def get_promt(self, n=10, sample_type='fix'):
        return get_prompt(self.samples_p, self.gt_p, self.transform, n, self.__len__(), sample_type=sample_type)


class TID2013Folder_Other_Type(data.Dataset):
    def __init__(self, root, index, transform, batch_size=11, istrain=True, types='SSIM'):
        print('TID Type: ', types)
        print('index', index)
        imgpath = os.path.join(root, 'distorted_images')
        csv_file = os.path.join(root, 'resNEWTest.csv')

        sample, gt = [], []
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (int(row['fileName'].split('_')[0][1:]) - 1) in index:
                    sample.append(os.path.join(imgpath, row['fileName']))
                    mos = np.array(float(row[types])).astype(np.float32)
                    gt.append(mos)
        # gt = normalization(gt)
        gt = list(np.array(gt) / 9)

        self.samples_p, self.gt_p = sample, gt
        print('gt', len(gt))

        self.samples, self.gt = split_array(sample, batch_size), split_array(gt, batch_size)
        if len(self.samples[-1]) != batch_size and istrain:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]
        self.transform = transform
        self.batch_size = batch_size

    def __getitem__(self, index):
        img_tensor, gt_tensor = get_item(self.samples, self.gt, index, self.transform, div=5)

        return img_tensor, gt_tensor, "", 'tid2013_other'

    def __len__(self):
        length = len(self.samples)
        return length

    def get_promt(self, n=10, sample_type='fix'):
        return get_prompt(self.samples_p, self.gt_p, self.transform, n, self.__len__(), sample_type=sample_type)

class TID2013Folder(data.Dataset):
    def __init__(self, root, index, transform, patch_num=1, batch_size=11, istrain=False):
        refpath = os.path.join(root, 'reference_images')
        refname = getTIDFileName(refpath, '.bmp.BMP')
        txtpath = os.path.join(root, 'mos_with_names.txt')
        fh = open(txtpath, 'r')
        imgnames = []
        target = []
        refnames_all = []
        for line in fh:
            line = line.split('\n')
            words = line[0].split()
            imgnames.append((words[1]))
            target.append(words[0])
            ref_temp = words[1].split("_")
            refnames_all.append(ref_temp[0][1:])
        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)

        sample = []
        gt = []
        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append(os.path.join(root, 'distorted_images', imgnames[item]))
                    gt.append(labels[item])
        # gt = normalization(gt)
        gt = list(np.array(gt) / 9)
        self.samples_p, self.gt_p = sample, gt

        self.samples, self.gt = split_array(sample, batch_size), split_array(gt, batch_size)
        if len(self.samples[-1]) != batch_size and istrain:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]
        self.transform = transform
        self.batch_size = batch_size
    
    def reshuffle(self):
        shuffle_sample, shuffle_gt = reshuffle(self.samples_p, self.gt_p)
        self.samples, self.gt = split_array(shuffle_sample, self.batch_size), split_array(shuffle_gt, self.batch_size)
        if len(self.samples[-1]) != self.batch_size:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]

    def __getitem__(self, index):
        img_tensor, gt_tensor = get_item(self.samples, self.gt, index, self.transform, div=5)

        return img_tensor, gt_tensor, "", 'tid2013'

    def __len__(self):
        length = len(self.samples)
        return length

    def get_promt(self, n=10, sample_type='fix'):
        return get_prompt(self.samples_p, self.gt_p, self.transform, n, self.__len__(), sample_type=sample_type)


class KADID(data.Dataset):
    def __init__(self, root, index, transform, batch_size=11, istrain=True):
        imgpath = os.path.join(root, 'images')

        csv_file = os.path.join(root, 'dmos.csv')

        sample, gt = [], []
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (int(row['dist_img'].split('_')[0].replace('I', '')) - 1) in index:
                    sample.append(os.path.join(imgpath, row['dist_img']))
                    mos = np.array(float(row['dmos'])).astype(np.float32)
                    gt.append(mos)
        # gt = normalization(gt)
        gt = list((np.array(gt) - 1) / 5)

        self.samples_p, self.gt_p = sample, gt

        self.samples, self.gt = split_array(sample, batch_size), split_array(gt, batch_size)
        if len(self.samples[-1]) != batch_size and istrain:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]
        self.transform = transform
        self.batch_size = batch_size
    
    def reshuffle(self):
        shuffle_sample, shuffle_gt = reshuffle(self.samples_p, self.gt_p)
        self.samples, self.gt = split_array(shuffle_sample, self.batch_size), split_array(shuffle_gt, self.batch_size)
        if len(self.samples[-1]) != self.batch_size:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]

    def __getitem__(self, index):
        img_tensor, gt_tensor = get_item(self.samples, self.gt, index, self.transform, div=5)

        return img_tensor, gt_tensor, "", 'kadid'

    def __len__(self):
        length = len(self.samples)
        return length

    def get_promt(self, n=10, sample_type='fix'):
        return get_prompt(self.samples_p, self.gt_p, self.transform, n, self.__len__(), sample_type=sample_type)


class KADID_Other(data.Dataset):
    def __init__(self, root, index, transform, batch_size=11, istrain=True, types='SSIM'):
        imgpath = os.path.join(root, 'images')
        csv_file = os.path.join(root, 'dmos.csv')
        
        print("Get Type: ", types)

        sample, gt = [], []
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (int(row['dist_img'].split('_')[0].replace('I', '')) - 1) in index:
                    sample.append(os.path.join(imgpath, row['dist_img']))
                    mos = np.array(float(row[types])).astype(np.float32)
                    gt.append(mos)
        # gt = normalization(gt)
        gt = list((np.array(gt) - 1) / 5)

        self.samples_p, self.gt_p = sample, gt

        self.samples, self.gt = split_array(sample, batch_size), split_array(gt, batch_size)
        if len(self.samples[-1]) != batch_size and istrain:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]
        self.transform = transform
        self.batch_size = batch_size
    
    def reshuffle(self):
        shuffle_sample, shuffle_gt = reshuffle(self.samples_p, self.gt_p)
        self.samples, self.gt = split_array(shuffle_sample, self.batch_size), split_array(shuffle_gt, self.batch_size)
        if len(self.samples[-1]) != self.batch_size:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]

    def __getitem__(self, index):
        img_tensor, gt_tensor = get_item(self.samples, self.gt, index, self.transform, div=5)

        return img_tensor, gt_tensor, "", 'kadid_other'

    def __len__(self):
        length = len(self.samples)
        return length

    def get_promt(self, n=10, sample_type='fix'):
        return get_prompt(self.samples_p, self.gt_p, self.transform, n, self.__len__(), sample_type=sample_type)
    
class TID_Multi_Dim(data.Dataset):
    def __init__(self, root, index, transform, batch_size=11, istrain=True, types='LPIPS'):
        print('TID Type: ', types)
        imgpath = os.path.join(root, 'distorted_images')

        csv_file = os.path.join(root, 'resNEWTest.csv')

        sample, gt = [], []
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (int(row['fileName'].split('_')[0][1:]) - 1) in index:
                    sample.append(os.path.join(imgpath, row['fileName']))
                    mos = np.array(float(row[types])).astype(np.float32)
                    gt.append(mos)
        # gt = normalization(gt)
        gt = list((np.array(gt) - 1) / 5)

        self.samples_p, self.gt_p = sample, gt
        print('gt', len(gt))

        self.samples, self.gt = split_array(sample, batch_size), split_array(gt, batch_size)
        if len(self.samples[-1]) != batch_size and istrain:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]
        self.transform = transform
        self.batch_size = batch_size
    
    def reshuffle(self):
        shuffle_sample, shuffle_gt = reshuffle(self.samples_p, self.gt_p)
        self.samples, self.gt = split_array(shuffle_sample, self.batch_size), split_array(shuffle_gt, self.batch_size)
        if len(self.samples[-1]) != self.batch_size:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]

    def __getitem__(self, index):
        img_tensor, gt_tensor = get_item(self.samples, self.gt, index, self.transform, div=5)

        return img_tensor, gt_tensor, "", 'tid2013_multi_dim'

    def __len__(self):
        length = len(self.samples)
        return length

    def get_promt(self, n=10, sample_type='fix'):
        return get_prompt(self.samples_p, self.gt_p, self.transform, n, self.__len__(), sample_type=sample_type)
    
    
class LIVEFolder(data.Dataset):

    def __init__(self, root, index, transform, batch_size=11, istrain=False):

        refpath = os.path.join(root, 'refimgs')
        refname = getFileName(refpath, '.bmp')

        jp2kroot = os.path.join(root, 'jp2k')
        jp2kname = self.getDistortionTypeFileName(jp2kroot, 227)

        jpegroot = os.path.join(root, 'jpeg')
        jpegname = self.getDistortionTypeFileName(jpegroot, 233)

        wnroot = os.path.join(root, 'wn')
        wnname = self.getDistortionTypeFileName(wnroot, 174)

        gblurroot = os.path.join(root, 'gblur')
        gblurname = self.getDistortionTypeFileName(gblurroot, 174)

        fastfadingroot = os.path.join(root, 'fastfading')
        fastfadingname = self.getDistortionTypeFileName(fastfadingroot, 174)

        imgpath = jp2kname + jpegname + wnname + gblurname + fastfadingname

        dmos = scipy.io.loadmat(os.path.join(root, 'dmos_realigned.mat'))
        labels = dmos['dmos'].astype(np.float32)

        orgs = dmos['orgs']
        refnames_all = scipy.io.loadmat(os.path.join(root, 'refnames_all.mat'))
        refnames_all = refnames_all['refnames_all']

        sample = []
        gt = []

        for i in range(0, len(index)):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = train_sel * ~orgs.astype(np.bool_)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[1].tolist()
            for j, item in enumerate(train_sel):
                sample.append(imgpath[item])
                gt.append(labels[0][item])
                
                # print(self.imgpath[item])
        # gt = normalization(gt)
        gt =list((np.array(gt) - 1) / 100)
        self.samples_p, self.gt_p = sample, gt

        self.samples, self.gt = split_array(sample, batch_size), split_array(gt, batch_size)
        if len(self.samples[-1]) != batch_size and istrain:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]
        self.transform = transform
        self.batch_size = batch_size
    
    def reshuffle(self):
        shuffle_sample, shuffle_gt = reshuffle(self.samples_p, self.gt_p)
        self.samples, self.gt = split_array(shuffle_sample, self.batch_size), split_array(shuffle_gt, self.batch_size)
        if len(self.samples[-1]) != self.batch_size:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]

    def __getitem__(self, index):
        img_tensor, gt_tensor = get_item(self.samples, self.gt, index, self.transform, div=5)

        return img_tensor, gt_tensor, "", 'live'

    def __len__(self):
        length = len(self.samples)
        return length


    def get_promt(self, n=10, sample_type='fix'):
        return get_prompt(self.samples_p, self.gt_p, self.transform, n, self.__len__(), sample_type=sample_type)


    def getDistortionTypeFileName(self, path, num):
        filename = []
        index = 1
        for i in range(0, num):
            name = '%s%s%s' % ('img', str(index), '.bmp')
            filename.append(os.path.join(path, name))
            index = index + 1
        return filename


class FLIVE(data.Dataset):
    def __init__(self, root, index, transform, batch_size=11, istrain=False):
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, 'labels_image.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['name'])
                mos = np.array(float(row['mos'])).astype(np.float32)
                mos_all.append(mos)

        sample, gt = [], []
        for i, item in enumerate(index):
            sample.append(os.path.join(root, imgname[item]))
            gt.append(mos_all[item])
        gt = normalization(gt)

        self.samples_p, self.gt_p = sample, gt

        self.samples, self.gt = split_array(sample, batch_size), split_array(gt, batch_size)
        if len(self.samples[-1]) != batch_size and istrain:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]
        self.transform = transform
        self.batch_size = batch_size
    
    def reshuffle(self):
        shuffle_sample, shuffle_gt = reshuffle(self.samples_p, self.gt_p)
        self.samples, self.gt = split_array(shuffle_sample, self.batch_size), split_array(shuffle_gt, self.batch_size)
        if len(self.samples[-1]) != self.batch_size:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]

    def __getitem__(self, index):
        img_tensor, gt_tensor = get_item(self.samples, self.gt, index, self.transform, div=5)

        return img_tensor, gt_tensor, "", 'flive'

    def __len__(self):
        length = len(self.samples)
        return length

    def get_promt(self, n=10, sample_type='fix'):
        return get_prompt(self.samples_p, self.gt_p, self.transform, n, self.__len__(), sample_type=sample_type)


class SPAQ(data.Dataset):
    def __init__(self, root, index, transform, column=6, batch_size=11, istrain=False):
        sample = []
        gt = []

        xls_file = os.path.join(root, 'Annotations', 'MOS_and_Image_attribute_scores.xlsx')
        workbook = load_workbook(xls_file)
        booksheet = workbook.active
        rows = booksheet.rows
        
        for count, row in enumerate(rows, 2):
            if count - 2 in index:
                sample.append(os.path.join(root, 'img_resize', booksheet.cell(row=count, column=1).value))
                mos = booksheet.cell(row=count, column=column).value
                mos = np.array(mos)
                mos = mos.astype(np.float32)
                gt.append(mos)
            if count == 11126:
                break
        # gt = normalization(gt)
        gt = list(np.array(gt) / 100)
        self.samples_p, self.gt_p = sample, gt
        self.column = column
        # print('get column', self.column)

        self.samples, self.gt = split_array(sample, batch_size), split_array(gt, batch_size)
        if len(self.samples[-1]) != batch_size and istrain:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]
        self.transform = transform
        self.batch_size = batch_size
    
    def reshuffle(self):
        shuffle_sample, shuffle_gt = reshuffle(self.samples_p, self.gt_p)
        self.samples, self.gt = split_array(shuffle_sample, self.batch_size), split_array(shuffle_gt, self.batch_size)
        if len(self.samples[-1]) != self.batch_size:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]

    def __getitem__(self, index):
        img_tensor, gt_tensor = get_item(self.samples, self.gt, index, self.transform, div=5)

        return img_tensor, gt_tensor, "", 'spaq_{}'.format(self.column)

    def __len__(self):
        length = len(self.samples)
        return length

    def get_promt(self, n=10, sample_type='fix'):
        return get_prompt(self.samples_p, self.gt_p, self.transform, n, self.__len__(), sample_type=sample_type)
    
class UWIQA(data.Dataset):
    def __init__(self, root, index, transform, batch_size=11, istrain=False):
        sample = []
        gt = []

        xls_file = os.path.join(root, 'IQA_Value.xlsx')
        workbook = load_workbook(xls_file)
        booksheet = workbook.active
        rows = booksheet.rows
        
        for count, row in enumerate(rows, 2):
            if count - 2 in index:
                sample.append(os.path.join(root, 'img', '{}.png'.format(str(booksheet.cell(row=count, column=1).value).zfill(4))))
                mos = booksheet.cell(row=count, column=2).value
                mos = np.array(mos)
                mos = mos.astype(np.float32)
                gt.append(mos)

        # gt = normalization(gt)
        gt = list(np.array(gt) / 1)
        self.samples_p, self.gt_p = sample, gt

        self.samples, self.gt = split_array(sample, batch_size), split_array(gt, batch_size)
        if len(self.samples[-1]) != batch_size and istrain:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]
        self.transform = transform
        self.batch_size = batch_size
    
    def reshuffle(self):
        shuffle_sample, shuffle_gt = reshuffle(self.samples_p, self.gt_p)
        self.samples, self.gt = split_array(shuffle_sample, self.batch_size), split_array(shuffle_gt, self.batch_size)
        if len(self.samples[-1]) != self.batch_size:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]

    def __getitem__(self, index):
        img_tensor, gt_tensor = get_item(self.samples, self.gt, index, self.transform, div=5)

        return img_tensor, gt_tensor, self.samples[index], 'UWIQA'

    def __len__(self):
        length = len(self.samples)
        return length

    def get_promt(self, n=10, sample_type='fix'):
        return get_prompt(self.samples_p, self.gt_p, self.transform, n, self.__len__(), sample_type=sample_type)


def split_array(arr, m):
    return [arr[i:i + m] for i in range(0, len(arr), m)]


class BID(data.Dataset):
    def __init__(self, root, index, transform, batch_size=11, istrain=False):

        imgname = []
        mos_all = []

        xls_file = os.path.join(root, 'DatabaseGrades.xlsx')
        workbook = load_workbook(xls_file)
        booksheet = workbook.active
        rows = booksheet.rows
        count = 1
        for row in rows:
            count += 1
            img_num = booksheet.cell(row=count, column=1).value
            img_name = "DatabaseImage%04d.JPG" % (img_num)
            imgname.append(img_name)
            mos = booksheet.cell(row=count, column=2).value
            mos = np.array(mos)
            mos = mos.astype(np.float32)
            mos_all.append(mos)
            if count == 587:
                break

        sample, gt = [], []
        for i, item in enumerate(index):
            sample.append(os.path.join(root, imgname[item]))
            gt.append(mos_all[item])
        # gt = normalization(gt)
        gt = list(np.array(gt) / 9)
        self.samples_p, self.gt_p = sample, gt

        self.samples, self.gt = split_array(sample, batch_size), split_array(gt, batch_size)
        if len(self.samples[-1]) != batch_size and istrain:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]
        self.transform = transform
        self.batch_size = batch_size
    
    def reshuffle(self):
        shuffle_sample, shuffle_gt = reshuffle(self.samples_p, self.gt_p)
        self.samples, self.gt = split_array(shuffle_sample, self.batch_size), split_array(shuffle_gt, self.batch_size)
        if len(self.samples[-1]) != self.batch_size:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]

    def __getitem__(self, index):
        img_tensor, gt_tensor = get_item(self.samples, self.gt, index, self.transform, div=5)

        return img_tensor, gt_tensor, "", 'bid'

    def __len__(self):
        length = len(self.samples)
        return length

    def get_promt(self, n=10, sample_type='fix'):
        return get_prompt(self.samples_p, self.gt_p, self.transform, n, self.__len__(), sample_type=sample_type)


class PIQ2023(data.Dataset):
    def __init__(self, root, index, transform, batch_size=11, istrain=True):
        #  index is not used.
        imgname = []
        mos_all = []
        img_path = os.path.join(root, 'img')
        if istrain:
            csv_file = os.path.join(root, 'Test split', 'Device Split', 'DeviceSplit_Train_Scores_Exposure.csv')
        else:
            csv_file = os.path.join(root, 'Test split', 'Device Split', 'DeviceSplit_Test_Scores_Exposure.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['IMAGE PATH'].replace('\\', '/'))
                mos = np.array(float(row['JOD'])).astype(np.float32)
                mos_all.append(mos)

        sample, gt = [], []
        for i, item in enumerate(range(len(imgname))):
            sample.append(os.path.join(img_path, imgname[item]))
            gt.append(mos_all[item])
        gt = normalization(gt)

        self.samples_p, self.gt_p = sample, gt

        self.samples, self.gt = split_array(sample, batch_size), split_array(gt, batch_size)
        if len(self.samples[-1]) != batch_size and istrain:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]
        self.transform = transform
        self.batch_size = batch_size
    
    def reshuffle(self):
        shuffle_sample, shuffle_gt = reshuffle(self.samples_p, self.gt_p)
        self.samples, self.gt = split_array(shuffle_sample, self.batch_size), split_array(shuffle_gt, self.batch_size)
        if len(self.samples[-1]) != self.batch_size:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]

    def __getitem__(self, index):
        img_tensor, gt_tensor = get_item(self.samples, self.gt, index, self.transform, div=5)

        return img_tensor, gt_tensor, "", 'PIQ2023'

    def __len__(self):
        length = len(self.samples)
        return length

    def get_promt(self, n=10, sample_type='fix'):
        return get_prompt(self.samples_p, self.gt_p, self.transform, n, self.__len__(), sample_type=sample_type)

class GFIQA_20k(data.Dataset):
    def __init__(self, root, index, transform, batch_size=11, istrain=True):
        imgname = []
        mos_all = []
        img_path = os.path.join(root, 'image')
        csv_file = os.path.join(root, 'mos.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['img_name'])
                mos = np.array(float(row['mos'])).astype(np.float32)
                mos_all.append(mos)

        sample, gt = [], []
        for i, item in enumerate(index):
            sample.append(os.path.join(img_path, imgname[item]))
            gt.append(mos_all[item])
        # gt = normalization(gt)
        gt = list(np.array(gt) / 1)

        self.samples_p, self.gt_p = sample, gt

        self.samples, self.gt = split_array(sample, batch_size), split_array(gt, batch_size)
        if len(self.samples[-1]) != batch_size and istrain:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]
        self.transform = transform
        self.batch_size = batch_size
    
    def reshuffle(self):
        shuffle_sample, shuffle_gt = reshuffle(self.samples_p, self.gt_p)
        self.samples, self.gt = split_array(shuffle_sample, self.batch_size), split_array(shuffle_gt, self.batch_size)
        if len(self.samples[-1]) != self.batch_size:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]

    def __getitem__(self, index):
        img_tensor, gt_tensor = get_item(self.samples, self.gt, index, self.transform, div=5)

        return img_tensor, gt_tensor, "", 'GFIQA_20k'

    def __len__(self):
        length = len(self.samples)
        return length

    def get_promt(self, n=10, sample_type='fix'):
        return get_prompt(self.samples_p, self.gt_p, self.transform, n, self.__len__(), sample_type=sample_type)

class AGIQA_3k(data.Dataset):
    def __init__(self, root, index, transform, batch_size=11, istrain=True):
        imgname = []
        mos_all = []
        img_path = os.path.join(root, 'img')
        csv_file = os.path.join(root, 'data.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['name'])
                mos = np.array(float(row['mos_quality'])).astype(np.float32)
                mos_all.append(mos)

        sample, gt = [], []
        for i, item in enumerate(index):
            sample.append(os.path.join(img_path, imgname[item]))
            gt.append(mos_all[item])
        # gt = normalization(gt)
        gt = list(np.array(gt) / 5)

        self.samples_p, self.gt_p = sample, gt

        self.samples, self.gt = split_array(sample, batch_size), split_array(gt, batch_size)
        if len(self.samples[-1]) != batch_size and istrain:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]
        self.transform = transform
        self.batch_size = batch_size
    
    def reshuffle(self):
        shuffle_sample, shuffle_gt = reshuffle(self.samples_p, self.gt_p)
        self.samples, self.gt = split_array(shuffle_sample, self.batch_size), split_array(shuffle_gt, self.batch_size)
        if len(self.samples[-1]) != self.batch_size:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]

    def __getitem__(self, index):
        img_tensor, gt_tensor = get_item(self.samples, self.gt, index, self.transform, div=5)

        return img_tensor, gt_tensor, self.samples[index], 'AGIQA_3k'

    def __len__(self):
        length = len(self.samples)
        return length

    def get_promt(self, n=10, sample_type='fix'):
        return get_prompt(self.samples_p, self.gt_p, self.transform, n, self.__len__(), sample_type=sample_type)


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
