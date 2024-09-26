import torch
import numpy as np


class Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, sample):
        if isinstance(sample, dict):
            img = sample['img']
            gt = sample['gt']
            img = (img - self.mean) / self.var
            sample = {'img': img, 'gt': gt}
        else:
            sample = (sample - self.mean) / self.var

        return sample



class RandHorizontalFlip(object):
    def __init__(self, prob_aug):
        self.prob_aug = prob_aug

    def __call__(self, sample):
        p_aug = np.array([self.prob_aug, 1 - self.prob_aug])
        prob_lr = np.random.choice([1, 0], p=p_aug.ravel())

        if isinstance(sample, dict):
            img = sample['img']
            gt = sample['gt']
            
            if prob_lr > 0.5:
                img = np.fliplr(img).copy()
            sample = {'img': img, 'gt': gt}
        else:
            if prob_lr > 0.5:
                sample = np.fliplr(sample).copy()
        return sample


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        if isinstance(sample, dict):
            img = sample['img']
            gt = np.array(sample['gt'])
            img = torch.from_numpy(img).type(torch.FloatTensor)
            gt = torch.from_numpy(gt).type(torch.FloatTensor)
            sample = {'img': img, 'gt': gt}
        else:
            sample = torch.from_numpy(sample).type(torch.FloatTensor)
        return sample
