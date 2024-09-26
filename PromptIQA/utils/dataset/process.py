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

import numpy as np

class TwentyCrop(object):
    def __init__(self, crop_height=224, crop_width=224):
        self.crop_height = crop_height
        self.crop_width = crop_width

    def __call__(self, sample):
        n, m = sample.shape[1], sample.shape[2] # 获取图像的高度和宽度
        h_stride = (n - self.crop_height) // 4 # 计算垂直方向上的步长
        w_stride = (m - self.crop_width) // 4 # 计算水平方向上的步长

        crops = []
        for h_step in range(5):
            for w_step in range(4):
                # 计算每个裁剪的起始点
                h_start = h_step * h_stride
                w_start = w_step * w_stride
                # 裁剪图像
                crop = sample[:, h_start:h_start+self.crop_height, w_start:w_start+self.crop_width]
                crops.append(crop)

        # 将裁剪的列表转换为numpy数组
        crops = np.stack(crops) # 这将创建一个形状为[20, 3, crop_height, crop_width]的数组
        return crops
    
class FiveCrop(object):
    def __init__(self, size=224):
        self.size = size  # 裁剪图片的尺寸

    def __call__(self, sample):
        # 确保输入的sample是期望的格式
        if isinstance(sample, np.ndarray) and sample.shape[0] == 3:
            c, h, w = sample.shape
            crop_h, crop_w = self.size, self.size

            # 计算裁剪的起始点
            tl = sample[:, 0:crop_h, 0:crop_w]  # 左上角
            tr = sample[:, 0:crop_h, w - crop_w:]  # 右上角
            bl = sample[:, h - crop_h:, 0:crop_w]  # 左下角
            br = sample[:, h - crop_h:, w - crop_w:]  # 右下角
            center = sample[:, h//2 - crop_h//2:h//2 + crop_h//2, w//2 - crop_w//2:w//2 + crop_w//2]  # 中心

            # 将五个裁剪合并到一个numpy数组中
            crops = np.stack([tl, tr, bl, br, center])
            return crops
        else:
            raise ValueError("输入的sample不是期望的格式或尺寸。")

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
