import torch.utils.data as data
from PromptIQA.utils.dataset.loader_tools import *

class PromptIQADataset(data.Dataset):
    def __init__(self, dataset_name):
        super().__init__()
        self.dataset_name = dataset_name
        
    def reshuffle(self):
        shuffle_sample, shuffle_gt = reshuffle(self.samples_p, self.gt_p)
        self.samples, self.gt = split_array(shuffle_sample, self.batch_size), split_array(shuffle_gt, self.batch_size)
        if len(self.samples[-1]) != self.batch_size:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]
    
    def __getitem__(self, index):
        img_tensor, gt_tensor = get_item(self.samples, self.gt, index, self.transform, div=5)
        return img_tensor, gt_tensor, self.samples[index], self.dataset_name

    def __len__(self):
        length = len(self.samples)
        return length

    def get_promt(self, n=10, sample_type='fix'):
        return get_prompt(self.samples_p, self.gt_p, self.transform, n, self.__len__(), sample_type=sample_type)
    