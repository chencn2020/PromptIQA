import torch
import torchvision

from utils.dataset_test import folders
from utils.dataset_test.process import ToTensor, Normalize, RandHorizontalFlip

class Data_Loader(object):
    """Dataset class for IQA databases"""

    def __init__(self, config, path, img_indx, istrain=True, column=2):

        self.batch_size = config.batch_size
        self.istrain = istrain
        dataset = config.dataset

        if istrain:
            transforms=torchvision.transforms.Compose([Normalize(0.5, 0.5), RandHorizontalFlip(prob_aug=0.5), ToTensor()])
        else:
            transforms=torchvision.transforms.Compose([Normalize(0.5, 0.5), ToTensor()])

        if dataset == 'livec':
            self.data = folders.LIVEC(root=path, index=img_indx, transform=transforms)
        elif dataset == 'koniq10k':
            self.data = folders.Koniq10k(root=path, index=img_indx, transform=transforms)
        elif dataset == 'bid':
            self.data = folders.BID(root=path, index=img_indx, transform=transforms)
        elif dataset == 'spaq':
            self.data = folders.SPAQ(root=path, index=img_indx, transform=transforms, column=column)
        elif dataset == 'flive':
            self.data = folders.FLIVE(root=path, index=img_indx, transform=transforms)
        elif dataset == 'csiq':
            self.data = folders.CSIQ(root=path, index=img_indx, transform=transforms)
        elif dataset == 'live':
            self.data = folders.LIVEFolder(root=path, index=img_indx, transform=transforms)
        elif dataset == 'tid2013':
            self.data = folders.TID2013Folder(root=path, index=img_indx, transform=transforms)
        elif dataset == 'kadid':
            self.data = folders.KADID(root=path, index=img_indx, transform=transforms)
        else:
            raise Exception("Only support livec, koniq10k, bid, spaq.")

    def get_data(self):
        dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=self.istrain, num_workers=16, drop_last=self.istrain)
        return dataloader
    
    def get_samples(self):
        return self.data
        
    def get_prompt(self, n=5):
        print('Get {} images for prompting.'.format(n))
        prompt_data = self.data.get_promt(n=n)
        return torch.utils.data.DataLoader(prompt_data, batch_size=prompt_data.__len__(), shuffle=False)