import torch
import torchvision

from utils.dataset import folders
from utils.dataset.process import ToTensor, Normalize, RandHorizontalFlip

class Data_Loader():
    """Dataset class for IQA databases"""

    def __init__(self, batch_size, dataset, path, img_indx, istrain=True, column=2, dist_type=None, types='SSIM'):

        self.batch_size = batch_size
        self.istrain = istrain
        # print('1. column --------', column, dataset)

        if istrain:
            transforms=torchvision.transforms.Compose([Normalize(0.5, 0.5), RandHorizontalFlip(prob_aug=0.5), ToTensor()])
        else:
            transforms=torchvision.transforms.Compose([Normalize(0.5, 0.5), ToTensor()])

        if dataset == 'livec':
            self.data = folders.LIVEC(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain)
        elif dataset == 'koniq10k':
            self.data = folders.Koniq10k(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain)
        elif dataset == 'uhdiqa':
            self.data = folders.uhdiqa(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain)
        elif dataset == 'bid':
            self.data = folders.BID(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain)
        elif dataset == 'spaq':
            self.data = folders.SPAQ(root=path, index=img_indx, transform=transforms, column=column, batch_size=batch_size, istrain=istrain)
        elif dataset == 'flive':
            self.data = folders.FLIVE(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain)
        elif dataset == 'csiq':
            self.data = folders.CSIQ(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain, dist_type=dist_type)
        elif dataset == 'csiq_other':
            self.data = folders.csiq_other(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain, dist_type=dist_type, types=types)
        elif dataset == 'live':
            self.data = folders.LIVEFolder(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain)
        elif dataset == 'tid2013':
            self.data = folders.TID2013Folder(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain)
        elif dataset == 'tid2013_other':
            self.data = folders.TID2013Folder_Other_Type(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain, types=types)
        elif dataset == 'kadid':
            self.data = folders.KADID(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain)
        elif dataset == 'kadid_other':
            self.data = folders.KADID_Other(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain, types=types)
        elif dataset == 'PIQ2023':
            self.data = folders.PIQ2023(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain)
        elif dataset == 'GFIQA_20k':
            self.data = folders.GFIQA_20k(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain)
        elif dataset == 'AGIQA_3k':
            self.data = folders.AGIQA_3k(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain)
        elif dataset == 'AIGCIQA2023':
            self.data = folders.AIGCIQA2023(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain)
        elif dataset == 'UWIQA':
            self.data = folders.UWIQA(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain)
        elif dataset == 'CGIQA6k':
            self.data = folders.CGIQA6k(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain)
        elif dataset == 'AIGCIQA3W':
            self.data = folders.AIGCIQA3W(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain)
        elif dataset == 'tid2013_multi_dim':
            self.data = folders.TID_Multi_Dim(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain)
        else:
            raise Exception("Only support livec, koniq10k, bid, spaq.")

    def get_data(self):
        dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=self.istrain, num_workers=16, drop_last=self.istrain)
        return dataloader
    
    def get_samples(self):
        return self.data
        
    def get_prompt(self, n=5, sample_type='fix'):
        # print('Get {} images for prompting.'.format(n))
        prompt_data = self.data.get_promt(n=n, sample_type=sample_type)
        return torch.utils.data.DataLoader(prompt_data, batch_size=prompt_data.__len__(), shuffle=False)