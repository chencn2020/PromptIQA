import torch
from scipy import stats
import numpy as np
from models import monet as MoNet
from models import gc_loss as GC_Loss
from utils.dataset import data_loader
import json
import random
import os
from tqdm import tqdm


def get_data(dataset, data_path='./utils/dataset/dataset_info.json'):
    '''
        Load dataset information from the json file.
    '''
    with open(data_path, 'r') as data_info:
        data_info = json.load(data_info)
    path, img_num = data_info[dataset]
    img_num = list(range(img_num))

    # Random choose 80% for traning and 20% for testing.
    random.shuffle(img_num)
    train_index = img_num[0:int(round(0.8 * len(img_num)))]
    test_index = img_num[int(round(0.8 * len(img_num))):len(img_num)]

    return path, train_index, test_index


def cal_srocc_plcc(pred_score, gt_score):
    srocc, _ = stats.spearmanr(pred_score, gt_score)
    plcc, _ = stats.pearsonr(pred_score, gt_score)

    return srocc, plcc


class Solver:
    def __init__(self, config):
        
        path, train_index, test_index = get_data(dataset=config.dataset)

        train_loader = data_loader.Data_Loader(config, path, train_index, istrain=True)
        test_loader = data_loader.Data_Loader(config, path, test_index, istrain=False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

        self.promt_data_loader = train_loader.get_prompt()
        # for i, j in self.promt_data_loader:
        #     print(j)
        print('Traning data number: ', len(train_index))
        print('Testing data number: ', len(test_index))
        
        if config.loss == 'MAE':
            self.loss = torch.nn.L1Loss().cuda()
        elif config.loss == 'MSE':
            self.loss = torch.nn.MSELoss().cuda()
        elif config.loss == 'GC':
            self.loss = GC_Loss.GC_Loss(queue_len=int(len(train_index) * config.queue_ratio))
        else:
            raise 'Only Support MAE, MSE and GC loss.'

        print('Loading MoNet...')
        self.MoNet = MoNet.MoNet().cuda()
        self.MoNet.train(True)

        self.epochs = config.epochs
        self.optimizer = torch.optim.Adam(self.MoNet.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.T_max, eta_min=config.eta_min)

        self.model_save_path = os.path.join(config.save_path, 'best_model.pkl')

    def train(self):
        """Training"""
        best_srocc = 0.0
        best_plcc = 0.0
        print('----------------------------------')
        print('Epoch\tTrain_Loss\tTrain_SROCC\tTrain_PLCC\tTest_SROCC\tTest_PLCC')
        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []

            for index, (img, label) in enumerate(tqdm(self.train_data)):
                img = img.cuda()
                label = label.cuda()

                # last_item = label[:, -1]
                # sorted_last_item, indices = torch.sort(last_item, descending=False)
                # img = img[indices].cuda()
                # label = label[indices].cuda()

                self.optimizer.zero_grad()
                pred, label = self.MoNet(img, label)

                pred_scores = pred_scores + pred.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()

                loss = self.loss(pred.squeeze(), label.float().detach())
                epoch_loss.append(loss.item())
                
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

            train_srocc, train_plcc = cal_srocc_plcc(pred_scores, gt_scores)
            test_srocc, test_plcc = self.test()
            if test_srocc + test_plcc > best_srocc + best_plcc:
                best_srocc = test_srocc
                best_plcc = test_plcc
                torch.save(self.MoNet.state_dict(), self.model_save_path)
                print('Model saved in: ', self.model_save_path)

            print('{}\t{}\t{}\t{}\t{}\t{}'.format(t + 1, round(np.mean(epoch_loss), 4), round(train_srocc, 4),
                                                  round(train_plcc, 4), round(test_srocc, 4), round(test_plcc, 4)))

        print('Best test SROCC {}, PLCC {}'.format(round(best_srocc, 6), round(best_plcc, 6)))

        return best_srocc, best_plcc

    def test(self):
        """Testing"""
        self.MoNet.train(False)
        pred_scores, gt_scores = [], []

        with torch.no_grad():
            for img, label in self.promt_data_loader:
                img = img.cuda()
                label = label.cuda()

                # print(img.shape)
                self.MoNet.forward_prompt(img, label)

            for img, label in tqdm(self.test_data):
                img = img.cuda()
                label = label.cuda()[:, 2]

                pred = self.MoNet.inference(img)

                pred_scores = pred_scores + pred.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()

        test_srocc, test_plcc = cal_srocc_plcc(pred_scores, gt_scores)
        self.MoNet.train(True)
        return test_srocc, test_plcc
