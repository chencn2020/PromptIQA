import sys
import argparse
import builtins
import os
import random
import shutil
import time
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import ConcatDataset

from PromptIQA.utils import log_writer
from PromptIQA.utils.dataset import data_loader
from PromptIQA.utils.toolkit import *
from PromptIQA.models import promptiqa

import warnings
warnings.filterwarnings('ignore')

loger_path = None

def init(config):
    global loger_path
    if config.dist_url == "env://" and config.world_size == -1:
        config.world_size = int(os.environ["WORLD_SIZE"])

    config.distributed = config.world_size > 1 or config.multiprocessing_distributed

    print("config.distributed", config.distributed)

    loger_path = os.path.join(config.save_path, "log")
    if not os.path.isdir(loger_path):
        os.makedirs(loger_path)
    sys.stdout = log_writer.Logger(os.path.join(loger_path, "training_logs.log"))
    print("All train and test data will be saved in: ", config.save_path)
    print("----------------------------------")
    print(
        "Begin Time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    )
    printArgs(config, loger_path)
    setup_seed(config.seed)

    # Save the traning files.
    file_backup = os.path.join(config.save_path, "training_files")
    if not os.path.isdir(file_backup):
        os.makedirs(file_backup)
    shutil.copy(
        os.path.basename(__file__),
        os.path.join(file_backup, os.path.basename(__file__)),
    )
    
    shutil.copy(
        os.path.basename('train.sh'),
        os.path.join(file_backup, 'train.sh'),
    )

    save_folder_list = ["PromptIQA"]
    for save_folder in save_folder_list:
        save_folder_path = os.path.join(file_backup, save_folder)
        if os.path.exists(save_folder_path):
            shutil.rmtree(save_folder_path)
        shutil.copytree(save_folder, save_folder_path)

def main(config):
    init(config)
    ngpus_per_node = torch.cuda.device_count()
    if config.multiprocessing_distributed:
        config.world_size = ngpus_per_node * config.world_size

        print(config.world_size, ngpus_per_node, ngpus_per_node)
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        # Simply call main_worker function
        main_worker(config.gpu, ngpus_per_node, config)

    print("End Time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

def main_worker(gpu, ngpus_per_node, args):
    
    if gpu == 0:
        loger_path = os.path.join(args.save_path, "log")
        sys.stdout = log_writer.Logger(os.path.join(loger_path, "training_logs_GPU0.log")) # The print info will be saved here
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    seed = args.seed + args.rank * 100
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print('seed', seed, 'rank', args.rank)

    print('Take Model: ', args.model)
    if args.model == 'promptiqa':
        model = promptiqa.PromptIQA()
    else:
        raise NotImplementedError('Only PromptIQA')

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
            print("Model Distribute.")
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    
    criterion = nn.L1Loss().cuda(args.gpu)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    prompt_num = args.batch_size - 1 # the number of prompts is batch_size / num_gpus - 1
    print('prompt_num', prompt_num)

    train_data_list, train_prompt_list, test_data_list = [], {}, [] # train_prompt_list save the ISPP from different datasets
    train_ori_data = []
    for dataset in args.dataset: # loading the datasets
        print('---Load ', dataset)
        path, train_index, test_index = get_data(dataset=dataset, split_seed=args.seed)
        
        train_dataset = data_loader.Data_Loader(args.batch_size, dataset, path, train_index, istrain=True)
        train_ori_data.append(train_dataset)
        train_data_list.append(train_dataset.get_samples()) # get the data_folder
        train_prompt_list[dataset] = train_dataset.get_prompt(prompt_num, 'fix') # The ISPP for testing is sampled from training data and is fixed.

        test_dataset = data_loader.Data_Loader(args.batch_size, dataset, path, test_index, istrain=False)
        test_data_list.append(test_dataset.get_samples())
            
    print('train_prompt_list', train_prompt_list.keys())
    combined_train_samples = ConcatDataset(train_data_list) # combine the training and testing dataset
    combined_test_samples = ConcatDataset(test_data_list)

    print("train_dataset", len(combined_train_samples))
    print("test_dataset", len(combined_test_samples))

    train_sampler = torch.utils.data.distributed.DistributedSampler(combined_train_samples)
    test_sampler = torch.utils.data.distributed.DistributedSampler(combined_test_samples)

    train_loader = torch.utils.data.DataLoader(
        combined_train_samples,
        batch_size=1, # please keep the bs to 1. More details about the bs can be found in ```folders.py```
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        drop_last=True,
        sampler=train_sampler,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        combined_test_samples,
        batch_size=1,
        shuffle=(test_sampler is None),
        num_workers=args.workers,
        sampler=test_sampler,
        drop_last=False,
        pin_memory=True,
    )

    best_srocc, best_plcc = 0, 0
    weight = {}
    for data in train_prompt_list.keys(): # the loss weight for different datasets
        weight[data] = 1
        
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        print('Weight: ', weight)
        pred_scores, gt_scores = train(train_loader, model, criterion, optimizer, args, epoch, weight)
        
        # gather all the results from all gpus
        gt_scores = gather_together(gt_scores)  
        gt_scores = [item for sublist in gt_scores for item in sublist]
        pred_scores = gather_together(pred_scores) 
        pred_scores = [item for sublist in pred_scores for item in sublist]
        train_srocc, train_plcc = cal_srocc_plcc(pred_scores, gt_scores)

        print(
            "Train SROCC: {}, Train PLCC: {}".format(
                round(train_srocc, 4), round(train_plcc, 4)
            )
        )

        print('reshuffle data.') # reorganize the training batch
        for d_re in train_data_list:
            d_re.reshuffle()

        pred_scores, gt_scores, path = test(
            test_loader, model, train_prompt_list
        )
        print('Summary---')

        # gather all the testing results from all gpus
        gt_scores = gather_together(gt_scores)
        pred_scores = gather_together(pred_scores) 

        gt_score_dict, pred_score_dict = {}, {}
        for sublist in gt_scores:
            for k, v in sublist.items():
                if k not in gt_score_dict:
                    gt_score_dict[k] = v
                else:
                    gt_score_dict[k] = gt_score_dict[k] + v
        
        for sublist in pred_scores:
            for k, v in sublist.items():
                if k not in pred_score_dict:
                    pred_score_dict[k] = v
                else:
                    pred_score_dict[k] = pred_score_dict[k] + v

        gt_score_dict = dict(sorted(gt_score_dict.items()))
        test_srocc, test_plcc = 0, 0
        for k, v in gt_score_dict.items():
            test_srocc_, test_plcc_ = cal_srocc_plcc(gt_score_dict[k], pred_score_dict[k])
            print('\tDataset: {} Test SROCC: {}, PLCC: {}'.format(k, round(test_srocc_, 4), round(test_plcc_, 4)))
            test_srocc += test_srocc_
            test_plcc += test_plcc_
            
        print('test_srocc + test_plcc / 2', (test_srocc + test_plcc) / 2)
        
        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            if test_srocc + test_plcc > best_srocc + best_plcc:
                best_srocc, best_plcc = test_srocc, test_plcc
                save_checkpoint(
                    {
                        "state_dict": model.state_dict(),
                        "prompt_num":prompt_num,
                    },
                    is_best=True,
                    filename=os.path.join(args.save_path, f'best_model_{epoch + 1}.pth.tar'),
                )
                print("Best Model Saved.")

    print('Best SROCC: {}, PLCC: {}'.format(best_srocc, best_plcc))

def test(test_loader, model, promt_data_loader, reverse=False):
    """Training"""
    pred_scores = {}
    gt_scores = {}
    path = []

    batch_time = AverageMeter("Time", ":6.3f")
    srocc = AverageMeter("SROCC", ":6.2f")
    plcc = AverageMeter("PLCC", ":6.2f")
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, srocc, plcc],
        prefix="Testing ",
    )

    model.train(False)
    with torch.no_grad():
        for index, (img_or, label_or, paths, dataset_type) in enumerate(test_loader):
            dataset_type = dataset_type[0]
            prompt_dataset = promt_data_loader[dataset_type]
            t = time.time()
            
            has_prompt = False
            if hasattr(model.module, 'check_prompt'):
                has_prompt =  model.module.check_prompt(dataset_type)
            
            if not has_prompt:
                for img, label in prompt_dataset:
                    img = img.squeeze(0).cuda()
                    label = label.squeeze(0).cuda()
                    if reverse == 2:
                        label = torch.rand_like(label[:, -1]).cuda()
                    else:
                        label = label[:, -1].cuda() if not reverse else (1 - label[:, -1].cuda())
                    model.module.forward_prompt(img, label.reshape(-1, 1), dataset_type)

            img = img_or.squeeze(0).cuda()
            label = label_or.squeeze(0).cuda()[:, 2]

            pred = model.module.inference(img, dataset_type)

            if dataset_type not in pred_scores:
                pred_scores[dataset_type] = []

            if dataset_type not in gt_scores:
                gt_scores[dataset_type] = []

            pred_scores[dataset_type] = pred_scores[dataset_type] + pred.cpu().tolist()
            gt_scores[dataset_type] = gt_scores[dataset_type] + label.cpu().tolist()
            path = path + list(paths)

            batch_time.update(time.time() - t)

            if index % 100 == 0:
                for k, v in pred_scores.items():
                    test_srocc, test_plcc = cal_srocc_plcc(pred_scores[k], gt_scores[k])
                srocc.update(test_srocc)
                plcc.update(test_plcc)

                progress.display(index)

    model.module.clear()
    model.train(True)
    return pred_scores, gt_scores, path


def train(train_loader, model, loss_fun, optimizer, args, epoch, weight):
    """Training"""
    print("----------------------------------")
    print("Epoch\tTrain_Loss\tTrain_SROCC\tTrain_PLCC\tTest_SROCC\tTest_PLCC")
    epoch_loss = []
    pred_scores = []
    gt_scores = []

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch),
    )

    end = time.time()

    random_flipping_rate = args.random_flipping_rate
    random_scale_rate = args.random_scale_rate
    
    for index, (img, label, _, data_name) in enumerate(train_loader):
        img = img.squeeze(0)
        label = label.squeeze(0)

        data_time.update(time.time() - end)

        optimizer.zero_grad()

        random_scale_ = random.uniform(0, 1)
        if random_scale_ < random_scale_rate: # random scale
            scale = random.uniform(float(torch.max(label[:, -1], dim=-1).values), 1)
            label[:, -1] = label[:, -1] / scale

        random_flipping_ = random.uniform(0, 1)
        if random_flipping_ < random_flipping_rate: # random reverse
            label[:, -1] = 1 - label[:, -1]
            
        assert (label[:, -1] >= 0).all(), "{}, {}".format(data_name, label[:, -1])
        assert (label[:, -1] <= 1).all(), "{}, {}".format(data_name, label[:, -1])

        pred, label_new = model(img, label[:, -1].reshape(-1, 1))

        loss = loss_fun(pred.squeeze(), label_new.float().detach())
        loss = loss * weight[data_name[0]]
        epoch_loss.append(loss.item())

        losses.update(loss.item(), img.size(0))

        loss.backward()
        optimizer.step()
        
        if random_scale_ < random_scale_rate: # random scale
            pred = pred * scale
            label_new = label_new * scale

        if random_flipping_ < random_flipping_rate:
            pred_scores = pred_scores + (1 - pred).cpu().tolist()
            gt_scores = gt_scores + (1 - label_new).cpu().tolist()
        else:
            pred_scores = pred_scores + pred.cpu().tolist()
            gt_scores = gt_scores + label_new.cpu().tolist()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if index % 100 == 0:
            progress.display(index)

    return pred_scores, gt_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=570908,
        help="Random seeds for result reproduction.",
    )

    # data related
    parser.add_argument(
        "--dataset",
        dest="dataset",
        nargs='+', default=None
    )

    parser.add_argument(
        "--lr", dest="lr", type=float, default=1e-5, help="Learning random_flipping_rate"
    )

    parser.add_argument(
        "--weight_decay",
        dest="weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay",
    )
    parser.add_argument(
        "--batch_size", dest="batch_size", type=int, default=44, help="Batch size"
    )
    parser.add_argument(
        "--epochs", dest="epochs", type=int, default=50, help="Epochs for training"
    )
    parser.add_argument(
        "--T_max",
        dest="T_max",
        type=int,
        default=50,
        help="Hyper-parameter for CosineAnnealingLR",
    )
    parser.add_argument(
        "--eta_min",
        dest="eta_min",
        type=int,
        default=0,
        help="Hyper-parameter for CosineAnnealingLR",
    )

    parser.add_argument(
        "-j",
        "--workers",
        default=32,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 32)",
    )

    parser.add_argument(
        "--world-size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://224.66.41.62:23456",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="Use multi-processing distributed training to launch "
             "N processes per node, which has N GPUs. This is the "
             "fastest way to use PyTorch for either single node or "
             "multi node data parallel training",
    )

    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument("--random_flipping_rate", default=0.1, type=float)
    parser.add_argument("--random_scale_rate", default=0.5, type=float)
    parser.add_argument("--model", default='promptiqa', type=str)
    parser.add_argument("--save_path", dest="save_path", type=str, default="./save_logs/Matrix_Comparation_Koniq_bs_25", help="The path where the model and logs will be saved.")

    config = parser.parse_args()

    main(config)
