import math
import os
import random
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
from scipy import stats
import json

@torch.no_grad()
def gather_together(data):
    dist.barrier()
    world_size = dist.get_world_size()
    gather_data = [None for _ in range(world_size)]
    dist.all_gather_object(gather_data, data)
    return gather_data

def printArgs(args, savePath):
    with open(os.path.join(savePath, "args_info.log"), "w") as f:
        print("--------------args----------------")
        f.write("--------------args----------------\n")
        for arg in vars(args):
            print(
                format(arg, "<20"), format(str(getattr(args, arg)), "<")
            )  # str, arg_type
            f.write(
                "{}\t{}\n".format(
                    format(arg, "<20"), format(str(getattr(args, arg)), "<")
                )
            )  # str, arg_type

        print("----------------------------------")
        f.write("----------------------------------")


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    warnings.warn(
        "You have chosen to seed training. "
        "This will turn on the CUDNN deterministic setting, "
        "which can slow down your training considerably! "
        "You may see unexpected behavior when restarting "
        "from checkpoints."
    )


def get_data(dataset, split_seed, data_path='./PromptIQA/utils/dataset/dataset_info.json'):
    """
        Load dataset information from the json file.
    """
    with open(data_path, "r") as data_info:
        data_info = json.load(data_info)
    path, img_num = data_info[dataset]
    img_num = list(range(img_num))

    random.seed(split_seed)
    random.shuffle(img_num)

    train_index = img_num[0: int(round(0.8 * len(img_num)))]
    test_index = img_num[int(round(0.8 * len(img_num))): len(img_num)]

    print('Split_seed', split_seed)
    print('train_index', train_index[:10], len(train_index))
    print('test_index', test_index[:10], len(test_index))

    return path, train_index, test_index


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    # if is_best:
    #     shutil.copyfile(filename, "model_best.pth.tar")


def cal_srocc_plcc(pred_score, gt_score):
    try:
        srocc, _ = stats.spearmanr(pred_score, gt_score)
        plcc, _ = stats.pearsonr(pred_score, gt_score)
    except:
        srocc, plcc = 0, 0

    return srocc, plcc


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
