import os
import time
import argparse

import torchvision
import torch
import torch.nn as nn

from util import AverageMeter
from encoder import SmallAlexNet
from align_uniform import align_loss, uniform_loss_prelog
from tqdm import tqdm
from collections import defaultdict
import copy

import matplotlib.pyplot as plt


class TwoAugUnsupervisedDatasetLbl(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""

    def __init__(self, dataset, transform, lblmap=None):
        self.dataset = dataset
        self.transform = transform
        self.lblmap = copy.deepcopy(lblmap)

    def __getitem__(self, index):
        image, lbl = self.dataset[index]
        lbl2return = lbl if self.lblmap is None else self.lblmap[lbl]
        return self.transform(image), self.transform(image), lbl2return

    def __len__(self):
        return len(self.dataset)

def parse_option():
    parser = argparse.ArgumentParser('STL-10 Representation Learning with Alignment and Uniformity Losses')

    parser.add_argument('--align_w', type=float, default=1, help='Alignment loss weight')
    parser.add_argument('--unif_w', type=float, default=1, help='Uniformity loss weight')
    parser.add_argument('--align_alpha', type=float, default=2, help='alpha in alignment loss')
    parser.add_argument('--unif_t', type=float, default=2, help='t in uniformity loss')

    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=400, help='Number of training epochs')
    parser.add_argument('--iter', type=int, default=0, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate. Default is linear scaling 0.12 per 256 batch size')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='Learning rate decay rate')
    parser.add_argument('--lr_decay_epochs', default=[155, 170, 185], nargs='*', type=int,
                        help='When to decay learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 weight decay')
    parser.add_argument('--feat_dim', type=int, default=128, help='Feature dimensionality')

    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers to use')
    parser.add_argument('--log_interval', type=int, default=40, help='Number of iterations between logs')
    parser.add_argument('--gpus', default=[0], nargs='*', type=int,
                        help='List of GPU indices to use, e.g., --gpus 0 1 2 3')

    parser.add_argument('--data_folder', type=str, default='./data', help='Path to data')
    parser.add_argument('--result_folder', type=str, default='./results', help='Base directory to save model')

    opt = parser.parse_args("")

    if opt.lr is None:
        opt.lr = 0.12 * (opt.batch_size / 256)

    opt.gpus = list(map(lambda x: torch.device('cuda', x), opt.gpus))

    opt.save_folder = os.path.join(
        opt.result_folder,
        f"cifar100_{opt.epochs}_4_sideinformation_align{opt.align_w:g}alpha{opt.align_alpha:g}_unif{opt.unif_w:g}t{opt.unif_t:g}_iter{opt.iter}"
    )
    os.makedirs(opt.save_folder, exist_ok=True)

    return opt




if __name__ == "__main__":
    opt = parse_option()

