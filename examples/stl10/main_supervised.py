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
import numpy as np
import copy
import json

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
    parser.add_argument('--exp_file', type=str, default=None, help='labels file')
    parser.add_argument('--align_w', type=float, default=1, help='Alignment loss weight')
    parser.add_argument('--unif_w', type=float, default=1, help='Uniformity loss weight')
    parser.add_argument('--align_alpha', type=float, default=2, help='alpha in alignment loss')
    parser.add_argument('--unif_t', type=float, default=2, help='t in uniformity loss')
    

    parser.add_argument('--batch_size', type=int, default=768, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--iter', type=int, default=0, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate. Default is linear scaling 0.12 per 256 batch size')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='Learning rate decay rate')
    parser.add_argument('--lr_decay_epochs', default=[155,170,185], nargs='*', type=int,
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

    opt = parser.parse_args()

    if opt.lr is None:
        opt.lr = 0.12 * (opt.batch_size / 256)

    opt.gpus = list(map(lambda x: torch.device('cuda', x), opt.gpus))

    opt.save_folder = os.path.join(
        opt.result_folder,
        f"cifar100_series_coarse_supervised_{opt.iter}_{opt.epochs}" if exp.file is None else f"cifar100_series_coarse_supervised_{opt.iter}_{opt.epochs}"
    )
    os.makedirs(opt.save_folder, exist_ok=True)

    return opt


def get_data_loader(opt, lblmap):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(32, scale=(0.08, 1)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
            (0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
        ),
    ])
    dataset = TwoAugUnsupervisedDatasetLbl(
        torchvision.datasets.CIFAR100(opt.data_folder, 'train', download=True), 
        transform=transform, 
        lblmap=lblmap )
    
    return torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_workers,
                                       shuffle=True, pin_memory=True)

def get_data_loader_sup(opt, lblmap):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(70),
        torchvision.transforms.CenterCrop(32),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
            (0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
        ),
    ])
    dataset = TwoAugUnsupervisedDatasetLbl(
        torchvision.datasets.CIFAR100(opt.data_folder, 'train', download=True), 
        transform=transform, 
        lblmap=lblmap )
    
    return torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_workers,
                                       shuffle=True, pin_memory=True)

import itertools


def main():
    opt = parse_option()
    
    if opt.exp_file is None:
        transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(32, scale=(0.08, 1)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
                    (0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
                ),
            ])

        arr =np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                                       3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                       6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                                       0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                                       5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                                       16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                                       10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                                       2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                                      16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                                      18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
        old_lbls = list(range(100))
        old2new = {i:arr[i] for i in range(len(arr))}
        new_lbls = list(np.unique(arr))
        count = len(new_lbls)

    else:
        print(f"Loading experiment file {opt.exp_file}") 
        transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(32, scale=(0.08, 1)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
                    (0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
                ),
            ])

        old_lbls = list(range(100))
        
        with open(opt.exp_file,"r") as f:
            exp_file_data = json.load(f)
        
        labels_2_keep = exp_file_data["labels"]
        
        print(f"Labels to keep:{labels_2_keep}")
        
        old2new = {}
        count = 0
        for old_lbl in old_lbls:
            if old_lbl in labels_2_keep: 
                old2new[old_lbl] = count
                count += 1

        for old_lbl in old_lbls:
            if old_lbl not in labels_2_keep: 
                old2new[old_lbl] = count

        new_lbls = list(range(count+1))
    
    print(f'Optimize: {opt.align_w:g} * loss_align(alpha={opt.align_alpha:g}) + {opt.unif_w:g} * loss_uniform(t={opt.unif_t:g})')

    torch.cuda.set_device(opt.gpus[0])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    encoder = SmallAlexNet(feat_dim=opt.feat_dim, cifar=True).to(opt.gpus[0])

    n_classes = 100 if True else 10
    classifier = torch.nn.Linear(opt.feat_dim, n_classes).to(opt.gpus[0])
    optim = torch.optim.Adam( itertools.chain(classifier.parameters(), encoder.parameters()), lr=1e-2 )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, gamma=opt.lr_decay_rate, milestones=opt.lr_decay_epochs)

    loader = get_data_loader_sup(opt, old2new)

    loss_meter = AverageMeter('total_loss')
    it_time_meter = AverageMeter('iter_time')
    for epoch in range(opt.epochs):
        loss_meter.reset()
        it_time_meter.reset()
        t0 = time.time()
        for ii, (im_x, im_y, lbl) in enumerate(loader):
            optim.zero_grad()

            x = encoder(im_x.to(opt.gpus[0]))
            # group according to new_lbls
            loss = torch.nn.functional.cross_entropy( classifier(x), lbl.to(opt.gpus[0]))
            loss_meter.update(loss, x.shape[0])
            loss.backward()
            optim.step()
            it_time_meter.update(time.time() - t0)

            if ii % opt.log_interval == 0:
                print(f"Epoch {epoch}/{opt.epochs}\tIt {ii}/{len(loader)}\t" +
                      f"{loss_meter}\t{it_time_meter}")
            t0 = time.time()

        scheduler.step()
    ckpt_file = os.path.join(opt.save_folder, 'encoder.pth')
    torch.save(encoder.state_dict(), ckpt_file)
    print(f'Saved to {ckpt_file}')


if __name__ == '__main__':
    main()
