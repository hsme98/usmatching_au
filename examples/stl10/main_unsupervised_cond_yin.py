import os
import time
import argparse

import torchvision
import torch
import torch.nn as nn

from util import AverageMeter
from encoder import SmallerPredAlexNet
from align_uniform import align_loss, uniform_loss_prelog
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import copy

import matplotlib.pyplot as plt


class TwoAugUnsupervisedDatasetLbl(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""

    def __init__(self, dataset, transform, lblmap=None):
        self.dataset = dataset
        # load the images on GPU
        self.dataset= dataset
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

    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--epochs', type=int, default=400, help='Number of training epochs')
    parser.add_argument('--iter', type=int, default=0, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate. Default is linear scaling 0.12 per 256 batch size')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='Learning rate decay rate')
    parser.add_argument('--lr_decay_epochs', default=[155,175,185, 255, 300, 355], nargs='*', type=int,
                        help='When to decay learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 weight decay')
    parser.add_argument('--feat_dim', type=int, default=32, help='Feature dimensionality')

    parser.add_argument('--num_workers', type=int, default=40, help='Number of data loader workers to use')
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
        f"cifar100_unsupervisedcond_yin_newlbls_smllr_{opt.feat_dim}_{opt.iter}_{opt.epochs}"
    )
    os.makedirs(opt.save_folder, exist_ok=True)

    return opt


opt = parse_option()

opt.gpus[0]

"""
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

"""

arr =[ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13]

classes_2_keep = [18,19,3, 5, 6]

labels_2_keep = [idx for idx, el in enumerate(arr) if el in classes_2_keep] 

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
# labels_2_keep = list(range(25))
# labels_2_keep = [0,1,2,3]

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


def get_data_loader(opt):
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
        lblmap=old2new )
    
    return torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_workers,
                                       shuffle=True, pin_memory=True)

print(f'Optimize: {opt.align_w:g} * loss_align(alpha={opt.align_alpha:g}) + {opt.unif_w:g} * loss_uniform(t={opt.unif_t:g})')

torch.cuda.set_device(opt.gpus[0])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

encoder = SmallerPredAlexNet(len(new_lbls),feat_dim=opt.feat_dim).to(opt.gpus[0])

optim = torch.optim.Adam(encoder.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, gamma=opt.lr_decay_rate,
                                                 milestones=opt.lr_decay_epochs)

loader = get_data_loader(opt)
align_meter = AverageMeter('align_loss')
unif_meter = AverageMeter('uniform_loss')
loss_meter = AverageMeter('total_loss')
it_time_meter = AverageMeter('iter_time')

for epoch in range(opt.epochs):
    align_meter.reset()
    unif_meter.reset()
    loss_meter.reset()
    it_time_meter.reset()
    t0 = time.time()
    for ii, (im_x, im_y, lbl) in enumerate(loader):
        optim.zero_grad()
        lbl_onehot = torch.nn.functional.one_hot(lbl, num_classes=len(new_lbls)).type(torch.FloatTensor).to(opt.gpus[0])
        
        x, y = encoder(torch.cat([im_x.to(opt.gpus[0]), im_y.to(opt.gpus[0])]), torch.cat([lbl_onehot, lbl_onehot])).chunk(2)
        
        align_loss_val = align_loss(x, y, alpha=opt.align_alpha)
        # group according to new_lbls

        z = torch.cat( [x, y])
        lbl_z = torch.cat([lbl, lbl])
        unif_losses = torch.cat([uniform_loss_prelog(z[lbl_z==new_lbl]) for new_lbl in new_lbls])
        unif_loss_val = torch.log( torch.mean(unif_losses) )
        
        loss = align_loss_val * opt.align_w + unif_loss_val * opt.unif_w
        align_meter.update(align_loss_val, x.shape[0])
        unif_meter.update(unif_loss_val)
        loss_meter.update(loss, x.shape[0])
        loss.backward()
        optim.step()
        it_time_meter.update(time.time() - t0)
        if ii % opt.log_interval == 0:
            print(f"Epoch {epoch}/{opt.epochs}\tIt {ii}/{len(loader)}\t" +
                  f"{align_meter}\t{unif_meter}\t{loss_meter}\t{it_time_meter}")
        t0 = time.time()
    scheduler.step()

ckpt_file = os.path.join(opt.save_folder, 'encoder.pth')
torch.save(encoder.state_dict(), ckpt_file)
print(f'Saved to {ckpt_file}')