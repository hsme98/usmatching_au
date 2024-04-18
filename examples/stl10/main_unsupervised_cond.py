import os
import time
import argparse
import datetime

import torchvision
import torch
import torch.nn as nn

from util import AverageMeter, prepare_imagenet
from encoder import SmallAlexNet
from align_uniform import align_loss,uniform_loss, uniform_loss_prelog
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import copy
import json
import math
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
    parser.add_argument('--dataset', type=str, default=None, help='dataset to train cifar100, imagenet')

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
    
    parser.add_argument('--folds', type=int, default=1, help="number of folds for cross validation")

    parser.add_argument('--data_folder', type=str, default='./data', help='Path to data')
    parser.add_argument('--result_folder', type=str, default='./results', help='Base directory to save model')
    
    opt = parser.parse_args()

    if opt.lr is None:
        opt.lr = 0.12 * (opt.batch_size / 256)

    opt.gpus = list(map(lambda x: torch.device('cuda', x), opt.gpus))
    
    opt.save_folder = os.path.join(
        opt.result_folder,
        f"experiment_{int(datetime.datetime.now().timestamp() * 1000 )}"
    )
    os.makedirs(opt.save_folder)
    
    # Convert opt to a dictionary and save it as JSON
    opt_dict = vars(opt)
    json_path = os.path.join(opt.save_folder, 'options.json')
    with open(json_path, 'w') as f:
        json.dump(opt_dict, f, indent=4)
    
    print(f"Results will be saved under {opt.save_folder}")
     
    return opt




def get_datasets(opt, lblmap):
    
    if opt.dataset == "cifar100":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(32, scale=(0.08, 1)), # make this 0.2 later
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
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(64, scale=(0.08, 1)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            torchvision.transforms.RandomGrayscale(p=0.2),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
                (0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
            ),
        ])


        imagenet_train, _ = prepare_imagenet(opt)
        
        dataset = TwoAugUnsupervisedDatasetLbl(
            imagent_train,
            transform=transform, 
            lblmap=lblmap )
    
    
    # generate folds based on the dataset object and return the split as the 
    if opt.folds < 2:
        return [(dataset,None)]
    else:
        # calculate the indices of each fold and the validation
        fold_size = len(dataset) // opt.folds
        val_folds = [list(range(i * fold_size + i, (i+1) * fold_size)) for i in range(opt.folds-1)]
        
        full_folds = [( set(range(len(dataset))).difference(val_fold) , val_fold ) for val_fold in val_folds]
        
        return [(torch.utils.data.Subset(dataset, fold),torch.utils.data.Subset(dataset, val_fold)) for fold, val_fold in full_folds]


def calc_loss(encoder , im_x, im_y, lbl, opt):
    x, y = encoder(torch.cat([im_x.to(opt.gpus[0]), im_y.to(opt.gpus[0])])).chunk(2)

    align_loss_val = align_loss(x, y, alpha=opt.align_alpha)
    
    if opt.exp_file is None:
        unif_loss_val =  (uniform_loss(x, t=opt.unif_t) + uniform_loss(y, t=opt.unif_t)) / 2
    else:
         z = torch.cat( [x, y])
        lbl_z = torch.cat([lbl, lbl])
        unif_losses = torch.cat([uniform_loss_prelog(z[lbl_z==new_lbl], t=opt.unif_t) for new_lbl in new_lbls])
        unif_loss_val = torch.log( torch.mean(unif_losses) )

    loss =  align_loss_val * opt.align_w + unif_loss_val * opt.unif_w


def main():
    opt = parse_option()
    assert(opt.dataset in ["cifar100", "imagenet"]) 
    old_lbls = list(range(100)) if opt.dataset == "cifar100" else list(range(200))
    
    if opt.exp_file is None:
        labels_2_keep = old_lbls
        old2new = None
        new_lbls = None
    else:
        print(f"Loading experiment file {opt.exp_file}") 
        
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
    
    for fold_idx, (dataset, dataset_val) in enumerate(folds):
        encoder = SmallAlexNet(feat_dim=opt.feat_dim, inp_size=32 if opt.dataset=="cifar100" else 64).to(opt.gpus[0])

        optim = torch.optim.Adam(encoder.parameters(), lr=opt.lr)
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, 
                                                         gamma=opt.lr_decay_rate,
                                                         milestones=opt.lr_decay_epochs)
        
        loader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=opt.batch_size, 
                                             num_workers=opt.num_workers,
                                             shuffle=True,
                                             pin_memory=True)
        
        loader_val = torch.utils.data.DataLoader(dataset_val, 
                                                 batch_size=opt.batch_size, 
                                                 num_workers=opt.num_workers,
                                                 shuffle=False, 
                                                 pin_memory=True)
        loss_meter = AverageMeter('total_loss')
        val_loss_meter = AverageMeter('val_loss')
        best_val_loss = math.inf
        best_encoder = None

        for epoch in range(opt.epochs):
            encoder.train()
            loss_meter.reset()
            for ii, (im_x, im_y, lbl) in enumerate(loader):
                optim.zero_grad()
                loss = calc_loss(encoder, im_x, im_y, lbl, opt)
                loss_meter.update(loss, len(lbl))
                loss.backward()
                optim.step()
                
                if ii % opt.log_interval == 0:
                    print(f"Epoch {epoch}/{opt.epochs}\tIt {ii}/{len(loader)}\t" +
                          f"\t{loss_meter}")

            scheduler.step()
            
            
            if dataset_val is not None:
                encoder.eval()
                val_loss_meter.reset() 
                # calculate the validation accuracy
                with torch.no_grad():
                    for im_x, im_y, lbl in enumerate(loader_val):
                        optim.zero_grad()
                        loss = calc_loss(encoder, im_x, im_y, lbl, opt)
                        val_loss_meter.update(loss, lbl.shape[0])

                if val_loss.avg < best_val_loss:
                    best_val_loss = val_loss.avg
                    best_encoder = copy.deepcopy(encoder)

                print(f"Validation: Epoch {epoch}/{opt.epochs}\t{val_loss.avg}\t{best_val_loss}")
        
        """
            save the best encoder and the last encoder
        """
        torch.save(encoder.state_dict(), os.path.join(opt.save_folder, f'encoder_{fold_idx}.pth'))
        
        if dataset_val is not None:
            torch.save(best_encoder.state_dict(), os.path.join(opt.save_folder, f'best_encoder_{fold_idx}.pth'))

            with open(os.path.join(opt.save_folder, f"folds_{fold_idx}.txt"),"w") as f:
                f.write(f"{best_val_loss}\n")

if __name__ == '__main__':
    main()
