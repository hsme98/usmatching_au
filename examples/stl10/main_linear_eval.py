"""
    Here we  do the linear evaluation, the old labels are provided to the linear objective as one hot
"""
import time
import argparse

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from util import AverageMeter
from encoder import SmallAlexNet
import json
import copy
from tqdm import tqdm
import numpy as np

def parse_option():
    parser = argparse.ArgumentParser('STL-10 Representation Learning with Alignment and Uniformity Losses')
    parser.add_argument('encoder_checkpoint', type=str, help='Encoder checkpoint to evaluate')
    parser.add_argument('--feat_dim', type=int, default=128, help='Encoder feature dimensionality')
    parser.add_argument('--layer_index', type=int, default=-2, help='Evaluation layer')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='Learning rate decay rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80', help='When to decay learning rate')
    
    parser.add_argument("--iter", type=int, default=0, help="iteration number")
    parser.add_argument('--num_workers', type=int, default=6, help='Number of data loader workers to use')
    parser.add_argument('--log_interval', type=int, default=40, help='Number of iterations between logs')
    parser.add_argument('--gpus', default=[0], nargs='*', type=int,
                        help='List of GPU indices to use, e.g., --gpus 0 1 2 3')

    parser.add_argument('--data_folder', type=str, default='./data', help='Path to data')

    opt = parser.parse_args()

    if opt.lr is None:
        opt.lr = 0.12 * (opt.batch_size / 256)

    if opt.gpus[0] != "cpu":
        opt.gpu = torch.device('cuda', opt.gpus[0])
    opt.lr_decay_epochs = list(map(int, opt.lr_decay_epochs.split(',')))

    return opt


class DatasetModifiedLbl(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""

    def __init__(self, dataset, lblmap=None):
        self.dataset = dataset
        self.lblmap = copy.deepcopy(lblmap)

    def __getitem__(self, index):
        image, lbl = self.dataset[index]
        lbl2return = lbl if self.lblmap is None else self.lblmap[lbl]
        return image, lbl2return

    def __len__(self):
        return len(self.dataset)
    
    
class DatasetModifiedLblandLbl(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""

    def __init__(self, dataset, lblmap):
        self.dataset = dataset
        self.lblmap = copy.deepcopy(lblmap)

    def __getitem__(self, index):
        image, lbl = self.dataset[index]
        return image, self.lblmap[lbl], lbl

    def __len__(self):
        return len(self.dataset)

class DatasetSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, lbl2keep):
        self.lbl_map = {lbl:lbl_idx for lbl_idx, lbl in enumerate(lbl2keep)}
        self.dataset = [(datum,self.lbl_map[lbl]) for datum,lbl in dataset if lbl in lbl2keep]
    def __getitem__(self, index):
        return self.dataset[index]
    def __len__(self):
        return len(self.dataset)

def get_data_loaders(opt,lblmap):
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(32, scale=(0.08, 1)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
            (0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
        ),
        torchvision.transforms.RandomHorizontalFlip()
    ])
    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(35),
        torchvision.transforms.CenterCrop(32),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
            (0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
        ),
    ])
    
    train_dataset = DatasetModifiedLblandLbl( torchvision.datasets.CIFAR100(opt.data_folder, 'train', download=True, transform=train_transform), lblmap=lblmap )
    val_dataset =  DatasetModifiedLblandLbl( torchvision.datasets.CIFAR100(opt.data_folder, 'test', transform=val_transform), lblmap=lblmap)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
                                               num_workers=opt.num_workers, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size,
                                             num_workers=opt.num_workers, pin_memory=True)
    return train_loader, val_loader


def validate_comb(opt, encoder, classifier, val_loader):
    correct = 0
    with torch.no_grad():
        for images, labels_mod, labels_act in val_loader:
            pred = classifier(torch.cat( (encoder(images.to(opt.gpus[0]), layer_index=opt.layer_index), torch.nn.functional.one_hot(labels_mod.to(opt.gpus[0]), num_classes=len(new_lbls))), dim=1)).argmax(dim=1)
            correct += (pred.cpu() == labels_act).sum().item()
    return correct / len(val_loader.dataset)

def validate(opt, encoder, classifier, val_loader):
    correct = 0
    with torch.no_grad():
        for images, labels_act in val_loader:
            pred = classifier( encoder(images.to(opt.gpus[0]), layer_index=opt.layer_index).flatten(1) ).argmax(dim=1)
            correct += (pred.cpu() == labels_act).sum().item()
    return correct / len(val_loader.dataset)


if __name__ == "__main__":
    
    """
    old_lbls = list(range(100))
    labels_2_keep = list(range(25))
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
    
    
    
    opt=parse_option()
    opt.gpu=opt.gpus[0]
    if opt.gpus[0] != "cpu":
        torch.cuda.set_device(opt.gpus[0])
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    encoder= SmallAlexNet(feat_dim=opt.feat_dim, cifar=True).to(opt.gpus[0])
    encoder.eval()
    train_loader, val_loader = get_data_loaders(opt, old2new)
    
    with torch.no_grad():
        sample, _ = train_loader.dataset.dataset[0]
        eval_numel = encoder(sample.unsqueeze(0).to(opt.gpus[0]), layer_index=opt.layer_index).numel()
    print(f'Feature dimension: {eval_numel}')

    MODEL_2_LOAD = opt.encoder_checkpoint
    print(f"loading {MODEL_2_LOAD}")
    if opt.gpus[0] == "cpu":
        encoder.load_state_dict(torch.load(MODEL_2_LOAD, map_location=torch.device("cpu")))
    else:
        encoder.load_state_dict(torch.load(MODEL_2_LOAD))
    encoder = encoder.to(opt.gpus[0])


    classifier =  nn.Linear( eval_numel + len(new_lbls), len(old_lbls)).to(opt.gpus[0])

    optim = torch.optim.Adam(classifier.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, 
                                                     gamma=opt.lr_decay_rate,
                                                     milestones=opt.lr_decay_epochs)

    val_accs = []
    loss_meter = AverageMeter('loss')
    it_time_meter = AverageMeter('iter_time')
    for epoch in tqdm(range(opt.epochs)):
        loss_meter.reset()
        it_time_meter.reset()
        t0 = time.time()
        for ii, (images, labels_mod, labels) in enumerate(train_loader):
            optim.zero_grad()
            labels_mod_onehot = torch.nn.functional.one_hot(labels_mod, num_classes=len(new_lbls)).to(opt.gpus[0])
            with torch.no_grad():
                feats = encoder(images.to(opt.gpus[0]), layer_index=opt.layer_index).flatten(1)


            logits = classifier(torch.cat((feats, labels_mod_onehot), dim=1))
            loss = F.cross_entropy(logits, labels.to(opt.gpus[0]))
            loss_meter.update(loss, images.shape[0])
            loss.backward()
            optim.step()
            it_time_meter.update(time.time() - t0)
            
            if ii % opt.log_interval == 0:
                print(f"Epoch {epoch}/{opt.epochs}\tIt {ii}/{len(train_loader)}\t{loss_meter}\t{it_time_meter}")
            t0 = time.time()
        scheduler.step()
        
        val_acc = validate_comb(opt,encoder,classifier,val_loader)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch}/{opt.epochs}\tval_acc {val_acc*100:.4g}%")
    print(f"Best validation accuracy {max(val_accs)}")
        
    save_path = os.path.join(os.path.dirname(opt.encoder_checkpoint),f"result_newlbl_{os.path.basename(opt.encoder_checkpoint)}_{opt.iter}.json") 
    with open(save_path,"w") as f:
        json.dump({"val_accs":val_accs},f)
    
    