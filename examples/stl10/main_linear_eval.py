"""
    Here we  do the linear evaluation, the old labels are provided to the linear objective as one hot
"""
import time
import argparse

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Subset
import os

from util import AverageMeter, prepare_imagenet
from encoder import SmallAlexNet
import json
import copy
from tqdm import tqdm
import numpy as np

def parse_option():
    parser = argparse.ArgumentParser('STL-10 Representation Learning with Alignment and Uniformity Losses')
    parser.add_argument('exp_fold', type=str, help='path to the experiment folder')
    parser.add_argument('--label_file', type=str, help='label file in case it is unsupervised learning')
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
    parser.add_argument('--folds', type=str, default=1, help="number of folds")
    parser.add_argument('--train_split', type=float, default=0.8, help='split between the validation and training set')

    opt = parser.parse_args()


    if opt.lr is None:
        opt.lr = 0.12 * (opt.batch_size / 256)

    if opt.gpus[0] != "cpu":
        opt.gpu = torch.device('cuda', opt.gpus[0])
    opt.lr_decay_epochs = list(map(int, opt.lr_decay_epochs.split(',')))

    return opt

    
    
class DatasetModifiedLblandLbl(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""

    def __init__(self, dataset, lblmap, transform=None):
        self.dataset = dataset
        self.lblmap = copy.deepcopy(lblmap)
        self.transform = transform

    def __getitem__(self, index):
        image, lbl = self.dataset[index]
        image = self.transform(image) if self.transform is not None else image
        return image, self.lblmap[lbl], lbl

    def __len__(self):
        return len(self.dataset)

def get_datasets(opt,lblmap, opt_exp):
    if opt_exp["dataset"] == "cifar100":
        val_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(35),
            torchvision.transforms.CenterCrop(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
                (0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
            ),
        ])


        dataset = DatasetModifiedLblandLbl( torchvision.datasets.CIFAR100(opt.data_folder, 'test', transform=val_transform), lblmap=lblmap)
    elif opt_exp["dataset"] == "imagenet":
        val_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(70),
            torchvision.transforms.CenterCrop(64),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
                (0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
            ),
        ])

        _, imagenet_val = prepare_imagenet(opt)

        dataset = DatasetModifiedLblandLbl(imagenet_val,transform=val_transform,lblmap=lblmap)
    else:
        raise ValueError

    if opt.folds == 1:
        perm = np.random.permutation(len(dataset))
        train_split, val_split = perm[:int(len(dataset) * opt.train_split)], perm[int(len(dataset) * opt.train_split):]
        train_dataset, val_dataset = Subset(dataset, train_split), Subset(dataset, val_split)
        datasets = [(train_dataset, val_dataset)]
    else:
        # here we do the validation logic as we did earlier
        fold_size = len(dataset) // opt.folds
        val_folds = [list(range(i * fold_size + i, (i + 1) * fold_size)) for i in range(opt.folds - 1)]
        full_folds = [(list(set(range(len(dataset))).difference(val_fold)), val_fold) for val_fold in val_folds]
        datasets = [(torch.utils.data.Subset(dataset, fold), torch.utils.data.Subset(dataset, val_fold)) for fold, val_fold
                in full_folds]
    return datasets


def validate_comb(opt, encoder, classifier, val_loader, num_classes):
    correct = 0
    with torch.no_grad():
        for images, labels_mod, labels_act in val_loader:
            pred = classifier(torch.cat( (encoder(images.to(opt.gpus[0]), layer_index=opt.layer_index), torch.nn.functional.one_hot(labels_mod.to(opt.gpus[0]), num_classes=num_classes)), dim=1) ).argmax(dim=1)
            correct += (pred.cpu() == labels_act).sum().item()
    return correct / len(val_loader.dataset)


if __name__ == "__main__":
    opt=parse_option()
    opt.gpu=opt.gpus[0]


    with open(os.path.join(opt.exp_fold,"options.json"),"r") as f:
        exp_opt = json.load(f)

    if exp_opt["exp_file"] is None:
        assert(opt.label_file is not None)
        label_file = opt.label_file
    else:
        label_file = exp_opt["exp_file"]

    with open(label_file,"r") as f:
        lbl_map = json.load(f)["label_map"]

    # create the old2new map
    if opt.gpus[0] != "cpu":
        torch.cuda.set_device(opt.gpus[0])
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    encoder= SmallAlexNet(feat_dim=opt.feat_dim, inp_size=32 if exp_opt["dataset"]=="cifar100" else 64).to(opt.gpus[0])
    encoder.eval()
    folds = get_datasets(opt, lbl_map)

    for fold_idx, (train_dataset, val_dataset) in folds:
        print(f"Starting fold {fold_idx}/{len(folds)}")
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=opt.batch_size,
                                                   num_workers=opt.num_workers,
                                                   shuffle=True,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=opt.batch_size,
                                                 num_workers=opt.num_workers,
                                                 shuffle=False,
                                                 pin_memory=True)

        if fold_idx == 0:
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

        classifier =  nn.Linear( eval_numel + max(lbl_map.values()), max(lbl_map.keys())).to(opt.gpus[0])

        optim = torch.optim.Adam(classifier.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,
                                                         gamma=opt.lr_decay_rate,
                                                         milestones=opt.lr_decay_epochs)

        val_accs = []
        loss_meter = AverageMeter('loss')
        for epoch in tqdm(range(opt.epochs)):
            loss_meter.reset()
            t0 = time.time()
            for ii, (images, labels_mod, labels) in enumerate(train_loader):
                optim.zero_grad()
                labels_mod_onehot = torch.nn.functional.one_hot(labels_mod, num_classes=max(lbl_map.keys())).to(opt.gpus[0])
                with torch.no_grad():
                    feats = encoder(images.to(opt.gpus[0]), layer_index=opt.layer_index).flatten(1)

                logits = classifier(torch.cat((feats, labels_mod_onehot), dim=1))
                loss = F.cross_entropy(logits, labels.to(opt.gpus[0]))
                loss_meter.update(loss, images.shape[0])
                loss.backward()
                optim.step()

                if ii % opt.log_interval == 0:
                    print(f"Epoch {epoch}/{opt.epochs}\tIt {ii}/{len(train_loader)}\t{loss_meter}")
            scheduler.step()

            val_acc = validate_comb(opt,encoder,classifier,val_loader, max(lbl_map.values()))
            val_accs.append(val_acc)

            print(f"Epoch {epoch}/{opt.epochs}\tval_acc {val_acc*100:.4g}%")
        print(f"Best validation accuracy {max(val_accs)}")

        save_path = os.path.join(os.path.dirname(opt.encoder_checkpoint),f"result_{fold_idx}_{os.path.basename(opt.encoder_checkpoint)}_{opt.iter}.json")
        with open(save_path,"w") as f:
            json.dump({"lbl_map":lbl_map,"params": classifier.state_dict(),"fold_idx":fold_idx,"val_accs":val_accs},f)
    
    