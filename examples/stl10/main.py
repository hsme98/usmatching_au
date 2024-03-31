import itertools
import os
import time
import argparse
import json
from datetime import datetime
from argparse import Namespace

import torchvision
import torch
import torch.nn as nn

from tqdm import tqdm
from util import AverageMeter, TwoAugUnsupervisedDatasetLbl
from encoder import SmallAlexNet
from align_uniform import align_loss, uniform_loss, uniform_loss_prelog

def parse_option():
    parser = argparse.ArgumentParser('STL-10 Representation Learning with Alignment and Uniformity Losses')
    parser.add_argument("experiment_file", type=str, help="path to the experiment json file")
    parser.add_argument("method", type=str, help="one of unsupervised_cond, unsupervised, supervised")

    parser.add_argument("--dataset", type=str, default="cifar100", help="one of cifar10, cifar100, stl10")
    parser.add_argument("--temp", type=float, default=1, help="temperature for the experiments")
    parser.add_argument('--labels', type=int, default=[0,1], nargs="*", help='which labels assists the unsupervised learning')

    parser.add_argument('--batch_size', type=int, default=768, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate. Default is linear scaling 0.12 per 256 batch size')
    # parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='Learning rate decay rate')
    # parser.add_argument('--lr_decay_epochs', default=[155, 170, 185], nargs='*', type=int,
    #                    help='When to decay learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 weight decay')
    parser.add_argument('--feat_dim', type=int, default=128, help='Feature dimensionality')

    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers to use')
    parser.add_argument('--log_interval', type=int, default=40, help='Number of iterations between logs')
    parser.add_argument('--gpus', default=[0], nargs='*', type=int,
                        help='List of GPU indices to use, e.g., --gpus 0 1 2 3')
    parser.add_argument('--n_eval', type=int, default=10, help="number of linear evaluations")
    parser.add_argument('--epochs_eval', type=int, default=100, help="number of epochs for training linear head")

    parser.add_argument('--data_folder', type=str, default='./data', help='Path to data')
    parser.add_argument('--result_folder', type=str, default='./results', help='Base directory to save model')

    opt = parser.parse_args()

    # set learning rate
    if opt.lr is None:
        opt.lr = 0.12 * (opt.batch_size / 256)

    # overwrite console configuration with json
    with open(opt.experiment_file, "r") as file:
        json_config = json.load(file)

    for key, value in json_config.items():
        setattr(opt, key, value)

    assert(opt.method in ["unsupervised", "supervised", "unsupervised_cond"])
    assert(opt.dataset in ["cifar10", "cifar100", "stl10"])

    if opt.dataset== "cifar100":
        assert(all( [lbl < 100 for lbl in opt.labels]))
    else:
        assert(all([lbl < 10 for lbl in opt.labels]))

    opt.gpus = list(map(lambda x: torch.device('cuda', x), opt.gpus))

    # Get the current timestamp
    current_timestamp = datetime.now()

    # Format the timestamp to include milliseconds
    # Format: YYYY-MM-DD_HH-MM-SS-MS (MS = milliseconds)
    formatted_timestamp = current_timestamp.strftime('%Y-%m-%d_%H-%M-%S-') + str(current_timestamp.microsecond // 1000)


    opt.save_folder = os.path.join(
        opt.result_folder,
        f"experiment_{formatted_timestamp}"
    )
    os.makedirs(opt.save_folder, exist_ok=True)

    def custom_serializer(obj):
        """A custom JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, Namespace):
            return vars(obj)  # Convert Namespace to dictionary
        elif isinstance(obj,torch.device):
            return str(obj)
        raise TypeError(f"Type {type(obj)} not serializable")


    # now we save the opt file as json
    with open(os.path.join(opt.save_folder, "conf.json"), "w") as file:
        json.dump(vars(opt), file, indent=4, default=custom_serializer)

    n_labels = 100 if opt.dataset == "cifar100" else 10
    old_lbls = list(range(n_labels))

    old2new = {}
    count = 0
    for old_lbl in old_lbls:
        if old_lbl in opt.labels:
            old2new[old_lbl] = count
            count += 1

    for old_lbl in old_lbls:
        if old_lbl not in opt.labels:
            old2new[old_lbl] = count

    opt.old2new = old2new
    return opt


def get_data_loader(opt):

    base_img_size = 64 if opt.dataset == "stl10" else 32

    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(base_img_size, scale=(0.08, 1)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
            (0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
        ),
    ])

    if opt.dataset == "cifar100":
        dataset_base = torchvision.datasets.CIFAR100(opt.data_folder, 'train', download=True)
    elif opt.dataset == "cifar10":
        dataset_base = torchvision.datasets.CIFAR10(opt.data_folder, 'train', download=True)
    elif opt.dataset == "stl10":
        dataset_base = torchvision.datasets.STL10(opt.data_folder,"train", download=True)

    dataset = TwoAugUnsupervisedDatasetLbl(dataset_base, transform=transform)

    return torch.utils.data.DataLoader(dataset,
                                       batch_size=opt.batch_size,
                                       num_workers=opt.num_workers,
                                       shuffle=True, pin_memory=True)

def get_data_loaders_val(opt):
    base_img_size = 64 if opt.dataset == "stl10" else 32

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(base_img_size, scale=(0.08, 1)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
            (0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
        ),
        torchvision.transforms.RandomHorizontalFlip()
    ])
    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(base_img_size),
        torchvision.transforms.CenterCrop(base_img_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
            (0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
        ),
    ])

    if opt.dataset == "stl10":
        train_dataset = torchvision.datasets.STL10(opt.data_folder, 'train', download=True, transform=train_transform)
        val_dataset = torchvision.datasets.STL10(opt.data_folder, 'test', transform=val_transform)
    elif opt.dataset == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(opt.data_folder, 'train', download=True, transform=train_transform)
        val_dataset = torchvision.datasets.CIFAR10(opt.data_folder, 'test', transform=val_transform)
    elif opt.dataset == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(opt.data_folder, 'train', download=True, transform=train_transform)
        val_dataset = torchvision.datasets.CIFAR100(opt.data_folder, 'test', transform=val_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
                                               num_workers=opt.num_workers, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size,
                                             num_workers=opt.num_workers, pin_memory=True)
    return train_loader, val_loader


def validate_comb(opt, encoder, classifier, val_loader):
    correct = 0
    with torch.no_grad():
        for images, labels_mod, labels_act in val_loader:
            pred = classifier(torch.cat( (encoder(images.to(opt.gpus[0]),
                                                  layer_index=opt.layer_index).flatten(1),
                                          torch.nn.functional.one_hot(labels_mod.to(opt.gpus[0]),
                                                                      num_classes=len(opt.labels)+1)),
                                         dim=1)).argmax(dim=1)
            correct += (pred.cpu() == labels_act).sum().item()
    return correct / len(val_loader.dataset)

def main():
    opt = parse_option()

    print(f'Optimize: {opt.temp}')

    torch.cuda.set_device(opt.gpus[0])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    encoder = SmallAlexNet(feat_dim=opt.feat_dim, inp_size=64 if opt.dataset=="stl10" else 32).to(opt.gpus[0])

    classifier = None
    if opt.method == "supervised":
        n_classes = 100 if opt.dataset == "cifar100" else 10
        classifier = torch.nn.Linear(opt.feat_dim, n_classes)
        optim = torch.optim.Adam( itertools.chain(classifier.parameters(), encoder.parameters()), lr=1e-2 )
    else:
        optim = torch.optim.Adam(encoder.parameters(), lr=1e-2)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, gamma=opt.lr_decay_rate, milestones=opt.lr_decay_epochs)

    loader = get_data_loader(opt)

    loss_meter = AverageMeter('total_loss')
    it_time_meter = AverageMeter('iter_time')
    for epoch in range(opt.epochs):
        loss_meter.reset()
        it_time_meter.reset()
        t0 = time.time()
        for ii, (im_x, im_y, lbl) in enumerate(loader):
            optim.zero_grad()

            if opt.method == "unsupervised":
                x, y = encoder(torch.cat([im_x.to(opt.gpus[0]), im_y.to(opt.gpus[0])])).chunk(2)
                align_loss_val = align_loss(x, y)
                unif_loss_val = (uniform_loss(x, t=opt.temp) + uniform_loss(y, t=opt.temp)) / 2
                loss = align_loss_val * opt.temp + unif_loss_val

                loss_meter.update(loss, x.shape[0])
                loss.backward()
                optim.step()
                it_time_meter.update(time.time() - t0)

            elif opt.method == "unsupervised_cond":
                x, y = encoder(torch.cat([im_x.to(opt.gpus[0]), im_y.to(opt.gpus[0])])).chunk(2)
                align_loss_val = align_loss(x, y, alpha=opt.align_alpha)
                # group according to new_lbls
                z = torch.cat([x, y])
                lbl_z = torch.cat([lbl, lbl])
                unif_losses = torch.cat([uniform_loss_prelog(z[lbl_z == new_lbl]) for new_lbl in range(opt.labels)])
                unif_loss_val = torch.log(torch.mean(unif_losses))

                loss = align_loss_val * opt.temp + unif_loss_val
                loss_meter.update(loss, x.shape[0])
                loss.backward()
                optim.step()
            elif opt.method == "supervised":
                x, y = encoder(torch.cat([im_x.to(opt.gpus[0]), im_y.to(opt.gpus[0])])).chunk(2)
                # group according to new_lbls
                z = torch.cat([x, y])
                lbl_z = torch.cat([lbl, lbl])
                loss = torch.nn.functional.cross_entropy( classifier(z), lbl_z)
                loss_meter.update(loss, x.shape[0])
                loss.backward()
                optim.step()

            if ii % opt.log_interval == 0:
                print(f"Epoch {epoch}/{opt.epochs}\tIt {ii}/{len(loader)}\t" +
                      f"{loss_meter}\t{it_time_meter}")
            t0 = time.time()

        # scheduler.step()
    ckpt_file = os.path.join(opt.save_folder, 'encoder.pth')
    torch.save(encoder.state_dict(), ckpt_file)
    print(f'Saved to {ckpt_file}')

    """
        run linear evaluation on a predesignated number of times, so that we do not worry about it
    """

    print("Starting linear evaluation")
    encoder.eval()
    train_loader, val_loader = get_data_loaders_val(opt)

    with torch.no_grad():
        sample, _ = train_loader.dataset.dataset[0]
        eval_numel = encoder(sample.unsqueeze(0).to(opt.gpus[0]), layer_index=opt.layer_index).numel()
    print(f'Feature dimension: {eval_numel}')


    val_accs_all = []
    for validation_idx in range(opt.n_val):
        print(f"Starting evaluation {validation_idx}/{opt.n_val}")
        classifier = nn.Linear(eval_numel + len(opt.labels) + 1, 10).to(opt.gpus[0])

        optim = torch.optim.Adam(classifier.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, gamma=opt.lr_decay_rate,
                                                         milestones=opt.lr_decay_epochs)
        val_accs = []
        for epoch in tqdm(range(opt.epochs_eval)):
            loss_meter.reset()
            it_time_meter.reset()
            t0 = time.time()
            for ii, (images, labels_mod, labels) in enumerate(train_loader):
                optim.zero_grad()
                with torch.no_grad():
                    feats = encoder(images.to(opt.gpus[0]), layer_index=opt.layer_index).flatten(1)

                logits = classifier(torch.cat((feats, torch.nn.functional.one_hot(labels_mod.to(opt.gpus[0]),
                                                                                  num_classes=len(
                                                                                      opt.labels) + 1)), dim=1))


                loss = torch.nn.functional.cross_entropy(logits, labels.to(opt.gpus[0]))
                loss_meter.update(loss, images.shape[0])
                loss.backward()
                optim.step()
                it_time_meter.update(time.time() - t0)
                if ii % opt.log_interval == 0:
                    print(f"Epoch {epoch}/{opt.epochs}\tIt {ii}/{len(train_loader)}\t{loss_meter}\t{it_time_meter}")
                t0 = time.time()
            scheduler.step()
            val_acc = validate_comb(opt, encoder, classifier, val_loader)
            val_accs.append(val_acc)
            print(f"Epoch {epoch}/{opt.epochs}\tval_acc {val_acc * 100:.4g}%")
        print(f"Best validation accuracy {max(val_accs)} for {validation_idx}/{opt.n_val}")
        val_accs_all.append(val_accs)
        with open(os.path.join(opt.save_folder, "val_results.json"), "w") as f:
            json.dump(val_accs_all, f)


if __name__ == '__main__':
    main()
