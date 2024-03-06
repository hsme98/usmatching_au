import os
import time
import argparse
from itertools import chain

import torchvision
import torch
import torch.nn as nn

from util import AverageMeter, AugUnsupervisedDataset, load_transforms
from encoder import SmallAlexNet
from align_uniform import align_loss, uniform_loss



def parse_option():
    parser = argparse.ArgumentParser('STL-10 Representation Learning with Alignment and Uniformity Losses')

    parser.add_argument('--align_w', type=float, default=1, help='Alignment loss weight')
    parser.add_argument('--unif_w', type=float, default=1, help='Uniformity loss weight')
    parser.add_argument('--align_alpha', type=float, default=2, help='alpha in alignment loss')
    parser.add_argument('--unif_t', type=float, default=2, help='t in uniformity loss')

    parser.add_argument('--batch_size', type=int, default=768, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
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

    parser.add_argument('--iter', type=int, default=0, help='identifier for the experiment')
    parser.add_argument('--shared', action='store_true', help='uses the same encoder in all augmentations and data')
    parser.add_argument('--data_folder', type=str, default='./data', help='Path to data')
    parser.add_argument('--result_folder', type=str, default='./results', help='Base directory to save model')
    parser.add_argument('--transforms', type=str, default=None)
    opt = parser.parse_args()
    assert(opt.transforms is not None)

    if opt.lr is None:
        opt.lr = 0.12 * (opt.batch_size / 256)

    opt.gpus = list(map(lambda x: torch.device('cuda', x), opt.gpus))

    opt.save_folder = os.path.join(
        opt.result_folder,
        f"result_{os.path.basename(opt.transforms).split('.')[0]}_{opt.iter}_{opt.shared}"
    )
    os.makedirs(opt.save_folder, exist_ok=True)

    return opt


#
#   transforms should be a list of transforms
#
def get_data_loader(opt, transforms):
    # normalization transforms
    transforms_norm = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
            (0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
        ),
    ]
    )

    dataset = AugUnsupervisedDataset( torchvision.datasets.STL10(opt.data_folder, 'train+unlabeled', download=True),
                                      transforms=transforms, norm_transform=transforms_norm )

    return torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_workers,
                                       shuffle=True, pin_memory=True)


def main():
    opt = parse_option()

    print(f'Optimize: {opt.align_w:g} * loss_align(alpha={opt.align_alpha:g}) + {opt.unif_w:g} * loss_uniform(t={opt.unif_t:g})')

    torch.cuda.set_device(opt.gpus[0])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    transforms = load_transforms(opt.transforms)
    loader = get_data_loader(opt, transforms=transforms)

    # for each transform allocate another encoder
    # should we use one single encoder or multiple encoders???

    if opt.shared:
        encoders = [nn.DataParallel(SmallAlexNet(feat_dim=opt.feat_dim).to(opt.gpus[0]), opt.gpus)]
    else:
        encoders = [nn.DataParallel(SmallAlexNet(feat_dim=opt.feat_dim).to(opt.gpus[0]), opt.gpus) for transform_idx in range(len(transforms)+1)]

    optim = torch.optim.SGD( chain.from_iterable([encoder.parameters() for encoder in encoders]) ,
                            lr=opt.lr,
                            momentum=opt.momentum,
                             weight_decay=opt.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, gamma=opt.lr_decay_rate,
                                                     milestones=opt.lr_decay_epochs)

    loss_meter = AverageMeter('total_loss')
    it_time_meter = AverageMeter('iter_time')
    for epoch in range(opt.epochs):
        loss_meter.reset()
        it_time_meter.reset()
        t0 = time.time()
        for ii, imgs in enumerate(loader):
            optim.zero_grad()
            if opt.shared:
                encoded = [encoders[0](img.to(opt.gpus[0])) for img in imgs]
            else:
                encoded = [encoder(img.to(opt.gpus[0])) for encoder, img in zip(encoders,imgs)]

            losses = torch.zeros(len(encoded[1:]))
            for encoding_idx,encoding in enumerate(encoded[1:]):
                align_loss_val = align_loss(encoded[0], encoding, alpha=opt.align_alpha)
                unif_loss_val = (uniform_loss(encoded[0], t=opt.unif_t) + uniform_loss(encoding, t=opt.unif_t)) / 2
                losses[encoding_idx] = align_loss_val * opt.align_w + unif_loss_val * opt.unif_w

            loss = torch.sum(losses)
            loss_meter.update(loss, imgs[0].shape[0])
            loss.backward()
            optim.step()
            it_time_meter.update(time.time() - t0)
            if ii % opt.log_interval == 0:
                print(f"Epoch {epoch}/{opt.epochs}\tIt {ii}/{len(loader)}\t" +
                      f"{loss_meter}\t{it_time_meter}")
            t0 = time.time()
        scheduler.step()
    ckpt_file = os.path.join(opt.save_folder, 'encoder.pth')
    torch.save(encoders, ckpt_file)
    print(f'Saved to {ckpt_file}')


if __name__ == '__main__':
    main()
