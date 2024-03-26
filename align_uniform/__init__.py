import torch


def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def uniform_loss_prelog(x, t=2):
     return torch.pdist(x, p=2).pow(2).mul(-t).exp()
    
__all__ = ['align_loss', 'uniform_loss']
