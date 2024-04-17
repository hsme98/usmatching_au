r"""
Adapted from https://github.com/HobbitLong/CMC/blob/f25c37e49196a1fe7dc5f7b559ed43c6fce55f70/models/alexnet.py
"""

import torch.nn as nn
import torch
import pdb

class L2Norm(nn.Module):
    def forward(self, x):
        return x / x.norm(p=2, dim=1, keepdim=True)

    
class SimpleNetwork(nn.Module):
    def __init__(self, in_channel=3, feat_dim=128, cifar=False):
        super(SimpleNetwork, self).__init__()
        # conv_block_1
        blocks = []
        
        blocks.append(nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))

        # conv_block_2
        blocks.append(nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))
        
        inp_size = 7 if cifar else 23
        blocks.append(nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * inp_size * inp_size, feat_dim *2 , bias=False),
                nn.BatchNorm1d(feat_dim * 2),
                nn.ReLU(inplace=True),
            ))

        blocks.append(nn.Sequential(
            nn.Linear(2 * feat_dim, feat_dim),
            L2Norm(),
        ))

        self.blocks = nn.ModuleList(blocks)
        self.init_weights_()
           
    def init_weights_(self):
        def init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, 0, 0.02)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init)

    def forward(self, x, *, layer_index=-1):
        if layer_index < 0:
            layer_index += len(self.blocks)
        for layer in self.blocks[:(layer_index + 1)]:
            x = layer(x)
        return x
        
import torch
import torch.nn as nn
import torch.nn.functional as F

class Hypernetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Hypernetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.fc1(x)

class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)

class SmallerPredAlexNet(nn.Module):
    def __init__(self, pred_dim, in_channel=3, feat_dim=128):
        super(SmallerPredAlexNet, self).__init__()
        self.in_channel = in_channel
        self.feat_dim = feat_dim

        self.hypernet_1 = Hypernetwork(input_dim=pred_dim, output_dim=96 * self.in_channel * 3 * 3)
        self.hypernet_2 = Hypernetwork(input_dim=pred_dim, output_dim=192 * 96 * 3 * 3)
        self.hypernet_3 = Hypernetwork(input_dim=pred_dim, output_dim=384 * 192 * 3 * 3)
        self.hypernet_4 = Hypernetwork(input_dim=pred_dim, output_dim=384 * 384 * 3 * 3)
        self.hypernet_5 = Hypernetwork(input_dim=pred_dim, output_dim=192 * 384 * 3 * 3)

        # Blocks for later fc layers
        blocks = [
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(192 * 3 * 3 + pred_dim, 2 * feat_dim , bias=False),
                nn.BatchNorm1d(2 * feat_dim),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Linear(2 * feat_dim, 2 * feat_dim, bias=False),
                nn.BatchNorm1d(2 * feat_dim),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Linear(2 * feat_dim, feat_dim),
                L2Norm(),
            )
        ]

        self.blocks = nn.ModuleList(blocks)
        self.init_weights_()

    def init_weights_(self):
        def init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        self.apply(init)

    def forward(self, x, y, layer_index=-1):
        
        if layer_index < 0:
            layer_index += len(self.blocks)
        # Apply hypernets and convolutional operations
        conv_weights = [
            self.hypernet_1(y).view(x.shape[0],96, self.in_channel, 3, 3),
            self.hypernet_2(y).view(x.shape[0],192, 96, 3, 3),
            self.hypernet_3(y).view(x.shape[0],384, 192, 3, 3),
            self.hypernet_4(y).view(x.shape[0],384, 384, 3, 3),
            self.hypernet_5(y).view(x.shape[0],192, 384, 3, 3)
        ]
        
        def apply_conv2d(inp, weight, **param):
            prev_shape = inp.shape
            inp_ = inp.reshape(1, prev_shape[0]* prev_shape[1], prev_shape[2], prev_shape[3] )
            weight_ = weight.reshape(-1, weight.shape[2] ,weight.shape[3], weight.shape[4])
            y = F.conv2d(inp_, weight_, groups=prev_shape[0],**param)
            new_shape = y.shape
            return y.reshape(prev_shape[0], -1, new_shape[2], new_shape[3])
        
        # Applying initial conv layer
        x = apply_conv2d(x, conv_weights[0], padding=1, stride=1)
        x = F.relu(F.batch_norm(x, torch.randn(96).to(x.device), torch.ones(96).to(x.device)))
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        # Subsequent conv layers
        for i in range(1, 5):
            x = apply_conv2d(x, conv_weights[i], padding=1, stride=1)
            if i != 4:  # Last conv layer does not have max pooling
                x = F.relu(F.batch_norm(x, torch.randn(x.size(1)).to(x.device), torch.ones(x.size(1)).to(x.device)))
                if i % 2 == 1:
                    x = F.max_pool2d(x, kernel_size=3, stride=2)
        
        # Apply fully connected layers
        x = self.blocks[0](torch.cat( (x.flatten(1), y) , dim=1) )
        for layer in self.blocks[1:layer_index + 1]:
            x = layer(x)

        return x

    
class SmallerAlexNetLbls(nn.Module):
    def __init__(self, n_lbls,in_channel=3, feat_dim=128, cifar=False):
        super(SmallerAlexNetLbls, self).__init__()
        self.n_lbls = n_lbls
        blocks = []

        # conv_block_1
        blocks.append(nn.Sequential(
            nn.Conv2d(in_channel, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))

        # conv_block_2
        blocks.append(nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))

        # conv_block_3
        blocks.append(nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        ))

        # conv_block_4
        blocks.append(nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        ))

        # conv_block_5
        blocks.append(nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))

        if not cifar:
            # fc6
            blocks.append(nn.Sequential(
                nn.Linear(192 * 7 * 7 + self.n_lbls, 2 * feat_dim, bias=False),  # 256 * 6 * 6 if 224 * 224
                nn.BatchNorm1d(2 *feat_dim),
                nn.ReLU(inplace=True),
            ))
        else:
            # fc6
            blocks.append(nn.Sequential(
                nn.Linear(192 * 3 * 3 + self.n_lbls, 2 * feat_dim, bias=False),  # 256 * 6 * 6 if 224 * 224
                nn.BatchNorm1d(2 * feat_dim),
                nn.ReLU(inplace=True),
            ))

        # fc7
        blocks.append(nn.Sequential(
            nn.Linear(2 * feat_dim, 2 * feat_dim, bias=False),
            nn.BatchNorm1d(2 * feat_dim),
            nn.ReLU(inplace=True),
        ))

        # fc8
        blocks.append(nn.Sequential(
            nn.Linear(2 * feat_dim, feat_dim),
            L2Norm(),
        ))

        self.blocks = nn.ModuleList(blocks)
        self.init_weights_()
        
        
    def init_weights_(self):
        def init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, 0, 0.02)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init)

    def forward(self, x, lbl, *, layer_index=-1):
        lbl_oneh = torch.nn.functional.one_hot(lbl, num_classes=self.n_lbls )
        if layer_index < 0:
            layer_index += len(self.blocks)
        for layer_idx, layer in enumerate(self.blocks[:(layer_index + 1)]):
            if layer_idx == 5: # fc7
                x = layer(torch.cat((x.flatten(1), lbl_oneh),dim=1))
            else:
                x = layer(x)
        return x

class SmallerAlexNet(nn.Module):
    def __init__(self, in_channel=3, feat_dim=128, cifar=False):
        super(SmallerAlexNet, self).__init__()

        blocks = []

        # conv_block_1
        blocks.append(nn.Sequential(
            nn.Conv2d(in_channel, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))

        # conv_block_2
        blocks.append(nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))

        # conv_block_3
        blocks.append(nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        ))

        # conv_block_4
        blocks.append(nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        ))

        # conv_block_5
        blocks.append(nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))

        if not cifar:
            # fc6
            blocks.append(nn.Sequential(
                nn.Flatten(),
                nn.Linear(192 * 7 * 7, 2 * feat_dim, bias=False),  # 256 * 6 * 6 if 224 * 224
                nn.BatchNorm1d(2 *feat_dim),
                nn.ReLU(inplace=True),
            ))
        else:
            # fc6
            blocks.append(nn.Sequential(
                nn.Flatten(),
                nn.Linear(192 * 3 * 3, 2 * feat_dim, bias=False),  # 256 * 6 * 6 if 224 * 224
                nn.BatchNorm1d(2 * feat_dim),
                nn.ReLU(inplace=True),
            ))

        # fc7
        blocks.append(nn.Sequential(
            nn.Linear(2 * feat_dim, 2 * feat_dim, bias=False),
            nn.BatchNorm1d(2 * feat_dim),
            nn.ReLU(inplace=True),
        ))

        # fc8
        blocks.append(nn.Sequential(
            nn.Linear(2 * feat_dim, feat_dim),
            L2Norm(),
        ))

        self.blocks = nn.ModuleList(blocks)
        self.init_weights_()

    def init_weights_(self):
        def init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, 0, 0.02)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init)

    def forward(self, x, *, layer_index=-1):
        if layer_index < 0:
            layer_index += len(self.blocks)
        for layer in self.blocks[:(layer_index + 1)]:
            x = layer(x)
        return x
    
class SmallAlexNet(nn.Module):
    def __init__(self, in_channel=3, feat_dim=128, cifar=False):
        super(SmallAlexNet, self).__init__()

        blocks = []

        # conv_block_1
        blocks.append(nn.Sequential(
            nn.Conv2d(in_channel, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))

        # conv_block_2
        blocks.append(nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))

        # conv_block_3
        blocks.append(nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        ))

        # conv_block_4
        blocks.append(nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        ))

        # conv_block_5
        blocks.append(nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))

        if not cifar:
            # fc6
            blocks.append(nn.Sequential(
                nn.Flatten(),
                nn.Linear(192 * 7 * 7, 4096, bias=False),  # 256 * 6 * 6 if 224 * 224
                nn.BatchNorm1d(4096),
                nn.ReLU(inplace=True),
            ))
        else:
            # fc6
            blocks.append(nn.Sequential(
                nn.Flatten(),
                nn.Linear(192 * 3 * 3, 4096, bias=False),  # 256 * 6 * 6 if 224 * 224
                nn.BatchNorm1d(4096),
                nn.ReLU(inplace=True),
            ))

        # fc7
        blocks.append(nn.Sequential(
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        ))

        # fc8
        blocks.append(nn.Sequential(
            nn.Linear(4096, feat_dim),
            L2Norm(),
        ))

        self.blocks = nn.ModuleList(blocks)
        self.init_weights_()

    def init_weights_(self):
        def init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, 0, 0.02)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init)

    def forward(self, x, *, layer_index=-1):
        if layer_index < 0:
            layer_index += len(self.blocks)
        for layer in self.blocks[:(layer_index + 1)]:
            x = layer(x)
        return x


class SmallAlexNetLbls(nn.Module):
    def __init__(self, n_lbls, in_channel=3, feat_dim=128, cifar=False):
        super(SmallAlexNetLbls, self).__init__()
        self.n_lbls = n_lbls
        blocks = []

        # conv_block_1
        blocks.append(nn.Sequential(
            nn.Conv2d(in_channel, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))

        # conv_block_2
        blocks.append(nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))

        # conv_block_3
        blocks.append(nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        ))

        # conv_block_4
        blocks.append(nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        ))

        # conv_block_5
        blocks.append(nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Flatten()
        ))

        if not cifar:
            # fc6
            blocks.append(nn.Sequential(
                nn.Linear(192 * 7 * 7 + n_lbls, 4096 , bias=False),  # 256 * 6 * 6 if 224 * 224
                nn.BatchNorm1d(4096),
                nn.ReLU(inplace=True),
            ))
        else:
            # fc6
            blocks.append(nn.Sequential(
                nn.Linear(192 * 3 * 3 + n_lbls, 4096 , bias=False),  # 256 * 6 * 6 if 224 * 224
                nn.BatchNorm1d(4096),
                nn.ReLU(inplace=True),
            ))


        # fc7
        blocks.append(nn.Sequential(
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        ))

        # fc8
        blocks.append(nn.Sequential(
            nn.Linear(4096, feat_dim),
            L2Norm(),
        ))

        self.blocks = nn.ModuleList(blocks)
        self.init_weights_()

    def init_weights_(self):
        def init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, 0, 0.02)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init)

    def forward(self, x, lbl, *, layer_index=-1):
        lbl_oneh = torch.nn.functional.one_hot(lbl, num_classes=self.n_lbls )
        if layer_index < 0:
            layer_index += len(self.blocks)
        for layer_idx, layer in enumerate(self.blocks[:(layer_index + 1)]):
            if layer_idx == 5: # fc7
                x = layer(torch.cat((x, lbl_oneh),dim=1))
            else:
                x = layer(x)
        return x