import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from yacs.config import CfgNode as CN

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.data.size(0), -1)


class CombConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, dropout=0.1, bias=False):
        super().__init__()
        self.add_module('layer1', ConvLayer(in_channels, out_channels, kernel))
        self.add_module('layer2', DWConvLayer(out_channels, out_channels, stride=stride))

    def forward(self, x):
        return super().forward(x)


class DWConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super().__init__()
        out_ch = out_channels

        groups = in_channels
        kernel = 3
        # print(kernel, 'x', kernel, 'x', out_channels, 'x', out_channels, 'DepthWise')

        self.add_module('dwconv', nn.Conv3d(groups, groups, kernel_size=3,
                                            stride=stride, padding=1, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm3d(groups))

    def forward(self, x):
        return super().forward(x)


class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1, bias=False):
        super().__init__()
        out_ch = out_channels
        groups = 1
        # print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)
        self.add_module('conv', nn.Conv3d(in_channels, out_ch, kernel_size=kernel,
                                          stride=stride, padding=kernel // 2, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm3d(out_ch))
        self.add_module('relu', nn.ReLU6(True))

    def forward(self, x):
        return super().forward(x)


class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False, dwconv=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0  # if upsample else in_channels
        for i in range(n_layers):  # i是block里的第i层
            outch, inch, link = self.get_link(i + 1, in_channels, growth_rate, grmul)
            self.links.append(link)
            use_relu = residual_out
            if dwconv:
                layers_.append(CombConvLayer(inch, outch))
            else:
                layers_.append(ConvLayer(inch, outch))

            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
        # print("Blk out =",self.out_channels)
        self.layers = nn.ModuleList(layers_)

    def forward(self, x):
        layers_ = [x]

        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)

        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or \
                    (i == t - 1) or (i % 2 == 1):
                out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out


class HarDNet(nn.Module):
    def __init__(self, cf):
        super().__init__()
        input_channel = cf.input_channel
        depth_wise = False
        arch = cf.HARDNET.ARCH
        class_num = cf.num_cls_classes
        first_ch = [32, 64]
        second_kernel = 3
        max_pool = False
        grmul = 1.7
        drop_rate = 0.5

        # HarDNet68
        ch_list = [128, 256, 320, 640, 1024]
        gr = [14, 16, 20, 40, 160]
        n_layers = [8, 16, 16, 16, 4]
        downSamp = [1, 0, 1, 1, 0]

        if arch == 85:
            # HarDNet85
            first_ch = [48, 96]
            ch_list = [192, 256, 320, 480, 720, 1280]
            gr = [24, 24, 28, 36, 48, 256]
            n_layers = [8, 16, 16, 16, 16, 4]
            downSamp = [1, 0, 1, 0, 1, 0]
        elif arch == 39:
            # HarDNet39
            first_ch = [24, 48]
            ch_list = [96, 320, 640, 1024]
            grmul = 1.6
            gr = [16, 20, 64, 160]
            n_layers = [4, 16, 8, 4]
            downSamp = [1, 1, 1, 0]

        if depth_wise:
            second_kernel = 1
            max_pool = False
            drop_rate = 0.05

        blks = len(n_layers)
        self.base = nn.ModuleList([])

        # First Layer: Standard Conv3x3, Stride=2
        self.base.append(
            ConvLayer(in_channels=input_channel, out_channels=first_ch[0], kernel=3,
                      stride=2, bias=False))

        # Second Layer
        self.base.append(ConvLayer(first_ch[0], first_ch[1], kernel=second_kernel))

        # Maxpooling or DWConv3x3 downsampling
        if max_pool:
            self.base.append(nn.MaxPool3d(kernel_size=3, stride=2, padding=1))
        else:
            self.base.append(DWConvLayer(first_ch[1], first_ch[1], stride=2))

        # Build all HarDNet blocks
        ch = first_ch[1]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i], dwconv=depth_wise)
            ch = blk.get_out_ch()
            self.base.append(blk)

            if i == blks - 1 and arch == 85:
                self.base.append(nn.Dropout(0.1))

            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]
            if downSamp[i] == 1:
                if max_pool:
                    self.base.append(nn.MaxPool3d(kernel_size=2, stride=2))
                else:
                    self.base.append(DWConvLayer(ch, ch, stride=2))

        ch = ch_list[blks - 1]
        self.base.append(
            nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                Flatten(),
                nn.Linear(ch, ch),
                nn.ReLU(),
                nn.Dropout(drop_rate)))
        self.base.append(nn.Linear(ch, class_num))

    def forward(self, x, clinical_infos=None):
        for layer in self.base:
            x = layer(x)
        if clinical_infos is not None:
            clinical_infos = clinical_infos.float()
            clinical_infos = self.activate(self.fc_clinic(clinical_infos))
            x = torch.cat([x, clinical_infos], dim=1)
            x = self.pre_norm(x)
            x = self.fc1(x)
            x = self.activate(x)
            x = self.fc2(x)
        return x


class ClsLoss(nn.Module):
    def __init__(self, cf):
        super(ClsLoss, self).__init__()
        self.class_loss_func = nn.CrossEntropyLoss(weight=torch.tensor(cf.loss_weight))

    def forward(self, predict, label):
        class_loss = self.class_loss_func(predict, torch.argmax(label, dim=1))
        return class_loss


class Net(nn.Module):
    def __init__(self, cf, logger):
        super(Net, self).__init__()
        self.cf = cf
        self.logger = logger
        self.net = HarDNet(cf)
        self.loss_func = ClsLoss(cf=self.cf)

    def forward(self, batch_inputs, phase):
        if phase != 'test':
            images, labels = batch_inputs['inputs'], batch_inputs['label']
            clinical_features = batch_inputs.get('clinical_features', None)
            return self.train_val_forward(images, labels, clinical_features)

        else:
            images = batch_inputs['inputs']
            clinical_features = batch_inputs.get('clinical_features', None)
            return self.test_forward(images, clinical_features)

    def train_val_forward(self, images, labels, clinical_infos=None):
        out = self.net(images, clinical_infos=clinical_infos)
        loss = self.loss_func(out, labels)
        return {'loss': loss}

    def test_forward(self, images, clinical_infos=None):
        out = self.net(images, clinical_infos=clinical_infos)
        out = F.softmax(out, dim=1)
        return {'predict_label': out}


if __name__ == '__main__':
    HARDNET = CN()
    HARDNET.ARCH = 39
    model = HarDNet()