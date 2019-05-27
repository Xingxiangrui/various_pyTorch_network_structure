"""
hierarchical graph attention network, grouping network use fc.
"""

import torch
import torchvision.models as models
from torch import nn
import mymodels.utils as utils
import torch
from torch import nn
import torch.nn.functional as F


class BGATLayer(nn.Module):
    """
    Batch GATLayer, modified from:
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha):
        super(BGATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(in_features, out_features))
        nn.init.xavier_uniform(self.W.data, gain=1.414)  # fixme
        self.a = nn.Parameter(torch.zeros(2 * out_features, 1))
        nn.init.xavier_uniform(self.a.data, gain=1.414)  # fixme
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.beta = nn.Parameter(data=torch.ones(1))
        self.register_parameter('beta', self.beta)

    def forward(self, x):
        # [Batchs ,Nodes ,Channels ]
        B, N, C = x.size()
        # h = torch.bmm(x, self.W.expand(B, self.in_features, self.out_features))  # [B,N,C]
        h = torch.matmul(x, self.W)  # [B,N,C]
        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, C), h.repeat(1, N, 1)], dim=2).view(B, N, N,
                                                                                                  2 * self.out_features)  # [B,N,N,2C]
        # temp = self.a.expand(B, self.out_features * 2, 1)
        # temp2 = torch.matmul(a_input, self.a)
        attention = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # [B,N,N]

        attention = F.softmax(attention, dim=2)  # [B,N,N]
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.bmm(attention, h)  # [B,N,N]*[B,N,C]-> [B,N,C]
        out = F.elu(h_prime + segelf.beta * h)
        return out


class NHeadsBGATLayer(nn.Module):
    def __init__(self, nheads, aggregate, in_features, out_features, dropout=0, alpha=0.2):
        super(NHeadsBGATLayer, self).__init__()
        self.aggregate = aggregate
        self.gats = nn.ModuleList(
            [BGATLayer(in_features=in_features, out_features=out_features, dropout=dropout, alpha=alpha) for _ in
             range(nheads)])

    def forward(self, x):
        if self.aggregate == 'mean':
            x = torch.stack([att(x) for att in self.attentions], dim=2)
            x = torch.mean(x, dim=3).squeeze(3)
        elif self.aggregate == 'concat':
            x = torch.cat([att(x) for att in self.attentions], dim=2)  # [B,N,nheads*C]
        else:
            raise Exception()
        return x

class Head(nn.Module):
    def __init__(self, groups, nclasses, nclasses_per_group, group_channels, class_channels):
        super(Head, self).__init__()
        self.groups = groups
        self.nclasses = nclasses
        self.nclasses_per_group = nclasses_per_group
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.reduce_fc = nn.Sequential(utils.BasicLinear(in_channels=2048, out_channels=1024),
                                       utils.BasicLinear(in_channels=1024, out_channels=group_channels), )
        self.group_fcs = nn.ModuleList(
            [utils.ResidualLinearBlock(in_channels=group_channels, reduction=2, out_channels=group_channels)
             for _ in range(groups)])

        self.gat1 = BGATLayer(in_features=group_channels, out_features=group_channels, dropout=0, alpha=0.2)
        #fixme fc layer to replace GALayer1
        self.fc_replace_GALayer1=utils.BasicLinear(in_channels=groups*group_channels, out_channels=groups*group_channels)
        self.class_fcs = nn.ModuleList(
            [utils.BasicLinear(in_channels=group_channels, out_channels=class_channels) for _ in range(nclasses)])
        self.gat2s = nn.ModuleList(
            [BGATLayer(in_features=class_channels, out_features=class_channels, dropout=0, alpha=0.2) for _ in
             range(groups)])
        #fixme fc layer to replace GALayer2
        self.fc_replace_GALayer2=nn.ModuleList(
            [utils.BasicLinear(in_channels=class_channels, out_channels=class_channels)
             for _ in range (groups)])


        self.fcs = nn.ModuleList(
            [nn.Sequential(
                utils.ResidualLinearBlock(in_channels=class_channels, reduction=2, out_channels=class_channels),
                nn.Linear(in_features=class_channels, out_features=1)
            ) for _ in range(nclasses)])

    def forward(self, x):
        x = self.gmp(x).view(x.size(0), x.size(1))  #output dim [B,2048]
        x = self.reduce_fc(x)  # [ B,  Group_channels ]
        x = torch.stack([self.group_fcs[i](x) for i in range(self.groups)], dim=1)  #output dim [B, Groups, Group_channels ]
        #x = self.gat1(x)  #output dim [B, Groups , Group_channels]
        #FIXME change from gatlayer to fc layer
        '''
        batch_size = x.size(2)
        group_channels = x.size(0)
        x=x.reshape([batch_size, self.groups*group_channels ])

        x=self.fc_replace_GALayer1(x)

        x=x.reshape([batch_size, self.groups , group_channels])
        '''
        # x = x.permute(1, 0, 2)  # [N, B, Group_channels ]
        count = 0
        outside = []
        for i in range(self.groups):
            inside = []
            for j in range(self.nclasses_per_group[i]):
                inside.append(self.class_fcs[count](x[:, i, :]))  # [B,Group_channels]
                count += 1
            inside = torch.stack(inside, dim=1)  # [B, nclasses_per_group ,Group_channels]
            #inside = self.gat2s[i](inside)  # [B, nclasses_per_group, Group_channels]
            outside.append(inside)
        x = torch.cat(outside, dim=1)  # [B,nclasses, Group_channels]
        x = torch.cat([self.fcs[i](x[:, i, :]) for i in range(self.nclasses)], dim=1)  # [B,nclasses]
        return x


class HGAT_FC(nn.Module):
    def __init__(self, backbone, groups, nclasses, nclasses_per_group, group_channels, class_channels):
        super(HGAT_FC, self).__init__()
        self.groups = groups
        self.nclasses = nclasses
        self.nclasses_per_group = nclasses_per_group
        self.group_channels = group_channels
        self.class_channels = class_channels
        if backbone == 'resnet101':
            model = models.resnet101(pretrained=False)
            print('load pretrained model...')
            model.load_state_dict(torch.load('./resnet101-5d3b4d8f.pth'))
        elif backbone == 'resnet50':
            model = models.resnet50(pretrained=False)
            print('load pretrained model...')
            model.load_state_dict(torch.load('./resnet50-5d3b4d8f.pth'))
        else:
            raise Exception()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4, )
        self.heads = Head(self.groups, self.nclasses, self.nclasses_per_group, group_channels=self.group_channels,
                          class_channels=self.class_channels)
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, x, inp):
        x = self.features(x)  # [B,2048,H,W]
        x = self.heads(x)
        return x

    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.features.parameters(), 'lr': lrp},
            {'params': self.heads.parameters(), 'lr': lr},
        ]


if __name__ == '__main__':
    model = HGAT_FC(backbone='resnet101', groups=12, nclasses=80,
                 nclasses_per_group=[1, 8, 5, 10, 5, 10, 7, 10, 6, 6, 5, 7], group_channels=512, class_channels=256)

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    print('')
    x = torch.zeros(2, 3, 448, 448).random_(0, 10)
    out = model(x)
    # model=models.(pretrained=False)
    # print('Number of model parameters: {}'.format(
    #     sum([p.data.nelement() for p in model.parameters()])))
