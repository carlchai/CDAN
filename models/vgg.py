import torch
import torchvision
import torch.nn as nn

from models.backbone_base import Backbone

from collections import OrderedDict
from torch import nn
from torchvision.models import resnet18,resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3
import torch
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        out = self.relu(x + residual)
        return out

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.layers = get_layers('vgg16')
        # down block
        self.block1 = self.layers[0]
        self.block2 = self.layers[1]
        self.block3 = self.layers[2]
        self.block4 = self.layers[3]
        self.block5 = self.layers[4]
        self.relu = nn.ReLU(inplace=True)

        # merge change information
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,bias=False)
        self.conv1_ = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,bias=False)
        self.ex1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,bias=False)

        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,bias=False)
        self.conv2_ = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,bias=False)
        self.ex2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,bias=False)

        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,bias=False)
        self.conv3_ = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,bias=False)
        self.ex3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,bias=False)

        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,bias=False)
        self.conv4_ = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,bias=False)
        self.ex4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,bias=False)

        # merge spatial features
        self.conv_block1 = conv_block(512, 512, 512)
        self.conv_block2 = conv_block(256, 512, 512)
        self.conv_block3 = conv_block(128, 512, 512)
        self.conv_block4 = conv_block(64, 512, 512)

        #self.up = nn.ConvTranspose2d(512,512, 2, stride=2)
        self.reduce = nn.Sequential(
            nn.Conv2d(2560, 512, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU(inplace=True)
        self.initial(self.block1)


    def Average_filter(self,t1,t2,conv,ex,combine):
        H,W = t1.shape[2],t1.shape[3]
        zero = torch.zeros_like(t1)
        AM = torch.mean(t1-t2,dim=(2,3),keepdim=True)
        AM = AM.repeat(1,1,H,W)

        cm1 = conv(self.relu(t1-AM))
        cm2 = conv(self.relu(t2-AM))
        exchange = ex(torch.max(t1,t2)-torch.min(t1,t2))
        t = combine(t1+t2)
        info = cm1+cm2+exchange+t
        return info


    def initial(self, *models):
        for m in models:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = OrderedDict()
        t0,t1 = torch.split(x,3,dim=1)
        #layer1  torch.Size([4, 64, 192, 256])
        t0_1 = self.block1(t0)
        t1_1 = self.block1(t1)
        layer1 = self.Average_filter(t0_1,t1_1,self.conv1,self.ex1,self.conv1_)

        # layer2  torch.Size([4, 64, 192, 256])
        t0_2 = self.block2(t0_1)
        t1_2 = self.block2(t1_1)
        layer2 = self.Average_filter(t0_2, t1_2,self.conv2,self.ex2,self.conv2_)

        # layer3  torch.Size([4, 128, 96, 128])
        t0_3 = self.block3(t0_2)
        t1_3 = self.block3(t1_2)
        layer3 = self.Average_filter(t0_3, t1_3,self.conv3,self.ex3,self.conv3_)

        # layer4 torch.Size([4, 256, 48, 64])
        t0_4 = self.block4(t0_3)
        t1_4 = self.block4(t1_3)
        layer4 = self.Average_filter(t0_4, t1_4,self.conv4,self.ex4,self.conv4_)

        # layer5 torch.Size([4, 512, 24, 32])
        t0_5 = self.block5(t0_4)
        t1_5 = self.block5(t1_4)
        layer5 = self.Average_filter(t0_5, t1_5,self.conv4,self.ex4,self.conv4_)


        l5 = self.conv_block1(layer5)
        l4 = self.conv_block1(layer4)
        l3 = self.conv_block2(layer3)
        l2 = self.conv_block3(layer2)
        l1 = self.conv_block4(layer1)

        l5 = F.interpolate(l5,l1.shape[-2:],mode="bilinear")
        l4 = F.interpolate(l4,l1.shape[-2:],mode="bilinear")
        l3 = F.interpolate(l3,l1.shape[-2:],mode="bilinear")
        l2 = F.interpolate(l2,l1.shape[-2:],mode="bilinear")

        layer_list = [l1,l2,l3,l4,l5]

        single = torch.cat(layer_list,dim=1)  #4 2560 192 256
        out['out'] = self.reduce(single)
        #
        # #stage1
        # l5_up = self.up(l5)
        # l4_up = self.up(l4+l5_up)
        # l3_up = self.up(l3+l4_up)
        # l2_up = self.up(l2+l3_up+l1)
        # l1_up = self.up(l2_up)
        #
        #
        # l5_up2 = self.up_o1(l5)
        # l4_up2 = self.up_o2(l4)
        # l3_up2 = self.up_o3(l3)
        # l2_up2 = self.up_o4(l2)
        # l1_up2 = self.up_o4(l1)
        #
        # mergeAll = l1_up+l5_up2+l4_up2+l3_up2+l2_up2+l1_up2
        # x = self.relu(mergeAll)
        # out = self.head(x)
        return out



class conv_block(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3,stride=1, padding=1, bias=False)
    def forward(self, x):
        x = self.conv1(x)
        output = self.conv2(x)
        return output


class VGG(Backbone):
    def __init__(self, name):
        assert name in ['vgg16', 'vgg16_bn']
        self.name = name
        super(VGG, self).__init__(get_layers(name))


def get_layers(name):
    if name == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
        layer0 = model.features[:5]
        layer1 = model.features[5:10]
        layer2 = model.features[10:17]
        layer3 = model.features[17:24]
        layer4 = model.features[24:]
    elif name == 'vgg16_bn':
        model = torchvision.models.vgg16_bn(pretrained=True)
        layer0 = model.features[:7]
        layer1 = model.features[7:14]
        layer2 = model.features[14:24]
        layer3 = model.features[24:34]
        layer4 = model.features[34:]
    else:
        raise ValueError(name)
    return [layer0, layer1, layer2, layer3, layer4]




if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tensor = torch.randn([4, 6, 512, 512]).to(device)
    vgg16 = VGG16()
    classifier = DeepLabHead(512, 2)
    model = DeepLabV3(vgg16,classifier,aux_classifier=None).to(device)
    output = model(tensor)






