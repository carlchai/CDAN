from collections import OrderedDict
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18,resnet50,resnet
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

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

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.layers = get_layers('resnet18')
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



        t0_1 = self.block1(t0)
        t1_1 = self.block1(t1)
        layer1 = self.Average_filter(t0_1,t1_1,self.conv1,self.ex1,self.conv1_)
        visual(t0_1.cpu(),t1_1.cpu(),layer1.cpu())


        # layer2  torch.Size([4, 64, 192, 256])
        t0_2 = self.block2(t0_1)
        t1_2 = self.block2(t1_1)
        layer2 = self.Average_filter(t0_2, t1_2,self.conv1,self.ex1,self.conv1_)
        visual(t0_2.cpu(), t1_2.cpu(), layer2.cpu())


        # layer3  torch.Size([4, 128, 96, 128])
        t0_3 = self.block3(t0_2)
        t1_3 = self.block3(t1_2)
        layer3 = self.Average_filter(t0_3, t1_3,self.conv2,self.ex2,self.conv2_)
        visual(t0_3.cpu(), t1_3.cpu(), layer3.cpu())


        # layer4 torch.Size([4, 256, 48, 64])
        t0_4 = self.block4(t0_3)
        t1_4 = self.block4(t1_3)
        layer4 = self.Average_filter(t0_4, t1_4,self.conv3,self.ex3,self.conv3_)
        visual(t0_4.cpu(), t1_4.cpu(), layer4.cpu())


        # layer5 torch.Size([4, 512, 24, 32])
        t0_5 = self.block5(t0_4)
        t1_5 = self.block5(t1_4)
        layer5 = self.Average_filter(t0_5, t1_5,self.conv4,self.ex4,self.conv4_)
        visual(t0_5.cpu(), t1_5.cpu(), layer5.cpu())


        l5 = self.conv_block1(layer5)
        l4 = self.conv_block2(layer4)
        l3 = self.conv_block3(layer3)
        l2 = self.conv_block4(layer2)
        l1 = self.conv_block4(layer1)

        l5 = F.interpolate(l5,l1.shape[-2:],mode="bilinear")
        l4 = F.interpolate(l4,l1.shape[-2:],mode="bilinear")
        l3 = F.interpolate(l3,l1.shape[-2:],mode="bilinear")

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


def visual(t1,t2,a):
    t1 = t1[0]
    t2 = t2[0]
    a = a[0]
    im1 = np.squeeze(t1.detach().numpy())  # 把tensor变成numpy
    im2 = np.squeeze(t2.detach().numpy())  # 把tensor变成numpy
    ima = np.squeeze(a.detach().numpy())  # 把tensor变成numpy
    # [C, H, W] -> [H, W, C]
    im1 = np.transpose(im1, [1, 2, 0])
    im2 = np.transpose(im2, [1, 2, 0])
    ima = np.transpose(ima, [1, 2, 0])
    h,w,c = im1.shape
    plt.figure(figsize=(h,w))
    ax1 = plt.subplot(1, 3, 1)
    im1 = np.mean(im1,axis=2)
    plt.imshow(im1, cmap='OrRd')
    plt.axis('off')
    ax2 = plt.subplot(1, 3, 2)
    im2 = np.mean(im2, axis=2)
    plt.imshow(im2, cmap='OrRd')
    plt.axis('off')
    ax3 = plt.subplot(1, 3, 3)
    ima = np.mean(ima, axis=2)
    plt.imshow(ima, cmap='OrRd')
    plt.axis('off')
    plt.show()

def get_layers(name):
    if 'resnet18' == name:
        replace_stride_with_dilation = [False, False, False]
        model = resnet.__dict__[name](
            pretrained=True,
            replace_stride_with_dilation=replace_stride_with_dilation)
    elif 'resnet50' == name:
        replace_stride_with_dilation = [False, True, True]
        model = resnet50(replace_stride_with_dilation=replace_stride_with_dilation)
    else:
        raise ValueError(name)
    layer0 = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
    layer1 = model.layer1
    layer2 = model.layer2
    layer3 = model.layer3
    layer4 = model.layer4
    return [layer0, layer1, layer2, layer3, layer4]


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

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tensor = torch.randn([4, 6, 768, 1024]).to(device)
    res18 = ResNet18()
    classifier = DeepLabHead(512, 2)
    model = DeepLabV3(res18,classifier,aux_classifier=None).to(device)
    output = model(tensor)


