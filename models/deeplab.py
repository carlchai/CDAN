import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 函数模块
from torchvision.models.segmentation import deeplabv3_resnet50





class DeepLabHead(nn.Sequential):  # 定义 DeepLabHead 类，继承自 nn.Sequential
    def __init__(self, in_channels, num_classes):  # 初始化方法，参数包括输入通道数和类别数
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, 256),  # 添加 ASPP 模块
            nn.Conv2d(256, 256, 3, padding=1, bias=False),  # 添加卷积层
            nn.BatchNorm2d(256),  # 添加批量归一化层
            nn.ReLU(),  # 添加 ReLU 激活函数
            nn.Conv2d(256, num_classes, 1)  # 添加最后的卷积层，用于类别预测
        )


class ASPP(nn.Module):  # 定义 ASPP（空洞空间金字塔池化）类，继承自 nn.Module
    def __init__(self, in_channels, out_channels, atrous_rates=None):  # 初始化方法，参数包括输入通道数、输出通道数和空洞率列表
        super(ASPP, self).__init__()
        if atrous_rates is None:  # 如果没有提供空洞率列表，则使用默认值
            atrous_rates = [6, 12, 18]

        layers = []  # 创建一个空列表，用于存放 ASPP 模块的层
        # 添加一个卷积层、批量归一化层和 ReLU 激活函数
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))
        for rate in atrous_rates:  # 遍历空洞率列表
            layers.append(ASPPConv(in_channels, out_channels, rate))  # 添加 ASPPConv 层，使用当前空洞率

        self.convs = nn.ModuleList(layers)  # 将 layers 列表转换为 ModuleList
        self.global_pooling = nn.Sequential(  # 定义全局平均池化层
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.out_conv = nn.Sequential(  # 定义输出卷积层
            nn.Conv2d(out_channels * (2 + len(atrous_rates)), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):  # 定义 ASPP 类的前向传播方法
        x_pool = self.global_pooling(x)  # 对输入 x 进行全局平均池化
        x_pool = F.interpolate(x_pool, size=x.shape[2:], mode='bilinear', align_corners=False)  # 将池化结果上采样到原始尺寸
        x_aspp = [x_pool] + [conv(x) for conv in self.convs]  # 对输入 x 应用 ASPPConv 层
        x = torch.cat(x_aspp, dim=1)  # 将上采样的全局池化结果和 ASPPConv 层的结果沿通道维度拼接
        return self.out_conv(x)  # 应用输出卷积层


class ASPPConv(nn.Sequential):  # 定义 ASPPConv 类，继承自 nn.Sequential
    def __init__(self, in_channels, out_channels, dilation):  # 初始化方法，参数包括输入通道数、输出通道数和空洞率
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),  # 添加带空洞的卷积层
            nn.BatchNorm2d(out_channels),  # 添加批量归一化层
            nn.ReLU()  # 添加 ReLU 激活函数
        )

if __name__=='main':
    head = DeepLabHead(64,2)
