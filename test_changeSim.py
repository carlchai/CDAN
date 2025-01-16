from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3

import utils.metric_util
from models.AverageFilter import ResNet18
from torch.nn import functional as F
from utils.helpers import *
from utils.metrics import generate_matrix, get_mean_iou, get_recall, get_precision
from utils import metric_util
from utils.dataloaders import get_data_loader,get_VL_CMU_CD_Raw,get_pcd_raw,get_ChangeSim
import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import warnings
import loss
import torchvision.transforms as T
from  models.DeepLabV3 import resnet18_mtf_msf_deeplabv3
warnings.filterwarnings("ignore")
from collections import OrderedDict
import math
from PIL import Image

def CD_evaluate(model, data_loader, device, save_imgs_dir=None):
    model.eval()
    metric_logger = utils.metric_util.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Prec', utils.metric_util.SmoothedValue(window_size=1, fmt='{value:.3f} ({global_avg:.3f})'))
    metric_logger.add_meter('Rec', utils.metric_util.SmoothedValue(window_size=1, fmt='{value:.3f} ({global_avg:.3f})'))
    metric_logger.add_meter('Acc', utils.metric_util.SmoothedValue(window_size=1, fmt='{value:.3f} ({global_avg:.3f})'))
    metric_logger.add_meter('F1score', utils.metric_util.SmoothedValue(window_size=1, fmt='{value:.4f} ({global_avg:.4f})'))
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            if isinstance(output, OrderedDict):
                output = output['out']
            mask_pred = torch.topk(output.data, 1, dim=1)[1][:, 0]
            mask_gt = (target > 0)[:, 0]
            precision, recall, accuracy, f1score = utils.metric_util.CD_metric_torch(mask_pred, mask_gt)
            metric_logger.Prec.update(precision.mean(), n=len(precision))
            metric_logger.Rec.update(recall.mean(), n=len(precision))
            metric_logger.Acc.update(accuracy.mean(), n=len(precision))
            metric_logger.F1score.update(f1score.mean(), n=len(f1score))
            if save_imgs_dir:
                assert len(precision) == 1, "save imgs needs batch_size=1"
                output_pil = data_loader.dataset.get_pil(image[0], mask_gt, mask_pred)
                output_pil.save(os.path.join(save_imgs_dir, "{}_{}.png".format(utils.metric_util.get_rank(), metric_logger.F1score.count)))
        metric_logger.synchronize_between_processes()

    print("{} {} Total: {} Metric Prec: {:.4f} Recall: {:.4f} F1: {:.4f}".format(
        header,
        data_loader.dataset.name,
        metric_logger.F1score.count,
        metric_logger.Prec.global_avg,
        metric_logger.Rec.global_avg,
        metric_logger.F1score.global_avg
    ))
    return metric_logger.F1score.global_avg


def get_scheduler_function(name, total_iters, final_lr=0):
    print("LR Scheduler: {}".format(name))
    if name == 'cosine':
        return lambda step: ((1 + math.cos(step * math.pi / total_iters)) / 2) * (1 - final_lr) + final_lr
    elif name == 'linear':
        return lambda step: 1 - (1 - final_lr) / total_iters * step
    elif name == 'exp':
        return lambda step: (1 - step / total_iters) ** 0.9
    elif name == 'none':
        return lambda step: 1
    else:
        raise ValueError(name)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch change detection', add_help=add_help)
    parser.add_argument('--model', default='resnet18_mtf_msf_deeplabv3', help='model')
    parser.add_argument('--mtf', default='iade', help='choose branches to use')
    parser.add_argument('--msf', default=4, type=int, help='the number of MSF layers')
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)',
                        dest='weight_decay')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr-scheduler', default='cosine', type=str, help='the lr scheduler')
    parser.add_argument('--input-size', default=640, type=int, metavar='N',
                        help='the input-size of images')
    parser.add_argument('--eval-every', default=1, type=int, metavar='N',
                        help='eval the model every n epoch')
    parser.add_argument('--loss', default='bi', type=str, help='the training loss')
    parser.add_argument('--randomflip', default=0.5, type=float, help='random flip input')
    parser.add_argument('--randomrotate', dest="randomrotate", action="store_true", help='random rotate input')
    parser.add_argument('--randomcrop', dest="randomcrop", action="store_true", help='random crop input')

    return parser



class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, is_bn=True):
        x = self.conv_relu(x)
        if is_bn:
            x = self.bn(x)
        return x



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.down1 = Downsample(4, 64)
        self.down2 = Downsample(64, 128)
        self.down3 = Downsample(128, 256)
        self.conv = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn = nn.BatchNorm2d(512)
        self.last = nn.Conv2d(512, 1, 3, 1)

    def forward(self, anno, img):
        x = torch.cat([anno, img], dim=1)  # batch*6*H*W
        x = self.down1(x, is_bn=False)
        x = self.down2(x)
        x = F.dropout2d(self.down3(x))
        x = F.dropout2d(F.leaky_relu(self.conv(x)))
        x = F.dropout2d(self.bn(x))
        x = torch.sigmoid(self.last(x))
        return x


def Train():
    best = -1
    f1_list = []
    loss_list = []
    for epoch in range(args.epochs):
        l=[]
        metric_logger = metric_util.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', metric_util.SmoothedValue(window_size=1, fmt='{value}'))
        metric_logger.add_meter('f1score',metric_util.SmoothedValue(window_size=1, fmt='{value:.4f} ({global_avg:.4f})'))
        with tqdm(total=len(train_dataloader)) as pbar:
            for images,target in train_dataloader:
                optimizer.zero_grad()
                d_optimizer.zero_grad()
                image, target = images.to(device), target.to(device)  # 4 6 768 1024  ,  4 1 768 1024
                image1, image2 = torch.split(image, 3, dim=1)
                difference = image1 - image2
                output = model(image)['out']
                real_pred = dis(difference, target)
                d_real_loss = dis_criterion(real_pred, torch.ones_like(real_pred, device=device))
                generator = torch.argmax(output, 1, keepdim=True)
                d_fake = dis(difference, generator)
                d_fake_loss = dis_criterion(d_fake, torch.zeros_like(d_fake, device=device))
                discriminate_loss = d_fake_loss + d_real_loss
                discriminate_loss.backward()
                loss = criterion(output, target[:, 0])
                loss.backward()
                d_optimizer.step()
                optimizer.step()
                mask_pred = torch.topk(output.data, 1, dim=1)[1][:, 0]# 4 768 1024
                mask_gt = (target > 0)[:, 0]# 4 768 1024
                _, _, _, f1score = metric_util.CD_metric_torch(mask_pred, mask_gt)
                metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
                metric_logger.f1score.update(f1score.mean(), n=len(f1score))
                l.append(loss.item())
                pbar.set_postfix(epoch=f'{epoch:.1f}',loss=f'{loss.item():.4f}',f1score=f'{f1score.mean():.4f}')  # 更新进度条的后缀以显示平均损失
                pbar.update(1)
        if epoch % args.eval_every == 0:
            f1score = CD_evaluate(model, train_dataloader, device=device)
            f1_list.append(f1score)
            loss_list.append(sum(l)/len(l))
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args
        }
        if f1score > best:
            best = f1score
            metric_util.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'best.pth'))
        metric_util.save_on_master(
            checkpoint,
            os.path.join(args.output_dir, 'checkpoint.pth'))
    print(f1_list,loss_list)



def save_img(model,model_path,save_imgs_dir,data_loader,device):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    metric_logger = utils.metric_util.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Prec',utils.metric_util.SmoothedValue(window_size=1, fmt='{value:.3f} ({global_avg:.3f})'))
    metric_logger.add_meter('Rec', utils.metric_util.SmoothedValue(window_size=1, fmt='{value:.3f} ({global_avg:.3f})'))
    metric_logger.add_meter('Acc', utils.metric_util.SmoothedValue(window_size=1, fmt='{value:.3f} ({global_avg:.3f})'))
    metric_logger.add_meter('F1score',utils.metric_util.SmoothedValue(window_size=1, fmt='{value:.4f} ({global_avg:.4f})'))
    m_precision = []
    m_recall = []
    m_acc = []
    m_f1 = []
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 254, header='Save:'):
            image, target = image.to(device), target.to(device)
            output = model(image)
            if isinstance(output, OrderedDict):
                output = output['out']
            mask_pred = torch.topk(output.data, 1, dim=1)[1][:, 0]
            mask_gt = (target > 0)[:, 0]
            precision, recall, accuracy, f1score = utils.metric_util.CD_metric_torch(mask_pred, mask_gt)
            m_precision.append(precision)
            m_recall.append(recall)
            m_acc.append(accuracy)
            m_f1.append(f1score)
            pre = sum(m_precision)/len(m_precision)
            rec = sum(m_recall)/len(m_recall)
            acc = sum(m_acc)/len(m_acc)
            f1s = sum(m_f1)/len(m_f1)
            metric_logger.Prec.update(precision.mean(), n=len(precision))
            metric_logger.Rec.update(recall.mean(), n=len(precision))
            metric_logger.Acc.update(accuracy.mean(), n=len(precision))
            metric_logger.F1score.update(f1score.mean(), n=len(f1score))
            assert len(precision) == 1, "save imgs needs batch_size=1"
            output_pil,mask,pred,t0 = data_loader.dataset.get_pil(image[0], mask_gt, mask_pred)
            output_pil.save(os.path.join(save_imgs_dir, "{}_{}.png".format(utils.metric_util.get_rank(), metric_logger.F1score.count)))
            mask.save(os.path.join(save_imgs_dir+"1", "{}_{}.png".format(utils.metric_util.get_rank(), metric_logger.F1score.count)))
            pred.save(os.path.join(save_imgs_dir+"2", "{}_{}.png".format(utils.metric_util.get_rank(), metric_logger.F1score.count)))
            t0.save(os.path.join(save_imgs_dir+"3", "{}_{}.png".format(utils.metric_util.get_rank(), metric_logger.F1score.count)))
        print("precision:{}  recall: {}  accuracy:{} f1score:{}".format(pre, rec, acc, f1s))


def  get_GT(model,model_path,save_imgs_dir1,save_imgs_dir2,data_loader,device):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    with torch.no_grad():
        count = 1
        for image, target in data_loader:
            image, target = image.to(device), target.to(device)
            pil_t0,pil_t1 = data_loader.dataset.get_T(image[0])
            pil_t0.save(save_imgs_dir1+"/{}_{}.png".format(utils.metric_util.get_rank(),count))
            pil_t1.save(save_imgs_dir2+"/{}_{}.png".format(utils.metric_util.get_rank(),count))
            count = count+1

class Bicriterion(nn.Module):
    def __init__(self, mask_loss):
        super(Bicriterion, self).__init__()
        self.mask_loss = mask_loss

    def forward(self, output, target):
        mask_loss = self.mask_loss(output, target)
        return mask_loss

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed_torch(seed=777)
    batchsize = 6
    args = get_args_parser().parse_args()
    args.num_classes = 2
    args.output_dir = 'PCD/raw/'
    trainset =get_pcd_raw(args,train=True)
    testset = get_pcd_raw(args,train=False)
    train_dataloader = torch.utils.data.DataLoader(trainset,batch_size=batchsize,num_workers=8)
    test_dataloader = torch.utils.data.DataLoader(testset,batch_size=1,num_workers=8)
    res18 = ResNet18()
    classifier = DeepLabHead(512, 2)
    # model = resnet18_mtf_msf_deeplabv3(args)
    # model.to(device)
    model = DeepLabV3(res18, classifier, aux_classifier=None).to(device)
    dis = Discriminator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, 0.999),weight_decay=args.weight_decay)
    d_optimizer = torch.optim.Adam(dis.parameters(), lr=2e-4, betas=(0.5, 0.999))
    ratio = trainset.get_mask_ratio()
    loss_weight = torch.tensor(ratio).cuda()
    criterion = Bicriterion(nn.CrossEntropyLoss(loss_weight))
    dis_criterion = nn.BCELoss()
    lambda_lr = get_scheduler_function(args.lr_scheduler, args.epochs * len(train_dataloader), final_lr=0.2 * args.lr)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    #Train()
    model_path = "ChangeSim/dust/output_changesim_AF/best.pth"
    save_imgs_dir = "output_img"
    save_img(model, model_path, save_imgs_dir, test_dataloader, device)




