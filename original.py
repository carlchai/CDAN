import utils.metric_util
from utils.parser import args
#from models.Models import SNUNet_ECAM
from torch import nn
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
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
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

class ConcatImages(object):
    def __call__(self, image, target):
        image = torch.cat(image, dim=0)
        return image, target



def Train():
    best = -1
    for epoch in range(args.epochs):
        metric_logger = metric_util.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', metric_util.SmoothedValue(window_size=1, fmt='{value}'))
        metric_logger.add_meter('f1score',metric_util.SmoothedValue(window_size=1, fmt='{value:.4f} ({global_avg:.4f})'))
        for images,target in train_dataloader:
            image, target = images.to(device), target.to(device)
            output = model(image)
            loss = criterion(output,target[:,0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if isinstance(output, OrderedDict):
                output = output['out']
            mask_pred = torch.topk(output.data, 1, dim=1)[1][:, 0]
            mask_gt = (target > 0)[:, 0]
            _, _, _, f1score = metric_util.CD_metric_torch(mask_pred, mask_gt)
            metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
            metric_logger.f1score.update(f1score.mean(), n=len(f1score))
        if epoch % args.eval_every == 0:
            f1score = CD_evaluate(model, train_dataloader, device=device)
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



#VLCMUCD

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed_torch(seed=777)
    batchsize = 4
    args = get_args_parser().parse_args()
    args.num_classes = 2
    args.output_dir = 'ChangeSim/raw/output_cs_c3po'
    trainset =get_ChangeSim(args,train=True)
    testset = get_ChangeSim(args,train=False)
    train_dataloader = torch.utils.data.DataLoader(trainset,batch_size=1,num_workers=8)
    test_dataloader = torch.utils.data.DataLoader(testset,batch_size=1,num_workers=8)
    model = resnet18_mtf_msf_deeplabv3(args)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, 0.999),
                                 weight_decay=args.weight_decay)
    lambda_lr = get_scheduler_function(args.lr_scheduler, args.epochs * len(train_dataloader), final_lr=0.2*args.lr)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    ratio = trainset.get_mask_ratio()
    loss_weight = torch.tensor(ratio).cuda()
    criterion = loss.get_loss(args.loss, loss_weight)
    model_path = "ChangeSim/raw/output_cs_c3po/best.pth"
    save_imgs_dir = "ChangeSim/raw/output_cs_c3po_img"
    Train()
    save_img(model,model_path,save_imgs_dir,test_dataloader,device)




