
from os.path import join as pjoin
import os

import PIL
import torch.utils.data as data
import torchvision
from PIL import Image,ImageOps
from utils import transforms as tr
from utils.helpers import path_sort
import glob
import utils.transforms2 as T
import torch
from utils.dataset import CDDataset

def get_image_path(data_dir):
    """
    Dataset structure
    /
    ├───Train
    │     ├─── A(image_l)
    │     ├─── B(image_r)
    │     └─── label
    ├───Val
    │     ├─── A(image_l)
    │     ├─── B(image_r)
    │     └─── label
    Args:
        data_dir: The root of data directory.

    Returns: data_path

    """
    data_path = []

    image_files_l = glob.glob("{}/A/*.png".format(data_dir))
    image_files_r = glob.glob("{}/B/*.png".format(data_dir))
    label_files = glob.glob("{}/OUT/*.png".format(data_dir))

    image_files_l = path_sort(image_files_l)
    image_files_r = path_sort(image_files_r)
    label_files = path_sort(label_files)

    for image_l, image_r, label in zip(image_files_l, image_files_r, label_files):
        data_path.append({"image_l": image_l, "image_r": image_r, "label": label})

    return data_path


def cdd_loader(image_l_path, image_r_path, label_path, aug):
    """
    Image loader.
    Args:
        image_l_path: Image path of A
        image_r_path: Image path of B
        label_path: Image path of label
        aug: Whether use data augmentation
    Returns: image_l, image_r, label

    """
    image_l = Image.open(image_l_path)
    image_r = Image.open(image_r_path)
    label = Image.open(label_path)

    sample = {'image_l': image_l, 'image_r': image_r, 'label': label}

    if aug:
        sample = tr.train_transforms(sample)
    else:
        sample = tr.test_transforms(sample)

    return sample['image_l'], sample['image_r'], sample['label']


class CDDloader(data.Dataset):

    def __init__(self, data_path, aug=False):
        self.data_path = data_path
        self.aug = aug
        self.gt = []

    def __getitem__(self, index):
        image_l_path = self.data_path[index]['image_l']
        image_r_path = self.data_path[index]['image_r']
        label_path = self.data_path[index]['label']

        return cdd_loader(image_l_path, image_r_path, label_path, self.aug)
    def __len__(self):
        return len(self.data_path)


    def get_mask_raw(self,index):
        image_l_path = self.data_path[index]['image_l']
        image_r_path = self.data_path[index]['image_r']
        label_path = self.data_path[index]['label']
        self.gt = cdd_loader(image_l_path, image_r_path, label_path, self.aug)
        return self.gt

    def get_mask_ratio(self):
        print(self.gt)
        all_count = 0
        mask_count = 0
        for i in range(len(self.gt)):
            mask = self.get_mask_raw(i)
            target = (T.ToTensor(mask) != 0).long()
            mask_count += target.sum()
            all_count += target.numel()
        mask_ratio = mask_count / float(all_count)
        background_ratio = (all_count - mask_count) / float(all_count)
        return [mask_ratio, background_ratio]



def get_data_loader(data_dir, batch_size, aug=True):
    """
    get torch  DataLoader
    Args:
        data_dir: The root of data directory.
        batch_size: The number image of one batch
        aug: Whether use data augmentation

    Returns: torch DataLoader

    """
    data_path = get_image_path(data_dir)
    dataset = CDDloader(data_path, aug=aug)

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         num_workers=8)

    return loader,dataset


class VL_CMU_CD_Raw(CDDataset):
    # all images are 1024x768
    def __init__(self, root, transforms=None, revert_transforms=None):
        super(VL_CMU_CD_Raw, self).__init__(root, transforms)
        self.root = root
        self.gt, self.t0, self.t1 = self._init_data_list()
        self._transforms = transforms
        self._revert_transforms = revert_transforms

    def _init_data_list(self):
        gt = []
        t0 = []
        t1 = []
        sub_class = list(f for f in os.listdir(self.root) if os.path.isdir(pjoin(self.root, f)))
        for c in sub_class:
            img_root = pjoin(self.root, c, 'RGB')
            mask_root = pjoin(self.root, c, 'GT')
            for f in os.listdir(mask_root):
                if self._check_validness(f):
                    gt.append(pjoin(mask_root, f))
                    t0.append(pjoin(img_root, f.replace("gt", "1_")))
                    t1.append(pjoin(img_root, f.replace("gt", "2_")))
        return gt, t0, t1

    def get_raw(self, index):
        imgs, mask = super(VL_CMU_CD_Raw, self).get_raw(index)

        # x == 255 is sky
        mask = mask.point(lambda x: int(0 < x < 255) * 255)
        return imgs, mask


class ChangeSim(CDDataset):
    def __init__(self, root,train=True ,transforms=None, revert_transforms=None):
        super(ChangeSim, self).__init__(root, transforms)
        self.root = root
        self.gt, self.t0, self.t1 = self._init_data_list()
        self._transforms = transforms
        self.istrain = train
        self.num = 0
        self._revert_transforms = revert_transforms

    def _init_data_list(self):
        gt = []
        t0 = []
        t1 = []
        for file in os.listdir(os.path.join(self.root, 'mask')):
            if self._check_validness(file):
                gt.append(pjoin(self.root, 'mask', file))
                t0.append(pjoin(self.root, 't0', file))
                t1.append(pjoin(self.root, 't1', file))
        return gt, t0, t1

    def get_raw(self, index):
        imgs, mask = super(ChangeSim, self).get_raw(index)
        mask = mask.point(lambda x: int(0 < x < 255) * 255)
        return imgs, mask

def get_transforms(args, train, size_dict=None):
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)
    if size_dict is not None:
        assert args.input_size in size_dict, "input_size: {}".format(size_dict.keys())
        input_size = size_dict[args.input_size]
    else:
        input_size = args.input_size

    mode = "Train" if train else "Test"
    print("{} Aug:".format(mode))
    augs = []
    if train:
        if args.randomcrop:
            if args.input_size == 256:
                augs.append(T.Resize(286))
                augs.append(T.RandomCrop(input_size))
            elif args.input_size == 512:
                augs.append(T.Resize((572, 572)))
                augs.append(T.RandomCrop(512))
            else:
                raise ValueError(args.input_size)
        else:
            augs.append(T.Resize(input_size))
        augs.append(T.RandomHorizontalFlip(args.randomflip))
    else:
        augs.append(T.Resize(input_size))
    augs.append(T.ToTensor())
    augs.append(T.Normalize(mean=mean, std=std))
    augs.append(T.ConcatImages())
    transforms = T.Compose(augs)
    revert_transforms = T.Compose([
        T.SplitImages(),
        T.RevertNormalize(mean=mean, std=std),
        T.ToPILImage()
    ])
    return transforms, revert_transforms

def get_VL_CMU_CD_Raw(args, train=True):
    mode = 'train' if train else 'test'
    raw_root = './dataset/VL-CMU-CD'
    size_dict = {
        512: (512, 512),
        768: (768, 1024),
        640: (640, 480)
    }
    transforms, revert_transforms = get_transforms(args, train, size_dict)
    dataset = VL_CMU_CD_Raw(os.path.join(raw_root, mode),
        transforms=transforms, revert_transforms=revert_transforms)
    print("VL_CMU_CD_Raw {}: {}".format(mode, len(dataset)))
    return dataset


def get_ChangeSim(args, train=True):
    mode = 'train' if train else 'test'
    raw_root = './dataset/ChangeSim/vis'
    size_dict = {
        640: (480,640),
    }
    transforms, revert_transforms = get_transforms(args, train, size_dict)
    dataset = ChangeSim(raw_root,train,
        transforms=transforms, revert_transforms=revert_transforms)
    print("Change_Sim {}: {}".format(mode, len(dataset)))
    return dataset


class PCD_Raw(CDDataset):
    # all images are 224x1024
    # object: black(0)  ->   white(255)  ->  True
    #                 invert           toTensor
    def __init__(self, root, num=0, train=True, transforms=None, revert_transforms=None):
        super(PCD_Raw, self).__init__(root, transforms)
        assert num in [0, 1, 2, 3, 4]
        self.root = root
        self.num = num
        self.istrain = train
        self.gt, self.t0, self.t1 = self._init_data_list()
        self._transforms = transforms
        self._revert_transforms = revert_transforms

    def _init_data_list(self):
        gt = []
        t0 = []
        t1 = []
        for file in os.listdir(os.path.join(self.root, 'mask')):
            if self._check_validness(file):
                idx = int(file.split('.')[0])
                img_is_test = self.num * 2 <= (idx % 10) < (self.num + 1) * 2
                if (self.istrain and not img_is_test) or (not self.istrain and img_is_test):
                    gt.append(pjoin(self.root, 'mask', file))
                    t0.append(pjoin(self.root, 't0', file))
                    t1.append(pjoin(self.root, 't1', file))
        return gt, t0, t1

    def get_raw(self, index):
        imgs, mask = super(PCD_Raw, self).get_raw(index)
        #mask = mask.point(lambda x: int(0 < x < 255) * 255)
        mask = PIL.ImageOps.invert(mask)
        return imgs, mask

class LCD_Raw(CDDataset):
    # all images are 224x1024
    # object: black(0)  ->   white(255)  ->  True
    #                 invert           toTensor
    def __init__(self, root, num=0, train=True, transforms=None, revert_transforms=None):
        super(LCD_Raw, self).__init__(root, transforms)
        assert num in [0, 1, 2, 3, 4]
        self.root = root
        self.num = num
        self.istrain = train
        self.gt, self.t0, self.t1 = self._init_data_list()
        self._transforms = transforms
        self._revert_transforms = revert_transforms

    def _init_data_list(self):
        gt = []
        t0 = []
        t1 = []
        for file in os.listdir(os.path.join(self.root, 'mask')):
            if self._check_validness(file):
                gt.append(pjoin(self.root, 'mask', file))
                t0.append(pjoin(self.root, 't0', file))
                t1.append(pjoin(self.root, 't1', file))
        return gt, t0, t1

    def get_raw(self, index):
        imgs, mask = super(LCD_Raw, self).get_raw(index)
        #mask = mask.point(lambda x: int(0 < x < 255) * 255)
        #mask = PIL.ImageOps.invert(mask)
        return imgs, mask

def get_pcd_raw(args, num=0, train=True):
    assert num in [0, 1, 2, 3, 4]
    root = './dataset/TSUNAMI'
    input_size = args.input_size
    size_dict = {
        224: (224, 1024),
        256: (224, 256),
        512: (512,512),
        640:(480,640)
    }
    assert input_size in size_dict, "input_size: {}".format(size_dict.keys())
    transforms, revert_transforms = get_transforms(args, train, size_dict)
    dataset = PCD_Raw(root, num, train, transforms=transforms, revert_transforms=revert_transforms)
    dataset.name = 'TSUNAMI'
    mode = "Train" if train else "Test"
    print("changesim_{} {}: {}".format( num, mode, len(dataset)))
    return dataset

def get_changesim_raw(args, num=0, train=True):
    assert num in [0, 1, 2, 3, 4]
    root = './dataset/ChangeSim/raw'
    input_size = args.input_size
    size_dict = {
        224: (224, 1024),
        256: (224, 256),
        448: (448, 2048),
        640:(480,640)
    }
    assert input_size in size_dict, "input_size: {}".format(size_dict.keys())
    transforms, revert_transforms = get_transforms(args, train, size_dict)
    dataset = PCD_Raw(root, num, train, transforms=transforms, revert_transforms=revert_transforms)
    dataset.name = 'changesim'
    mode = "Train" if train else "Test"
    print("changesim_{} {}: {}".format( num, mode, len(dataset)))
    return dataset

def get_LevirCD_raw(args, num=0, train=True):
    assert num in [0, 1, 2, 3, 4]
    root = './dataset/LEVIR-CD/train' if train else './dataset/LEVIR-CD/test'
    input_size = args.input_size
    size_dict = {
        512: (512, 512),
        256: (256, 256),
        1024:(1024, 1024)
    }
    assert input_size in size_dict, "input_size: {}".format(size_dict.keys())
    transforms, revert_transforms = get_transforms(args, train, size_dict)
    dataset = LCD_Raw(root, num, train, transforms=transforms, revert_transforms=revert_transforms)
    dataset.name = 'LEVIR-CD'
    mode = "Train" if train else "Test"
    print("LEVIR-CD_{} {}: {}".format( num, mode, len(dataset)))
    return dataset