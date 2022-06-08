import functools
import collections
import os
import cv2
import random

import albumentations
from albumentations import pytorch as AT
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn


cudnn.benchmark = True


class memoized(object):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    '''
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__

    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)


@memoized
def get_labels(base_dir):
    files = os.listdir(base_dir + "train")
    files_noext = [i.split(".")[0] for i in files]
    labels = pd.read_csv(base_dir + 'train_labels.csv')
    return labels.loc[labels["id"].isin(files_noext)]

def get_img_class_dict(base_dir):
    labels = get_labels(base_dir)
    img_class_dict = {k:v for k, v in zip(labels.id, labels.label)}
    return img_class_dict

def seed_everything(seed=323):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def plot_sample_images(base_dir):
    fig = plt.figure(figsize=(25, 4))
    train_imgs = os.listdir(base_dir + "train")
    labels = get_labels(base_dir)
    for idx, img in enumerate(np.random.choice(train_imgs, 20)):
        ax = fig.add_subplot(2, 20//2, idx+1, xticks=[], yticks=[])
        im = Image.open(base_dir + "train/" + img)
        plt.imshow(im)
        lab = labels.loc[labels['id'] == img.split('.')[0], 'label'].values[0]
        ax.set_title('Label: %s'%lab)


class CancerDataset(Dataset):
    def __init__(self, dataset=None, datafolder=None, datatype='train',
                 transform=transforms.Compose([transforms.CenterCrop(64), transforms.ToTensor()]),
                 labels_dict={}):
        self.dataset = dataset
        self.datafolder = datafolder
        self.datatype = datatype
        self.image_files_list = [s for s in os.listdir(datafolder)]
        self.transform = transform
        self.labels_dict = labels_dict
        if self.datatype == 'train':
            self.labels = [labels_dict[i.split('.')[0]] for i in self.image_files_list]
        else:
            self.labels = [0 for _ in range(len(self.image_files_list))]

    @classmethod
    def attack(cls, dataset, transform=transforms.Compose([transforms.CenterCrop(64), transforms.ToTensor()])):
        return cls(dataset, None, 'attack', transform)
        
    def __len__(self):
        return len(self.image_files_list)

    def __getitem__(self, idx):
        if self.datatype != 'attack':
            img_name = os.path.join(self.datafolder, self.image_files_list[idx])
            img = cv2.imread(img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = self.transform(image=img)
            image = image['image']

            img_name_short = self.image_files_list[idx].split('.')[0]

            if self.datatype == 'train':
                label = self.labels_dict[img_name_short]
            else:
                label = 0
            return image, label
        else:
            return self.dataset[idx], 0


def get_dataloader(base_dir):
    img_class_dict = get_img_class_dict(base_dir)
    data_transforms_test = albumentations.Compose([
        albumentations.Resize(224, 224),
        albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        AT.ToTensorV2()
    ])
    dataset = CancerDataset(datafolder=base_dir + 'train/', datatype='test', transform=data_transforms_test, labels_dict=img_class_dict)
    # prepare data loaders (combine dataset and sampler)
    return torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)

def get_model(base_dir):
    model_conv = ptcv_get_model("cbam_resnet50", pretrained=True)
    model_conv.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    model_conv.last_linear = nn.Sequential(nn.Dropout(0.6), nn.Linear(in_features=2048, out_features=512, bias=True), nn.SELU(),
                                           nn.Dropout(0.8),  nn.Linear(in_features=512, out_features=1, bias=True))
    # don't think we need this line
    # model_conv.eval()
    saved_dict = torch.load(base_dir + 'model.pt', map_location=torch.device('cpu'))
    model_conv.load_state_dict(saved_dict)
    return model_conv