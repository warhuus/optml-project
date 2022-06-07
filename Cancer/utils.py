import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import random
from PIL import Image

from albumentations import pytorch as AT
from pytorchcv.model_provider import get_model as ptcv_get_model


cudnn.benchmark = True


SEED = 323
BASE_DIR = 'input/'


files = os.listdir(BASE_DIR + "train")
files_noext = [i.split(".")[0] for i in files]
labels = pd.read_csv(BASE_DIR + 'train_labels.csv')
labels = labels.loc[labels["id"].isin(files_noext)]
img_class_dict = {k:v for k, v in zip(labels.id, labels.label)}


def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def plot_sample_images():
    fig = plt.figure(figsize=(25, 4))
    train_imgs = os.listdir(BASE_DIR + "train")
    for idx, img in enumerate(np.random.choice(train_imgs, 20)):
        ax = fig.add_subplot(2, 20//2, idx+1, xticks=[], yticks=[])
        im = Image.open(BASE_DIR + "train/" + img)
        plt.imshow(im)
        lab = labels.loc[labels['id'] == img.split('.')[0], 'label'].values[0]
        ax.set_title('Label: %s'%lab)


class CancerDataset(Dataset):
    def __init__(self, dataset=None, datafolder=None, datatype='train', transform=transforms.Compose([transforms.CenterCrop(64), transforms.ToTensor()]), labels_dict={}):
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

data_transforms = albumentations.Compose([
    albumentations.Resize(224, 224),
    albumentations.RandomRotate90(p=0.5),
    albumentations.Transpose(p=0.5),
    albumentations.Flip(p=0.5),
    albumentations.OneOf([
        albumentations.CLAHE(clip_limit=2), albumentations.Sharpen(), albumentations.Emboss(), 
        albumentations.RandomBrightness(), albumentations.RandomContrast(),
        albumentations.JpegCompression(), albumentations.Blur(), albumentations.GaussNoise()], p=0.5), 
    albumentations.HueSaturationValue(p=0.5), 
    albumentations.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=45, p=0.5),
    albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    AT.ToTensorV2()
    ])

data_transforms_test = albumentations.Compose([
    albumentations.Resize(224, 224),
    albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    AT.ToTensorV2()
    ])

data_transforms_tta0 = albumentations.Compose([
    albumentations.Resize(224, 224),
    albumentations.RandomRotate90(p=0.5),
    albumentations.Transpose(p=0.5),
    albumentations.Flip(p=0.5),
    albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    AT.ToTensorV2()
    ])

data_transforms_tta1 = albumentations.Compose([
    albumentations.Resize(224, 224),
    albumentations.RandomRotate90(p=1),
    albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    AT.ToTensorV2()
    ])

data_transforms_tta2 = albumentations.Compose([
    albumentations.Resize(224, 224),
    albumentations.Transpose(p=1),
    albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    AT.ToTensorV2()
    ])

data_transforms_tta3 = albumentations.Compose([
    albumentations.Resize(224, 224),
    albumentations.Flip(p=1),
    albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    AT.ToTensorV2()
    ])

def get_dataloader():
    dataset = CancerDataset(datafolder=BASE_DIR + 'train/', datatype='test', transform=data_transforms_test, labels_dict=img_class_dict)
    # prepare data loaders (combine dataset and sampler)
    return torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)

def get_model(model_path: str = 'model.pt'):
    model_conv = ptcv_get_model("cbam_resnet50", pretrained=True)
    # model_conv = ptcv_get_model("bam_resnet50", pretrained=True)
    # model_conv = pretrainedmodels.se_resnext101_32x4d()
    model_conv.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    model_conv.last_linear = nn.Sequential(nn.Dropout(0.6), nn.Linear(in_features=2048, out_features=512, bias=True), nn.SELU(),
                                           nn.Dropout(0.8),  nn.Linear(in_features=512, out_features=1, bias=True))
    # don't think we need this line
    # model_conv.eval()
    saved_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model_conv.load_state_dict(saved_dict)
    return model_conv