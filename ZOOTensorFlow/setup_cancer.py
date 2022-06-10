from matplotlib.pyplot import get
import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import pickle

import albumentations
from albumentations import pytorch as AT
import cv2
import PIL.Image
import pandas as pd
import onnx


data_transforms_test = albumentations.Compose([
    albumentations.Resize(224, 224),
    albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    AT.ToTensorV2()
])

def get_labels(base_dir):
    files = os.listdir(os.path.join(base_dir, "train")) + os.listdir(os.path.join(base_dir, "test"))
    files_noext = [i.split(".")[0] for i in files]
    labels = pd.read_csv(os.path.join(base_dir, 'train_labels.csv'))
    return labels.loc[labels["id"].isin(files_noext)]

def get_img_class_dict(base_dir):
    labels = get_labels(base_dir)
    img_class_dict = {k:v for k, v in zip(labels.id, labels.label)}
    return img_class_dict

def load_images(datapath, img_class_dict):
    data = []
    labels = []

    for f in os.listdir(datapath):
        img_name = os.path.join(datapath, f)
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = data_transforms_test(image=img)
        image = image['image']

        img_name_short = f.split('.')[0]
        label = img_class_dict[img_name_short]

        data += [image.numpy()]
        labels += [np.array(label)]

    return data, labels


class CancerDataset:
    def __init__(self):
        
        datapath = os.path.join('Cancer', 'input', 'train')
        test_datapath = os.path.join('Cancer', 'input', 'test')

        if not os.path.exists(datapath):
            raise FileNotFoundError()

        img_class_dict = get_img_class_dict(os.path.join('Cancer', 'input'))

        data, labels = load_images(datapath, img_class_dict)
        train_data = np.array(data, dtype=np.float32)
        train_labels = np.array(labels)

        data, labels = load_images(test_datapath, img_class_dict)
        self.test_data = np.array(data, dtype=np.float32)
        self.test_labels = np.array(labels)

        VALIDATION_SIZE = 100
        
        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]


def get_cancer_model():
    return onnx.load(os.path.join("Cancer", "converted_onnx_model.onnx"))