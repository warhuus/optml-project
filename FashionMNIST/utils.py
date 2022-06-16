import torch.nn as nn
from torch.utils.data import Dataset
import torch
import numpy as np

import math

class FashionCNN(nn.Module):
    
    def __init__(self):
        super(FashionCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out

def get_model(path: str, device) -> nn.Module:
    model = FashionCNN()
    model = model.to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

sigmoid = lambda x: 1 / (1 + math.exp(-x))

class LossCancer(object):
 
    def __init__(self, model, img, img_shape, true_lbl, device) -> None:
        self.device = device
        self.model = model
        if isinstance(img, torch.Tensor):
            self.true_img = img.to(self.device)
        elif isinstance(img, np.ndarray):
            self.true_img = img
        else:
            raise TypeError
        try:
            self.true_lbl = true_lbl[0]
        except IndexError:
            self.true_lbl = true_lbl
        self.lambda_ = 0.9
        self.shape = img_shape
	
    def __call__(self, delta):
        if isinstance(self.true_img, np.ndarray):
            self.true_img = torch.tensor(self.true_img)
        input = torch.reshape(self.true_img + torch.tensor(delta).to(self.device), self.shape)
        with torch.no_grad():
            if self.device == torch.device('cpu'):
                output = self.model(input.type(torch.FloatTensor))
            else:
                output = self.model(input.type(torch.cuda.FloatTensor))
        pr = output[:, 0][0].cpu().numpy()
        fx = sigmoid(pr)
        return ((fx - 1 - self.true_lbl) ** 2 +
                self.lambda_ * np.linalg.norm(delta, 0))

class LossFashionMnist(LossCancer):

    def __init__(self, target_class: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.target_class = target_class

    def __call__(self, delta):
        if isinstance(self.true_img, np.ndarray):
            self.true_img = torch.tensor(self.true_img)
        input = torch.reshape(self.true_img + torch.tensor(delta).to(self.device), (-1, *self.shape))
        with torch.no_grad():
            if torch.cuda.is_available():
                output = self.model(input.type(torch.cuda.FloatTensor))
            else:
                output = self.model(input.type(torch.FloatTensor))
        pr = torch.nn.functional.softmax(output[0], dim=0).cpu().numpy()
        maxZxi = max(np.delete(pr, self.target_class))
        Zxt = pr[self.target_class]
        return max(maxZxi - Zxt, 0) + self.lambda_ * np.linalg.norm(delta, 0, axis=1)
