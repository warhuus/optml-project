import torch.nn as nn
from torch.utils.data import Dataset
import torch
import numpy as np
import torchvision.models as models
from torchvision import transforms

import math

class ResNetFeatrueExtractor18(nn.Module):
    def __init__(self, pretrained = True):
        super(ResNetFeatrueExtractor18, self).__init__()
        model_resnet18 = models.resnet18(pretrained=pretrained)
        self.conv1 = model_resnet18.conv1
        self.bn1 = model_resnet18.bn1
        self.relu = model_resnet18.relu
        self.maxpool = model_resnet18.maxpool
        self.layer1 = model_resnet18.layer1
        self.layer2 = model_resnet18.layer2
        self.layer3 = model_resnet18.layer3
        self.layer4 = model_resnet18.layer4
        self.avgpool = model_resnet18.avgpool
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        return out

def get_model(path: str, device) -> nn.Module:
    model = ResNetFeatrueExtractor18()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    model.to(device)
    return model


class LossFashionMnist(object):
    def __init__(self, model, img, img_shape, true_lbl, device) -> None:
        self.device = device
        self.model = model
        self.true_img = img
        try:
            self.true_lbl = true_lbl[0]
        except IndexError:
            self.true_lbl = true_lbl
        self.lambda_ = 0.9
        self.shape = img_shape
        self.transform = transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1))

    def __call__(self, delta):
        self.pr = self.pr_func(delta)
        max_Fx_i_neq_t0 = np.log(max(np.delete(self.pr, self.true_lbl)) + 1e-6)
        Fx_t0 = np.log(self.pr[self.true_lbl] + 1e-6)
        return max(Fx_t0 - max_Fx_i_neq_t0, 0) + self.lambda_ * np.linalg.norm(delta, 2, axis=1)

    def pr_func(self, delta):
        input = self.transform(torch.tensor(np.reshape(self.true_img + delta, (-1, *self.shape)))).to(self.device)
        with torch.no_grad():
            if self.device == torch.device('cpu'):
                output = self.model(input.type(torch.FloatTensor))
            else:
                output = self.model(input.type(torch.cuda.FloatTensor))
        return torch.nn.functional.softmax(output[0], dim=0).cpu().numpy()


def generate_data(test_loader: Dataset, model: torch.nn, device: torch.device):

    inputs = []
    targets = []
    num_label = 10
    model.to(device)
    num_samples_per_label = np.zeros(num_label)

    for data in test_loader:

        # get the data and label
        image, label = data[0], data[1]

        # evaluate the model
        image_3channels = transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1))(image)
        with torch.no_grad():
            if device == torch.device('cpu'):
                output = model(image_3channels.type(torch.FloatTensor)).to(device)
            else:
                output = model(image_3channels.type(torch.cuda.FloatTensor)).to(device)

        pr = torch.nn.functional.softmax(output[0], dim=0).cpu().numpy()
        pred_class = np.argmax(pr)

        if pred_class != int(label):
            continue

        if num_samples_per_label[label] >= 2:
            continue

        inputs.append(image.numpy().ravel())
        targets.append(int(label))

        num_samples_per_label[label] += 1

        if (num_samples_per_label == 2).all():
            break

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets
