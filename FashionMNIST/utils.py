import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.models as models
from torchvision import transforms


class ResNetFeatrueExtractor18(nn.Module):
    """
    Resnet ResNet model for fashion-mnist data. Model architecture and
    training steup is courtesy of https://github.com/JiahongChen/resnet-pytorch.
    """
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


class LossFashionMnist(object):
    def __init__(self, model: nn.Module, img: np.ndarray, img_shape: tuple,
                 true_lbl: int, device: torch.device, lamb: float, norm: int) -> None:
        """ Loss function for a non-targeted attach with fashion-MNIST data """
        self.device = device
        self.model = model
        self.true_img = img
        self.true_lbl = true_lbl
        self.lamb = lamb
        self.norm = norm
        self.shape = img_shape
        self.transform = transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1))

    def __call__(self, delta):
        """ Calculate attack loss function """

        # calc probabilities from model
        self.pr = self.pr_func(delta)

        # calc both terms in loss function
        max_Fx_i_neq_t0 = np.log(max(np.delete(self.pr, self.true_lbl)) + 1e-6)
        Fx_t0 = np.log(self.pr[self.true_lbl] + 1e-6)

        # return both terms plus regularizer
        return max(Fx_t0 - max_Fx_i_neq_t0, 0) + self.lamb * np.linalg.norm(delta, self.norm, axis=1)

    def pr_func(self, delta):
        """ Get model output in the form of probabilities per class """

        # transform input and duplicate color channel (to fit with resnet model)
        input = self.transform(torch.tensor(np.reshape(self.true_img + delta, (-1, *self.shape)))).to(self.device)

        # get output and take softmax to get probabilities
        with torch.no_grad():
            if self.device == torch.device('cpu'):
                output = self.model(input.type(torch.FloatTensor))
            else:
                output = self.model(input.type(torch.cuda.FloatTensor))
        return torch.nn.functional.softmax(output[0], dim=0).cpu().numpy()


def get_model(path: str, device: torch.device) -> nn.Module:
    """ Load trained resnet model """
    model = ResNetFeatrueExtractor18()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    model.to(device)
    return model

def generate_data(test_loader: Dataset, model: nn.Module, device: torch.device):
    """ Get 20 images that the pretrained model classifies correctly """

    inputs = []
    targets = []
    num_label = 10
    model.to(device)
    num_samples_per_label = np.zeros(num_label)

    for data in test_loader:

        # get the data and true label
        image, label = data[0], data[1]

        # use the data and model to get the predicted label
        image_3channels = transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1))(image)
        with torch.no_grad():
            if device == torch.device('cpu'):
                output = model(image_3channels.type(torch.FloatTensor)).to(device)
            else:
                output = model(image_3channels.type(torch.cuda.FloatTensor)).to(device)

        pr = torch.nn.functional.softmax(output[0], dim=0).cpu().numpy()
        pred_class = np.argmax(pr)

        # skip if the predicted label is not equal to the true label
        if pred_class != int(label):
            continue
        
        # we only want two samples per class
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
