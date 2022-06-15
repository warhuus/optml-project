import matplotlib.pyplot as plt
import functools
import collections
import os
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
from torch.utils.data import Dataset
import urllib


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


def get_labels():
    # Get imagenet class mappings
    #url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
    #urllib.request.urlretrieve(url, filename) 
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return categories


def get_label(name):
    id_to_label = {
        'n03028079' : 497, # church 
        'n02979186' : 482, # casette player 
        'n01440764' : 0, # tench (a kind of fish) 
        'n03888257' : 701, # parachute 
        'n03417042' : 569, # garbage truck 
        'n03445777' : 574, # golf ball 
        'n03425413' : 571, # gas pump 
        'n03000684' : 491, # chain saw 
        'n03394916' : 566, # French horn 
        'n02102040' : 217 # English springer (a kind of dog)
    }
    return id_to_label[name]
    
    
class ImageNet(Dataset):
    def __init__(self, datafolder="imagenette2/val", transform=None):
        self.datafolder = datafolder
        self.transform = transform
        
        self.image_files_list = []
        for root, subdirs, files in os.walk(datafolder):
            for f in files:
                self.image_files_list.append(
                    (os.path.join(root, f), get_label(root.split("/")[-1]))
                )
        
    def __len__(self):
        return len(self.image_files_list)
    
    def __getitem__(self, idx):
        path, label = self.image_files_list[idx]
        
        img = Image.open(path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0)
        
        return tensor, label
    
    
def get_transform(model):
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    return transform
    
    
def get_model(device):
    model = timm.create_model('convnext_base', pretrained=True).to(device)
    model.eval()
    return model


def print_top_predictions(probabilities, k=5):
    categories = get_labels()
    
    topk_prob, topk_catid = torch.topk(probabilities, 5)
    for i in range(topk_prob.size(0)):
        print(categories[topk_catid[i]], topk_prob[i].item())

    
def predict(model, data):
    with torch.no_grad():
        out = model(data)
    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    return probabilities
