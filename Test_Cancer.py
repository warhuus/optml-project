import enum
import copy
import os
from ZORO.benchmarkfunctions import SparseQuadric
from ZORO.optimizers import *
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader,Dataset
from torchvision import datasets, transforms
import Cancer.utils as cancer_utils
import FashionMNIST.utils as fashionmnist_utils
import albumentations
from albumentations import pytorch as AT
import math

DATASET = 'Cancer'
# DATASET = 'FashionMNIST'
TARGET_CLASS = 1            # for attacking the FashionMNIST model

sigmoid = lambda x: 1 / (1 + math.exp(-x))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class LossCancer(object):
 
	def __init__(self, model, img, img_shape, true_lbl, device) -> None:
		self.device = device
		self.model = model
		self.true_img = img.to(self.device)
		self.true_lbl = true_lbl[0]
		self.lambda_ = 0.9
		self.shape = img_shape
	
	def __call__(self, delta):
		input = torch.reshape(self.true_img + torch.tensor(delta).to(self.device), self.shape)
		with torch.no_grad():
			if torch.cuda.is_available():
				output = self.model(input.type(torch.cuda.FloatTensor))
			else:
				output = self.model(input.type(torch.FloatTensor))
		pr = output[:, 0][0].cpu().numpy()
		fx = sigmoid(pr)
		return ((fx - 1 - self.true_lbl) ** 2 +
				self.lambda_ * np.linalg.norm(delta, 0))

class LossFashionMnist(LossCancer):

    def __init__(self, target_class: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.target_class = target_class

    def __call__(self, delta):
        input = torch.reshape(self.true_img + torch.tensor(delta).to(self.device), self.shape)
        with torch.no_grad():
            output = self.model(input.type(torch.cuda.FloatTensor))
        pr = torch.nn.functional.softmax(output[0], dim=0).cpu().numpy()
        maxZxi = max(np.delete(pr, self.target_class))
        Zxt = pr[self.target_class]
        return max(maxZxi - Zxt, 0) + self.lambda_ * np.linalg.norm(delta, 0)

if DATASET == 'Cancer':

    # get model
    model = cancer_utils.get_model('Cancer', device)

    # get data
    dataset = cancer_utils.get_dataset(os.path.join('Cancer', 'input'))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
    data, target = next(iter(dataloader))

    # get objective function
    obj_func = LossCancer(model=model, img=data.flatten(), true_lbl=target,
                          img_shape=data.shape, device=device)

elif DATASET == 'FashionMNIST':

    # get model
    model = fashionmnist_utils.get_model(os.path.join('FashionMNIST', 'model.pt'), device)

    # get data
    dataset = datasets.FashionMNIST(
        os.path.join("FashionMNIST", "data"), download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)

    target = -1
    while int(target) != TARGET_CLASS:
        data, target = next(iter(dataloader))

    # get objective function
    obj_func = LossFashionMnist(model=model, target_class=TARGET_CLASS, img=data.flatten(),
                                true_lbl=target, img_shape=data.shape, device=device)

else:
    raise NotImplementedError

n = 2000
s = int(0.1*n)

# Choose initialization
x0    = np.random.randn(np.prod(data.shape))
x0    = 100 * x0/np.linalg.norm(x0)
xx0   = copy.deepcopy(x0)

sparsity = s
#sparsity = int(0.1*len(x0)) # This is a decent default, if no better estimate is known. 

# Parameters for ZORO. Defaults are fine in most cases
params = {"step_size":1.0, "delta": 0.0001, "max_cosamp_iter": 10, 
		  "cosamp_tol": 0.5,"sparsity": sparsity,
		  "num_samples": int(np.ceil(np.log(len(x0))*sparsity))}

performance_log_ZORO = [[0, obj_func(x0)]]


# initialize optimizer object
opt  = ZORO(x0, obj_func, params, function_budget= int(1e6))
print(f'Type x0:{x0, type(obj_func)}')
# the actual optimization routine
termination = False
print("Here")
while termination is False:
	
	# optimization step
	# solution_ZORO = False until a termination criterion is met, in which 
	# case solution_ZORO = the solution found.
	# termination = False until a termination criterion is met.
	# If ZORO terminates because function evaluation budget is met, 
	# termination = B
	# If ZORO terminated because the target accuracy is met,
	# termination= T.

	evals_ZORO, solution_ZORO, termination = opt.step()
	# save some useful values
	print(f'This is opt:{opt.fd}')
	print(f'This is evals: {evals_ZORO}')
	
	performance_log_ZORO.append( [evals_ZORO,np.mean(opt.fd)] )
	print(f'This is opt:{opt.fd}')
	# print some useful values
	opt.report( 'Estimated f(x_k): %f  function evals: %d\n' %
	(np.mean(opt.fd), evals_ZORO) )

# for idx, (data, target) in enumerate(iter(dataloader)):
