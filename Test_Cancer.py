import enum
import copy
import os
from ZORO.benchmarkfunctions import SparseQuadric
from ZORO.optimizers import *
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader,Dataset
import Cancer.utils as cancer_utils
import ImageNet.utils as imagenet_utils
import albumentations
from albumentations import pytorch as AT
import math

DATASET = 'Cancer'
# DATASET = 'ImageNet'

sigmoid = lambda x: 1 / (1 + math.exp(-x))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if DATASET == 'Cancer':

    # get model
    model = cancer_utils.get_model('Cancer', device)

    # get data
    dataset = cancer_utils.get_dataset(os.path.join('Cancer', 'input'))

elif DATASET == 'ImageNet':

    raise NotImplementedError

    # get model
    model = imagenet_utils.get_model(device)

    # get data
    transform = imagenet_utils.get_transform(model)
    dataset = imagenet_utils.ImageNet(datafolder=os.path.join('ImageNet', 'imagenette2', 'val'),
                                      transform=transform)

else:
    raise NotImplementedError

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
data, target = next(iter(dataloader))

class LossCancer(object):

	def __init__(self, model, img, img_shape, true_lbl, device) -> None:
		self.device = device
		self.model = model
		self.true_img = img.to(self.device)
		self.true_lbl = true_lbl
		self.lambda_ = 0.9
		self.shape = img_shape
	
	def __call__(self, delta):
		input = torch.reshape(self.true_img + torch.tensor(delta).to(self.device), self.shape)
		print(type(input))
		print(f'type delta:{type(delta)}')
		print(f'type true_img:{type(self.true_img)}')
		with torch.no_grad():
			output = self.model(input.type(torch.cuda.FloatTensor))
		pr = output[:, 0][0].cpu().numpy()
		fx = sigmoid(pr)
		return ((fx - 1 - self.true_lbl) ** 2 +
				self.lambda_ * np.linalg.norm(delta, 0))
		 

obj_func = LossCancer(model=model, img=data.flatten(), true_lbl=target, img_shape=data.shape, device=device)

n = 2000
s = int(0.1*n)

# Choose initialization
x0    = torch.randn(np.prod(data.shape))
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
	performance_log_ZORO.append( [evals_ZORO,np.mean(opt.fd)] )
	# print some useful values
	opt.report( 'Estimated f(x_k): %f  function evals: %d\n' %
	(np.mean(opt.fd), evals_ZORO) )

# for idx, (data, target) in enumerate(iter(dataloader)):
