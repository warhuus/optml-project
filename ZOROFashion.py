import os

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit
import torch
from torchvision import datasets, transforms

import FashionMNIST.utils as fashionmnist_utils
from ZORO import optimizers
import random

np.random.seed(42)
torch.manual_seed(42)
random.seed(0)

# Parameters to search for ZORO attack
params = {
    "step_size": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
    "delta": [1e-3, 1e-4, 1e-5], 
    "max_cosamp_iter": [5, 10, 15, 20, 25],
    "cosamp_tol": [0.5], 
    "prop_sparsity": [0.05, 0.1, 0.15, 0.20, 0.25], 
    "lamb" : [0.05], 
    "norm" : [2],
    "function_budget": [1e5]
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = fashionmnist_utils.get_model(os.path.join('FashionMNIST', 'model', 'resnet.pt'), device)

class ZOROExperiment:

    def __init__(self, step_size, delta, max_cosamp_iter, cosamp_tol, prop_sparsity, lamb, norm, function_budget):
        self.step_size = step_size
        self.delta = delta
        self.max_cosamp_iter = max_cosamp_iter
        self.cosamp_tol = cosamp_tol
        self.prop_sparsity = prop_sparsity
        self.lamb = lamb
        self.norm = norm
        self.model = model
        self.device = device
        self.function_budget = function_budget
            
    def score(self, X, y):
        return self.loss
    
    def get_params(self, deep=True):
        return {
            # Parameters for ZORO. 
            "step_size": self.step_size,
            "delta": self.delta,
            "max_cosamp_iter": self.max_cosamp_iter,
            "cosamp_tol": self.cosamp_tol,
            "prop_sparsity": self.prop_sparsity,
            "lamb" : self.lamb,
            "norm" : self.norm,
            "function_budget": self.function_budget
        }
    
    def set_params(self, **kwargs):
        for parameter, value in kwargs.items():
            setattr(self, parameter, value)
        return self
    
    def fit(self, X, y):
        self.report = []
        
        params = {
            "step_size": self.step_size,
            "delta": self.delta,
            "max_cosamp_iter": self.max_cosamp_iter,
            "cosamp_tol": self.cosamp_tol,
            "prop_sparsity": self.prop_sparsity,
            "lamb" : self.lamb,
            "norm" : self.norm,
            "function_budget": self.function_budget
        }
        
        params["sparsity"] = int(params["prop_sparsity"] * X.shape[1])
        params["num_samples"] = int(np.ceil(np.log(X.shape[1])*params["sparsity"]))

        # Compute attack loss for each data point individually
        for i in range(len(X)):
            x0           = X[i, :] # 1D
            xx0          = x0.copy()
            label        = y[i]

            obj_func = fashionmnist_utils.LossFashionMnist(
                model=self.model,
                img=np.expand_dims(xx0, 0),
                img_shape=(1, 28, 28),
                true_lbl=label,
                device=self.device,
                lamb=params['lamb'],
                norm=params['norm']
            )

            # initialize optimizer object
            self.report.append([{"evals": 0, "x": x0, "y": label, "loss": obj_func(np.expand_dims(x0, 0))[0]}])
            opt = optimizers.ZORO(x0, obj_func, params, function_budget=self.function_budget, function_target=0.001)

            # the optimization routine
            termination = False
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
                pr_solution_ZORO = obj_func.pr
                # save some useful values
                self.report[-1].append({"evals" : evals_ZORO, "x": solution_ZORO, "loss": np.mean(opt.fd), "y": pr_solution_ZORO})
                # print some useful values
                opt.report( f'Estimated f(x_{i}): %f  function evals: %d\n' %
                   (np.mean(opt.fd), evals_ZORO) )
        self.loss = sum([self.report[i][-1]["loss"] for i in range(len(self.report))]) / len(self.report)

clf_search = sklearn.model_selection.RandomizedSearchCV(
    estimator = ZOROExperiment(**params),
    param_distributions = params,
    n_iter = 100, # Run 50 random trials
    n_jobs = 20, # Run 10 jobs at once
    refit=True,
    cv = ShuffleSplit(n_splits=1, train_size=19, random_state=42)
)

# get data
dataset = datasets.FashionMNIST(
    os.path.join("FashionMNIST", "data"), download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
X, y = fashionmnist_utils.generate_data(dataloader, model, device)

search_results = clf_search.fit(X, y) # Make sure to clear the output of this cell before saving

# We see NaNs when numerical errors due to overflow occur (indicates a terrible hyperparam combination)
pd.DataFrame(search_results.cv_results_).sort_values("mean_test_score").to_csv('grid_search_results.csv')

savename = f"ZORO_L_{''.join(str(params['lamb']).split('.'))}_report.pt"
torch.save(clf_search.best_estimator_.report, os.path.join('ouputs', savename))
