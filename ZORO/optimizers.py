#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:46:35 2021

Code for ZORO, by Cai, McKenzie ,Yin, and Zhang

"""

import numpy as np
import numpy.linalg as la
from .base import BaseOptimizer
from .Cosamp import cosamp


class ZORO(BaseOptimizer):
    
    '''
    ZORO for black box optimization. 
    '''
    
    def __init__(self, x0, f, params, function_budget=10000, prox=None,
                 function_target=None):
        
        super().__init__()
        
        self.function_evals = 0
        self.function_budget = function_budget
        self.function_target = function_target
        self.f = f
        self.x = x0
        self.n = len(x0)
        self.t = 0
        self.delta = params["delta"]
        self.sparsity = params["sparsity"]
        self.step_size = params["step_size"]
        self.num_samples = params["num_samples"]
        self.prox = prox
        # Define sampling matrix
        # TODO (?): add support for other types of random sampling directions
        Z = 2*(np.random.rand(self.num_samples, self.n) > 0.5) - 1

        cosamp_params = {"Z": Z, "delta": self.delta, "maxiterations": 10,
                         "tol": 0.5, "sparsity": self.sparsity}
        self.cosamp_params = cosamp_params

    # Handle the (potential) proximal operator
    def Prox(self, x):
        if self.prox is None:
            return x
        else:
            return self.prox.prox(x, self.step_size)
       
    def CosampGradEstimate(self):
        '''
        Gradient estimation sub-routine.
        '''
      
        maxiterations = self.cosamp_params["maxiterations"]
        Z = self.cosamp_params["Z"]
        delta = self.cosamp_params["delta"]
        sparsity = self.cosamp_params["sparsity"]
        tol = self.cosamp_params["tol"]
        num_samples = np.size(Z, 0)
        x = self.x
        f = self.f
        #y = np.zeros(num_samples)
        
        f_x = f(np.expand_dims(x, axis=0))
        self.function_evals += 1

        perturbed = np.repeat([x], len(Z), axis=0) + delta * Z
        f_perturbed = f(perturbed)

        y = (f_perturbed - f_x) / (np.sqrt(num_samples) * delta)
        self.function_evals += len(y)
        
        Z = Z/np.sqrt(num_samples)
        grad_estimate = cosamp(Z, y, sparsity, tol, maxiterations)
        
    
        return grad_estimate, f_x

    def step(self):
        '''
        Take step of optimizer
        '''
   
        grad_est, f_est = self.CosampGradEstimate()
        self.fd = f_est
        # Note that if no prox operator was specified then self.prox is the
        # identity mapping.
        self.x = self.Prox(self.x -self.step_size*grad_est)

        if self.reachedFunctionBudget(self.function_budget, self.function_evals):
            # if budget is reached return current iterate
            return self.function_evals, self.x, 'B'

        if self.function_target is not None:
            if self.reachedFunctionTarget(self.function_target, f_est):
                # if function target is reached terminate
                return self.function_evals, self.x, 'T'
 
        self.t += 1
        return self.function_evals, self.x, False
    
    
    
    
class AdaZORO(BaseOptimizer):
    
    '''
    ZORO with adaptive sampling for black box optimization. 
    '''
    
    def __init__(self, x0, f, params, function_budget=10000, prox=None,
                 function_target=None):
        
        super().__init__()
        
        self.function_evals = 0
        self.function_budget = function_budget
        self.function_target = function_target
        self.f = f
        self.x = x0
        self.n = len(x0)
        self.t = 0
        self.delta = params["delta"]
        self.sparsity = params["sparsity"]
        self.step_size = params["step_size"]
        #self.num_samples = params["num_samples"]
        self.num_samples_constant = params["num_samples_constant"]
        self.num_samples = self.update_num_samples()
        self.phi_cosamp = params["phi_cosamp"]
        self.phi_lstsq = params["phi_lstsq"]
        self.compessible_constant = params["compessible_constant"]
        self.prox = prox
        self.compessible = True
        
        self.saved_y = np.zeros(0)
        self.saved_function_estimate = 0
        
        # Define sampling matrix
        # TODO (?): add support for other types of random sampling directions
        Z = 2*(np.random.rand(self.num_samples, self.n) > 0.5) - 1

        cosamp_params = {"Z": Z, "delta": self.delta, "maxiterations": 10,
                         "tol": 0.5}
        self.cosamp_params = cosamp_params

    # Handle the (potential) proximal operator
    def Prox(self, x):
        if self.prox is None:
            return x
        else:
            return self.prox.prox(x, self.step_size)
        
    def update_num_samples(self):
        num_samples = int(np.ceil(self.num_samples_constant * self.sparsity * np.log(self.n)))
        return num_samples
       
        
    def CosampGradEstimate(self):
        '''
        Gradient estimation sub-routine.
        '''
        
        maxiterations = self.cosamp_params["maxiterations"]
        self.num_samples = self.update_num_samples()
        sparsity = self.sparsity
        delta = self.cosamp_params["delta"]
        tol = self.cosamp_params["tol"]
        Z = self.cosamp_params["Z"]
        Z = Z[0:self.num_samples,:]
        x = self.x
        f = self.f
        phi = self.phi_cosamp
        function_estimate = 0
        function_evals = 0
        
        y = np.zeros(self.num_samples)
        save_queries_num = len(self.saved_y)
        y[0:save_queries_num] = self.saved_y * np.sqrt(save_queries_num / self.num_samples)        
        
        # We could reuse the queries from letsq/cosamp in the same iteration
        # but we didn't do it for simplify the implementation
        f_x = f(np.expand_dims(x, axis=0))
        function_evals += 1
        function_estimate = f_x
        
        perturbed = np.repeat(
                [x], 
                self.num_samples-save_queries_num, 
                axis=0) \
            + delta * Z[save_queries_num:self.num_samples]
        f_perturbed = f(perturbed)
        
        y[save_queries_num:self.num_samples] = \
            (f_perturbed - f_x) / (np.sqrt(self.num_samples)*delta)
        function_evals += len(y)
                
        Z = Z/np.sqrt(self.num_samples)
        grad_estimate = cosamp(Z, y, sparsity, tol, maxiterations)
        
        
        # print('cosamp error',la.norm(Z @ grad_estimate - y)/la.norm(y)  )  ################
        if la.norm(Z @ grad_estimate - y)/la.norm(y) > phi:
            self.saved_y = y
            self.saved_function_estimate = function_estimate
            return grad_estimate, function_estimate, False, function_evals
        else:
            self.saved_y = np.zeros(0)
            self.saved_function_estimate = 0
            return grad_estimate, function_estimate, True, function_evals        
    
    
    
    def SparseLstSq(self, old_grad_est, compessible):
        '''
        Least square with fixed support
        '''
        
        Z = self.cosamp_params["Z"]
        delta = self.cosamp_params["delta"]
        
        if compessible == True:
            sparsity =  self.sparsity
            num_samples = sparsity
            old_support = (old_grad_est != 0)
            Z_restricted = np.zeros((num_samples,self.n))
            Z_restricted[:,old_support] = Z[0:num_samples,old_support]
        else:
            sparsity =  np.count_nonzero(old_grad_est)
            num_samples = int(self.compessible_constant * sparsity)
        #Z = self.cosamp_params["Z"]
        Z = Z[0:num_samples,:]
        x = self.x
        f = self.f
        phi = self.phi_lstsq
        y = np.zeros(num_samples)
        y_restricted = np.zeros(num_samples)
        function_estimate = 0
        function_evals = 0
        
        f_x = f(np.expand_dims(x, axis=0))
        function_estimate = f_x
        function_evals += 1

        if compessible == True:
            perturbed = np.repeat([x], len(Z), axis=0) + delta * Z   
            perturbed_restricted = np.repeat([x], len(Z_restricted), axis=0) + delta * Z_restricted

            f_perturbed = f(perturbed)
            f_perturbed_restricted = f(perturbed_restricted)

            y = (f_perturbed - f_x) / (np.sqrt(num_samples)*delta)
            y_restricted = (f_perturbed_restricted - f_x) / (np.sqrt(num_samples)*delta)
             
            function_evals += len(y) + len(y_restricted)
            
            Z = Z/np.sqrt(num_samples)
            grad_est_non_zeros,_ ,_ ,_ = la.lstsq(Z[:,old_support], y_restricted, rcond=None)
            
            # print('least sq error',la.norm(Z[:,old_support] @ grad_est_non_zeros - y)/la.norm(y) )  ################
            
            if la.norm(Z[:,old_support] @ grad_est_non_zeros - y)/la.norm(y) > phi:
                self.saved_y = y
                self.saved_function_estimate = function_estimate
                return grad_est_non_zeros, function_estimate, False, function_evals
            else:
                self.saved_y = np.zeros(0)
                self.saved_function_estimate = 0
                grad_est = np.zeros(self.n)
                grad_est[old_support] = grad_est_non_zeros
                return grad_est, function_estimate, True, function_evals
        else:
            save_queries_num = len(self.saved_y)
            if save_queries_num  >= num_samples:
                y = self.saved_y[0:num_samples] * np.sqrt(save_queries_num / num_samples)
            else:
                y[0:save_queries_num] = self.saved_y * np.sqrt(save_queries_num / num_samples)                
                perturbed = np.repeat([x], num_samples-save_queries_num, axis=0) + delta * Z[save_queries_num:num_samples]
                f_perturbed = f(perturbed)
                y[save_queries_num:num_samples] = (f_perturbed - f_x) / (np.sqrt(num_samples)*delta)
                function_evals += abs(save_queries_num - num_samples)
                    
            function_estimate = f_x
            
            Z = Z/np.sqrt(num_samples)
            grad_est,_ ,_ ,_ = la.lstsq(Z, y, rcond=None)

            self.saved_y = np.zeros(0)
            self.saved_function_estimate = 0
            return grad_est, function_estimate, True, function_evals



    def getMoreZ(self):
        '''
        Get more rows in Z matrix
        '''
        Z = self.cosamp_params["Z"]
        self.num_samples = self.update_num_samples()
        
        more_rows = self.num_samples - np.size(Z, 0)
        if more_rows > 0:
            Z_new = 2*(np.random.rand(more_rows, self.n) > 0.5) - 1
            self.cosamp_params["Z"] = np.concatenate((Z, Z_new), axis=0)
            


    def step(self):
        '''
        Take step of optimizer
        '''
        
        print('Current Sparsity: ', self.sparsity)
        good_est = False
        if (self.t > 0):
            grad_est, f_est, good_est, function_evals = self.SparseLstSq(self.grad_est, self.compessible)
            self.function_evals += function_evals 
        if good_est == True:
            self.grad_est = grad_est
            self.fd = f_est
            self.x = self.Prox(self.x - self.step_size * grad_est)
        else:
            grad_est, f_est, good_est, function_evals = self.CosampGradEstimate()
            self.function_evals += function_evals 
            while good_est == False:
                if self.num_samples <= int(self.compessible_constant * self.n):
                    self.sparsity += 1
                    self.getMoreZ()                    
                    grad_est, f_est, good_est, function_evals = self.CosampGradEstimate()
                    self.function_evals += function_evals 
                else:
                    self.compessible = False
                    grad_est, f_est, good_est, function_evals = self.SparseLstSq(np.ones(self.n), self.compessible)
                    self.function_evals += function_evals 
            #self.saved_y = np.zeros(0)
            #self.saved_function_estimate = 0
            self.grad_est = grad_est
            self.fd = f_est
            self.x = self.Prox(self.x - self.step_size * grad_est)

        if self.reachedFunctionBudget(self.function_budget, self.function_evals):
            # if budget is reached return current iterate
            return self.function_evals, self.sparsity, self.x, 'B'

        if self.function_target is not None:
            if self.reachedFunctionTarget(self.function_target, f_est):
                # if function target is reached terminate
                return self.function_evals, self.sparsity, self.x, 'T'
            
        self.t += 1
        return self.function_evals, self.sparsity, self.x, False 
    
