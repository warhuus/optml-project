# optml-project
This is a mini-project which focuses on practical implementation. It investigates an optimization algorithm
for a real-life machine learning application and gives us insight into that algorithm. We should provide empirical
evidence for the behavior of our chosen optimization algorithm or modification.

**It seems like we have chosen to work with this topic:** "How well do zero-order optimization methods do for ML applications, compared to standard first-order methods?"

## Outline for the project

### Introduction

Present the theory and general idea for zeroth-order optimization algorithms and first-order algorithms. Describe which algorithms we chose and why. Maybe go deeper into what each algorithm is about.

### Our contribution

We make a benchmark for zero-order optimization (ZOO) algorithms to compare them with first-order methods to see how well they apply in ML. **AND** we implement ourselves one of the algorithms against a real-world already trained model (like cancer cells CNN), that still stays novel enough. By doing so, we get to check the robustness of this latter and get a chance to understand the theory on our own. 

### Experiments

#### Results

### Discussion

### Conclusion

## Papers we could consider


1) Zeroth-order regularized optimization (ZORO): Approximately Sparse Gradients and Adaptive Sampling (HanQin Cai, [2021](https://arxiv.org/pdf/2003.13001.pdf)) ([GitHub](https://github.com/caesarcai/ZORO)) (Clear and understandable paper, with several references to other ZO-algos such as: ZO-SCD, ZO-SGD, ZO-AdaMM ...)

2) Sparse Perturbations for Improved Convergence in Stochastic Zeroth-Order Optimization (Mayumi Ohta, [2020](https://arxiv.org/pdf/2006.01759.pdf)) ([GitHub](https://github.com/StatNLP/sparse_szo))  (So this is Stochastic ZO optimization, interesting?)

3) ZOO: Zeroth Order Optimization Based Black-box Attacks to Deep Neural Networks without Training Substitute Models (Pin-Yu Chen et al. [2017](https://arxiv.org/pdf/1708.03999.pdf)) ([GitHub](https://github.com/as791/ZOO_Attack_PyTorch)) (This paper seems a bit overkill, but still interesting now that we understand how zero order optimization aimed at "falsifying" a models classifier). 
