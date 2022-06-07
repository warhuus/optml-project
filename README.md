# optml-project
This is a mini-project which focuses on practical implementation. It investigates an optimization algorithm
for a real-life machine learning application and gives us insight into that algorithm. We should provide empirical
evidence for the behavior of our chosen optimization algorithm or modification.

**It seems like we have chosen to work with this topic:** "How well do zero-order optimization methods do for ML applications?"
## Outline for the project

### Introduction

Present the theory and general idea for zeroth-order optimization algorithms. Describe which algorithms we chose and why. Maybe go deeper into what each algorithm is about.

### Our contribution

We make a benchmark for zero-order optimization (ZOO) algorithms to see how well they apply in ML applications. **AND** we implement ourselves one of the algorithms against a real-world already trained model (like cancer cells CNN), that still stays novel enough. By doing so, we get to check the robustness of this latter and get a chance to understand the theory on our own. 

### Experiments

#### Results

- Benchmark 
- Our own implementation of ZOO against the best [Kaggle CNN implementation](https://www.kaggle.com/competitions/histopathologic-cancer-detection/code?competitionId=11848&sortBy=scoreDescending).

### Discussion

### Conclusion

## Papers we could consider


[] Zeroth-order regularized optimization (ZORO): Approximately Sparse Gradients and Adaptive Sampling (HanQin Cai, [2021](https://arxiv.org/pdf/2003.13001.pdf)) ([GitHub](https://github.com/caesarcai/ZORO)) (Clear and understandable paper, with several references to other ZO-algos such as: ZO-SCD, ZO-SGD, ZO-AdaMM ...)

[] Sparse Perturbations for Improved Convergence in Stochastic Zeroth-Order Optimization (Mayumi Ohta, [2020](https://arxiv.org/pdf/2006.01759.pdf)) ([GitHub](https://github.com/StatNLP/sparse_szo))  (So this is Stochastic ZO optimization, interesting?)

[x] ZOO: Zeroth Order Optimization Based Black-box Attacks to Deep Neural Networks without Training Substitute Models (Pin-Yu Chen et al. [2017](https://arxiv.org/pdf/1708.03999.pdf)) ([GitHub](https://github.com/as791/ZOO_Attack_PyTorch)) (This paper seems a bit overkill, but still interesting now that we understand how zero order optimization aimed at "falsifying" a models classifier). 

## What we have done so far:
### Meeting 31-05

We have all read the [ZOO](https://arxiv.org/pdf/1708.03999.pdf) paper. Really great. Gave a _great insight_ into basic ZOO attack. Two method implementations could be considered: ZOO-Adam and ZOO-Newton. CiFar10 and MNIST datasets were attacked. Comparison between C&W White-box attack and their own black-box attack was presented. On top of that, performance comparison was made with both targeted and untargeted attacks. 

**Next goals**: 
- Implement ZOO-Adam or ZOO-Newton 
- Read two other papers 
- Choose the best Kaggle model for CNN

Thursday evening 02-06 meet in person with Chris and Simon Liu to start the project. 

**Next meeting** -> Tuesday 7th of June. 

### Meeting 07-05

Simon Liu will be working on SZOO implementation. Found a wonderful Kaggle CNN model for cancer cells. Simon Larsen will look at the paper [ZORO](https://arxiv.org/pdf/2003.13001.pdf). Chris is working on implementing ZOO.

Chris: Went through all ZOO. Pushed a utils file. Added ZOO stuff in a folder in the repo. Will continue to make changes in the utils and make it work with the cancer data. Thursday hopefully will get an idea of how it works. Utils is a bit cancer apparently.

_We can make seperate utils._

Simon Liu: Pushing Imagenet stuff now. Similar format to cancer (but different data download). 

**Next goals**:
- Get MNIST downloaded and choose a model for it (each paper uses its own constructed MNIST model)
- Implement everything 

**Next meeting** -> Friday 10th of June. 
