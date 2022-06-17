# Zero'th-order adversarial attacks with ZOO
This repository serves as a key deliverable for the final project for course CS-439 Optimization for Machine Learning at the Swiss Federal Institute of Technology, Lausanne (EPFL). It was completed by Simon Liu, Simon Larsen, and Christopher Warhuus in June of 2022.

## Outline
The project's main goal was to study zero'th order optmiization algorithms for attacking black-box classifiers. To this extent, we chose to study ZORO (https://arxiv.org/abs/2003.13001), and attacked two models. The first model was a simple linear classifier for synthetic isotropic Gaussian data, while the second was a ResNet classifier (courtesy of https://github.com/JiahongChen/resnet-pytorch) trained on Zalandao's Fashion-MNIST dataset (https://github.com/zalandoresearch/fashion-mnist). Licenses are included in the repository, where appropriate.

## Running attacks
To retrieve the data and model files, please run 'git lfs pull'.

For both models, we use cross-validation to tune ZORO's hyperparameters. To perform cross-validation and select and save the best results, run:

+ NDimensionalGaussians.ipynb for the isotropic Gaussian data. The notebook starts by generating a the synthetic Gaussian dataset.
+ ZOROFashion.py for the Fashion-MNIST data.

To create the visualizations, run the VisualizeResults.ipynb and VisualizeResultsFashion.ipynb for the Gaussian and Fashion-MNIST data, respectively.
