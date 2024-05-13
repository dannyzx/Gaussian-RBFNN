# Gaussian Radial Basis Functions Neural Networks (GRBFNNs)
The repository contains the code for the paper titled "Learning Active Subspaces and Discovering Important Features with Gaussian Radial Basis Functions Neural Networks".

In machine learning, achieving a balance between predictive performance and interpretability remains a challenge. We propose a modification to the radial basis function neural network (RBFNN) model by incorporating a learnable precision matrix into the Gaussian kernel of the RBFNN. Valuable information can be extracted from the learned precision matrix spectrum (i.e., eigenvectors and eigenvalues):

1. **Active subspace**: the directions of maximum variability of the model, aiding supervised dimensionality reduction tasks such as visualization.
2. **Feature importance ranking**: how individual features contribute to the predictive performance of the model.

Numerical experiments across regression, classification, and feature selection tasks demonstrate that the model maintains attractive prediction performance while providing transparent and interpretable results.
 
## Citation
```python
@article{DAGOSTINO2024106335,
title = {Learning active subspaces and discovering important features with Gaussian radial basis functions neural networks},
journal = {Neural Networks},
volume = {176},
pages = {106335},
year = {2024},
issn = {0893-6080},
doi = {https://doi.org/10.1016/j.neunet.2024.106335},
url = {https://www.sciencedirect.com/science/article/pii/S0893608024002594},
author = {Danny D’Agostino and Ilija Ilievski and Christine Annette Shoemaker},
keywords = {Explainable AI, Supervised learning, Dimensionality reduction, Feature selection, Radial Basis Function Neural Networks, Active subspace},
}
```



