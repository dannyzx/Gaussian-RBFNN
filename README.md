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
## Abstract
Providing a model that achieves a strong predictive performance and at the same time is
interpretable by humans is one of the most difficult challenges in machine learning research due to the
conflicting nature of these two objectives. To address this challenge, we propose a modification of the
Radial Basis Function Neural Network model by equipping its Gaussian kernel with a learnable precision
matrix. We show that precious information is contained in the spectrum of the precision matrix that
can be extracted once the training of the model is completed. In particular, the eigenvectors explain the
directions of maximum sensitivity of the model revealing the active subspace and suggesting potential
applications for supervised dimensionality reduction. At the same time, the eigenvectors highlight the
relationship in terms of absolute variation between the input and the latent variables, thereby allowing
us to extract a ranking of the input variables based on their importance to the prediction task enhancing
the model interpretability. We conducted numerical experiments for regression, classification, and feature
selection tasks, comparing our model against popular machine learning models and the state-of-the-art
deep learning-based embedding feature selection techniques. Our results demonstrate that the proposed
model does not only yield an attractive prediction performance with respect to the competitors but also
provides meaningful and interpretable results that potentially could assist the decision-making process in
real-world applications.


