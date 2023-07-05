# Learning Active Subspaces and Discovering Important Features with Gaussian Radial Basis Functions Neural Networks
This repository contains the code for the paper titled "Learning Active Subspaces and Discovering Important Features with Gaussian Radial Basis Functions Neural Networks". 
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


