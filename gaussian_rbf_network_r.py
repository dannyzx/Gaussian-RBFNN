import torch
import torch.nn as nn

class GaussianRBFNetworkR(nn.Module):
    def __init__(self, gaussian_regularizer, centers_regularizer, weights_regularizer, centers, n_features, n_outputs, n_centers=None):
        super(GaussianRBFNetworkR, self).__init__()

        self.gaussian_regularizer = gaussian_regularizer
        self.centers_regularizer = centers_regularizer
        self.weights_regularizer = weights_regularizer
        self.centers = centers
        self.n_centers = n_centers
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.task_type = task_type

        # Define learnable parameters

        self.precision_elements = nn.Parameter(torch.randn(self.n_features * (self.n_features + 1) // 2))

        if self.centers == None:
            self.centers = nn.Parameter(torch.randn(self.n_centers, self.n_features))
        
        
        self.weights = nn.Parameter(torch.randn(self.n_centers))
    
    
    def forward(self, X):
        
        lower_tri = torch.zeros((self.n_features, self.n_features))
        idx = torch.tril_indices(*lower_tri.shape)
        lower_tri[idx[0], idx[1]] = self.precision_elements
        precision_matrix = torch.matmul(lower_tri, lower_tri.T)
        
        diff = X[:, None, :] - self.centers.unsqueeze(0)
        Phi = torch.exp(-0.5 * torch.einsum('ijk, kl, ijl -> ij', diff, precision_matrix, diff))
        y_hat = Phi @ self.weights
        
        return y_hat

    def evaluate_loss(self, y, y_hat):

 
        mse_loss = nn.MSELoss()
        loss = mse_loss(y_hat, y)

        reg_gaussian = self.gaussian_regularizer * torch.linalg.norm(self.precision_elements)
        reg_centers = self.centers_regularizer * torch.linalg.norm(self.centers)
        reg_weights = self.weights_regularizer * torch.linalg.norm(self.weights)
        loss = loss + reg_gaussian + reg_centers + reg_weights
        return loss
    
    def feature_importance(self, precision_matrix):
        
        eigenvalues, eigenvectors = torch.linalg.eigh(precision_matrix)
        feature_importance = torch.matmul(torch.abs(eigenvectors), eigenvalues)
        return feature_importance / torch.sum(feature_importance), eigenvalues, eigenvectors