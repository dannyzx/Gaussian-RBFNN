#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 16:20:26 2022

@author: danny
"""
import numpy as np
import time
from sklearn.cluster import KMeans
from scipy.linalg import eigh
import torch
import math
torch.set_default_dtype(torch.float64)

class GRBF_NN_MC:
    
    def __init__(self, n_neurons,  
                 opt_method='adam',
                 batch_size=None,
                 centers_training=False,
                 centers_strategy='kmeans', 
                 n_epochs=10000, learning_rate=1e-3,
                 gaussian_regularizer=1e-3,
                 weights_regularizer=1e-3,
                 centers_regularizer=1e-3,
                 patience=10, verbose=False, seed=0):
        
        self.opt_method = opt_method
        self.batch_size= batch_size
        self.M = n_neurons
        self.centers_training = centers_training
        self.centers_strategy = centers_strategy
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.gaussian_regularizer = gaussian_regularizer
        self.weights_regularizer = weights_regularizer
        if self.centers_training == True:
            self.centers_regularizer = centers_regularizer
        else:
            self.centers_regularizer = 1e-16
        self.patience = patience
        self.verbose = verbose
        self.seed = seed
    
    def optimize(self, X, y, loader):

        torch.manual_seed(self.seed)

        if self.centers_training == True:
            n_params = int(self.M * self.C + self.D + self.D * (self.D - 1) / 2)
            centers = None
            kmeans = KMeans(n_clusters=self.M, random_state=seed, n_init='auto').fit(X)
            centers_init = torch.from_numpy(kmeans.cluster_centers_).flatten()
            #centers_init = X[torch.randperm(self.N)[:self.M]].flatten()
            params = torch.cat((centers_init, torch.randn(n_params)))
            
        else:
            n_params = int(self.M * self.C + self.D + self.D * (self.D - 1) / 2)
            if self.centers_strategy == 'k-means':
                kmeans = KMeans(n_clusters=self.M, random_state=seed, n_init='auto').fit(X)
                centers = torch.from_numpy(kmeans.cluster_centers_)
            if self.centers_strategy == 'random':
                centers = X[torch.randperm(self.N)[:self.M]].flatten()
            params = torch.randn(n_params)
        
        params.requires_grad = True
        
        if self.opt_method == 'LBFGS':
            optimizer = torch.optim.LBFGS([params],line_search_fn='strong_wolfe')
        if self.opt_method == 'SGD':
            optimizer = torch.optim.SGD([params], lr=self.learning_rate)#, momentum=0.9)
        if self.opt_method == 'Adam':
            optimizer = torch.optim.Adam([params], lr=self.learning_rate)
        f_best_j = float("inf") 
        convergence_fun = []
        count_f_no_impr = 0
        del X, y
        for j in range(self.n_epochs):
            
            for batch_index, batch in enumerate(loader):
                start = time.time()
                X_b, y_b = batch
                optimizer.zero_grad()
                loss, params= self.evaluate_loss(params, X_b, y_b, centers)
                loss.backward()
                optimizer.step(lambda:loss.item())
                fun = loss.item()
                end = time.time()
                if self.verbose == True:
                    
                    print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
                        j, self.n_epochs, batch_index+1, len(loader),
                        loss.item()
                        ))
                    print('time: ',"{:.5f}".format(end-start))
                    
            convergence_fun.append(fun)
            if math.isnan(fun) == True:
                break
            if fun < f_best_j- 1e-6:
                #print(fun, f_best_j)
                best_par = params.detach().clone()
                f_best_j = fun
                count_f_no_impr = 0 
            if fun >= f_best_j - 1e-6:
                
                count_f_no_impr += 1
                #print(fun, f_best_j, count_f_no_impr)
                if count_f_no_impr >= self.patience:
                    break
                
                
                #print(best_par, 'best par')
        return (f_best_j, best_par, centers, convergence_fun)
    
    def fit(self, X, y):
        
        self.D = X.shape[1]
        self.N = X.shape[0]
        self.C = y.shape[1]
        if self.batch_size == None or self.opt_method == 'LBFGS':
            self.batch_size = self.N
        dset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dset, 
                                    batch_size=self.batch_size, # choose your batch size
                                    shuffle=True)
        f_best = float("inf")  
        f_best, best_params, centers, convergence_fun = self.optimize(X, y, loader)  
        if self.centers_training == True:
            centers = best_params[:int(self.M*self.D)].reshape(self.M, self.D)
            w = best_params[int(self.M*self.D):int(self.M*self.D)+int(self.M*self.C)].reshape(self.M, self.C)
            l = best_params[int(self.M*self.D)+int(self.M*self.C):]
        else:
            w = best_params[:+int(self.M*self.C)].reshape(self.M, self.C)
            l = best_params[+int(self.M*self.C):]
    
        L = torch.zeros((self.D, self.D))
        idx = torch.tril_indices(*L.shape)
        L[idx[0], idx[1]] = l
        P = torch.matmul(L, L.T)
        return ((centers, w, P), convergence_fun)
    
    def design_matrix(self, X, centers, P):
        
        R = (X[:, None] - centers).reshape(-1, self.D)
        CR = torch.matmul(P, R[..., None]).squeeze(-1).reshape(X.shape[0], self.M, self.D)
        R1 = (X[:, None] - centers)
        Phi = torch.exp(-0.5*torch.einsum('ijk, ijk -> ij', R1,CR))
        return Phi 
    
    def evaluate_loss(self, params, X, y, centers):
        
        if self.centers_training == True:
            centers = params[:int(self.M*self.D)].reshape(self.M, self.D)
            w = params[int(self.M*self.D):int(self.M*self.D)+int(self.M*self.C)].reshape(self.M, self.C)
            l = params[int(self.M*self.D)+int(self.M*self.C):]
        else:
            w = params[:+int(self.M*self.C)].reshape(self.M, self.C)
            l = params[+int(self.M*self.C):]
        L = torch.zeros((self.D, self.D))
        idx = torch.tril_indices(*L.shape)
        L[idx[0], idx[1]] = l
        P = torch.matmul(L, L.T)
        Phi = self.design_matrix(X, centers, P)
        y_hat = torch.nn.functional.softmax(Phi @ w, dim=1)
        cr_loss = torch.nn.CrossEntropyLoss()
        err = cr_loss(y_hat, y)
        reg_l =  torch.linalg.norm(L)
        reg_c = torch.linalg.norm(centers)
        reg_w = torch.linalg.norm(w)
        loss = err + self.gaussian_regularizer * reg_l + self.weights_regularizer * reg_w + self.centers_regularizer * reg_c
        return loss, params

    def predict(self, X, parameters):
        centers, w, P = parameters[0], parameters[1], parameters[2]
        Phi = self.design_matrix(X, centers, P)
        S = Phi @ w
        y_hat = torch.nn.functional.softmax(S, dim=1)
        return y_hat
    
    def feature_importance(self, P):
        
        gammas, V = eigh(P)
        feature_importance = np.dot(np.abs(V), gammas)
        feature_importance = feature_importance / sum(feature_importance)
        return feature_importance, gammas[::-1], V[:, ::-1]
