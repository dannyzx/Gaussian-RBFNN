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
torch.set_default_dtype(torch.float64) ##

class RBFN_R:
    
    def __init__(self, n_neurons,  
                 opt_method,
                 batch_size,
                 centers_training,
                 interpolation,
                 centers_strategy, n_iter,learning_rate,
                 gaussian_regularizer,
                 weights_regularizer,
                 centers_regularizer,
                 patience, verbose, seed):
        
        self.opt_method = opt_method
        self.batch_size = batch_size
        self.M = n_neurons
        self.interpolation = interpolation
        self.centers_training = centers_training
        self.centers_strategy = centers_strategy
        self.n_iter = n_iter
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

        if self.interpolation ==True:
            centers = torch.clone(X)
            self.M = self.N
            n_params = int(self.M + self.D + self.D * (self.D - 1) / 2)
            params = torch.randn(n_params)
        else:
            if self.centers_training == True:
                n_params = int(self.M + self.D + self.D * (self.D - 1) / 2)
                centers = None
                kmeans = KMeans(n_clusters=self.M, random_state=0).fit(X)
                centers_init = torch.from_numpy(kmeans.cluster_centers_).flatten()
                params = torch.cat((centers_init, torch.randn(n_params)))
                
            else:
                n_params = int(self.M + self.D + self.D * (self.D - 1) / 2)
                if self.centers_strategy == 'k-means':
                    kmeans = KMeans(n_clusters=self.M, random_state=0).fit(X)
                    centers = torch.from_numpy(kmeans.cluster_centers_)
                if self.centers_strategy == 'random':
                    X[torch.randperm(self.N)[:self.M]].flatten()
                params = torch.randn(n_params)
        
        params.requires_grad = True
        
        if self.opt_method == 'LBFGS':
            optimizer = torch.optim.LBFGS([params])#,line_search_fn='strong_wolfe')
        if self.opt_method == 'SGD':
            optimizer = torch.optim.SGD([params], lr=self.learning_rate)#, momentum=0.9)
        if self.opt_method == 'Adam':
            optimizer = torch.optim.Adam([params], lr=self.learning_rate)
        f_best_j = float("inf") 
        convergence_fun = []
        count_f_no_impr = 0
        del X, y
        for j in range(self.n_iter):
            for batch_index, batch in enumerate(loader):
                X_b, y_b = batch
                optimizer.zero_grad()
                start = time.time()
                loss, params= self.evaluate_loss(params, X_b, y_b, centers)
                loss.backward()
                optimizer.step(lambda:loss.item())
                fun = loss.item()
                end = time.time()
                if self.verbose == True:
                    
                    print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
                        j, self.n_iter, batch_index+1, len(loader),
                        loss.item()
                        ))
                    print('time: ',"{:.5f}".format(end-start))
                # print('iter: ', j,  'loss: ', "{:.5f}".format(fun), 
                #       'time: ',"{:.5f}".format(end-start))
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
        if self.batch_size == None or self.opt_method == 'LBFGS':
            self.batch_size = self.N
        dset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dset, 
                                    batch_size=self.batch_size, # choose your batch size
                                    shuffle=True, drop_last=True)
        f_best, best_params, centers, convergence_fun = self.optimize(X, y, loader)  
        if self.centers_training == True:
            centers = best_params[:int(self.M*self.D)].reshape(self.M, self.D)
            w = best_params[int(self.M*self.D):int(self.M*self.D)+self.M].flatten()
            l = best_params[int(self.M*self.D)+self.M:]
        else:
            w = best_params[:self.M].flatten()
            l = best_params[self.M:]
    
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
            centers = params[: int(self.D*self.M)].reshape(self.M, self.D)
            w = params[int(self.D*self.M):int(self.D*self.M)+self.M]
            l = params[int(self.D*self.M)+self.M:]
        else:
            w = params[:self.M]
            l = params[self.M:]
        L = torch.zeros((self.D, self.D))
        idx = torch.tril_indices(*L.shape)
        L[idx[0], idx[1]] = l
        P = torch.matmul(L, L.T)
        Phi = self.design_matrix(X, centers, P)
        
        y_hat = Phi @ w

        err = torch.sum((y_hat - y).pow(2))
        reg_l =  torch.linalg.norm(L)
        reg_c = torch.linalg.norm(centers)
        reg_w = torch.linalg.norm(w)
        loss = 0.5 * err + self.gaussian_regularizer * 0.5 * reg_l + self.weights_regularizer * 0.5 * reg_w +self.centers_regularizer * 0.5 * reg_c
        return loss, params


    def predict(self, X, centers, w, P):
        
        Phi = self.design_matrix(X, centers, P)
        f = torch.einsum('ij, j -> i', Phi, w)
    
        return f
    
    def feature_importance(self, X, P, features_names):
        
        lambdas, U = eigh(P)
        UL = np.dot(np.abs(U), lambdas)
        FI = UL/sum(UL)
        return FI, lambdas, U


#example
# X_train, X_test, y_train, y_test = train_test_split(X, y, 
#                     test_size=0.2, random_state=42) 
# scaler_X = StandardScaler()
# X_train_st = scaler_X.fit_transform(X_train)
# X_test_st = scaler_X.transform(X_test)

# scaler_y = StandardScaler()
# y_train_st = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(len(y_train))
# y_test_st = scaler_y.transform(y_test.reshape(-1, 1)).reshape(len(y_test))

# N, D, M = len(X_train), len(X_train[0]), 8 # M number of neurons (i.e. centers)

# X_train_st1, y_train_st1 = torch.from_numpy(X_train_st), torch.from_numpy(y_train_st)

# gko_rbfn = RBFN_R(n_neurons=M, opt_method='LBFGS',
#                     batch_size=None,
#                     interpolation=True,# misleading..this means that M is fixed equal to N
#                    centers_training=False, 
#                    centers_strategy='k-means', n_iter=50000,
#                    learning_rate=0.01,
#                    gaussian_regularizer=1e-2, 
#                    weights_regularizer=0.001,
#                    centers_regularizer=1e-6,
#                    patience=100,
#                    verbose=True, 
#                    seed=i)
# par_opt, f_opt = gko_rbfn.fit(X_train_st1, y_train_st1)

# centers_opt, w_opt, P_opt = par_best[0], par_best[1], par_best[2]
# X_test_st1, y_test_st1 = torch.from_numpy(X_test_st), torch.from_numpy(y_test_st)
# y_train_fitted_st = np.asarray(gko_rbfn.predict(X_train_st1, centers_opt, w_opt, P_opt))

# y_test_pred_st = np.asarray(gko_rbfn.predict(X_test_st1, centers_opt, w_opt,P_opt))
# y_test_pred = scaler_y.inverse_transform(y_test_pred_st)
# y_train_fitted = scaler_y.inverse_transform(y_train_fitted_st)
# RMSE_test = np.sqrt(sum((y_test_pred - y_test)**2)/ len(y_test))
# RMSE_train = np.sqrt(sum((y_train_fitted - y_train)**2)/ len(y_train))
# print('RMSE test', RMSE_test, 'RMSE train', RMSE_train)
# FI, lambdas, U = gko_rbfn.feature_importance(X_train_st, P_opt, features_names)