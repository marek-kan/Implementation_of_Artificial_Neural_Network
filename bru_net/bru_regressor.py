# -*- coding: utf-8 -*-

import numpy as np

class BRURegressor():
    def __init__(self, input_shape=None, r=2, n_hidden_layers=1, n_units=5, learning_rate=0.1,
                 reg_lambda=0.1, beta=0.9):
        self.hidden_layers = n_hidden_layers
        self.units = n_units
        self.lr = learning_rate
        self.beta = beta
        self.reg_lambda = reg_lambda
        self.input_shape = input_shape
        self.r = r
        self.w = {}
        self.v = {}
        for layer in range(n_hidden_layers+1): # input layer + output layer
            if layer == 0:
                self.w.update({0: np.random.rand(input_shape[1]+1, self.units)/10}) # +1 - bias
                self.v.update({0: np.zeros(shape=(input_shape[1]+1, self.units))})
            elif layer==n_hidden_layers: #last layer
                self.w.update({layer: np.random.rand(self.units+1, 1)/10})
                self.v.update({layer: np.zeros(shape=(self.units+1, 1))})
            else:
                self.w.update({layer: np.random.rand(self.units+1, self.units)/10}) # +1 - bias
                self.v.update({layer: np.zeros(shape=(self.units+1, self.units))})
    
    def cost(self, x, y):
        # TODO cost with lambda
        pred = self.predict(x)
        return 1/(2*len(x)) * np.sum((pred - y)**2)
    
    def activation(self, z):
        return np.sign(z)*((self.r**2 * np.abs(z) + 1)**(1/self.r) - 1)
    
    def d_activation(self, z):
        return self.r*(self.r**2 * np.abs(z) + 1)**((1-self.r)/self.r)
    
    def step_forward(self, a, w, with_activation=True):
        if not with_activation:
            return np.dot(a, w)
        z = np.dot(a, w)
        return self.activation(z), z
    
    def add_bias(self, x):
        return np.c_[np.ones((x.shape[0],1)), x]
    
    def forward_prop(self, x):
        z = {}
        a = {}
        a_x = self.add_bias(x)
        a.update({0: a_x})
        
        _a, _z = self.step_forward(a_x, self.w[0])
        _a = self.add_bias(_a)
        z.update({1: _z})
        a.update({1: _a})
        
        for layer in range(1, self.hidden_layers+1):
            if layer!=self.hidden_layers:
                _a, _z = self.step_forward(a[layer], self.w[layer])
                _a = self.add_bias(_a)
                z.update({layer+1: _z})
                a.update({layer+1: _a})
            else:
                _z = self.step_forward(a[layer], self.w[layer], with_activation=False) # for output layer only linear combination
                z.update({layer+1: _z})
        return a, z
    
    def back_prop(self, x, y, m):
        a, z = self.forward_prop(x)
        deltas = {}
        grads = {} 
        y_pred = z[self.hidden_layers+1].reshape(m, 1) # z because linear unit (no activation)
        deltas.update({self.hidden_layers+1: 1/m * (y_pred - y) }) # compute delta for output layer
        
        # Backpropagate
        for layer in reversed(list(z.keys())): # for two hidden layers: [1, 2, 3]; input layer doesnt have "z" only final "activation"
            if layer==self.hidden_layers+1:
                grad =  a[layer-1].T.dot(deltas[layer])
                grads.update({layer-1: grad})
            else:
                da = deltas[layer+1].dot(self.w[layer].T)[:, 1:] # left out bias
                da = np.multiply(da, self.d_activation(z[layer]))
                deltas.update({layer: da})
                
                grad = a[layer-1].T.dot(da)
                grads.update({layer-1: grad})
            
        # Update
        for layer in grads.keys(): 
#           velocities
            self.v[layer][0, :] = self.beta * self.v[layer][0, :] + (1-self.beta) * grads[layer][0, :]
            self.v[layer][1:, :] = self.beta * self.v[layer][1:, :] + (1-self.beta) * grads[layer][1:, :]

#           weights update    
            self.w[layer][0, :] = self.w[layer][0, :] - self.lr * self.v[layer][0, :]
            self.w[layer][1:, :] = self.w[layer][1:, :]*(1 - self.lr*self.reg_lambda/m) - self.lr * self.v[layer][1:, :]
        self.costs.append(self.cost(x, y))
                
    def fit(self, x, y, n_iter=100, batch_size=64):
        self.costs = []
        m=batch_size
        y = y.reshape(-1, 1)
        for _ in range(n_iter):
            mask = np.random.choice(range(len(y)), batch_size)
            x_tr = x[mask]
            y_tr = y[mask]
            self.back_prop(x_tr, y_tr, m)
            
            if (_+1) % 100 == 0:
                print('Iteration: ', _+1)
                print('Cost: ', self.costs[-1])
            
    def predict(self, x):
        a = self.add_bias(x)
        for layer in self.w.keys():
            if layer!=self.hidden_layers:
                _a, _z = self.step_forward(a, self.w[layer])
                a = self.add_bias(_a)
            else:
                pred = self.step_forward(a, self.w[layer], with_activation=False)
        return pred

   


