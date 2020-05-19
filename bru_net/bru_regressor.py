# -*- coding: utf-8 -*-

import numpy as np

class BRU_Regressor():
    def __init__(self, r=2, n_hidden_layers=1, n_units=5, learning_rate=0.1, beta=0.9):
        self.hidden_layers = n_hidden_layers
        self.units = n_units
        self.lr = learning_rate
        self.b = beta
        self.r = r
        
    def activation(self, z):
#        g =  r*(r**2 * K.abs(x) + 1)**((1-r)/r)
        return np.sign(z)*((self.r**2 * np.abs(z) + 1)**(1/self.r) - 1)
    
    def step_forward(self, a, w):
        z = np.dot(a, w)
        return self.activation(z)

