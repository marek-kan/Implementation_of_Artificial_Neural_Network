# -*- coding: utf-8 -*-

import numpy as np

class BRU_Regressor():
    def __init__(self, input_shape= None, r=2, n_hidden_layers=1, n_units=5, learning_rate=0.1, beta=0.9):
        self.hidden_layers = n_hidden_layers
        self.units = n_units
        self.lr = learning_rate
        self.beta = beta
        self.r = r
        self.w = {}
        for layer in range(n_hidden_layers+2): # input layer + output layer
            if layer == 0:
                self.w.update({0: np.random.rand(self.units, input_shape[1]+1)/100}) # +1 - bias
            elif layer==n_hidden_layers+1: #last layer
                self.w.update({layer: np.random.rand(1, self.units+1)/100})
            else:
                self.w.update({layer: np.random.rand(self.units, self.units+1)/100}) # +1 - bias
           
        
    def activation(self, z):
#        g =  r*(r**2 * K.abs(x) + 1)**((1-r)/r)
        return np.sign(z)*((self.r**2 * np.abs(z) + 1)**(1/self.r) - 1)
    
    def step_forward(self, a, w, with_activation=True):
        if not with_activation:
            return np.dot(a, w.T)
        z = np.dot(a, w.T)
        return self.activation(z), z
    
    def add_bias(self, x):
        return np.c_[np.ones((x.shape[0],1)), x]
    
    def forward_prop(self, x):
        z = {}
        x = self.add_bias(x)
        a, _z = self.step_forward(x, self.w[0])
        z.update({0: _z})
        a = self.add_bias(a)
        for layer in range(1, self.hidden_layers+2):
            if layer!=self.hidden_layers+1:
                a, _z = self.step_forward(a, self.w[layer])
                z.update({layer: _z})
                a = self.add_bias(a)
            else:
                a = self.step_forward(a, self.w[layer], with_activation=False) # for output layer only linear combination
        return a,z
# =============================================================================
#   do fitu a podobne daj na np.float32 
# =============================================================================
if __name__=='__main__':
    reg = BRU_Regressor(input_shape=(1,4), n_hidden_layers=3)
    res, z = reg.forward_prop(np.random.rand(1,4))
    print(res)


