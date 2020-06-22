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
           
    
    def cost(self, x, y):
        pred = self.predict(x)
        return (1/(2*len(x)) * (pred - y)**2)[-1][-1]
    
    def activation(self, z):
        return np.sign(z)*((self.r**2 * np.abs(z) + 1)**(1/self.r) - 1)
    
    def d_activation(self, z):
        return self.r*(self.r**2 * np.abs(z) + 1)**((1-self.r)/self.r)
    
    def step_forward(self, a, w, with_activation=True):
        if not with_activation:
            return np.dot(a, w.T)
        z = np.dot(a, w.T)
        return self.activation(z), z
    
    def add_bias(self, x):
        return np.c_[np.ones((x.shape[0],1)), x]
    
    def forward_prop(self, x):
        z = {}
        a = {}
        x = self.add_bias(x)
        _a, _z = self.step_forward(x, self.w[0])
        z.update({0: _z})
        a.update({0: _a})
        a[0] = self.add_bias(a[0])
        for layer in range(1, self.hidden_layers+2):
            if layer!=self.hidden_layers+1:
                _a, _z = self.step_forward(a[layer-1], self.w[layer])
                _a = self.add_bias(_a)
                z.update({layer: _z})
                a.update({layer: _a})
            else:
                _a = self.step_forward(a[layer-1], self.w[layer], with_activation=False) # for output layer only linear combination
                a.update({layer: _a})
        return a, z
    
    def back_prop(self, x, y):
        m = len(x) #create in fit
        a, z = self.forward_prop(x)
        deltas = {}
        D = {}
        grads = {}
        deltas.update({self.hidden_layers+1: a[self.hidden_layers+1] - y}) # compute delta for output layer
        self.costs.append(self.cost(x, y))
        # Backpropagate
        for layer in reversed(list(z.keys())):
            if layer!=0:
                w = np.transpose(self.w[layer])
                d = deltas[layer+1] # take deltas from previous layer
                _z = self.add_bias(z[layer])
                try:
                    _delta = np.matmul(w.dot(d).T, self.d_activation(_z).T)
                except: # last hidden layer
                    _delta = np.matmul((w * d).T, self.d_activation(_z).T)
                deltas.update({layer:_delta})
                # need to correct this step
                _D = 1/m * (_delta * a[layer]) # Gradient
                D.update({layer: _D})
#        Update
        for layer in z.keys(): # for two layer NN z has [0, 1, 2], will skip input layer and update only hidden ones
            if layer!=0:
                self.w[layer] = self.w[layer] - self.lr * D[layer]
                
    def fit(self, x, y, n_iter=100):
        self.costs = []
        for _ in range(n_iter):
            self.back_prop(x, y)
            
    def predict(self, x):
        a, _ = self.forward_prop(x)
        return a[self.hidden_layers+1]

            
# =============================================================================
#   trosku pochybujem ze to je dobre spravene; checkni kurz a skontroluj step by step
#   hlavne to vynechavanie nulteho elementu
# =============================================================================
if __name__=='__main__':
    reg = BRU_Regressor(input_shape=(1,4), n_hidden_layers=2)
    x = np.random.rand(1,4)
    y = np.array([2.3])/10
    reg.fit(x, y)
#    res, z = reg.forward_prop(np.random.rand(1,4))
#    reg.back_prop(np.random.rand(1,4), np.array([2.3]))
#    print(res)


