# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 11:44:34 2024

@author: shingo
"""

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
import numpy as np
from dezero import Variable, Model
import dezero.functions as F
import dezero.layers as L
import matplotlib.pyplot as plt
import dezero.models as M

# dataset
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# hyper parameters
lr = 0.2
max_iter = 10000
hidden_size = 10

# model
class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)
        
    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y

model = TwoLayerNet(hidden_size, 1)


# training
for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_square_error(y, y_pred)
    
    model.cleargrads()
    loss.backward()
    
    for p in model.params():
        p.data -= lr * p.grad.data
    
    if i % 1000 == 0:
        print(loss)


# MLP model
model2 = M.MLP( (hidden_size, 1) )
for i in range(2*max_iter):
    y_pred = model2(x)
    loss = F.mean_square_error(y, y_pred)
    
    model2.cleargrads()
    loss.backward()
    
    for p in model2.params():
        p.data -= lr * p.grad.data
    
    if i % 1000 == 0:
        print(loss)
    
x_test = Variable( np.linspace(0, 1, 50).reshape(50,1) )
y_pred1 = model.forward(x_test)
y_pred2 = model2.forward(x_test)
plt.scatter( x, y, marker='.' )
plt.plot( x_test.data, y_pred1.data, color='red' )
plt.plot( x_test.data, y_pred2.data, color='blue' )