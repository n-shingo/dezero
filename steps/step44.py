# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 11:34:56 2024

@author: shingo
"""

import numpy as np
from dezero import Variable
import dezero.functions as F
import dezero.layers as L
import matplotlib.pyplot as plt

# dataset
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

l1 = L.Linear(10)
l2 = L.Linear(1)

def predict(x):
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y

lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_square_error(y, y_pred)
    
    l1.cleargrads()
    l2.cleargrads()
    loss.backward()
    
    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data
    
    if i % 100 == 0:
        print(loss)


x_test = Variable( np.linspace(0, 1, 50).reshape(50,1) )
y_pred = predict(x_test)
plt.scatter( x, y, marker='.' )
plt.plot( x_test.data, y_pred.data, color='red' )