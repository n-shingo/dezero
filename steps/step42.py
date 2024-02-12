# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 18:43:00 2024

@author: shingo
"""

import numpy as np
import dezero
from dezero import Variable
import dezero.functions as F

# data set
np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

W = Variable(np.zeros((1,1)))
b = Variable(np.zeros(1))

def predict(x):
    y = F.matmul(x, W) + b
    return y

lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = dezero.functions.mean_square_error(y, y_pred)
    
    W.cleargrad()
    b.cleargrad()
    loss.backward()
    
    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    print( W, b, loss)