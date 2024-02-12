# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:03:25 2024

@author: shingo
"""

import numpy as np
from dezero import Variable

def rosenbrock(x0, x1, a=1.0, b=100.0):
    y = b * (x1 - x0 ** 2) ** 2 + (a - x0)**2
    return y

x0 = Variable( np.array(0.0) )
x1 = Variable( np.array(2.0) )
lr = 0.001
iters = 10000


for i in  range(iters):
    print(x0, x1)
    
    y = rosenbrock(x0, x1)
    
    x0.cleargrad()
    x1.cleargrad()
    y.backward()
    
    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad
