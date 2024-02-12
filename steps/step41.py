# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 18:33:52 2024

@author: shingo
"""
import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.random.randn(2,3))
W = Variable(np.random.randn(3,4))

y = F.matmul(x, W)
y.backward()

print(x.grad.shape)
print(W.grad.shape)