# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 17:00:40 2024

@author: shingo
"""

import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output
    
    def foward(self, x):
        raise NotImplementedError()
        

class Square(Function):
    def forward(self, x):
        return x**2


x = Variable(np.array(10))
f = Square()
y = f(x)
print( type(y) )
print( y.data )