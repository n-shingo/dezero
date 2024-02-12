# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 18:10:31 2024

@author: shingo
"""

import numpy as np
from dezero import Variable

x0 = Variable(np.array([1,2,3]))
x1 = Variable(np.array([10]))

y = x0 + x1
print(y)

y.backward()
print(x1.grad)