# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:27:29 2024

@author: shingo
"""

import numpy as np
from dezero import Variable

x = Variable(np.array(2.0))
y = x**2
y.backward(create_graph=True)
gx = x.grad
x.cleargrad()

z = gx ** 3 + y
z.backward()
print(x.grad)