# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:53:36 2024

@author: shingo
"""

import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.sum(x)
y.backward()

print(y)
print(x.grad)