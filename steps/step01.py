# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 11:14:17 2024

@author: shingo
"""

class Variable:
    def __init__(self, data):
        self.data = data
        

import numpy as np

data = np.array(1.0)
x = Variable(data)
print(x.data)