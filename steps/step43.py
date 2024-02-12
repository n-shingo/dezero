# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 19:12:18 2024

@author: shingo
"""

import numpy as np
from dezero import Variable
import dezero.functions as F
import matplotlib.pyplot as plt

# dataset
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2*np.pi*x) + np.random.rand(100,1)

# params
I, H, O = 1, 10, 1
W1 = Variable( 0.01 * np.random.randn(I,H))
b1 = Variable( np.zeros(H))
W2 = Variable( 0.01 * np.random.randn(H,I))
b2 = Variable( np.zeros(O))


# predict func
def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y

lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_square_error(y, y_pred)
    
    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()
    
    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data
    
    if i % 1000 == 0:
        print(loss)



x_test = Variable( np.linspace(0, 1, 50).reshape(50,1) )
y_pred = predict(x_test)
plt.scatter( x, y, marker='.' )
plt.plot( x_test.data, y_pred.data, color='red' )


plt.show()
