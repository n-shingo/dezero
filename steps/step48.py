# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:24:59 2024

@author: shingo
"""

import math
import numpy as np
import dezero
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP
import matplotlib.pyplot as plt


x = np.array( [10,20])
y = np.array( [[5, 2], [10, 20], [5, 1]])
z = x/y
print(z)

# ハイパーパラメータ
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

# データ, モデル、最適化
x, t = dezero.datasets.get_spiral(train=True)
model = MLP( (hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)


data_size = len(x)
max_iter = math.ceil(data_size / batch_size)

avg_losses = []
for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0
    
    for i in  range(max_iter):
        batch_index = index[i*batch_size : (i+1)*batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]
        
        y = model(batch_x)
        loss = F.softmax_cross_entropy_simple(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        
        sum_loss += float(loss.data) * len(batch_t)
    
    avg_loss = sum_loss / data_size
    avg_losses.append(avg_loss)
    print( 'epoch %d, loss %.2f' % (epoch + 1, avg_loss) )

# プロット
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

# 教師データ
x1 = x[:,0]
x2 = x[:,1]
ax1.scatter( x1, x2, c=t)

# LOSS 遷移
ax2.plot( np.arange(max_epoch), avg_losses)
plt.show()