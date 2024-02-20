# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:30:46 2024

@author: shingo
"""

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
import math
import numpy as np
import dezero
from dezero import optimizers, DataLoader
import dezero.functions as F
from dezero.models import MLP
import matplotlib.pyplot as plt

max_epoch = 3
batch_size = 100

train_set = dezero.datasets.MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)
model = MLP( (1000, 10) )
optimizer = optimizers.SGD().setup(model)

if os.path.exists('my_mlp.npz'):
    model.load_weights('my_mlp.npz')

for epoch in range(max_epoch):
    sum_loss = 0
    
    for x, t, in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)
    
    print('epoch: {}, loss: {:.4f}'.format(
        epoch + 1, sum_loss/len(train_set)))

model.save_weights('my_mlp.npz')