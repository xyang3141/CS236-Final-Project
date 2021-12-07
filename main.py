# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 16:02:10 2021

@author: Nicole
"""

import xlrd
import pandas as pd
import tensorflow as tf
import numpy as np
import math

import logging
from PIL import Image

from tfrbm import BBRBM
from tfrbm import GBRBM

import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import Input
from tensorflow.python.keras import layers
from tensorflow.python.keras import Model

from sklearn.metrics import confusion_matrix
import seaborn as sns


#filename = 'D:/学习/2021autumn/CS236/homework/completion/Project/final product/data/fft_11648_20160519_6_1_7.csv'
train_file = 'D:/学习/2021autumn/CS236/homework/completion/Project/final product/data/channel1/abnormal_pd.csv'
test_file = 'D:/学习/2021autumn/CS236/homework/completion/Project/final product/data/channel1/normal_pd.csv'

'''
workbook = xlrd.open_workbook(filename)
sheet = workbook.sheet_by_index(0)
nrows = sheet.nrows
ncols = sheet.ncols
data = []
for i in range(ncols):
    col_data = []
    for j in range(nrows):
        col_data.append(float(sheet.cell_value(i, j)))
    data.append(col_data)
'''

train_df = pd.read_csv(train_file, engine='python')
train = train_df.to_numpy()
train = np.transpose(train)
test_df = pd.read_csv(test_file, engine='python')
test = test_df.to_numpy()
test = np.transpose(test)

#trainset = tf.data.Dataset.from_tensor_slices(train.astype(np.float32))   #size:(batch,dim)
#trainset = trainset.shuffle(1024, reshuffle_each_iteration=True)
#testset = tf.data.Dataset.from_tensor_slices(test.astype(np.float32))   #size:(batch,dim)
#testset = testset.shuffle(1024, reshuffle_each_iteration=True)
trainset = train
testset = test

x_dim = train.shape[-1]

'''
input_tensor = Input(shape=(x_dim,1))
lstm = layers.Bidirectional(layers.LSTM(units=64, return_sequences=False), input_shape=(x_dim, 1))
rbm = GBRBM(n_visible=x_dim, n_hidden=64) #n_hidden=128
output = lstm(input_tensor)
output = rbm.forward(output)
model = Model(input_tensor, output)
'''

rbm = GBRBM(n_visible=x_dim, n_hidden=64) #n_hidden=128
errors = rbm.fit(trainset, n_epoches=100, batch_size=10)
train_energy = rbm.get_energy(train)
#print('train_energy:', train_energy)
test_energy = rbm.get_energy(test)
#print('test_energy:', test_energy)

#train_energy_finite = train_energy[np.isfinite(train_energy)]
lower = -100#-9.4315
upper = 10#-9.4313
train_energy_tr = train_energy[train_energy>lower]
train_energy_tr = train_energy_tr[train_energy_tr<upper]
test_energy_tr = test_energy[test_energy>lower]
test_energy_tr = test_energy_tr[test_energy_tr<upper]

train_plot = np.random.choice(train_energy_tr, min(train_energy_tr.shape[0],test_energy_tr.shape[0]))
test_plot = np.random.choice(test_energy_tr, min(train_energy_tr.shape[0],test_energy_tr.shape[0]))

mean = 0.5 * (np.mean(train_plot) + np.mean(test_plot))
std = 0.5 * (np.std(train_plot) + np.std(test_plot))
train_plot = (train_plot - mean) / std + mean
test_plot = (test_plot - mean) / std + mean
x_min = int(min(np.min(train_plot), np.min(test_plot)))
x_max = math.ceil(max(np.max(train_plot), np.max(test_plot)))
#x_min = -2
x_plot = np.linspace(x_min, x_max, num=(x_max-x_min)*10+1)
'''
train_plot = (train_plot + 9.4313) * 1e5
test_plot = (test_plot + 9.4313) * 1e5
x_plot = np.linspace(-10, -7.4, num=27)
'''
fig = plt.figure()
hist = plt.hist([train_plot,test_plot], bins=x_plot, density=True, label=['normal','abnormal'])
plt.legend(fontsize='xx-large', loc='upper left')
#plt.ylabel('Density', rotation='horizontal', ha='right', va='center', fontsize='large')
plt.ylabel('Density', fontsize='xx-large')
plt.xlabel('Energy', fontsize='xx-large')
plt.xticks(ticks=np.around(x_plot, decimals=2), labels=np.around(x_plot, decimals=2), fontsize='large', rotation=60)
plt.yticks(fontsize='large')

threshold_plot = min(test_plot)
x_threshold = np.linspace(threshold_plot, threshold_plot, num=1000)
y_threshold = np.linspace(0, int(np.max(hist[0]))+1, num=1000)
plt.plot(x_threshold, y_threshold, linestyle='--', color='r', linewidth=2)

fig.savefig('D:/学习/2021autumn/CS236/homework/completion/Project/final product/energy.png', dpi=800)

tp = train_plot[train_plot<threshold_plot]
tn = test_plot[test_plot>threshold_plot]
fp = test_plot[test_plot<threshold_plot]
fn = train_plot[train_plot>threshold_plot]
n_tp = tp.shape[0]
n_tn = tn.shape[0]
n_fp = fp.shape[0]
n_fn = fn.shape[0]
n_p = n_tp + n_fn
n_n = n_tn + n_fp
r_tp = n_tp / n_p
r_tn = n_tn / n_n
r_fp = n_fp / n_p
r_fn = n_fn / n_n

train_label = train_plot.copy()
train_label.fill(0)
test_label = test_plot.copy()
test_label.fill(1)
y_label = np.concatenate((train_label, test_label))
train_pred = train_plot.copy()
test_pred = test_plot.copy()
for i in range(train_pred.shape[0]):
    if train_pred[i] < threshold_plot:
        train_pred[i] = 0
    else:
        train_pred[i] = 1
for i in range(test_pred.shape[0]):
    if test_pred[i] < threshold_plot:
        test_pred[i] = 0
    else:
        test_pred[i] = 1
'''
a1 = train_pred[train_pred<threshold_plot]
a1.fill(0)
a2 = train_pred[train_pred>threshold_plot]
a2.fill(1)
a3 = test_pred[test_pred<threshold_plot]
a3.fill(0)
a4 = test_pred[test_pred>threshold_plot]
a4.fill(1)
'''
y_pred = np.concatenate((train_pred,test_pred))
cf_matrix = confusion_matrix(y_label, y_pred)
fig2 = plt.figure()
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.4%', cmap='Blues')
fig2.savefig('D:/学习/2021autumn/CS236/homework/completion/Project/final product/confusion matrix.png', dpi=800)


from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
cm = confusion_matrix(y_label, y_pred, normalize = 'true')

normalize = True
cmap = 'RdPu'
classes = [0, 1]
title = 'Cofusion Matrix'
fig3, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
ax.figure.colorbar(im, ax = ax)
ax.set(xticks = np.arange(cm.shape[1]), yticks = np.arange(cm.shape[0]), xticklabels = classes, yticklabels = classes, ylabel = 'True label', xlabel = 'Predicted label', title = title)
plt.setp(ax.get_xticklabels(), rotation=45, ha = 'right', rotation_mode = 'anchor')
fmt = '.4f' if normalize else 'd'
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt), ha = 'center', va = 'center', color = 'white' if cm[i,j] > thresh else 'black')
        fig3.tight_layout()


#for i in np.random.choice(np.arange(test.shape[0]), 5, replace=False):
    #x = test[i]
    #rbm.get_energy(x)
    #print('x.shape:', x.shape)
    #x_tensor = tf.convert_to_tensor(x.reshape(1, x.shape[0]), dtype=tf.float32)
    #x_reconstructed_tensor = rbm.reconstruct(x_tensor)
    #x_reconstructed = x_reconstructed_tensor.numpy().reshape(28, 28)

    #Image.fromarray((x * 255).astype(np.uint8)).save(f'{i}_original.png')
    #Image.fromarray((x_reconstructed * 255).astype(np.uint8)).save(f'{i}_reconstructed.png')
