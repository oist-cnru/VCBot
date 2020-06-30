#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020  Hendry F. Chame <hendryfchame@gmail.com>
All rights reserved.

This source code is licensed under the 3-Clause BSD License found in the
LICENSE file in the root directory of this source tree. 

If a copy license was not distributed with this file, 
You can obtain one at https://opensource.org/licenses/BSD-3-Clause

Publication: 
    
    "Towards hybrid primary intersubjectivity: a neural robotics library for human science"
    Hendry F. Chame, Ahmadreza Ahmadi, Jun Tani
    
    Okinawa Institute of Science and Technology Graduate University (OIST)
    Cognitive Neurorobotics Research Unit (CNRU)
    1919-1, Tancha, Onna, Kunigami District, Okinawa 904-0495, Japan

"""


import numpy as np
import matplotlib.pyplot as plt

class TrainingPlot():
    
    def __init__(self, _mName, _train, _context):
        
        wW = 8                
        wH = 5
    
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(wW, wH))        
        
        fig.canvas.set_window_title('Agent {} training details'.format(_mName))
        fig.subplots_adjust(wspace=0.2, hspace=0.8)
        
        for i in range (2):
            for j in range (2):
                ax = axes[i][j]
                ax.grid(linestyle='dotted')    
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                ax.set_xlabel('epochs')
                
        times = np.cumsum(np.ones(_train.shape[0]))
        times = (times - 1) * _train[0,0]
        axes[0][0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        axes[0][0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
        titleY = 1.1
        font = {'family': 'serif',
                'color':  'darkred',
                'weight': 'normal',
                'size': 12,
                }
        axes[0][0].set_title('Posterior reconstruction error', y=titleY, fontdict=font)
        axes[0][0].plot(times, _train[:,2])
        axes[0][1].set_title('Prior reconstruction error', y=titleY, fontdict=font)
        axes[0][1].plot(times, _train[:,3], color='brown')
        axes[1][0].set_title('Regulation error', y=titleY, fontdict=font)
        axes[1][0].plot(times, _train[:,4], color='darkgreen')
        axes[1][1].set_title('Loss (Negative ELBO)', y=titleY, fontdict=font)
        axes[1][1].plot(times, _train[:,5], color='darkmagenta')
        plt.show()