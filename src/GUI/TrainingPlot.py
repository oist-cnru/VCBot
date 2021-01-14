#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 BSD 3-Clause License

  Copyright (c) 2020 Okinawa Institute of Science and Technology (OIST).
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     disclaimer in the documentation and/or other materials provided
     with the distribution.
   * Neither the name of Willow Garage, Inc. nor the names of its
     contributors may be used to endorse or promote products derived
     from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGE.

 Author: Hendry F. Chame <hendryfchame@gmail.com>

 Publication:

   Chame, H. F., Ahmadi, A., & Tani, J. (2020).
   A hybrid human-neurorobotics approach to primary intersubjectivity via
   active inference. Frontiers in psychology, 11.

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