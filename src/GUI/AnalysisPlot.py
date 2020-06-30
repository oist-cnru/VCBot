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

   "Towards hybrid primary intersubjectivity: a neural robotics 
   library for human science"

   Hendry F. Chame, Ahmadreza Ahmadi, Jun Tani

   Okinawa Institute of Science and Technology Graduate University (OIST)
   Cognitive Neurorobotics Research Unit (CNRU)
   1919-1, Tancha, Onna, Kunigami District, Okinawa 904-0495, Japan

"""


import numpy as np
import matplotlib.pyplot as plt

class AnalysisPlot():
    
    def __init__(self, _params):
        
        m = _params['m']
        e = _params['e']
        ut = _params['ut']
        delimiter = _params['delimiter']
        layer = _params['layer']
        form = _params['format']
        layerdetails = _params["layerdetails"]
        
        hum_pos = e['hum_pos']
        cur_pos = e['cur_pos']
        tgt_pos = e['tgt_pos']
        period = int(e['samplingperiod'])
        state = e['states']
        elbo = e['elbo']
        eName = e['datetime']
        
        # Getting the model dimensions
        d = []
        for _d in ut.trimString(m['d']).split(delimiter):
            d.append(int(_d))
        z = []
        for _z in ut.trimString(m['z']).split(delimiter):
            z.append(int(_z))           
        
        #getting number of features        
        featCount = 0
        for k in layerdetails.keys():
            if layerdetails[k]:
                featCount = featCount + 1
                
        layerId = []
        layerBuffer = []
        
        nLayers = len(d)
        
        for l in range(nLayers):
            layerBuffer.append(d[l]*4 + z[l]*10)
            
        if layer == 'All':
            nLayers = m['nlayers']
            for l in range(nLayers):
                layerId.append(l)            
        else:
            layerId.append(int(layer)-1)
        
        nplots = 5 + featCount*len(layerId)
            
        wW = 8                
        wH = 12
        
        nTimes = cur_pos.shape[0]        
        eT = np.linspace(0.0, period/1000.0, num=nTimes)
        
        fig, axes = plt.subplots(constrained_layout=False, nrows=nplots, ncols=1, figsize=(wW, wH))                           
        fig.canvas.set_window_title('Agent {} - Experiment {}'.format(m['name'], eName))
        fig.subplots_adjust(wspace=0.1, hspace=1.0)
        
        for i in range (nplots):
            ax = axes[i]
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            ax.get_yaxis().set_major_locator(plt.MaxNLocator(2))
            ax.get_xaxis().set_visible(False)
         
        plt.rc('font', size=7)
        titleX = 0.5
        titleY = 1.0
        font = {'family': 'serif',
                'color':  'darkred',
                'weight': 'bold',
                'size': 8,
                }        
        axes[0].set_title('Emergent behavior', x=titleX, y=titleY, fontdict=font)
        axes[0].plot(eT, cur_pos[:,0])
        axes[0].plot(eT, cur_pos[:,1])

        axes[1].set_title('Robot Intention', x=titleX, y=titleY, fontdict=font)
        axes[1].plot(eT, tgt_pos[:,0])
        axes[1].plot(eT, tgt_pos[:,1])
        
        axes[2].set_title('Human action', x=titleX, y=titleY, fontdict=font)
        axes[2].plot(eT, hum_pos[:,0])
        axes[2].plot(eT, hum_pos[:,1])

        axes[3].set_title('Mean squared error', x=titleX, y=titleY, fontdict=font)
        axes[3].plot(eT, np.mean(np.power(cur_pos - tgt_pos,2.0),1))

        axes[4].set_title('Regulation error (KL-Divergence)', x=titleX, y=titleY, fontdict=font)
        axes[4].plot(eT, elbo[:,2])
        
        i = 5
        if featCount > 0:
            j = 0
            for l in range(nLayers):            
                if l in layerId:                            
                    k = j
                    if layerdetails['dp']:
                        axes[i].set_title('dp (Layer {})'.format(l+1), x=titleX, y=titleY, fontdict=font)                    
                        self.plotVariable(axes[i], eT, state[:,k:k+d[l]], form)                        
                        i = i + 1
                    k = k + d[l]                                                
                    if layerdetails['hp']:
                        axes[i].set_title('hp (Layer {})'.format(l+1), x=titleX, y=titleY, fontdict=font)                    
                        self.plotVariable(axes[i], eT, state[:,k:k+d[l]], form)                        
                        i = i + 1
                    k = k + d[l]                            
                    if layerdetails['mp']:
                        axes[i].set_title('mp (Layer {})'.format(l+1), x=titleX, y=titleY, fontdict=font)                    
                        self.plotVariable(axes[i], eT, state[:,k:k+z[l]], form)                        
                        i = i + 1
                    k = k + z[l]                            
                    if layerdetails['lp']:                        
                        axes[i].set_title('lp (Layer {})'.format(l+1), x=titleX, y=titleY, fontdict=font)                    
                        self.plotVariable(axes[i], eT, state[:,k:k+z[l]], form)                        
                        i = i + 1
                    k = k + z[l]                            
                    if layerdetails['sp']:                    
                        axes[i].set_title('sp (Layer {})'.format(l+1), x=titleX, y=titleY, fontdict=font)                    
                        self.plotVariable(axes[i], eT, state[:,k:k+z[l]], form)                        
                        i = i + 1
                    k = k + z[l]                            
                    if layerdetails['np']:
                        axes[i].set_title('np (Layer {})'.format(l+1), x=titleX, y=titleY, fontdict=font)                    
                        self.plotVariable(axes[i], eT, state[:,k:k+z[l]], form)                        
                        i = i + 1
                    k = k + z[l]                            
                    if layerdetails['zp']:                    
                        axes[i].set_title('zp (Layer {})'.format(l+1), x=titleX, y=titleY, fontdict=font)                    
                        self.plotVariable(axes[i], eT, state[:,k:k+z[l]], form)                        
                        i = i + 1
                    k = k + z[l]                                                                    
                    if layerdetails['dq']:
                        axes[i].set_title('dq (Layer {})'.format(l+1), x=titleX, y=titleY, fontdict=font)                    
                        self.plotVariable(axes[i], eT, state[:,k:k+d[l]], form)                        
                        i = i + 1
                    k = k + d[l]                            
                    if layerdetails['hq']:
                        axes[i].set_title('hq (Layer {})'.format(l+1), x=titleX, y=titleY, fontdict=font)                    
                        self.plotVariable(axes[i], eT, state[:,k:k+d[l]], form)                        
                        i = i + 1
                    k = k + d[l]                            
                    if layerdetails['mq']:
                        axes[i].set_title('mq (Layer {})'.format(l+1), x=titleX, y=titleY, fontdict=font)                    
                        self.plotVariable(axes[i], eT, state[:,k:k+z[l]], form)                        
                        i = i + 1
                    k = k + z[l]                            
                    if layerdetails['lq']:
                        axes[i].set_title('lq (Layer {})'.format(l+1), x=titleX, y=titleY, fontdict=font)                    
                        self.plotVariable(axes[i], eT, state[:,k:k+z[l]], form)                        
                        i = i + 1
                    k = k + z[l]                            
                    if layerdetails['sq']:
                        axes[i].set_title('sq (Layer {})'.format(l+1), x=titleX, y=titleY, fontdict=font)                    
                        self.plotVariable(axes[i], eT, state[:,k:k+z[l]], form)                        
                        i = i + 1
                    k = k + z[l]                            
                    if layerdetails['nq']:
                        axes[i].set_title('nq (Layer {})'.format(l+1), x=titleX, y=titleY, fontdict=font)                    
                        self.plotVariable(axes[i], eT, state[:,k:k+z[l]], form)                        
                        i = i + 1
                    k = k + z[l]                            
                    if layerdetails['zq']:                        
                        axes[i].set_title('zq (Layer {})'.format(l+1), x=titleX, y=titleY, fontdict=font)                    
                        self.plotVariable(axes[i], eT, state[:,k:k+z[l]], form)                        
                        i = i + 1
                    k = k + z[l]                            

                j = j + layerBuffer[l]
        plt.show()
            
    def plotVariable(self, _ax, _times, _data, _format):
        
        if _format == 'Sum':            
            if len(_data.shape) > 1:
                _ax.plot(_times, np.sum(_data,1))
            else:
                _ax.plot(_times, _data)
        elif _format == 'Mean':
            if len(_data.shape) > 1:
                _ax.plot(_times, np.mean(_data,1))
            else:
                _ax.plot(_times, _data)
        else:
            for d in range(_data.shape[1]):
                _ax.plot(_times, _data[:,d])            
            
                
