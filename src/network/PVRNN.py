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

import math

class PVRNN(object):    
    
    def __init__(self, _d, _z, _t, _delimiter):
        
        self.d = []
        self.z = []
        self.t = []
        
        for _d in _d.split(_delimiter):
            self.d.append(int(_d))
            
        for _z in _z.split(_delimiter):
            self.z.append(int(_z))
            
        self.baseKeys = ['dp', 'hp', 'up', 'lp', 'sp', 'np', 'zp',\
                         'dq', 'hq', 'uq', 'lq', 'sq', 'nq', 'zq']
        self.nBaseKeys = len(self.baseKeys)
        #self.nLayers = 0
        self.nLayers = len(self.d)
                
#    def getStateBufferSize(self, _d, _z, _delimiter):
#        
#        size = 0
#        # the PV-RNN for the prior and the posterior distributions states are: 
#        # 2(h, d)
#        # 5(m, l, s, n, z)
#        self.d = []
#        self.z = []
#        self.nLayers = 0
#        for _d in _d.split(_delimiter):
#            self.d.append(int(_d))
#            size = size + int(_d)*4
#        for _z in _z.split(_delimiter):
#            self.z.append(int(_z))
#            size = size + int(_z)*10
#        self.nLayers = len(self.d)
#        
#        return size

    def e_getStateKey(self,_i): 
        
        l = math.floor(_i / 14)
        i = _i % self.nBaseKeys
        return '{}_{}'.format(self.baseKeys[i], l)
    
    
    def e_getStateMap(self,_v):        
        
        i = 0
        s = {}        
       
        for l in range(self.nLayers):
            
            d_ = self.d[l] 
            z_ = self.z[l] 
            i = 0
            j = 0
            for k in self.baseKeys:
                n = z_
                if j in [0,1,7,8]:
                    n = d_
                
                s['{}_{}'.format(k,l)] = _v[i:i+n]
                i = i + n
                j = j + 1

        return s
            
        
    def getdNum(self):
        
        return self.d        
    
    def getzNum(self):
        
        return self.d            
    
    def getTau(self, _v):
        
        return self.t    

