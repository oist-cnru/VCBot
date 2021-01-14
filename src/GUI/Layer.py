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

import tkinter as tk
from tkinter import ttk


class Layer():
            
    def __init__(self, _master=None, _context=None):
                
        #self.master =  tk.Toplevel(_master)
        self.master =  _master
        self.master.title("Layer parameters selection")
        self.context = _context
        self.state = self.context['layerdetails']         
        padx_ = self.context['padx']
        pady_ = self.context['pady']        
                
        self.f1 = ttk.Frame(self.master)                                
        self.f1_f1 = ttk.LabelFrame(self.f1, relief=tk.SUNKEN, text=" Prior distribution    ")                                
        self.f1_f2 = ttk.LabelFrame(self.f1, relief=tk.SUNKEN, text=" Posterior distribution")                                
        self.f1_f3 = ttk.Frame(self.f1)

        self.b_f3_select = ttk.Button(self.f1_f3, text="select",  command=self.doSelect)                                                
        self.b_f3_select.pack()
                
        self.dp  = tk.BooleanVar()
        self.hp  = tk.BooleanVar()
        self.mp  = tk.BooleanVar()
        self.lp  = tk.BooleanVar()
        self.sp  = tk.BooleanVar()
        self.np  = tk.BooleanVar()
        self.zp  = tk.BooleanVar()
        self.dq  = tk.BooleanVar()
        self.hq  = tk.BooleanVar()
        self.mq  = tk.BooleanVar()
        self.lq  = tk.BooleanVar()
        self.sq  = tk.BooleanVar()
        self.nq  = tk.BooleanVar()
        self.zq  = tk.BooleanVar()
        
        self.l_f1_f1_dp = ttk.Label(self.f1_f1, text="d")
        self.l_f1_f1_hp = ttk.Label(self.f1_f1, text="h")
        self.l_f1_f1_mp = ttk.Label(self.f1_f1, text="\u03BC")
        self.l_f1_f1_lp = ttk.Label(self.f1_f1, text="log \u03C3")
        self.l_f1_f1_sp = ttk.Label(self.f1_f1, text="\u03C3")
        self.l_f1_f1_np = ttk.Label(self.f1_f1, text="\u03B5")
        self.l_f1_f1_zp = ttk.Label(self.f1_f1, text="z")
        
        self.c_f1_f1_dp = ttk.Checkbutton(self.f1_f1, variable=self.dp, onvalue=True, offvalue=False)
        self.c_f1_f1_hp = ttk.Checkbutton(self.f1_f1, variable=self.hp, onvalue=True, offvalue=False)
        self.c_f1_f1_mp = ttk.Checkbutton(self.f1_f1, variable=self.mp, onvalue=True, offvalue=False)
        self.c_f1_f1_lp = ttk.Checkbutton(self.f1_f1, variable=self.lp, onvalue=True, offvalue=False)
        self.c_f1_f1_sp = ttk.Checkbutton(self.f1_f1, variable=self.sp, onvalue=True, offvalue=False)
        self.c_f1_f1_np = ttk.Checkbutton(self.f1_f1, variable=self.np, onvalue=True, offvalue=False)
        self.c_f1_f1_zp = ttk.Checkbutton(self.f1_f1, variable=self.zp, onvalue=True, offvalue=False)
        
        if not self.state is None:
            if self.state['dp']:
                self.c_f1_f1_dp.state(['selected'])
                self.dp.set(True)
            if self.state['hp']:
                self.c_f1_f1_hp.state(['selected'])
                self.hp.set(True)
            if self.state['mp']:
                self.c_f1_f1_mp.state(['selected'])
                self.mp.set(True)
            if self.state['lp']:
                self.c_f1_f1_lp.state(['selected'])
                self.lp.set(True)
            if self.state['sp']:
                self.c_f1_f1_sp.state(['selected'])
                self.sp.set(True)
            if self.state['np']:
                self.c_f1_f1_np.state(['selected'])
                self.np.set(True)
            if self.state['zp']:
                self.c_f1_f1_zp.state(['selected'])
                self.zp.set(True)
        else:
            self.c_f1_f1_dp.state(['selected'])
            self.dp.set(True)
            self.c_f1_f1_mp.state(['selected'])
            self.mp.set(True)
            self.c_f1_f1_sp.state(['selected'])
            self.sp.set(True)
            self.c_f1_f1_zp.state(['selected'])
            self.zp.set(True)
            
        
        self.l_f1_f2_dq = ttk.Label(self.f1_f2, text="d")
        self.l_f1_f2_hq = ttk.Label(self.f1_f2, text="h")
        self.l_f1_f2_mq = ttk.Label(self.f1_f2, text="\u03BC")
        self.l_f1_f2_lq = ttk.Label(self.f1_f2, text="log \u03C3")
        self.l_f1_f2_sq = ttk.Label(self.f1_f2, text="\u03C3")
        self.l_f1_f2_nq = ttk.Label(self.f1_f2, text="\u03B5")
        self.l_f1_f2_zq = ttk.Label(self.f1_f2, text="z")
        
        self.c_f1_f2_dq = ttk.Checkbutton(self.f1_f2, variable=self.dq, onvalue=True, offvalue=False)
        self.c_f1_f2_hq = ttk.Checkbutton(self.f1_f2, variable=self.hq, onvalue=True, offvalue=False)
        self.c_f1_f2_mq = ttk.Checkbutton(self.f1_f2, variable=self.mq, onvalue=True, offvalue=False)
        self.c_f1_f2_lq = ttk.Checkbutton(self.f1_f2, variable=self.lq, onvalue=True, offvalue=False)
        self.c_f1_f2_sq = ttk.Checkbutton(self.f1_f2, variable=self.sq, onvalue=True, offvalue=False)
        self.c_f1_f2_nq = ttk.Checkbutton(self.f1_f2, variable=self.nq, onvalue=True, offvalue=False)
        self.c_f1_f2_zq = ttk.Checkbutton(self.f1_f2, variable=self.zq, onvalue=True, offvalue=False)
        
        if not self.state is None:
            if self.state['dq']:
                self.dq.set(True)
                self.c_f1_f2_dq.state(['selected'])
            if self.state['hq']:
                self.hq.set(True)
                self.c_f1_f2_hq.state(['selected'])
            if self.state['mq']:
                self.mq.set(True)
                self.c_f1_f2_mq.state(['selected'])
            if self.state['lq']:
                self.lq.set(True)
                self.c_f1_f2_lq.state(['selected'])
            if self.state['sq']:
                self.sq.set(True)
                self.c_f1_f2_sq.state(['selected'])
            if self.state['nq']:
                self.nq.set(True)
                self.c_f1_f2_nq.state(['selected'])
            if self.state['zq']:
                self.zq.set(True)
                self.c_f1_f2_zq.state(['selected'])
        else:
            self.c_f1_f2_dq.state(['selected'])
            self.dq.set(True)
            self.c_f1_f2_mq.state(['selected'])
            self.mq.set(True)
            self.c_f1_f2_sq.state(['selected'])
            self.sq.set(True)
            self.c_f1_f2_zq.state(['selected'])
            self.zq.set(True)

        
        self.l_f1_f1_dp.grid(row=0, column=0, padx=padx_, pady=pady_,  sticky=tk.E)        
        self.l_f1_f1_hp.grid(row=1, column=0, padx=padx_, pady=pady_,  sticky=tk.E)        
        self.l_f1_f1_mp.grid(row=2, column=0, padx=padx_, pady=pady_,  sticky=tk.E)        
        self.l_f1_f1_lp.grid(row=3, column=0, padx=padx_, pady=pady_,  sticky=tk.E)        
        self.l_f1_f1_sp.grid(row=4, column=0, padx=padx_, pady=pady_,  sticky=tk.E)        
        self.l_f1_f1_np.grid(row=5, column=0, padx=padx_, pady=pady_,  sticky=tk.E)        
        self.l_f1_f1_zp.grid(row=6, column=0, padx=padx_, pady=pady_,  sticky=tk.E)        
        self.c_f1_f1_dp.grid(row=0, column=1, padx=padx_, pady=pady_,  sticky=tk.W)        
        self.c_f1_f1_hp.grid(row=1, column=1, padx=padx_, pady=pady_,  sticky=tk.W)        
        self.c_f1_f1_mp.grid(row=2, column=1, padx=padx_, pady=pady_,  sticky=tk.W)        
        self.c_f1_f1_lp.grid(row=3, column=1, padx=padx_, pady=pady_,  sticky=tk.W)        
        self.c_f1_f1_sp.grid(row=4, column=1, padx=padx_, pady=pady_,  sticky=tk.W)        
        self.c_f1_f1_np.grid(row=5, column=1, padx=padx_, pady=pady_,  sticky=tk.W)        
        self.c_f1_f1_zp.grid(row=6, column=1, padx=padx_, pady=pady_,  sticky=tk.W)        
                
        self.l_f1_f2_dq.grid(row=0, column=0, padx=padx_, pady=pady_,  sticky=tk.E)        
        self.l_f1_f2_hq.grid(row=1, column=0, padx=padx_, pady=pady_,  sticky=tk.E)        
        self.l_f1_f2_mq.grid(row=2, column=0, padx=padx_, pady=pady_,  sticky=tk.E)        
        self.l_f1_f2_lq.grid(row=3, column=0, padx=padx_, pady=pady_,  sticky=tk.E)        
        self.l_f1_f2_sq.grid(row=4, column=0, padx=padx_, pady=pady_,  sticky=tk.E)        
        self.l_f1_f2_nq.grid(row=5, column=0, padx=padx_, pady=pady_,  sticky=tk.E)        
        self.l_f1_f2_zq.grid(row=6, column=0, padx=padx_, pady=pady_,  sticky=tk.E)               
        self.c_f1_f2_dq.grid(row=0, column=1, padx=padx_, pady=pady_,  sticky=tk.W)        
        self.c_f1_f2_hq.grid(row=1, column=1, padx=padx_, pady=pady_,  sticky=tk.W)        
        self.c_f1_f2_mq.grid(row=2, column=1, padx=padx_, pady=pady_,  sticky=tk.W)        
        self.c_f1_f2_lq.grid(row=3, column=1, padx=padx_, pady=pady_,  sticky=tk.W)        
        self.c_f1_f2_sq.grid(row=4, column=1, padx=padx_, pady=pady_,  sticky=tk.W)        
        self.c_f1_f2_nq.grid(row=5, column=1, padx=padx_, pady=pady_,  sticky=tk.W)        
        self.c_f1_f2_zq.grid(row=6, column=1, padx=padx_, pady=pady_,  sticky=tk.W)        
                
        self.f1_f1.grid(row=0, column=0, padx=padx_, pady=pady_,  sticky=tk.E)        
        self.f1_f2.grid(row=0, column=1, padx=padx_, pady=pady_,  sticky=tk.E)        
        self.f1_f3.grid(columnspan=2)
        self.f1.pack(side=tk.BOTTOM, expand=1, fill=tk.BOTH,  padx=padx_, pady=pady_, anchor=tk.N)
        
    def doSelect(self):
        selectMap = {}
        
        selectMap['dp'] = self.dp.get()
        selectMap['hp'] = self.hp.get()
        selectMap['mp'] = self.mp.get()
        selectMap['lp'] = self.lp.get()
        selectMap['sp'] = self.sp.get()
        selectMap['np'] = self.np.get()
        selectMap['zp'] = self.zp.get()
        selectMap['dq'] = self.dq.get()
        selectMap['hq'] = self.hq.get()
        selectMap['mq'] = self.mq.get()
        selectMap['lq'] = self.lq.get()
        selectMap['sq'] = self.sq.get()
        selectMap['nq'] = self.nq.get()
        selectMap['zq'] = self.zq.get()
        
        self.context['layerdetails']  = selectMap  

        self.master.destroy()        
        
        
        

        
