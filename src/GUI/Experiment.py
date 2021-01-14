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



import time
import tkinter as tk
from tkinter import ttk
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

from collections import deque
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec

from network.PVRNN import PVRNN

import ctypes
from GUI.Message import Message
from GUI.Mouse import Mouse


class Experiment():
        
    def __init__(self, _name, _context, _master=None):

        self.name = _name                
        self.master = _master                
        self.context = _context
        self.nrl = self.context['nrl']
        self.ut = self.context['ut']        
        self.cwp = self.context['cwd']        
        self.delimiter = self.context['delimiter']
        self.modelConfigDir = self.context['modelconfigdir']
        self.datasetDir = self.context['datasetdir']
        self.experimentDir = self.context['experimentdir']        
        self.stateParser = None
        
        self.m = None
        self.d = None
        
        self.modelLoaded = False
                    
        f_padx = self.context['f_padx']
        f_pady = self.context['f_pady']
        padx_ = self.context['padx']
        pady_ = self.context['pady']        
        self.xLimMouse = [-65.0, 65.0]
        self.yLimMouse = [-65.0, 65.0]
        self.xLimPlot = [-55.0, 55.0]
        self.yLimPlot = [-55.0, 55.0]
        
        self.motionSaturation = 5
        
        self.aplhaFgPlot = 1.0
        self.aplhaBgPlot = 0.2        
        self.datasetColor = '#4affff'
        self.drawColor = 'brown'

        self.time_ms = self.ut.getCurrentTimeMS()        
        
        #--- Initialization         
        self.mouseIn = False
        self.drawHuman = False
        self.nActDof = 2        
        self.mc = 0.9                    
        self.step = 0        
        self.samplingPeriod = 0        
        self.showERLog = False
        self.stateBufferSize = 0
        
        # default values 
        self.wInit = '1.0e-5'
        self.postdiction_epochs = 15
        self.windowSize = 5
        self.alpha = 0.3
        self.beta1 = 0.1
        self.beta2 = 0.95
        self.nBuffdata = self.windowSize * self.nActDof
        self.expTimeSteps = 2000
        self.runExperiment = False                        
        self.plotBuffSize = 50 # for visualization purposes
        self.topDownId = 0
        self.ERStartTime = self.windowSize -1
        
        # maps 
        self.primSet = {}   
        self.state_index = {}
        
        # containers
        self.robotBuffer = deque(maxlen=self.plotBuffSize)    
        self.humanBuffer = deque(maxlen=self.plotBuffSize)    
        self.posWinBuffer = deque(maxlen=self.windowSize)    
        self.signal_list = []
        self.opt_elbo_list = []
        self.state_list = []
        self.tgt_pos_list = []
        self.cur_pos_list = []        
        self.hum_pos_list = []        
        self.hum_int_list = []        
        self.primCanvas = []  

        self.signalVar =  tk.BooleanVar() 
        self.sliderLabelVar = tk.StringVar()
        self.sliderLabelVar.set('{:.2f}'.format(self.mc))
        self.WStext = 'Workspace'
              
                
        # === BLOCK 1: Parameter section
        
        self.f_t1_f1_controls = ttk.Frame(self.master)
                
        self.f_t1_f1_f1 = ttk.LabelFrame(self.f_t1_f1_controls, relief=tk.SUNKEN, text=" Selection ")                                
        self.l_t1_f1_f1_N = ttk.Label(self.f_t1_f1_f1, text="Model:")
        self.l_t1_f1_f1_D = ttk.Label(self.f_t1_f1_f1, text="Dataset:")       
        self.l_t1_f1_f1_S = ttk.Label(self.f_t1_f1_f1, text="Signal:")
        self.l_t1_f1_f1_F = ttk.Label(self.f_t1_f1_f1, text="Format:")
        self.l_t1_f1_f1_Nv = ttk.Label(self.f_t1_f1_f1, text="               ")
        self.l_t1_f1_f1_Dv = ttk.Label(self.f_t1_f1_f1, text="               ")
        self.c_t1_f1_f1_S = ttk.Combobox(self.f_t1_f1_f1, values=['empty'], width= 20, state="readonly")
        self.c_t1_f1_f1_F = ttk.Combobox(self.f_t1_f1_f1, values=['Raw','Sum','Mean'], width= 20, state="readonly")
        self.l_t1_f1_f1_N.grid(row=0, column=0, padx=padx_, pady=pady_,  sticky=tk.E)        
        self.l_t1_f1_f1_D.grid(row=1, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        self.l_t1_f1_f1_S.grid(row=2, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        self.l_t1_f1_f1_F.grid(row=3, column=0, padx=padx_, pady=pady_,  sticky=tk.E)        
        self.l_t1_f1_f1_Nv.grid(row=0, column=1, padx=padx_, pady=pady_,  sticky=tk.W)        
        self.l_t1_f1_f1_Dv.grid(row=1, column=1, padx=padx_, pady=pady_,  sticky=tk.W)        
        self.c_t1_f1_f1_S.grid(row=2, column=1, padx=padx_, pady=pady_,  sticky=tk.W)
        self.c_t1_f1_f1_F.grid(row=3, column=1, padx=padx_, pady=pady_,  sticky=tk.W)
        self.c_t1_f1_f1_S.bind("<<ComboboxSelected>>", self.doComboSignal)
        self.c_t1_f1_f1_F.current(0)
        self.c_t1_f1_f1_F.bind("<<ComboboxSelected>>", self.doComboFormatSignal)
        
        self.f_t1_f1_f2 = ttk.LabelFrame(self.f_t1_f1_controls, relief=tk.SUNKEN, text=" ADAM ")          
        self.l_t1_f1_f2_ADAM_A = ttk.Label(self.f_t1_f1_f2, text="\u03B1:")
        self.l_t1_f1_f2_ADAM_B1 = ttk.Label(self.f_t1_f1_f2, text="\u03B2\u2081:")
        self.l_t1_f1_f2_ADAM_B2 = ttk.Label(self.f_t1_f1_f2, text="\u03B2\u2082:")        
        self.e_t1_f1_f2_ADAM_A = ttk.Entry(self.f_t1_f1_f2, width=10)
        self.e_t1_f1_f2_ADAM_B1 = ttk.Entry(self.f_t1_f1_f2, width=10)
        self.e_t1_f1_f2_ADAM_B2 = ttk.Entry(self.f_t1_f1_f2, width=10)                
        
        self.e_t1_f1_f2_ADAM_A.bind("<KeyRelease>", self.entryKeyRelease)
        self.e_t1_f1_f2_ADAM_B1.bind("<KeyRelease>", self.entryKeyRelease)
        self.e_t1_f1_f2_ADAM_B2.bind("<KeyRelease>", self.entryKeyRelease)    
        
        self.l_t1_f1_f2_ADAM_A.grid(row=0, column=0, padx=padx_, pady=pady_,  sticky=tk.E)        
        self.l_t1_f1_f2_ADAM_B1.grid(row=1, column=0, padx=padx_, pady=pady_,  sticky=tk.E)        
        self.l_t1_f1_f2_ADAM_B2.grid(row=2, column=0, padx=padx_, pady=pady_,  sticky=tk.E)        
        self.e_t1_f1_f2_ADAM_A.grid(row=0, column=1, padx=padx_, pady=pady_,  sticky=tk.W)        
        self.e_t1_f1_f2_ADAM_B1.grid(row=1, column=1, padx=padx_, pady=pady_,  sticky=tk.W)        
        self.e_t1_f1_f2_ADAM_B2.grid(row=2, column=1, padx=padx_, pady=pady_,  sticky=tk.W)        
               
        self.f_t1_f1_f3 = ttk.LabelFrame(self.f_t1_f1_controls, relief=tk.SUNKEN, text=" Postdiction ")
        self.l_t1_f1_f3_Epd = ttk.Label(self.f_t1_f1_f3, text="Enable:")
        self.l_t1_f1_f3_E = ttk.Label(self.f_t1_f1_f3, text="Epochs:")
        self.l_t1_f1_f3_Win = ttk.Label(self.f_t1_f1_f3, text="Window:")        
        self.l_t1_f1_f3_W = ttk.Label(self.f_t1_f1_f3, text="w:")
        self.c_t1_f1_f3_Epd = ttk.Combobox(self.f_t1_f1_f3, values=['yes','no'], width= 10, state="readonly")
        self.e_t1_f1_f3_E = ttk.Entry(self.f_t1_f1_f3, width=10)
        self.e_t1_f1_f3_Win = ttk.Entry(self.f_t1_f1_f3, width=15)
        self.e_t1_f1_f3_W = ttk.Entry(self.f_t1_f1_f3, width=15) 
        
        self.c_t1_f1_f3_Epd.bind("<<ComboboxSelected>>", self.doComboPostdiction)
        self.c_t1_f1_f3_Epd.set('yes')        
        self.e_enabled = True
        
        self.interationMode = False                
        
        self.e_t1_f1_f3_E.bind("<KeyRelease>", self.entryKeyRelease)
        self.e_t1_f1_f3_Win.bind("<KeyRelease>", self.entryKeyRelease)
        self.e_t1_f1_f3_W.bind("<KeyRelease>", self.entryKeyRelease)

        self.l_t1_f1_f3_Epd.grid(row=0, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        self.l_t1_f1_f3_E.grid(row=1, column=0, padx=padx_, pady=pady_,  sticky=tk.E)        
        self.l_t1_f1_f3_Win.grid(row=2, column=0, padx=padx_, pady=pady_,  sticky=tk.E)        
        self.l_t1_f1_f3_W.grid(row=3, column=0, padx=padx_, pady=pady_,  sticky=tk.E)      
        self.c_t1_f1_f3_Epd.grid(row=0, column=1, padx=padx_, pady=pady_,  sticky=tk.W)      
        self.e_t1_f1_f3_E.grid(row=1, column=1, padx=padx_, pady=pady_,  sticky=tk.W)        
        self.e_t1_f1_f3_Win.grid(row=2, column=1, padx=padx_, pady=pady_,  sticky=tk.W)        
        self.e_t1_f1_f3_W.grid(row=3, column=1, padx=padx_, pady=pady_,  sticky=tk.W)
        
        self.f_t1_f1_f4 = ttk.LabelFrame(self.f_t1_f1_controls, relief=tk.SUNKEN, text=" Experiment ")            
        self.l_t1_f1_f4_T = ttk.Label(self.f_t1_f1_f4, text="Time steps:")
        self.l_t1_f1_f4_P = ttk.Label(self.f_t1_f1_f4, text="Period in ms:")
        self.l_t1_f1_f4_Prim = ttk.Label(self.f_t1_f1_f4, text="Initial primitive:")
        self.l_t1_f1_f4_Mc = ttk.Label(self.f_t1_f1_f4, text="Motor compliance:")        
        self.e_t1_f1_f4_T = ttk.Entry(self.f_t1_f1_f4, width=10)
        self.e_t1_f1_f4_P = ttk.Entry(self.f_t1_f1_f4, width=10)
        self.c_t1_f1_f4_Prim = ttk.Combobox(self.f_t1_f1_f4, values=['empty'], width= 10, state="readonly", postcommand=self.initComboPrimitive)
        
        self.f_t1_f1_f4_f1 = ttk.Frame(self.f_t1_f1_f4)            
        self.l_t1_f1_f4_Mc_l = ttk.Label(self.f_t1_f1_f4_f1, textvariable=self.sliderLabelVar)

        self.s_t1_f1_f4_Mc = ttk.Scale(self.f_t1_f1_f4_f1, from_=0.0, to=1.0, orient=tk.HORIZONTAL,\
                                           length=100, value = self.mc, command=self.doScaleMotorCompliance)
        

        self.s_t1_f1_f4_Mc.grid(row=0, column=0, padx=0, pady=pady_,  sticky=tk.W)        
        self.l_t1_f1_f4_Mc_l.grid(row=0, column=1, padx=padx_, pady=pady_,  sticky=tk.W)        
        
        self.e_t1_f1_f4_T.bind("<KeyRelease>", self.entryKeyRelease)
        self.e_t1_f1_f4_P.bind("<KeyRelease>", self.entryKeyRelease)        
        self.c_t1_f1_f4_Prim.bind("<<ComboboxSelected>>", self.doComboPrimitive)
        
        self.l_t1_f1_f4_T.grid(row=0, column=0, padx=padx_, pady=pady_,  sticky=tk.E)        
        self.l_t1_f1_f4_P.grid(row=1, column=0, padx=padx_, pady=pady_,  sticky=tk.E)        
        self.l_t1_f1_f4_Prim.grid(row=2, column=0, padx=padx_, pady=pady_,  sticky=tk.E) 
        self.l_t1_f1_f4_Mc.grid(row=3, column=0, padx=padx_, pady=pady_,  sticky=tk.E)         
        self.e_t1_f1_f4_T.grid(row=0, column=1, padx=padx_, pady=pady_,  sticky=tk.W)        
        self.e_t1_f1_f4_P.grid(row=1, column=1, padx=padx_, pady=pady_,  sticky=tk.W)        
        self.c_t1_f1_f4_Prim.grid(row=2, column=1, padx=padx_, pady=pady_,  sticky=tk.W)        
        self.f_t1_f1_f4_f1.grid(row=3, column=1, padx=padx_, pady=pady_,  sticky=tk.W)         
        
        self.f_t1_f1_f4_f2 = ttk.Frame(self.f_t1_f1_f4)            
        self.b_t1_f1_f4_f2_Start = ttk.Button(self.f_t1_f1_f4_f2, text="Start",  command=self.doStart)        
        self.b_t1_f1_f4_f2_Pause = ttk.Button(self.f_t1_f1_f4_f2, text="Pause",  command=self.doPause)
        self.b_t1_f1_f4_f2_Reset = ttk.Button(self.f_t1_f1_f4_f2, text="Reset",  command=self.doReset)
        self.b_t1_f1_f4_f2_Save = ttk.Button(self.f_t1_f1_f4_f2, text="Save",  command=self.doSave)                
        self.b_t1_f1_f4_f2_Start.grid(row=0, column=0, padx=padx_, pady=pady_,  sticky=tk.N)
        self.b_t1_f1_f4_f2_Pause.grid(row=0, column=1, padx=padx_, pady=pady_,  sticky=tk.N) 
        self.b_t1_f1_f4_f2_Reset.grid(row=0, column=2, padx=padx_, pady=pady_,  sticky=tk.N)
        self.b_t1_f1_f4_f2_Save.grid(row=0, column=3, padx=padx_, pady=pady_,  sticky=tk.N)
        self.f_t1_f1_f4_f2.grid(columnspan=2)
                        
        self.f_t1_f1_f1.pack(side=tk.LEFT, expand=1, fill=tk.X, padx=padx_, pady=pady_, anchor=tk.N)
        self.f_t1_f1_f2.pack(side=tk.LEFT, expand=1, fill=tk.X, padx=padx_, pady=pady_, anchor=tk.N)
        self.f_t1_f1_f3.pack(side=tk.LEFT, expand=1, fill=tk.X, padx=padx_, pady=pady_, anchor=tk.N)
        self.f_t1_f1_f4.pack(side=tk.LEFT, expand=1, fill=tk.X, padx=padx_, pady=pady_, anchor=tk.N)
                
        # === BLOCK 2: Workspace
        
        self.f_t1_f2 = ttk.Frame(self.master, relief=tk.SUNKEN)
        cW = 5.5 
        cH = 5.5
        
        self.fig = plt.Figure(constrained_layout=False,figsize=(cW, cH), dpi=100, facecolor=(0.1, 0.1, 0.1))
        cHf1 = cH        
        cHf2 = cHf1/20.0
        cH = cHf1+cHf2

        fWidths = [cW]
        fHeights = [cHf1, cHf2]
        spec = gridspec.GridSpec(ncols=1, nrows=2, width_ratios=fWidths, height_ratios=fHeights, figure=self.fig)
        self.ax = self.fig.add_subplot(spec[0, 0])     
                
        # reducing the number of ticks in the axis to 3
        self.ax.xaxis.set_major_locator(plt.MaxNLocator(3)) 
        self.ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        self.ax.axis('off')
        self.ax.axis('square')
        self.axSignal = self.fig.add_subplot(spec[1, 0])                                  
        self.ax.set_xlim(self.xLimMouse)
        self.ax.set_ylim(self.yLimMouse)
        self.drawPrim, = self.ax.plot(0, 0, self.drawColor)
        self.tCtrMsg = 'Press control to interact'
        self.drawControl = self.ax.text(15.0, 65.0, '', style='italic', color='teal', bbox={'facecolor': 'teal', 'alpha': 0.5, 'pad': 10})
        self.drawWSTitle = self.ax.text(-55.0, 60.0, '', style='italic', color=self.datasetColor )       
                
        ## signal plot
        self.axSignal.axis('off')
        self.signalYmin = 0.0
        self.signalYmax = 20.0
                
        self.f_t1_f2_canvas = FigureCanvasTkAgg(self.fig, master=self.f_t1_f2)  # A tk.DrawingArea.        
        self.f_t1_f2_canvas.draw()
        self.f_t1_f2_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1, padx=f_padx)
        self.toolbar = NavigationToolbar2Tk(self.f_t1_f2_canvas, self.f_t1_f2)
        self.toolbar.update()    
        
        self.mouseH = Mouse(self.drawPrim, self.xLimMouse, self.yLimMouse)        
        self.fig.canvas.mpl_connect('key_press_event', self.keyPressed)                    
        self.fig.canvas.mpl_connect('key_release_event', self.keyReleased)         
        
        self.messenger = Message()
         
        self.f_t1_f1_controls.pack(side=tk.TOP, expand=0, fill=tk.X, padx=f_padx, pady=10, anchor=tk.N)
        self.f_t1_f2.pack(side=tk.TOP, expand=1, fill=tk.BOTH,  padx=f_padx, anchor=tk.N)
        
        # plot objects
        self.robotCanvas, = self.ax.plot(0.0, 0.0)
        self.robotEffCanvas, = self.ax.plot(0.0, 0.0, color='red', marker='o')        
        self.humanCanvas, = self.ax.plot(0.0, 0.0)
        self.signalCanvas, = self.axSignal.plot(0.0, 0.0, color='orangered')
        self.signalTitle = self.ax.text(-55.0, -63.0, '', style='italic', color=self.datasetColor)                

        self.cur_pos = np.array([0.0,0.0])        
        self.hum_pos = np.array([0.0,0.0])        
        
        # thread loop 
        self.timer = self.fig.canvas.new_timer(interval=10, callbacks=[(self.doLoop, [], {})])        
        self.timer.start()

    def doScaleMotorCompliance(self,val):
        
        self.sliderLabelVar.set('{:.2f}'.format(float(val)))
        self.mc = float(val)
        
    def doComboSignal(self, event):
        
        opt = self.c_t1_f1_f1_S.get()
        self.signal_list = []
        self.signalTitle.set_text('Signal: {}'.format(opt))
        self.f_t1_f2_canvas.get_tk_widget().focus_set()

    def doComboFormatSignal(self, event):
        
        self.f_t1_f2_canvas.get_tk_widget().focus_set()
        
    
    def processSignal(self):        
        
        idx = self.c_t1_f1_f1_S.current()
        format_idx = self.c_t1_f1_f1_F.current()
        d = 0.0
        if idx < 3:
            d = self.opt_elbo_list[-1][idx]
        else:
            s = self.nrl.e_getStateMap(self.state_list[-1])
            k = self.nrl.e_getStateKey(idx - 3)            
            d_raw = s[k]
            if format_idx == 0:
                d = d_raw[0]
            if format_idx == 1:
                d = np.sum(np.array(d_raw))
            else:
                d = np.mean(np.array(d_raw))
        self.signal_list.append(d)
        
    
    def doComboPostdiction(self, event):
        
        self.e_enabled = (self.c_t1_f1_f3_Epd.get() == 'yes')
        
        st = tk.DISABLED
        if self.interationMode:
            st = tk.NORMAL
            
        self.e_t1_f1_f3_E.configure(state=st)
        self.e_t1_f1_f3_Win.configure(state=st)
        self.e_t1_f1_f3_W.configure(state=st) 
        self.experimentReset()                    
    
                 
    def doStart(self):
        
        if not self.m is None:
            if not self.modelLoaded:
                self.doReset()    
            self.b_t1_f1_f4_f2_Start.configure(state= tk.DISABLED)
            self.b_t1_f1_f4_f2_Pause.configure(state= tk.NORMAL)
            self.b_t1_f1_f4_f2_Reset.configure(state= tk.DISABLED)
            self.b_t1_f1_f4_f2_Save.configure(state= tk.DISABLED)
            self.runExperiment = True        
            self.drawControl.set_text(self.tCtrMsg)
            self.drawWSTitle.set_text(self.WStext)  
            self.robotCanvas.axes.figure.canvas.draw()             
            self.f_t1_f2_canvas.get_tk_widget().focus_set()
            self.doComboSignal(None)
            self.updateObserver()
        
    def doPause(self):
        
        if not self.m is None:
            self.b_t1_f1_f4_f2_Start.configure(state= tk.NORMAL)
            self.b_t1_f1_f4_f2_Pause.configure(state= tk.DISABLED)        
            self.b_t1_f1_f4_f2_Reset.configure(state= tk.NORMAL)
            self.b_t1_f1_f4_f2_Save.configure(state= tk.NORMAL)
            self.runExperiment = False               
            self.drawControl.set_text('')  
            self.robotCanvas.axes.figure.canvas.draw()     
            self.f_t1_f2_canvas.get_tk_widget().focus_set()
            self.updateObserver()

    def doReset(self):
        
        if not self.m is None:          
            self.experimentReset()                    
            self.drawControl.set_text('')
            self.drawWSTitle.set_text('')  
            self.signalTitle.set_text('')  
            self.robotCanvas.axes.figure.canvas.draw()     
        
    def doSave(self, loop=False):
        
        if not self.m is None:
            print ("Saving experiment")
            _e = {}
            _e['model'] = self.m['name']
            _e['dataset'] = self.d['name']
            _e['numbertimes'] = '{}'.format(self.step)
            _e['samplingperiod'] = '{}'.format(self.d['samplingperiod'])
            _e['primitiveid'] = self.c_t1_f1_f4_Prim.get()
            _e['windowSize'] = self.e_t1_f1_f3_Win.get()
            _e['w'] = self.e_t1_f1_f3_W.get()
            _e['postdiction'] = '{}'.format(self.e_enabled)
            _e['epochs'] = self.e_t1_f1_f3_E.get()
            _e['alpha'] = self.e_t1_f1_f2_ADAM_A.get()
            _e['beta1'] = self.e_t1_f1_f2_ADAM_B1.get()
            _e['beta2'] = self.e_t1_f1_f2_ADAM_B2.get()
            _e['motorcompliance'] = '{}'.format(self.mc)
            _e['cur_pos'] = np.vstack(self.cur_pos_list)        
            _e['tgt_pos'] = np.vstack(self.tgt_pos_list)
            _e['hum_pos'] = np.vstack(self.hum_pos_list)
            _e['hum_int'] = np.vstack(self.hum_int_list)
            _e['states'] = np.vstack(self.state_list)
            _e['elbo'] = np.vstack(self.opt_elbo_list)        
            
            
            if self.ut.saveExperiment(self.experimentDir+'/'+self.m['name'], _e):
                if loop:
                    self.messenger.doInfo("The experiment has finished, data has been saved succesfully!")
                else:
                    self.messenger.doInfo("The experiment has been saved!")
                    self.updateObserver()
            else:
                self.messenger.doWarning("Error, the experiment was not saved!")
                
            self.b_t1_f1_f4_f2_Start.configure(state= tk.DISABLED)
            self.b_t1_f1_f4_f2_Pause.configure(state= tk.DISABLED)
            self.b_t1_f1_f4_f2_Reset.configure(state= tk.NORMAL)
            self.b_t1_f1_f4_f2_Save.configure(state= tk.DISABLED)
            self.interationMode = False
            self.updateObserver()
            
    
    def doLoop(self):
        
        ms_ = self.ut.getCurrentTimeMS()
        tDiff = ms_ - self.time_ms 
                
        if self.step == self.expTimeSteps and self.runExperiment:
            self.runExperiment = False
            self.doSave(True)
            
        if (tDiff > self.samplingPeriod and self.runExperiment):
            tgt_pos = np.array([0.0,0.0])        
            tgt_pos_buffer = np.zeros((self.nActDof,), dtype=float);
            dataOut = (ctypes.c_float * self.nActDof)(*tgt_pos_buffer)
            
            # generate the robot intention        
            self.nrl.e_generate(dataOut)                                        
            tgt_pos = np.frombuffer(dataOut, np.float32)
            new_cur_pos = tgt_pos                
                        
            # capture the human intention                     
            hXY, mouseIn = self.mouseH.getXY()            
            if self.interationMode:
                if not mouseIn:      
                    hXY = self.hum_pos
                self.master.configure(cursor="hand1")    
                self.humanBuffer.append(hXY)                        
                
                # merge robot and human intention
                new_cur_pos = self.mc*hXY + (1.0 - self.mc)*tgt_pos
                self.hum_pos = hXY
            else:
                self.master.configure(cursor="")    

            delta = new_cur_pos - self.cur_pos             
            xDeltaSat = min(self.motionSaturation, max(delta[0], -self.motionSaturation))
            yDeltaSat = min(self.motionSaturation, max(delta[1], -self.motionSaturation))                    
            self.cur_pos = self.cur_pos + np.array([xDeltaSat, yDeltaSat])
    
            self.posWinBuffer.append(self.cur_pos)
            opt_elbo = [0.0,0.0,0.0]
            if self.e_enabled:                                             
                if (self.step >= self.ERStartTime): 
                    
                    elbo_buffer = np.zeros((3,), dtype=float);
                    elboOut = (ctypes.c_float * 3)(*elbo_buffer)        
                    pos_win_1d = np.hstack(self.posWinBuffer)                
                    self.nrl.e_postdict((ctypes.c_float * self.nBuffdata)(*pos_win_1d), elboOut, self.showERLog)
                    opt_elbo = np.frombuffer(elboOut, np.float32).tolist()                    
    
            m_state = np.zeros((self.stateBufferSize,), dtype=float);
            m_stateOut = (ctypes.c_float * self.stateBufferSize)(*m_state)
            self.nrl.e_getState(m_stateOut)
            st_data = np.frombuffer(m_stateOut, np.float32)
                        
            # appending data to experiment containers    
            self.state_list.append(st_data)
            
            self.opt_elbo_list.append(opt_elbo)
            self.tgt_pos_list.append(tgt_pos)
            self.cur_pos_list.append(self.cur_pos)
            self.hum_pos_list.append(self.hum_pos)
            self.hum_int_list.append(self.interationMode)
                                    
            # get the most recent measurement       
            if len(self.humanBuffer) > 0 and self.drawHuman :
                vData = np.vstack(self.humanBuffer)
                self.humanCanvas.set_data(vData[:,0], vData[:,1])
                    
            #draw primitives
            for k in self.primSet.keys():
                p = self.primSet[k]
                vData = p['d']
                c = p['c']
                c.set_data(vData[:,0], vData[:,1])                         
                                        
            self.robotBuffer.append(self.cur_pos)            
            vData = np.vstack(self.robotBuffer)
            self.robotCanvas.set_data(vData[:,0], vData[:,1])                
            self.robotEffCanvas.set_data(vData[-1,0], vData[-1,1])                
            self.drawControl.set_text(self.tCtrMsg)    
            self.robotCanvas.axes.figure.canvas.draw()            
                
            self.processSignal()
            vData2 = np.vstack(self.signal_list)
            eT = np.linspace(0.0, (self.step*self.samplingPeriod)/1000.0, num=len(self.signal_list))              
            
            self.signalCanvas.set_data(eT, vData2)
            self.signalCanvas.axes.autoscale_view(tight=True)            
            self.signalCanvas.axes.relim()            
            self.signalCanvas.axes.figure.canvas.draw()                        
                    
            self.time_ms = ms_
            self.step = self.step + 1 
            

    def keyPressed(self, event):    
        
        if event.key == "control":                            
            if self.runExperiment:           
                self.tCtrMsg = ''
                self.interationMode = True                
            
    def keyReleased(self, event):          
        
        if event.key == "control":    
            self.tCtrMsg = 'Press control to interact'
            self.interationMode = False                        

    def allocatePrimitive(self, _d):        
        
        i = len(self.primSet.keys()) + 1
        c, = self.ax.plot(1000,1000, color=self.datasetColor, alpha=0.1)            
        self.primSet[i] = {'c':c, 'd':_d}            
            
    def clearPrimitives(self):   
        
        for k in self.primSet.keys():
            self.primSet[k]['c'].remove()            
        self.primSet.clear()        
        self.fig.canvas.draw() 
     
    def initComboPrimitive(self,reset=False):
        
        dList = []
        for i in range(len(self.d['data'])):
            dList.append('{}'.format(i+1))
        text = "empty"
        st = tk.DISABLED
        if len(dList) > 0:
            st = "readonly",
            text = dList[0]    
        self.c_t1_f1_f4_Prim["values"] = dList
        self.c_t1_f1_f4_Prim.configure(state=st)
        if not reset:
            self.c_t1_f1_f4_Prim.set(text)
    
    def doComboPrimitive(self, event):
        
        try:
            self.topDownId = self.ut.parseString(self.c_t1_f1_f4_Prim.get(), 'int', self.delimiter)[0] - 1
            self.b_t1_f1_f4_f2_Reset.configure(state= tk.NORMAL)
        except: 
            self.messenger.doWarning('The field \'Primitive\' should correspond to an integer value')
            
    def experimentReset(self):
        
        self.runExperiment  = False 
        mName = self.m['name']
        
        proceed = True        
        error_msg = ''        
        
        try:
            self.w = self.ut.parseString(self.e_t1_f1_f3_W.get(), 'float', self.delimiter)                    
        except: 
            error_msg = error_msg + 'The field \'W\' should be set to float value(s)\n'
            proceed = False            
        try:            
            self.postdiction_epochs = self.ut.parseString(self.e_t1_f1_f3_E.get(), 'int', self.delimiter)[0]                    
        except: 
            error_msg = error_msg + 'The field \'epochs\' should be set to an integer value\n'
            proceed = False            
        try:
            self.windowSize = self.ut.parseString(self.e_t1_f1_f3_Win.get(), 'int', self.delimiter)[0]
            self.nBuffdata = self.windowSize * self.nActDof
            self.ERStartTime = self.windowSize -1 
            self.posWinBuffer = deque(maxlen=self.windowSize)              
        except: 
            error_msg = error_msg + 'The field \'window\' should be set to an integer value\n'
            proceed = False                
        try:            
            self.alpha = self.ut.parseString(self.e_t1_f1_f2_ADAM_A.get(), 'float', self.delimiter)[0]                    
            self.beta1 = self.ut.parseString(self.e_t1_f1_f2_ADAM_B1.get(), 'float', self.delimiter)[0]                    
            self.beta2 = self.ut.parseString(self.e_t1_f1_f2_ADAM_B2.get(), 'float', self.delimiter)[0]                    
        except: 
            error_msg = error_msg + 'The fields \u03B1, \u03B2\u2081, \u03B2\u2082, should be set to an float values\n'
            proceed = False                    

        try:
            self.samplingPeriod = self.ut.parseString(self.e_t1_f1_f4_P.get(), 'int', self.delimiter)[0]                    
        except: 
            error_msg = error_msg + 'The field \'Period in ms\' should be set to an integer value\n'
            proceed = False     
            
        try:
            self.expTimeSteps = self.ut.parseString(self.e_t1_f1_f4_T.get(), 'int', self.delimiter)[0]                    
        except: 
            error_msg = error_msg + 'The field \'Time steps\' should be set to an integer value\n'
            proceed = False     

        try:
            self.topDownId = self.ut.parseString(self.c_t1_f1_f4_Prim.get(), 'int', self.delimiter)[0] - 1
        except: 
            error_msg = error_msg + 'The field \'Primitive\' should correspond to an integer value\n'
            proceed = False     

            
        if not proceed:
            self.messenger.doWarning(error_msg)
            return
            
        modelConfigPath = self.cwp + '/' + self.ut.modelPathString(self.modelConfigDir, mName)                

        self.step = 0
        self.humanBuffer.clear()
        self.robotBuffer.clear()
        self.opt_elbo_list = []
        self.state_list = []
        self.signal_list  = []
        self.cur_pos_list  = []
        self.tgt_pos_list  = []
        self.hum_pos_list  = []
        self.hum_int_list  = []
        self.cur_pos = np.array([0.0,0.0]) 
        self.robotCanvas.set_data(0.0, 0.0)                
        self.robotEffCanvas.set_data(0.0, 0.0)                
        self.robotCanvas.axes.figure.canvas.draw()            
        self.signalCanvas.set_data(0.0,0.0)
        self.signalCanvas.axes.autoscale_view(tight=True)            
        self.signalCanvas.axes.relim()            
        self.signalCanvas.axes.figure.canvas.draw()                                        
        
        self.nrl.newModel(modelConfigPath.encode('ascii')) 
        self.nrl.load()
        #self.stateBufferSize = self.nrl.getStateBufferSize(self.ut.trimString(self.m['d']), self.ut.trimString(self.m['z']), self.delimiter)
  
        self.stateParser = PVRNN(self.ut.trimString(self.m['d']), self.ut.trimString(self.m['z']), self.ut.trimString(self.m['t']), self.delimiter)
        self.nrl.setStateParser(self.stateParser)
        self.stateBufferSize = self.nrl.getStateBufferSize()
        
        storeStates = False
        storeER = False
        self.nrl.e_enable(self.topDownId,\
                            self.windowSize,
                            (ctypes.c_float * len(self.w))(*self.w),
                            self.expTimeSteps, 
                            self.postdiction_epochs,
                            (ctypes.c_float)(self.alpha),
                            (ctypes.c_float)(self.beta1),
                            (ctypes.c_float)(self.beta2), storeStates, storeER)
        
        sList = ['N-ELBO','Reconstruction', 'Regulation']
        for l in range(self.m['nlayers']):
            sList.append('d prior (Layer {})'.format(l+1))
            sList.append('h prior (Layer {})'.format(l+1))
            sList.append('\u03BC prior (Layer {})'.format(l+1))
            sList.append('log \u03C3 prior (Layer {})'.format(l+1))
            sList.append('\u03C3 prior (Layer {})'.format(l+1))
            sList.append('\u03B5 prior (Layer {})'.format(l+1))
            sList.append('z prior (Layer {})'.format(l+1))
            sList.append('d prior (Layer {})'.format(l+1))
            sList.append('h prior (Layer {})'.format(l+1))
            sList.append('\u03BC posterior (Layer {})'.format(l+1))
            sList.append('log \u03C3 posterior (Layer {})'.format(l+1))
            sList.append('\u03C3 posterior (Layer {})'.format(l+1))
            sList.append('\u03B5 posterior (Layer {})'.format(l+1))
            sList.append('z posterior (Layer {})'.format(l+1))
        self.c_t1_f1_f1_S["values"] = sList
        self.c_t1_f1_f1_S.set(sList[0])
        self.clearPrimitives()
        for p in self.d['data']:
            self.allocatePrimitive(p)
        self.initComboPrimitive(True)
        self.b_t1_f1_f4_f2_Start.configure(state= tk.NORMAL)
        self.b_t1_f1_f4_f2_Pause.configure(state= tk.DISABLED)
        self.b_t1_f1_f4_f2_Reset.configure(state= tk.DISABLED)
        self.b_t1_f1_f4_f2_Save.configure(state= tk.DISABLED)
        self.drawControl.set_text('')  
        self.f_t1_f2_canvas.get_tk_widget().focus_set()
        self.modelLoaded = True

    def entryKeyRelease(self,event):
                
        counter = 0
        if not self.e_t1_f1_f2_ADAM_A.get() == '':
            counter = counter + 1
        if not self.e_t1_f1_f2_ADAM_B1.get() == '':
            counter = counter + 1
        if not self.e_t1_f1_f2_ADAM_B2.get() == '':
            counter = counter + 1        
        if not self.e_t1_f1_f3_E.get() == '':
            counter = counter + 1
        if not self.e_t1_f1_f3_Win.get() == '':
            counter = counter + 1
        if not self.e_t1_f1_f3_W.get() == '':
            counter = counter + 1        
        if not self.e_t1_f1_f4_T.get() == '':
            counter = counter + 1
        if not self.e_t1_f1_f4_P.get() == '':
            counter = counter + 1

        self.b_t1_f1_f4_f2_Reset.configure(state= tk.NORMAL)
        st = tk.DISABLED
        if counter == 8:
            st = tk.NORMAL
        self.b_t1_f1_f4_f2_Start.configure(state=st)
        if self.runExperiment:
            self.f_t1_f2_canvas.get_tk_widget().focus_set()
        
        
    # === Observer design pattern methods
    
    def updateObserver(self):    
        
        self.context['e'] = self.runExperiment
        self.context['main'].update(self.name)
        
    def notify(self):
        
        if self.context['t'] or self.context['e']:
            return 
        
        self.m = self.context['m']
        self.modelLoaded = False
        self.d = None
        self.step = 0                

        self.b_t1_f1_f4_f2_Start.configure(state= tk.DISABLED)
        self.b_t1_f1_f4_f2_Pause.configure(state= tk.DISABLED)        
        self.b_t1_f1_f4_f2_Reset.configure(state= tk.DISABLED)
        self.b_t1_f1_f4_f2_Save.configure(state= tk.DISABLED)
               
        if self.m is None:
            
            self.f_t1_f2_canvas.get_tk_widget().focus_set()
            self.drawControl.set_text('')
            self.drawWSTitle.set_text('')  
            self.signalTitle.set_text('')  
            
        elif not self.m['train'] :                   
            
               self.m = None
               return
        else:
               dName = self.m['dsname']
               
               if len(dName) > 0:
                   self.d = self.ut.parseDataset(self.datasetDir + '/' + dName)
                   
                   if not self.d is None:
                       
                        self.b_t1_f1_f4_f2_Start.configure(state= tk.NORMAL)

                        self.e_t1_f1_f4_P.delete(0,tk.END)
                        self.e_t1_f1_f4_P.insert(tk.END, self.d['samplingperiod'])
                        self.l_t1_f1_f1_Nv.config(text=self.m['name'])
                        self.l_t1_f1_f1_Dv.config(text=self.d['name'])
                           
                        # default values

                        if self.ut.trimString(self.e_t1_f1_f2_ADAM_A.get().strip())  == '':
                               self.e_t1_f1_f2_ADAM_A.insert(tk.END, '{}'.format(self.alpha))
                        if self.ut.trimString(self.e_t1_f1_f2_ADAM_B1.get()) == '':
                            self.e_t1_f1_f2_ADAM_B1.insert(tk.END, '{}'.format(self.beta1))
                        if self.ut.trimString(self.e_t1_f1_f2_ADAM_B2.get()) == '':
                            self.e_t1_f1_f2_ADAM_B2.insert(tk.END, '{}'.format(self.beta2))
                        if self.ut.trimString(self.e_t1_f1_f3_W.get()) == '':
                            wText = ''
                            for i in range(self.m['nlayers']):    
                                wText = wText + self.wInit + self.delimiter
                            self.e_t1_f1_f3_W.insert(tk.END, wText[0:-1])
                        if self.ut.trimString(self.e_t1_f1_f3_Win.get()) == '':    
                            self.e_t1_f1_f3_Win.insert(tk.END, '{}'.format(self.windowSize))
                        if self.ut.trimString(self.e_t1_f1_f3_E.get()) == '':    
                            self.e_t1_f1_f3_E.insert(tk.END, '{}'.format(self.postdiction_epochs))    
                        if self.ut.trimString(self.e_t1_f1_f4_T.get()) == '':                
                            self.e_t1_f1_f4_T.insert(tk.END, '{}'.format(self.expTimeSteps))    
                        if self.ut.trimString(self.e_t1_f1_f4_P.get()) == '':                
                            self.e_t1_f1_f4_P.insert(tk.END, self.d['samplingperiod'])    
                            
                        self.entryKeyRelease(None)
                        self.initComboPrimitive()
                        
                   else:
                        self.m = None
                    
        
            
