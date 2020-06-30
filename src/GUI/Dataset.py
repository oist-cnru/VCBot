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


import time
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

from collections import deque
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

from GUI.Message import Message
from GUI.Mouse import Mouse

class Dataset():
        
    def __init__(self, _name, _context, _master=None):
                
        self.name = _name
        self.master = _master                
        self.context = _context
        self.ut = self.context['ut']        
        self.datasetDir = self.context['datasetdir']
        f_padx = self.context['f_padx']
        f_pady = self.context['f_pady']
        padx_ = self.context['padx']
        pady_ = self.context['pady']        
        wEntry = self.context['wentry']
        
        self.xLimMouse = [-55.0, 55.0]
        self.yLimMouse = [-55.0, 55.0]
        
        self.d = None
        self.m = None
        
        self.numberTimes = 0
        self.samplingPeriod = 0
        self.interationMode = False
        self.addPrimitiveMode = False
        self.addPrimitiveClicked = False        
        self.aplhaFgPlot = 1.0
        self.aplhaBgPlot = 0.2        
        self.datasetColor = '#4affff'
        self.drawColor = 'brown'
                        

        self.f_t1_f1_controls = ttk.Frame(self.master)
        
        # === BLOCK 1: Dataset creation 
        
        self.f_t1_f1 = ttk.LabelFrame(self.f_t1_f1_controls, relief=tk.SUNKEN, text=" Addition ")
        
        self.l_t1_f1_M = ttk.Label(self.f_t1_f1, text="Name:")
        self.l_t1_f1_L = ttk.Label(self.f_t1_f1, text="Primitive length:")
        self.l_t1_f1_S = ttk.Label(self.f_t1_f1, text="Period in ms:")
                
        self.e_t1_f1_M = ttk.Entry(self.f_t1_f1, width=wEntry)
        self.e_t1_f1_L = ttk.Entry(self.f_t1_f1, width=wEntry)
        self.e_t1_f1_S = ttk.Entry(self.f_t1_f1, width=wEntry)
                        
        self.f_t1_f1_f1 = ttk.Frame(self.f_t1_f1)
        self.b_t1_f1_clear = ttk.Button(self.f_t1_f1_f1, text="Clear",  command=self.doClearDataset)
        self.b_t1_f1_add = ttk.Button(self.f_t1_f1_f1, text="Add",  command=self.doAddDataset)        
                
        self.l_t1_f1_M.grid(row=0, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        self.e_t1_f1_M.grid(row=0, column=1, padx=padx_, pady=pady_,  sticky=tk.W)
        
        self.l_t1_f1_L.grid(row=1, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        self.e_t1_f1_L.grid(row=1, column=1, padx=padx_, pady=pady_,  sticky=tk.W)
        
        self.l_t1_f1_S.grid(row=2, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        self.e_t1_f1_S.grid(row=2, column=1, padx=padx_, pady=pady_,  sticky=tk.W)
                
        self.b_t1_f1_clear.grid(row=0, column=0, padx=padx_, pady=pady_)
        self.b_t1_f1_add.grid(row=0, column=1, padx=padx_, pady=pady_)        
        self.f_t1_f1_f1.grid(columnspan=2)
       
        # === BLOCK 2: Dataset selection
    
        self.f_t1_f2 = ttk.LabelFrame(self.f_t1_f1_controls, relief=tk.SUNKEN, text=" Selection ")
        self.l_t1_f2_M = ttk.Label(self.f_t1_f2, text="Name:")        
        self.c_t1_f2_M = ttk.Combobox(self.f_t1_f2, values=['empty'], state="readonly", postcommand=self.initComboDataset)
        
        self.c_t1_f2_M.bind("<<ComboboxSelected>>", self.doComboDataset)
        
        self.l_t1_f2_N = ttk.Label(self.f_t1_f2, text="Number:")
        self.l_t1_f2_D = ttk.Label(self.f_t1_f2, text="Details:")
        
        self.l_t1_f2_Nv = ttk.Label(self.f_t1_f2, text="-")
        self.l_t1_f2_Dv = ttk.Label(self.f_t1_f2, text="-")
        
        self.l_t1_f2_M.grid(row=0, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        self.c_t1_f2_M.grid(row=0, column=1, padx=padx_, pady=pady_,  sticky=tk.W)
        self.l_t1_f2_N.grid(row=1, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        self.l_t1_f2_Nv.grid(row=1, column=1, padx=padx_, pady=pady_,  sticky=tk.W)
        self.l_t1_f2_D.grid(row=2, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        self.l_t1_f2_Dv.grid(row=2, column=1, padx=padx_, pady=pady_,  sticky=tk.W)

        self.f_t1_f2_f1 = ttk.Frame(self.f_t1_f2)
        self.b_t1_f2_select = ttk.Button(self.f_t1_f2_f1, text="Select",  command=self.doSelectDataset)    
        self.b_t1_f2_edit = ttk.Button(self.f_t1_f2_f1, text="Edit",  command=self.doEditDataset)    
        self.b_t1_f2_remove = ttk.Button(self.f_t1_f2_f1, text="Remove",  command=self.doRemoveDataset)    

        self.b_t1_f2_select.configure(state=tk.DISABLED)
        self.b_t1_f2_edit.configure(state=tk.DISABLED)
        self.b_t1_f2_remove.configure(state=tk.DISABLED)
        
        self.b_t1_f2_select.grid(row=0, column=0, padx=padx_, pady=pady_)
        self.b_t1_f2_edit.grid(row=0, column=1, padx=padx_, pady=pady_)
        self.b_t1_f2_remove.grid(row=0, column=2, padx=padx_, pady=pady_)        
        self.f_t1_f2_f1.grid(columnspan=3)

        # === BLOCK 3: Dataset Edition 
        
        self.f_t1_f3 = ttk.LabelFrame(self.f_t1_f1_controls, relief=tk.SUNKEN, text=" Edition ")
        self.l_t1_f3_M = ttk.Label(self.f_t1_f3, text="Primitive:")        
        self.c_t1_f3_M = ttk.Combobox(self.f_t1_f3, values=['empty'], state="readonly", postcommand=self.initComboPrimitive)
        
        self.c_t1_f3_M.bind("<<ComboboxSelected>>", self.doComboPrimitive)
        
        self.l_t1_f3_M.grid(row=0, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        self.c_t1_f3_M.grid(row=0, column=1, padx=padx_, pady=pady_,  sticky=tk.W)
                
        self.f_t1_f3_f1 = ttk.Frame(self.f_t1_f3)
        self.b_t1_f3_add = ttk.Button(self.f_t1_f3_f1, text="Add",  command=self.doAddPrimitive)            
        self.b_t1_f3_remove = ttk.Button(self.f_t1_f3_f1, text="Remove",  command=self.doRemovePrimitive)    

        self.b_t1_f3_add.configure(state=tk.DISABLED)        
        self.b_t1_f3_remove.configure(state=tk.DISABLED)
        
        self.b_t1_f3_add.grid(row=0, column=0, padx=padx_, pady=pady_)        
        self.b_t1_f3_remove.grid(row=0, column=2, padx=padx_, pady=pady_)        
        self.f_t1_f3_f1.grid(columnspan=2)

        # === BLOCK 4: Workspace

        self.primSet = {}        
        
        self.f_t1_f4 = ttk.LabelFrame(self.master, relief=tk.SUNKEN, text=" Workspace ")
        self.cW = 5.5 
        self.cH = 5.5
     
        self.fig = plt.Figure(constrained_layout=False,figsize=(self.cW, self.cH), dpi=100, facecolor=(0.1, 0.1, 0.1))
        self.ax = self.fig.add_subplot(111)
        
        self.ax.set_xlim(self.xLimMouse)
        self.ax.set_ylim(self.yLimMouse)    
        self.ws_Top = self.ax.plot([self.xLimMouse[0],self.xLimMouse[1]],[self.yLimMouse[1],self.yLimMouse[1]], color='silver') 
        self.ws_bottom = self.ax.plot([self.xLimMouse[0],self.xLimMouse[1]],[self.yLimMouse[0],self.yLimMouse[0]], color='silver') 
        self.ws_left = self.ax.plot([self.xLimMouse[0],self.xLimMouse[0]],[self.yLimMouse[0],self.yLimMouse[1]], color='silver') 
        self.ws_right = self.ax.plot([self.xLimMouse[1],self.xLimMouse[1]],[self.yLimMouse[0],self.yLimMouse[1]], color='silver') 
        self.ax.axis('off')
        self.ax.axis('square')
        self.drawPrim, = self.ax.plot(0, 0, self.drawColor)
        self.drawTime = self.ax.text(15.0, 62.0, '', style='italic', bbox={'facecolor': self.drawColor, 'alpha': 0.5, 'pad': 10})
        self.drawControl = self.ax.text(15.0, 62.0, '', style='italic', bbox={'facecolor': self.datasetColor, 'alpha': 0.5, 'pad': 10})
        
        self.f_t1_f4_canvas = FigureCanvasTkAgg(self.fig, master=self.f_t1_f4)  # A tk.DrawingArea.        
        self.f_t1_f4_canvas.draw()
        self.f_t1_f4_canvas.get_tk_widget().pack(side=tk.TOP, anchor=tk.N, fill=tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar2Tk(self.f_t1_f4_canvas, self.f_t1_f4)
        self.toolbar.update()       
    
        self.mouseH = Mouse(self.drawPrim, self.xLimMouse, self.yLimMouse)
        self.fig.canvas.mpl_connect('key_press_event', self.keyPressed)                    
        self.fig.canvas.mpl_connect('key_release_event', self.keyReleased)
        self.timer = self.fig.canvas.new_timer(interval=10, callbacks=[(self.doLoop, [], {})])
        self.timer.start()

        self.drawBuffer = deque(maxlen=self.numberTimes)    
        self.time_ms = self.ut.getCurrentTimeMS()        
                
        self.messenger = Message()
        
        self.f_t1_f1.pack(side=tk.LEFT, expand=1, fill=tk.X, padx=f_padx, pady=f_pady, anchor=tk.NW)
        self.f_t1_f2.pack(side=tk.LEFT, expand=1, fill=tk.X, padx=f_padx, pady=f_pady, anchor=tk.NW)
        self.f_t1_f3.pack(side=tk.LEFT, expand=1, fill=tk.X, padx=f_padx, pady=f_pady, anchor=tk.NW)
        
        self.f_t1_f1_controls.pack(side=tk.TOP, expand=0, fill=tk.X, anchor=tk.N)
        self.f_t1_f4.pack(side=tk.TOP, expand=1, fill=tk.BOTH,  padx=f_padx, anchor=tk.N)
        
                    
    def keyPressed(self, event):    
        
        if event.key == "control":            
            self.addPrimitiveClicked = False
            if self.addPrimitiveMode:                
                self.interationMode = True                
            
    def keyReleased(self, event):  
        
        if event.key == "control":
            self.master.configure(cursor="")          
            self.interationMode = False            
            if self.addPrimitiveMode:
                self.handlePrimitiveAddition()
                                
    def doLoop(self):
        
        ms_ = self.ut.getCurrentTimeMS()
        tDiff = ms_ - self.time_ms 
        if (tDiff > self.samplingPeriod):
            hXY, mouseIn = self.mouseH.getXY()
            if self.interationMode and mouseIn:                              
                self.drawBuffer.append(hXY)              
                
            #draw current capture
            bLen = len(self.drawBuffer)
            redraw = False
            tMsg = ''
            tCtrMsg = ''
            
            if self.addPrimitiveClicked:
                redraw = True            
                tCtrMsg = 'Press the control key to start'
                
            if self.addPrimitiveMode:
                if bLen > 0:
                    redraw = True            
                    vData = np.vstack(self.drawBuffer)
                    self.drawPrim.set_data(vData[:,0], vData[:,1])                                                
                    self.master.configure(cursor="hand1")    
                    tMsg = 'Sampled steps = {}/{}'.format(bLen,self.numberTimes)                
                    tCtrMsg = ''
            else:
                self.drawPrim.set_data(5000, 5000)                                                

            #draw previous capture            
            for k in self.primSet.keys():
                redraw= True
                p = self.primSet[k]
                vData = p['d']
                c = p['c']
                c.set_data(vData[:,0], vData[:,1])                         
                
                if self.c_t1_f3_M.get() == str(k):                    
                    c.set(alpha=self.aplhaFgPlot)                
                else:
                    c.set(alpha=self.aplhaBgPlot)                
                        
            if redraw:
                self.refheshPlot = False
                self.drawTime.set_text(tMsg)                
                self.drawControl.set_text(tCtrMsg)                
                self.drawPrim.axes.autoscale_view()
                self.drawPrim.axes.figure.canvas.draw()                            
            
            self.time_ms = ms_        
        
    # === callback function 
    
    def callback(self):    
            
        self.f_t1_f4_canvas.get_tk_widget().focus_set()  
        self.c_t1_f2_M.state(['readonly'])              
        self.c_t1_f3_M.state(['readonly'])              
    
    # === Dataset adition methods 
        
    def doAddDataset(self):     
        
        nDataset = self.e_t1_f1_M.get()                                        
        if nDataset == "" or not nDataset.isalnum():
            self.messenger.doInfo("Please provide an alphanumeric name for the dataset!")
        elif self.ut.datasetExists(self.datasetDir, nDataset):
            self.messenger.doWarning("The dataset already exists!")
        else:        
            d = self.ut.datasetFactory()
            lnT = self.ut.parseString(self.e_t1_f1_L.get(), 'int', ' ')    
            lsR = self.ut.parseString(self.e_t1_f1_S.get(), 'int', ' ')                
            if (len(lnT) > 0  and len(lsR) > 0):
                d['name'] = nDataset
                d['numbertimes'] = '{}'.format(lnT[0])
                d['samplingperiod'] = '{}'.format(lsR[0])                
                self.d = None
                if(self.ut.saveDataset(self.datasetDir, d)):
                    self.messenger.doInfo("The dataset was added successfully!")
                    self.d = d
                    self.updateDatasetControls(text=nDataset)
                    self.clearPrimitives()
                else:
                    self.messenger.doWarning("Error the dataset could not be added!")                    
                
            else:                
                self.messenger.doWarning("Please provide integer numbers for the desired primitive length and sampling rate!")
    
    def doClearDataset(self): 
        
        self.e_t1_f1_M.delete(0, tk.END)        
        self.e_t1_f1_L.delete(0, tk.END)
        self.e_t1_f1_S.delete(0, tk.END)
    
    # === Dataset management methods
    
    def initComboDataset(self): 
        
        self.updateDatasetControls()
        
    def updateDatasetControls(self, text=''):
        
        dList = self.ut.getDirList(self.datasetDir)
        if len(dList) > 0 :                        
            self.c_t1_f2_M["values"] = dList
            self.c_t1_f2_M.configure(state=tk.NORMAL)
            self.c_t1_f2_M.set(text)
            
        t1 = t2 ='-'         
        s = tk.DISABLED
        if len(text) > 0:
            self.numberTimes = int(self.d['numbertimes'])
            self.samplingPeriod = int(self.d['samplingperiod'])
            t1 = '{} primitives'.format(len(self.d['data']))
            t2 = 'length {}, period {} ms'.format(self.numberTimes, self.samplingPeriod)
            s = tk.NORMAL                     
        self.l_t1_f2_Nv.config(text=t1)
        self.l_t1_f2_Dv.config(text=t2)
        self.b_t1_f2_edit.configure(state=s)
        self.b_t1_f2_remove.configure(state=s)
        if self.m is None:
            self.b_t1_f2_select.configure(state=tk.DISABLED)
        else:
            self.b_t1_f2_select.configure(state=s)
        self.c_t1_f2_M.state(['readonly'])              
        self.c_t1_f3_M.state(['readonly'])              
        
    def doComboDataset(self, _event):   
        
        dName = self.c_t1_f2_M.get()        
        if len(dName) > 0 :
            self.d = self.ut.parseDataset(self.datasetDir + '/' + dName)
            self.updateDatasetControls(text=dName)           
            self.clearPrimitives() 
            for p in self.d['data']:
                self.allocatePrimitive(p)
                                
            self.updatePrimitiveControls(enable=False)
            self.c_t1_f2_M.state(['readonly'])              
            self.c_t1_f3_M.state(['readonly'])              

    def doSelectDataset(self):
        
        if not self.d is None:
            dName = self.d['name']       
            self.updateObserver()        
            self.messenger.doInfo('The datset \'{}\' has been selected!'.format(dName))                         
        
    def doEditDataset(self):
        
        self.updatePrimitiveControls()
        
    def doRemoveDataset(self):   
                     
        nDataset = self.c_t1_f2_M.get()        
        if self.messenger.doYesNo("This operation cannot be undone. Are ou sure about removing the dataset \'{}\'".format(nDataset)):            
            datasetPath = self.datasetDir + '/' + nDataset
            if self.ut.removeDataset(datasetPath):                           
                self.messenger.doInfo('The dataset \'{}\' has been removed!'.format(nDataset))                            
                self.clearPrimitives()
                self.updatePrimitiveControls(enable=False)
                self.updateDatasetControls()                
            else:
                self.messenger.doWarning('The dataset \'{}\' could not be removed!'.format(nDataset))            
        self.d = None    
        self.updateObserver()

    # === Primitive edition methods ---------
        
    def initComboPrimitive(self):         
               
        self.f_t1_f4_canvas.get_tk_widget().focus_set()
    
    def handlePrimitiveAddition(self):
        
        pLen = len(self.drawBuffer)                
        if pLen  ==  self.numberTimes:                                                            
            self.allocatePrimitive(np.vstack(self.drawBuffer))
            self.updatePrimitiveControls()        
        if not self.ut.saveDataset(self.datasetDir, self.d):
            self.messenger.doWarning("Error, the dataset could not be updated!")
        self.addPrimitiveMode = False        
        self.updateAddPrimitiveControl()                
                
    def allocatePrimitive(self, _d):   
        
            i = len(self.primSet.keys()) + 1
            c, = self.ax.plot(1000,1000, color=self.datasetColor, alpha=0.1)            
            self.primSet[i] = {'c':c, 'd':_d}
            self.drawBuffer.clear()            
    
    def deallocatePrimitive(self, _i):  
        
            newPrimSet = {}
            k = 1
            for i in range(1,len(self.primSet.keys())+1):
                if not i == _i:
                    newPrimSet[k] = self.primSet[i] 
                    k = k + 1
                else:
                    self.primSet[i]['c'].remove()
            self.primSet = newPrimSet
            self.fig.canvas.draw()  
            
    def clearPrimitives(self):  
        
        for k in self.primSet.keys():
            self.primSet[k]['c'].remove()            
        self.primSet.clear()        
        self.fig.canvas.draw()                            
    
    def updatePrimitiveControls(self, enable=True):
        
        pList = []        
        pDataList = []  
        i = 1
        for k in self.primSet.keys():
            pList.append(i)
            pDataList.append(self.primSet[k]['d'])
            i = i + 1                
        nP = len(pDataList)
        self.d['data'] = pDataList   
        self.d['numberprims'] = '{}'.format(nP)
        
        if nP > 0 :            
            self.c_t1_f3_M["values"] = pList
            self.c_t1_f3_M.set(pList[-1])                
        
        remState = tk.DISABLED
        addState = tk.DISABLED
        if enable:
            addState = tk.NORMAL               
            if len(self.primSet.keys()) > 0 :   
                remState = tk.NORMAL            
        else:
            self.c_t1_f3_M.set('')        
            self.c_t1_f3_M["values"] = []
            
        self.b_t1_f3_add.config(state=addState)
        self.b_t1_f3_remove.config(state=remState)
        self.f_t1_f4_canvas.get_tk_widget().focus_set()
        
        
    def updateAddPrimitiveControl(self):      
        
        state = tk.NORMAL
        if self.addPrimitiveMode:
            state = tk.DISABLED
        self.b_t1_f3_add.configure(state=state)                           
        self.b_t1_f3_remove.configure(state=state)                           
            
                        
    def doComboPrimitive(self, _event):        
        
        self.f_t1_f4_canvas.get_tk_widget().focus_set()
        self.c_t1_f2_M.state(['readonly'])              
        self.c_t1_f3_M.state(['readonly'])              

                                    
    def doAddPrimitive(self):            
        
        self.addPrimitiveClicked = True
        self.f_t1_f4_canvas.get_tk_widget().focus_set()
        if (self.numberTimes > 0 and self.samplingPeriod > 0):            
            self.addPrimitiveMode = True                       
            self.updateAddPrimitiveControl()            
            self.drawBuffer = deque(maxlen=self.numberTimes)    
                
    def doRemovePrimitive(self):
        
        self.deallocatePrimitive(self.c_t1_f3_M.current()+1)
        self.updatePrimitiveControls()        
        if not self.ut.saveDataset(self.datasetDir, self.d):
             self.messenger.doWarning("Error, the dataset could not be updated!")
                
    # === Observer design pattern methods
    
    def updateObserver(self):
        
        if self.d is None:
            self.m = None
            self.context['m'] = self.m
            self.context['d'] = self.d
        elif not (self.context['m'] is None): 
            self.b_t1_f2_select.configure(state=tk.DISABLED)                                          
            if (self.context['m']['train']):
                self.m = None                
                self.context['m'] = self.m
            
        self.context['d'] = self.d
        self.context['main'].update(self.name)
    
    def notify(self):
        
        if self.context['t'] or self.context['e']:
            return 

        if not self.m == self.context['m']:                        
            stSelect = tk.NORMAL
            self.m = self.context['m']
            if not self.m is None:                                                         
                if len(self.m['dsname']) > 0 and self.m['dsname'] in self.c_t1_f2_M["values"]:
                    self.c_t1_f2_M.set(self.m['dsname'])
                    self.doComboDataset(None)
                    stSelect = tk.DISABLED                    
                else:
                    stSelect = tk.NORMAL                    
                    
            else:
                self.d = None
                stSelect = tk.DISABLED
            
            self.b_t1_f2_select.configure(state=stSelect)
                         
        
