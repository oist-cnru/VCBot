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
import math
import statistics
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

from collections import deque
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec


from GUI.Message import Message
from GUI.AnalysisPlot import AnalysisPlot
from GUI.Layer import Layer

class Analysis():
        
    def __init__(self, _name, _context, _master=None):
                    
        self.name = _name
        self.master = _master                
        self.context = _context
        self.root = self.context['main'].root
        self.nrl = self.context['nrl']
        self.ut = self.context['ut']        
        self.cwp = self.context['cwd']        
        self.delimiter = self.context['delimiter']
        self.modelConfigDir = self.context['modelconfigdir']
        self.datasetDir = self.context['datasetdir']
        self.experimentDir = self.context['experimentdir']        
        self.consoleBg =  self.context['consoleBg']
        self.consoleFg = self.context['consoleFg']    
        
        layerdetailsMap = {}
        layerdetailsMap['dp'] = True
        layerdetailsMap['hp'] = False
        layerdetailsMap['mp'] = True
        layerdetailsMap['lp'] = False
        layerdetailsMap['sp'] = True
        layerdetailsMap['np'] = False
        layerdetailsMap['zp'] = False
        layerdetailsMap['dq'] = True
        layerdetailsMap['hq'] = False
        layerdetailsMap['mq'] = True
        layerdetailsMap['lq'] = False
        layerdetailsMap['sq'] = True
        layerdetailsMap['nq'] = False
        layerdetailsMap['zq'] = False        
        
        self.context['layerdetails'] = layerdetailsMap 

        self.m = None
        self.e = None
                    
        f_padx = self.context['f_padx']
        padx_ = self.context['padx']
        pady_ = self.context['pady']        
        
        self.aplhaFgPlot = 1.0
        self.aplhaBgPlot = 0.2        
        self.datasetColor = 'darkgreen'
        self.drawColor = 'brown'

        # # === BLOCK 1: Parameters selection
        
        self.f_t1_f1 = ttk.Frame(self.master)        
        self.f_t1_f1_f1 = ttk.LabelFrame(self.f_t1_f1, relief=tk.SUNKEN, text=" Selection ")                                
        self.l_t1_f1_f1_N = ttk.Label(self.f_t1_f1_f1, text="Model:")
        self.l_t1_f1_f1_D = ttk.Label(self.f_t1_f1_f1, text="Dataset:")       
        self.l_t1_f1_f1_E = ttk.Label(self.f_t1_f1_f1, text="Experiment:")        
        self.l_t1_f1_f1_L = ttk.Label(self.f_t1_f1_f1, text="Layer:")        
        self.l_t1_f1_f1_P = ttk.Label(self.f_t1_f1_f1, text="Parameters:")        
        self.l_t1_f1_f1_F = ttk.Label(self.f_t1_f1_f1, text="Format:")        
        self.l_t1_f1_f1_Nv = ttk.Label(self.f_t1_f1_f1, text="               ")
        self.l_t1_f1_f1_Dv = ttk.Label(self.f_t1_f1_f1, text="               ")        
        self.c_t1_f1_f1_E = ttk.Combobox(self.f_t1_f1_f1, values=['empty'], width= 30, state="readonly")        
        self.c_t1_f1_f1_L = ttk.Combobox(self.f_t1_f1_f1, values=['empty'], width= 30, state="readonly")        
        self.b_t1_f1_f1_P = ttk.Button(self.f_t1_f1_f1, text="Select", command=self.doParamSelect)        
        self.c_t1_f1_f1_F = ttk.Combobox(self.f_t1_f1_f1, values=['Mean','Sum','Raw'], width= 20, state="readonly")        

        self.l_t1_f1_f1_N.grid(row=0, column=0, padx=padx_, pady=pady_,  sticky=tk.E)        
        self.l_t1_f1_f1_D.grid(row=1, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        self.l_t1_f1_f1_E.grid(row=2, column=0, padx=padx_, pady=pady_,  sticky=tk.E)        
        self.l_t1_f1_f1_L.grid(row=3, column=0, padx=padx_, pady=pady_,  sticky=tk.E)        
        self.l_t1_f1_f1_P.grid(row=4, column=0, padx=padx_, pady=pady_,  sticky=tk.E)        
        self.l_t1_f1_f1_F.grid(row=5, column=0, padx=padx_, pady=pady_,  sticky=tk.E)        
        self.l_t1_f1_f1_Nv.grid(row=0, column=1, padx=padx_, pady=pady_,  sticky=tk.W)        
        self.l_t1_f1_f1_Dv.grid(row=1, column=1, padx=padx_, pady=pady_,  sticky=tk.W)        
        self.c_t1_f1_f1_E.grid(row=2, column=1, padx=padx_, pady=pady_,  sticky=tk.W)
        self.c_t1_f1_f1_L.grid(row=3, column=1, padx=padx_, pady=pady_,  sticky=tk.W)
        self.b_t1_f1_f1_P.grid(row=4, column=1, padx=padx_, pady=pady_,  sticky=tk.W)
        self.c_t1_f1_f1_F.grid(row=5, column=1, padx=padx_, pady=pady_,  sticky=tk.W)
        
        self.c_t1_f1_f1_E.bind("<<ComboboxSelected>>", self.doComboExperiment)
        self.c_t1_f1_f1_L.bind("<<ComboboxSelected>>", self.doComboLayer)
        self.c_t1_f1_f1_F.bind("<<ComboboxSelected>>", self.doComboFormat)
        self.c_t1_f1_f1_E.current(0)
        self.c_t1_f1_f1_L.current(0)
        self.c_t1_f1_f1_F.current(0)
                
        self.f_t1_f1_f2 = ttk.Frame(self.f_t1_f1_f1)            
        self.b_t1_f1_f2_Statistics = ttk.Button(self.f_t1_f1_f2, text="Statistics",  command=self.doStatistics)        
        self.b_t1_f1_f2_Plot = ttk.Button(self.f_t1_f1_f2, text="Plot",  command=self.doPlot)        
        self.b_t1_f1_f2_Export = ttk.Button(self.f_t1_f1_f2, text="Export to CSV",  command=self.doExport)        
        self.b_t1_f1_f2_Remove = ttk.Button(self.f_t1_f1_f2, text="Remove",  command=self.doRemove)        
        self.b_t1_f1_f2_Statistics.grid(row=0, column=0, padx=padx_, pady=pady_,  sticky=tk.N)
        self.b_t1_f1_f2_Plot.grid(row=0, column=1, padx=padx_, pady=pady_,  sticky=tk.N)
        self.b_t1_f1_f2_Export.grid(row=0, column=2, padx=padx_, pady=pady_,  sticky=tk.N)
        self.b_t1_f1_f2_Remove.grid(row=0, column=3, padx=padx_, pady=pady_,  sticky=tk.N)
        self.f_t1_f1_f2.grid(columnspan=2)        
        self.f_t1_f1_f1.pack(side=tk.TOP, expand=1, fill=tk.X, padx=f_padx, pady=10, anchor=tk.N)
                                
        # === BLOCK 1: std output

        self.f_t1_f2 = ttk.LabelFrame(self.master,relief=tk.SUNKEN, text=" Output ")    
        self.s_t1_f2_bar = ttk.Scrollbar(self.f_t1_f2)
        self.t_t1_f2_val = tk.Text(self.f_t1_f2, bg=self.consoleBg, fg=self.consoleFg)
        self.s_t1_f2_bar.pack(side=tk.RIGHT, fill=tk.Y)
        self.t_t1_f2_val.pack(side=tk.TOP, fill=tk.BOTH, expand="yes")
        self.s_t1_f2_bar.config(command=self.t_t1_f2_val.yview)
        self.t_t1_f2_val.config(yscrollcommand=self.s_t1_f2_bar.set)
                
        self.messenger = Message(self.t_t1_f2_val)
            
        self.f_t1_f1.pack(side=tk.TOP, expand=0, fill=tk.X, padx=f_padx, anchor=tk.N)
        self.f_t1_f2.pack(side=tk.TOP, expand=1, fill=tk.BOTH,  padx=f_padx, anchor=tk.N)
        
                
    
    def doParamSelect(self):       
        
        popUp = tk.Toplevel(self.root)
        Layer(popUp, self.context)
            
    def doComboExperiment(self, event):
        
        eName = self.c_t1_f1_f1_E.get()
        basedir = self.experimentDir + '/' + self.m['name'] + '/' + eName        
        self.e = self.ut.parseExperiment(basedir)
        self.messenger.logConsole("Experiment \'{}\' selected!".format(eName))
        
    def doComboLayer(self, event):
        
        lName = self.c_t1_f1_f1_L.get() 
        self.messenger.logConsole("Layer \'{}\' selected!".format(lName))
        
    def doComboFormat(self, event):
        
        fName = self.c_t1_f1_f1_F.get() 
        self.messenger.logConsole("Format \'{}\' selected!".format(fName))


    def updateControls(self, reset=False):
        
        if not self.m is None:
            basedir = self.experimentDir + '/' + self.m['name']
            eList = self.ut.getDirList(basedir)            
            st = tk.NORMAL
            self.c_t1_f1_f1_E["values"] = eList
            if len(eList) > 0:
                self.c_t1_f1_f1_E.set(eList[0])
                basedir = self.experimentDir + '/' + self.m['name'] + '/' + self.c_t1_f1_f1_E.get()        
                self.e = self.ut.parseExperiment(basedir)
            else:
                self.c_t1_f1_f1_E.set('')
                st = tk.DISABLED        
            
            self.l_t1_f1_f1_Nv.configure(text=self.m['name'])
            self.l_t1_f1_f1_Dv.configure(text=self.m['dsname'])     
            self.b_t1_f1_f2_Statistics.configure(state=st)        
            self.b_t1_f1_f2_Plot.configure(state=st)        
            self.b_t1_f1_f2_Export.configure(state=st)       
            self.b_t1_f1_f2_Remove.configure(state=st)        
    
            if reset:
                layerList = ['All']
                for d in range(self.m['nlayers']):
                    layerList.append('{}'.format(d+1))
        
                self.c_t1_f1_f1_L["values"] = layerList
                self.c_t1_f1_f1_L.current(0)
        else:            
            self.c_t1_f1_f1_E["values"] = ['Empty']
            self.c_t1_f1_f1_L["values"] = ['All']
            self.c_t1_f1_f1_L.current(0)
        
    def doStatistics(self):
        
        if self.e == None:
            self.messenger.doWarning("Error: no experiment selected!")
            return
        
        state = np.float64(self.e['states'])
        elbo = np.float64(self.e['elbo'])
        
        self.experimentKeys = ['datetime',\
                              'model',
                              'dataset',
                              'numbertimes',
                              'samplingperiod',                              
                              'primitiveid',
                              'windowSize',
                              'w',
                              'postdiction',
                              'epochs',
                              'alpha',
                              'beta1',
                              'beta2']


        samplingperiod = int(self.e['samplingperiod'])
        time_ = int(self.e['numbertimes'])*samplingperiod/1000.0
        msg =       'Experiment Results\n'
        msg = msg + '====================\n'
        msg = msg + '{:<12}{:<15}{:<12}{:<12}{:<12}{:<12}{:<12}\n'.format('Context','Variable', 'Neuron', 'Dist.','Sum','Mean','stdev')
        msg = msg + '------------------------------------------------------------------------------------\n'
        context = 'Global'            
        msg = msg + '{:<12}{:<15}{:<12}{:<12}{:<12}{:<12}{:<12}\n'.format(context,'Time(s)','', '',time_,'','')
        sum_ = np.sum(elbo[:,0])
        mean_ = statistics.mean(elbo[:,0])
        stdev_ = statistics.stdev(elbo[:,0]) 
        msg = msg + '{:<12}{:<15}{:<12}{:<12}{:<12,.3f}{:<12,.3f}{:<12,.3f}\n'.format(context,'ELBO', '', '', sum_, mean_, stdev_)
        sum_ = np.sum(elbo[:,1])
        mean_ = statistics.mean(elbo[:,1])
        stdev_ = statistics.stdev(elbo[:,1]) 
        msg = msg + '{:<12}{:<15}{:<12}{:<12}{:<12,.3f}{:<12,.3f}{:<12,.3f}\n'.format(context,'Reconstruction','', '', sum_, mean_, stdev_)
        sum_ = np.sum(elbo[:,2])
        mean_ = statistics.mean(elbo[:,2])
        stdev_ = statistics.stdev(elbo[:,2]) 
        msg = msg + '{:<12}{:<15}{:<12}{:<12}{:<12,.3f}{:<12,.3f}{:<12,.3f}\n'.format(context,'Regulation','', '', sum_, mean_, stdev_)
        
        d = []
        for _d in self.ut.trimString(self.m['d']).split(self.delimiter):
            d.append(int(_d))
        z = []
        for _z in self.ut.trimString(self.m['z']).split(self.delimiter):
            z.append(int(_z))
        
        layerData_list = []
        for l in range(self.m['nlayers']):
            iterators = [d[l],d[l],z[l],z[l],z[l],z[l],z[l],d[l],d[l],z[l],z[l],z[l],z[l],z[l]]
            labels = ['d', 'h', '\u03BC', 'log \u03C3', '\u03C3', '\u03B5', 'z', 'd', 'h', '\u03BC', 'log \u03C3', '\u03C3', '\u03B5','z']            

            h = 0
            layerData_l_list = []
            k = 0
            for i in iterators:
                for j in range(i):
                    jh = j + h
                    sum_ = np.sum(state[:,jh])
                    mean_ = statistics.mean(state[:,jh])
                    stdev_ = statistics.stdev(state[:,jh]) 
                    #layerMap = {'k':'{}{}'.format(prefix[k],j),'s':sum_, 'm':mean_, 'sd':stdev_}
                    dist_ = 'Prior'
                    if k >= 7:
                        dist_ = 'Posterior'
                    layerMap = {'k':labels[k],'n':j+1,'dist': dist_, 's':sum_, 'm':mean_, 'sd':stdev_}
                    layerData_l_list.append(layerMap)
                h = h + i
                k = k + 1
            layerData_list.append(layerData_l_list)
        
        for l in range(self.m['nlayers']):
            msg = msg + '------------------------------------------------------------------------------------\n'
            list_l = layerData_list[l]
            for v in range(len(list_l)):
                context = 'Layer #{}'.format(l+1)
                map_ = list_l[v]
                msg = msg + '{:<12}{:<15}{:<12}{:<12}{:<12,.3f}{:<12,.3f}{:<12,.3f}\n'.format(context,map_['k'],map_['n'], map_['dist'], map_['s'], map_['m'], map_['sd'])

        msg = msg + '====================\n'

        
        self.messenger.logConsole(msg,False,False)
        
        
    def doPlot(self):
        
        if self.e == None:
            self.messenger.doWarning("Error: no experiment selected!")
            return
            
        params = {}
        params['m'] = self.m
        params['e'] = self.e
        params['ut'] = self.ut
        params['delimiter'] = self.delimiter        
        params['layer'] = self.c_t1_f1_f1_L.get()
        params['format'] = self.c_t1_f1_f1_F.get()
        params["layerdetails"] = self.context['layerdetails']
        
        AnalysisPlot(params)
        

    def doExport(self):

        basedir = self.experimentDir + '/' + self.m['name'] + '/' + self.c_t1_f1_f1_E.get()                 
        expDir = self.messenger.doDirSelection()
        if self.ut.exportExperimentCsv(basedir, expDir):
            self.messenger.doInfo("CSV data files were succesfully exported to the folder: \'{}\'".format(expDir))            
        else:
            self.messenger.doWarning("An error ocurred when exporting data!")

    def doRemove(self):
        
        eName = self.c_t1_f1_f1_E.get() 
        if len(eName) == 0:
            return        
        basedir = self.experimentDir + '/' + self.m['name'] + '/' + eName
        if self.messenger.doYesNo("This operation cannot be undone. Do you want to proceed?'"):
            if self.ut.removeDir(basedir):                    
                #self.messenger.doInfo("The experiment was successfully removed!'")
                self.messenger.logConsole("The experiment was successfully removed!'")
                self.updateControls()
            else:
                self.messenger.doWarning("Error: the experiment could not be removed!'")
                
    # === Observer design pattern methods
       
    def updateObserver(self):    
        
        print("Analysis: Nothing to be informed")
        
    def notify(self):
        
        if self.context['t'] or self.context['e']:
            return 

        self.m = self.context['m']                
        self.updateControls(True)
            
            
            
            
        
            

    
    
