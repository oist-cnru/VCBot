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
from ttkthemes import themed_tk as tthk
from GUI.Modeling import Modeling
from GUI.Dataset import Dataset
from GUI.Training import Training
from GUI.Experiment import Experiment
from GUI.Analysis import Analysis
from GUI.Help import Help 
from GUI.About import About
from GUI.Message import Message 
import time

class Splash(tk.Toplevel):
    
    def __init__(self, parent_, title_):
        
        tk.Toplevel.__init__(self, parent_)
        self.title(title_)
        self.center_window(600,373)
        #self.overrideredirect(1) #to remove the window border
        self.photo = tk.PhotoImage(file="images/splash.png")
        self.labelbanner = ttk.Label(self, image=self.photo, background="#1a1a1a")
        self.labelbanner.pack()
        self.update()
        time.sleep(2)
    
    def center_window(self, width=300, height=200):
        
        # get screen width and height
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        
        # calculate position x and y coordinates
        x = (screen_width/2) - (width/2)
        y = (screen_height/2) - (height/2)
        self.geometry('%dx%d+%d+%d' % (width, height, x, y))

class Main():
    
    def __init__(self, _context):                
        
        self.m = None 
        self.d = None
        self.t = False
        self.e = False
        self.debug = False
        
        self.context = _context
        self.appTitle = self.context['appTitle']        
        self.controlBg = self.context['controlBg']
        self.controlFg = self.context['controlFg']
        self.consoleBg = self.context['consoleBg']
        self.appTheme = self.context['themeid']
        self.rootBg = self.context['rootBg']
        self.minH = self.context['minH']
        self.minW = self.context['minW']        
        self.f_padx = self.context['f_padx']                
        self.context['m'] = self.m
        self.context['m'] = self.d
        self.context['t'] = self.t
        self.context['e'] = self.e
                        
        self.root = tthk.ThemedTk()                        
        #self.root.state('zoomed')
        #self.root.attributes('-zoomed', True)
        
        self.root.set_theme(self.appTheme)
        self.root.configure(bg=self.rootBg)        

        self.center_window(self.minW, self.minH)
        self.root.withdraw()
        
        splash = Splash(self.root, self.appTitle)
        self.context['main'] = self        
        
        self.root.wm_title(self.appTitle)
        self.root.protocol("WM_DELETE_WINDOW", self.doClose)
        self.root.bind('<Escape>', self.doEscape)
        
        self.msg = Message()
        
        photo = tk.PhotoImage(file="images/banner.png")
        self.labelbanner = tk.Label(self.root, image=photo, bg='#0f0f0f')
        self.labelbanner.pack(side=tk.TOP, anchor=tk.NW)
                
        ttkStyle = ttk.Style()
        ttkStyle.map('TCombobox', fieldbackground=[('readonly',self.controlBg)], foreground=[('readonly',self.controlFg)])
        ttkStyle.configure("TEntry", fieldbackground=self.controlBg, foreground=self.controlFg)
        ttkStyle.configure("Text", fieldbackground=self.controlBg, foreground=self.controlFg) 
        
        self.notebook = ttk.Notebook(self.root)

        self.tabModeling = ttk.Frame(self.notebook)        
        self.tabDataset = ttk.Frame(self.notebook)
        self.tabTraining = ttk.Frame(self.notebook)
        self.tabExperiment = ttk.Frame(self.notebook)
        self.tabAnalysis = ttk.Frame(self.notebook)        
        self.tabHelp = ttk.Frame(self.notebook)        
        self.tabAbout = ttk.Frame(self.notebook)        

        self.msgNovelty = {'Modeling':False, 'Dataset':False, 'Training':False, 'Experiment':False, 'Analysis':False, 'Help': False, 'About':False}
        
        self.modeling = Modeling("Modeling", self.context, self.tabModeling)        
        self.dataset = Dataset("Dataset", self.context, self.tabDataset)
        self.training = Training("Training", self.context, self.tabTraining)
        self.experiment = Experiment("Experiment", self.context, self.tabExperiment)
        self.analysis = Analysis("Analysis", self.context, self.tabAnalysis)
        self.help = Help("Help", self.context, self.tabHelp)
        self.about = About("About", self.context, self.tabAbout)
                
        self.notebook.add(self.tabModeling, text="Modeling")
        self.notebook.add(self.tabDataset, text="Dataset")
        self.notebook.add(self.tabTraining, text="Training")
        self.notebook.add(self.tabExperiment, text="Experiment")
        self.notebook.add(self.tabAnalysis, text="Analysis")
        self.notebook.add(self.tabHelp, text="Help")
        self.notebook.add(self.tabAbout, text="About")

        self.notebook.tab(2, state=tk.DISABLED)
        self.notebook.tab(3, state=tk.DISABLED)
        self.notebook.tab(4, state=tk.DISABLED)
        
        self.notebook.pack(expand=1, side=tk.TOP, fill=tk.BOTH, anchor=tk.N) 

        self.statusFrame =  ttk.Frame(self.root)
        self.status = ttk.Label(self.statusFrame, text="Program ready!", anchor=tk.W)        
        self.status.pack(side=tk.BOTTOM, anchor=tk.SW, padx = self.f_padx)
        self.statusFrame.pack(expand=0, fill=tk.X, anchor=tk.SW)
        
        ## finished loading so destroy splash
        splash.withdraw()
        splash.destroy()

		## show window again
        print ("Main window stared")
        self.root.deiconify()
        tk.mainloop()        
        
        
        
    def center_window(self, width=300, height=200):
        
        # get screen width and height
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
    
        # calculate position x and y coordinates
        x = (screen_width/2) - (width/2)
        y = (screen_height/2) - (height/2)
        self.root.geometry('%dx%d+%d+%d' % (width, height, x, y))
                
    # === Observer design pattern methods

    def isEmitter(self, _key):
        novelty = False
        if self.msgNovelty[_key]:
            novelty = True
            self.msgNovelty[_key] = False
        return novelty
            
                    
    def update(self, _who):
        
        self.msgNovelty[_who] = True
        
        self.mChange = False
        self.dChange = False
        self.eChange = False
        self.tChange = False        
        
        self.mChange = not self.m == self.context['m']
        
        self.dChange = False
        if not (self.d is None or self.context['d'] is None):
            self.dChange = not self.d['name'] == self.context['d']['name']
        else:
            self.dChange = not self.d == self.context['d']
            
        self.eChange = not self.e == self.context['e']        
        self.tChange = not self.t == self.context['t']        
        
        if self.debug:
            print("")
            print("Who : ", _who)
            print("Model change: ", self.mChange)
            print("Dataset change: ", self.dChange)
            print("Traiing change: ", self.tChange)
            print("Experiment change: ", self.eChange)
            
        self.m = self.context['m']
        self.d = self.context['d']        
        self.e = self.context['e']        
        self.t = self.context['t']        
        
        # update the status bar
        statusText = 'Program ready!'
        if self.mChange:
            if not self.m is None:
                statusText = 'Model selected!'
        if self.dChange:
            if not self.d is None:
                statusText = 'Dataset selected!'
        if self.tChange:
            if self.t:
                statusText = 'Training stated !'
        if self.eChange:
            if self.e:
                statusText = 'Experiment stated !'
            
        self.status.config(text=statusText)
        
        self.notify()
        
    def notify(self):
        
        stModeling = tk.NORMAL
        stDataset = tk.NORMAL
        stTraining = tk.NORMAL
        stExperiment = tk.NORMAL
        stAnalysis = tk.NORMAL                            
        stHelp = tk.NORMAL
        stAbout = tk.NORMAL                            
        
        if not self.isEmitter("Modeling"):
            self.modeling.notify()
        if not self.isEmitter("Dataset"):
            self.dataset.notify()
        if not self.isEmitter("Training"):
            self.training.notify()
        if not self.isEmitter("Experiment"):
            self.experiment.notify()
        if not self.isEmitter("Analysis"):
            self.analysis.notify()
        
        # managing tabs activation
        if self.t:
            stModeling = tk.DISABLED
            stDataset = tk.DISABLED
            stExperiment = tk.DISABLED
            stAnalysis = tk.DISABLED
        elif self.e:
            stModeling = tk.DISABLED
            stDataset = tk.DISABLED
            stTraining = tk.DISABLED
            stAnalysis = tk.DISABLED
            stHelp = tk.DISABLED
            stAbout = tk.DISABLED                            
        elif not (self.m is None or self.m['train']):
                stAnalysis = tk.DISABLED
                stExperiment = tk.DISABLED
                if self.d is None:
                    stTraining = tk.DISABLED                    
        elif self.m is None: 
            stTraining = tk.DISABLED
            stAnalysis = tk.DISABLED
            stExperiment = tk.DISABLED

        self.notebook.tab(0, state=stModeling)
        self.notebook.tab(1, state=stDataset)
        self.notebook.tab(2, state=stTraining)
        self.notebook.tab(3, state=stExperiment)
        self.notebook.tab(4, state=stAnalysis)        
        self.notebook.tab(5, state=stHelp)        
        self.notebook.tab(6, state=stAbout)        
            
       
    def doEscape(self, event):
        self.doClose()
        
    def doClose(self):
                
        if self.msg.doYesNo("Do you wish to quit?"):      
            self.root.withdraw()                          
            self.root.destroy()
                    
            

        
        
