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

import os
import tkinter as tk
from tkinter import ttk
import numpy as np
from GUI.TrainingPlot import TrainingPlot
from GUI.Message import Message

class Modeling():
    
    def __init__(self, _name, _context, _master=None):   
        
        self.name = _name
        self.master = _master        
        self.context = _context
        self.ut = self.context['ut']        
        self.delimiter = self.context['delimiter']
        self.modelConfigDir = self.context['modelconfigdir']
        self.modelDataDir = self.context['modeldatadir']
        self.datasetDir = self.context['datasetdir']
        self.experimentDir = self.context['experimentdir']
        self.consoleBg =  self.context['consoleBg']
        self.consoleFg = self.context['consoleFg']
    
        self.cwp = self.context['cwd']
        f_padx = self.context['f_padx']
        f_pady = self.context['f_pady']
        padx_ = self.context['padx']
        pady_ = self.context['pady']        
        wEntry = self.context['wentry']
        
        self.m = None
        self.d = None
        
        # === BLOCK 1: Model Creation
        
        self.f_t1_controls = ttk.Frame(self.master)
        
        self.f_t1_f1 = ttk.LabelFrame(self.f_t1_controls, relief=tk.SUNKEN, text=" Create a model ")
        
        self.l_t1_f1_M = ttk.Label(self.f_t1_f1, text="Name:")
        self.l_t1_f1_N = ttk.Label(self.f_t1_f1, text="Network type:")
        self.l_t1_f1_D = ttk.Label(self.f_t1_f1, text="D units:")
        self.l_t1_f1_Z = ttk.Label(self.f_t1_f1, text="Z units:")
        self.l_t1_f1_w1 = ttk.Label(self.f_t1_f1, text="Regulation w (t=1):")       
        self.l_t1_f1_w = ttk.Label(self.f_t1_f1, text="Regulation w:")  
        self.l_t1_f1_T = ttk.Label(self.f_t1_f1, text="Time constant:")
        #self.l_t1_f1_dSof = ttk.Label(self.f_t1_f1, text="Dim/sofmax:")
                        
        self.e_t1_f1_M = ttk.Entry(self.f_t1_f1, width=wEntry)
        self.c_t1_f1_N = ttk.Combobox(self.f_t1_f1, values=['PV-RNN','PV-RNN Beta'], width= 20, state="readonly")
        self.e_t1_f1_D = ttk.Entry(self.f_t1_f1, width=wEntry)
        self.e_t1_f1_Z = ttk.Entry(self.f_t1_f1, width=wEntry)
        self.e_t1_f1_w1 = ttk.Entry(self.f_t1_f1, width=wEntry)        
        self.e_t1_f1_w1.configure(state=tk.DISABLED)
        self.e_t1_f1_w = ttk.Entry(self.f_t1_f1, width=wEntry)
        self.e_t1_f1_T = ttk.Entry(self.f_t1_f1, width=wEntry)
        
        self.c_t1_f1_N.bind("<<ComboboxSelected>>", self.doNetworkType)
        #self.e_t1_f1_dSof = ttk.Entry(self.f_t1_f1, width=wEntry)
        #self.e_t1_f1_dSof.insert(tk.END, '10')        
                
        self.f_t1_f1_f1 = ttk.Frame(self.f_t1_f1)
        self.b_t1_f1_clear = ttk.Button(self.f_t1_f1_f1, text="Clear",  command=self.doClear)
        self.b_t1_f1_add = ttk.Button(self.f_t1_f1_f1, text="Add",  command=self.doAdd)        
        
        self.l_t1_f1_M.grid(row=0, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        self.e_t1_f1_M.grid(row=0, column=1, padx=padx_, pady=pady_,  sticky=tk.W)

        self.l_t1_f1_N.grid(row=1, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        self.c_t1_f1_N.grid(row=1, column=1, padx=padx_, pady=pady_,  sticky=tk.W)

        self.l_t1_f1_D.grid(row=2, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        self.e_t1_f1_D.grid(row=2, column=1, padx=padx_, pady=pady_,  sticky=tk.W)
        
        self.l_t1_f1_Z.grid(row=3, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        self.e_t1_f1_Z.grid(row=3, column=1, padx=padx_, pady=pady_,  sticky=tk.W)
               
        self.l_t1_f1_w1.grid(row=4, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        self.e_t1_f1_w1.grid(row=4, column=1, padx=padx_, pady=pady_,  sticky=tk.W)

        self.l_t1_f1_w.grid(row=5, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        self.e_t1_f1_w.grid(row=5, column=1, padx=padx_, pady=pady_,  sticky=tk.W)
        
        self.l_t1_f1_T.grid(row=6, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        self.e_t1_f1_T.grid(row=6, column=1, padx=padx_, pady=pady_,  sticky=tk.W)
        
        #self.l_t1_f1_dSof.grid(row=5, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        #self.e_t1_f1_dSof.grid(row=5, column=1, padx=padx_, pady=pady_,  sticky=tk.W)
        
        self.b_t1_f1_clear.grid(row=0, column=0, padx=padx_, pady=pady_)
        self.b_t1_f1_add.grid(row=0, column=1, padx=padx_, pady=pady_)        
        self.f_t1_f1_f1.grid(columnspan=2)
       
        # === BLOCK 2: Model details

        self.f_t1_f2 = ttk.LabelFrame(self.f_t1_controls, relief=tk.SUNKEN, text=" Select a model")
        self.l_t1_f2_M = ttk.Label(self.f_t1_f2, text="Name:")
        self.l_t1_f2_N = ttk.Label(self.f_t1_f2, text="Network type:")
        self.l_t1_f2_D = ttk.Label(self.f_t1_f2, text="D units:")
        self.l_t1_f2_Z = ttk.Label(self.f_t1_f2, text="Z units:")        
        self.l_t1_f2_w1 = ttk.Label(self.f_t1_f2, text="Regulation w (t=1):")       
        self.l_t1_f2_w = ttk.Label(self.f_t1_f2, text="Regulation w:")        
        self.l_t1_f2_T = ttk.Label(self.f_t1_f2, text="Time constant:")
        #self.l_t1_f2_dSof = ttk.Label(self.f_t1_f2, text="Dim/sofmax:")
        self.c_t1_f2_M = ttk.Combobox(self.f_t1_f2, values=['empty'], state="readonly", postcommand=self.initSelection)
        self.c_t1_f2_M.bind("<<ComboboxSelected>>", self.doComboSelection)
                
        self.l_t1_f2_Nv = ttk.Label(self.f_t1_f2, text="-")
        self.l_t1_f2_Dv = ttk.Label(self.f_t1_f2, text="-")
        self.l_t1_f2_Zv = ttk.Label(self.f_t1_f2, text="-")
        self.l_t1_f2_w1v = ttk.Label(self.f_t1_f2, text="-")        
        self.l_t1_f2_wv = ttk.Label(self.f_t1_f2, text="-")        
        self.l_t1_f2_Tv = ttk.Label(self.f_t1_f2, text="-")
        #self.l_t1_f2_dSofv = ttk.Label(self.f_t1_f2, text="-")
                        
        self.f_t1_f2_f1 = ttk.Frame(self.f_t1_f2)
        self.b_t1_f2_select = ttk.Button(self.f_t1_f2_f1, text="Select",  command=self.doSelect)
        self.b_t1_f2_details = ttk.Button(self.f_t1_f2_f1, text="Details",  command=self.doDetails)                
        self.b_t1_f2_remove = ttk.Button(self.f_t1_f2_f1, text="Remove",  command=self.doRemove)
        
        self.b_t1_f2_select.configure(state=tk.DISABLED)
        self.b_t1_f2_details.configure(state=tk.DISABLED)
        self.b_t1_f2_remove.configure(state=tk.DISABLED)
        
        self.l_t1_f2_M.grid(row=0, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        self.c_t1_f2_M.grid(row=0, column=1, padx=padx_, pady=pady_,  sticky=tk.W)

        self.l_t1_f2_N.grid(row=1, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        self.l_t1_f2_Nv.grid(row=1, column=1, padx=padx_, pady=pady_,  sticky=tk.W)
        
        self.l_t1_f2_D.grid(row=2, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        self.l_t1_f2_Dv.grid(row=2, column=1, padx=padx_, pady=pady_,  sticky=tk.W)
        
        self.l_t1_f2_Z.grid(row=3, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        self.l_t1_f2_Zv.grid(row=3, column=1, padx=padx_, pady=pady_,  sticky=tk.W)
        
        self.l_t1_f2_w1.grid(row=4, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        self.l_t1_f2_w1v.grid(row=4, column=1, padx=padx_, pady=pady_,  sticky=tk.W)

        self.l_t1_f2_w.grid(row=5, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        self.l_t1_f2_wv.grid(row=5, column=1, padx=padx_, pady=pady_,  sticky=tk.W)
                
        self.l_t1_f2_T.grid(row=6, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        self.l_t1_f2_Tv.grid(row=6, column=1, padx=padx_, pady=pady_,  sticky=tk.W)
        
        #self.l_t1_f2_dSof.grid(row=5, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        #self.l_t1_f2_dSofv.grid(row=5, column=1, padx=padx_, pady=pady_,  sticky=tk.W)        

        self.b_t1_f2_select.grid(row=0, column=0, padx=padx_, pady=pady_,  sticky=tk.NW)
        self.b_t1_f2_details.grid(row=0, column=1, padx=padx_, pady=pady_,  sticky=tk.NW)        
        self.b_t1_f2_remove.grid(row=0, column=2, padx=padx_, pady=pady_,  sticky=tk.NW)
        self.f_t1_f2_f1.grid(columnspan=2,  sticky=tk.NW)
                       
        # === BLOCK 2: std output
        
        self.f_t1_f3 = ttk.LabelFrame(self.master,relief=tk.SUNKEN, text=" Output ")        
        self.s_t1_f3_bar = ttk.Scrollbar(self.f_t1_f3)
        self.t_t1_f3_val = tk.Text(self.f_t1_f3, bg=self.consoleBg, fg=self.consoleFg)
        
        self.s_t1_f3_bar.pack(side=tk.RIGHT, fill=tk.Y)
        self.t_t1_f3_val.pack(side=tk.TOP, fill=tk.BOTH, expand="yes")
        self.s_t1_f3_bar.config(command=self.t_t1_f3_val.yview)
        self.t_t1_f3_val.config(yscrollcommand=self.s_t1_f3_bar.set)
                
        self.messenger = Message(self.t_t1_f3_val)
        
        self.f_t1_f1.pack(side=tk.LEFT, expand=1, fill=tk.X, padx=0, pady=f_pady, anchor=tk.W)
        self.f_t1_f2.pack(side=tk.LEFT, expand=1, fill=tk.X, padx=0, pady=f_pady, anchor=tk.W)
        self.f_t1_controls.pack(side=tk.TOP, expand=0, fill=tk.X, padx=f_padx, pady=f_pady, anchor=tk.N)
        self.f_t1_f3.pack(side=tk.TOP, expand=1, fill=tk.BOTH, padx=f_padx, pady=f_pady, anchor=tk.N)
        
    
    def doNetworkType(self, event):
        
        st = tk.DISABLED
        if (self.c_t1_f1_N.get() == 'PV-RNN Beta'):
            st = tk.NORMAL
            
        self.e_t1_f1_w1.configure(state=st)
        
    def updateModelList(self):
        
        model_list =  self.ut.getModelList(self.modelConfigDir)
        
        if not model_list == None and len(model_list) > 0 :                
            self.c_t1_f2_M["values"] = model_list
            self.c_t1_f2_M.configure(state=tk.NORMAL)
            
        self.c_t1_f2_M.set('')
 
    def initSelection(self): 
                
        self.updateModelList()
                       
    def updateSelectionFields(self, m):
                
        text_Dv = '-'
        text_Nv = '-'
        text_Zv = '-'
        text_w1v = '-'
        text_wv = '-'
        #text_dSofv = '-'
        text_Tv = '-'
        stDetails = tk.DISABLED        
        stRemove = tk.DISABLED
            
        if not m is None:     
            
            text_Dv = m['d']
            text_Nv = m['network']
            text_Zv = m['z']
            text_w1v = m['w1']
            text_wv = m['w']
            
            #text_dSofv = m['dsoft']
            text_Tv = m['t']
            stRemove = tk.NORMAL
            
            #self.l_t1_f2_Dv.config(text=m['d'])
            #self.l_t1_f2_Zv.config(text=m['z'])            
            #text_w1v
            #self.l_t1_f2_wv.config(text=m['w'])
            ##self.l_t1_f2_dSofv.config(text=m['dsoft'])
            #self.l_t1_f2_Tv.config(text=m['t'])
            self.b_t1_f2_select.configure(state=tk.NORMAL)
                                       
            if not self.getTrainingData(m) is None:
                stDetails = tk.NORMAL
                
            
        self.l_t1_f2_Dv.config(text=text_Dv)
        self.l_t1_f2_Nv.config(text=text_Nv)
        self.l_t1_f2_Zv.config(text=text_Zv)
        self.l_t1_f2_w1v.config(text=text_w1v)            
        self.l_t1_f2_wv.config(text=text_wv)
        #self.l_t1_f2_dSofv.config(text=text_dSofv)
        self.l_t1_f2_Tv.config(text=text_Tv)        
        self.b_t1_f2_details.configure(state=stDetails)
        self.b_t1_f2_remove.configure(state=stRemove) 
        
    def getTrainingData(self, m):
        
        prevTrain = None
        
        if not m is None:
            mPath = m['modelpath']                
            if len(mPath) > 0:
                prevTrain = self.ut.parseTrainingData(mPath)
            
        return prevTrain   
    
    def checkDatasetAssociation(self):
        
        if self.m is None:
            return False
        
        if not self.getTrainingData(self.m) is None:  
                      
            if self.ut.parseDataset(self.datasetDir + '/' + self.m['datapath'].split(os.sep)[-1]) is None:
                
                # remove model-dataset association since the dataset has been deleted
                m_old = self.m
                mName = self.m['name']
                self.removeModel(mName)
                self.m = self.ut.modelFactory()
                self.m['modelpath'] = m_old['modelpath']
                self.m['name'] = m_old['name']
                self.m['network'] = m_old['network']
                self.m['d'] = m_old['d']
                self.m['z'] = m_old['z']
                self.m['w1'] = m_old['w1']
                self.m['w'] = m_old['w']
                self.m['t'] = m_old['t']
                
                if not self.addModel(mName, self.m):
                    return False
                
                mPathString = self.ut.modelPathString(self.modelConfigDir, self.m['name'])
                
                if not self.ut.saveModel(mPathString, self.m):
                    return False
                
                self.m = self.getModelData(mName)
                
        return True

    def getModelData(self,mName):
                
        if len(mName) > 0:
            confPathString = self.ut.modelPathString(self.modelConfigDir, mName)
            m = self.ut.parseModel(confPathString)                                        
            if m == None:             
                 self.messenger.doInfo('The model \'{}\' seems to be corrupted please consider removing it!'.format(mName))
            return m
        
    def doComboSelection(self,event):
        
        mName = self.c_t1_f2_M.get()        
        self.m = self.getModelData(mName)        
        self.updateSelectionFields(self.m)                                                          
        self.c_t1_f2_M.state(['readonly'])
                
        
    def doDetails(self):
                
        if self.m == None:
            return
        
        mName= self.m['name']                
        t = self.getTrainingData(self.m)
        
        if t is None:
            self.messenger.doInfo('The model \'{}\' has no previous training !'.format(mName))                        
        else:                
            TrainingPlot(mName, t, self.context)            
        
    def doSelect(self):        
        
        mName = self.c_t1_f2_M.get()        
        self.m = self.getModelData(mName)        
        if not self.m is None:
            #checking for the existence of the dataset for trained models
            if not self.checkDatasetAssociation():
                self.m = None
                return
            self.messenger.logConsole('The model \'{}\' has been selected!'.format(mName))            
        self.updateObserver()    
        self.b_t1_f2_select.configure(state=tk.DISABLED)
        
    def removeModel(self, _name):

            mPathString = self.ut.modelPathString(self.modelConfigDir, _name)
            mDataPath = self.modelDataDir + '/' + _name
            mExpPath = self.experimentDir + '/' + _name
            
            return (self.ut.removeFile(mPathString) and self.ut.removeDir(mDataPath) and self.ut.removeDir(mExpPath))

    def doRemove(self):        
        
        mName = self.c_t1_f2_M.get()        
        if self.messenger.doYesNo("This operation cannot be undone. Are ou sure about removing the model \'{}\'".format(mName)):
            if self.removeModel(mName): 
                self.messenger.logConsole('The model \'{}\' has been removed!'.format(mName))            
                self.c_t1_f2_M.config(text='')
                self.l_t1_f2_Nv.config(text='-')
                self.l_t1_f2_Dv.config(text='-')
                self.l_t1_f2_Zv.config(text='-')            
                self.l_t1_f2_w1v.config(text='-')
                self.l_t1_f2_wv.config(text='-')
                #self.l_t1_f2_dSofv.config(text='-')
                self.l_t1_f2_Tv.config(text='-')
                self.b_t1_f2_select.configure(state=tk.DISABLED)
                self.b_t1_f2_details.configure(state=tk.DISABLED)
                self.b_t1_f2_remove.configure(state=tk.DISABLED)
                self.updateModelList()                                        
            else:
                self.messenger.doWarning('The model \'{}\' could not be removed!'.format(mName))            
                
        self.m = None
        self.updateObserver()
                
    def doClear(self):             
        
        self.e_t1_f1_M.delete(0, tk.END)
        self.e_t1_f1_D.delete(0, tk.END)
        self.e_t1_f1_Z.delete(0, tk.END)
        self.e_t1_f1_w1.delete(0, tk.END)
        self.e_t1_f1_w.delete(0, tk.END)
        self.e_t1_f1_T.delete(0, tk.END)
        self.c_t1_f1_N.set('')
        #self.e_t1_f1_dSof.delete(0, tk.END)
        
    
    def addModel(self, _name, _m):
        
        modelPathString = self.ut.modelPathString(self.modelConfigDir, _name)
        
        if not self.ut.createDir(self.experimentDir+'/'+_name):
            self.messenger.doWarning("Error the model could not be added!")
            return False
        if not self.ut.createDir(self.modelDataDir+'/'+_name):
            self.messenger.doWarning("Error the model could not be added!")
            return False
        
        return self.ut.saveModel(modelPathString, _m)
        
    def doAdd(self):            
        
        mName = self.e_t1_f1_M.get()                            
        modelPathString = self.ut.modelPathString(self.modelConfigDir, mName)
        network = self.c_t1_f1_N.get()
        self.m = None
        self.updateObserver()
        if mName == "" or not mName.isalnum():
            self.messenger.doInfo("Please provide an alphanumeric name for the model!")
        elif network == "":
            self.messenger.doInfo("Please select a network type!")
        elif self.ut.fileExists(modelPathString):
            self.messenger.doWarning("The name already exists!")
        else:        
            
            list_Z = self.ut.parseString(self.e_t1_f1_Z.get(), 'int', self.delimiter)    
            list_D = self.ut.parseString(self.e_t1_f1_D.get(), 'int', self.delimiter)    
            list_w = self.ut.parseString(self.e_t1_f1_w.get(), 'float', self.delimiter)
            list_w1 = self.ut.parseString(self.e_t1_f1_w1.get(), 'float', self.delimiter)
            list_t = self.ut.parseString(self.e_t1_f1_T.get(), 'int', self.delimiter)    
            network = self.ut.trimString(self.c_t1_f1_N.get()).lower()
            network = self.ut.trimString(network, '-')
            
            
            #list_dSof  = self.ut.parseString(self.e_t1_f1_dSof.get(), 'float', self.delimiter)    
            if network == 'pvrnn':
                list_w1 = list_w
            
            len_list =[len(list_Z), len(list_D), len(list_w1), len(list_w), len(list_t)]
            
            msg = ''
            
            if (np.sum(np.array(len_list)) % 5 == 0) :
    
                m = self.ut.modelFactory()
                                
                m['modelpath'] = self.cwp + os.sep + self.modelDataDir.replace('/',os.sep) + os.sep + mName
                
                zText = ''
                dText = ''
                wText = ''
                w1Text = ''
                tText = ''
                
                for z in self.e_t1_f1_Z.get().split(self.delimiter):
                    zText = zText + z.strip() + self.delimiter   
                for d in self.e_t1_f1_D.get().split(self.delimiter):
                    dText = dText + d.strip() + self.delimiter  
                for w in self.e_t1_f1_w.get().split(self.delimiter):
                    wText = wText + w.strip() + self.delimiter       
                for w1 in self.e_t1_f1_w1.get().split(self.delimiter):
                    w1Text = w1Text + w1.strip() + self.delimiter   
                for t in self.e_t1_f1_T.get().split(self.delimiter):
                    tText = tText + t.strip() + self.delimiter   

                m['d'] = dText[0:-1]
                m['z'] = zText[0:-1]
                m['w'] = wText[0:-1]
                if network == 'pvrnnbeta':
                    m['w1'] = w1Text[0:-1]
                else:
                    m['w1'] = wText[0:-1]
                m['t'] = tText[0:-1]
                m['network'] = network
                
                if not self.addModel(mName, m):
                    self.messenger.doWarning("Error the model could not be added!")
                else:                                      
                    #self.messenger.doInfo("The model was added successfully!")
                    msg = msg + 'The model was added successfully!\n\n'
                    msg = msg + 'Network Configuration:\n\n'
                    
                    msg = msg + 'Model name: {}\n'.format(mName)
                    msg = msg + 'Network: {}\n'.format(network)
                    for l in range(len(list_D), 0, -1):                        
                        msg = msg + '\n----[Layer #{}]----\n\n'.format(l)                        
                        msg = msg + 'D units: {} \n'.format(list_D[l-1])
                        msg = msg + 'Z units: {} \n'.format(list_Z[l-1])     
                        
                        if network == 'pvrnnbeta':
                            msg = msg + 'Regulation w (t=1): {} \n'.format(list_w1[l-1])
                            
                        msg = msg + 'Regulation w: {} \n'.format(list_w[l-1])
                        msg = msg + 'Time const: {}\n'.format(list_t[l-1])                
                    msg = msg + '\n----[ output ]----\n\n'
                    #msg = msg + 'Dim/sofmax: {}\n\n'.format(list_dSof[0])         
                                        
                    m['name'] = mName
                    self.m = m              
                    self.updateModelList()                                        
                    self.updateSelectionFields(self.m)
                    self.c_t1_f2_M.set(mName)
                    
            else:
                self.messenger.doWarning("The model could not be added, please check the output!")
                
                msg = msg + 'Check failed! \n\n'
                msg = msg + 'Please insert the layer values separated by comma.\n'
                msg = msg + 'For example, for a Two-layers PV-RNN framework you could insert something like this:\n\n'
                msg = msg + 'D units: 40,10\n'
                msg = msg + 'Z units: 4,1\n'            
                msg = msg + 'Regulation w: 0.01,0.01\n'
                msg = msg + 'Time const: 2,5\n'
                msg = msg + '\nIn case the network type is PV-RNN Beta, you could additionally set\nRegulation w (t=1): 0.01,0.01\n'
                #msg = msg + 'Dim/sofmax: 10\n'                                        
                
                    
            self.messenger.logConsole(msg)
        
        
    # === Observer design pattern methods
    
    def updateObserver(self):
                
        if not self.m is None:
            self.d = None
            if self.m['train']:
                self.d = self.ut.parseDataset(self.datasetDir + '/' + self.m['datapath'].split(os.sep)[-1])                    
            self.context['d'] = self.d     
                        
        self.context['m'] = self.m                             
        self.context['main'].update(self.name)

    
    def notify(self):
        
        if self.context['t'] or self.context['e']:
            return 

        if not self.context['m'] == self.m:                                    
            self.m = self.context['m']
            self.updateSelectionFields(self.m)        
            self.b_t1_f2_select.configure(state=tk.NORMAL)                                          
        elif not self.m is None:
            stDetails = tk.DISABLED       
            if self.m['train']:
                stDetails = tk.NORMAL
            self.b_t1_f2_details.configure(state=stDetails)
                
                
            
        

        
                
            
        
