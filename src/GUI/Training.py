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
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk
from GUI.Message import Message
from IPython.utils.capture import capture_output
from GUI.TrainingPlot import TrainingPlot
import numpy as np
import ctypes


class ConsoleRedirect(object):
   
    def __init__(self, _text_ctrl):
        
        self.output = _text_ctrl
        self.scrollDown = False
        self.cleared = True
    
    def setScroolDown(self, _flag):
        
        self.scrollDown = _flag
    
    def clear(self):
        
        self.output.delete(1.0, tk.END)
    
    def write(self, string):
        
        self.cleared = False
        self.output.insert(tk.END, string)
        if self.scrollDown:
            self.output.see(tk.END)
            
        
class Training():
    
    def __init__(self,  _name, _context, _master=None):        
        
        self.name = _name
        self.master = _master
        self.context = _context
        self.nrl = self.context['nrl']
        self.ut = self.context['ut']
        self.cwp = self.context['cwd']
        self.datasetDir = self.context['datasetdir']
        self.modelConfigDir = self.context['modelconfigdir']
        self.delimiter = self.context['delimiter']
        self.consoleBg =  self.context['consoleBg']
        self.consoleFg = self.context['consoleFg']    
        self.refreshFactor = 100
        self.greedy = tk.BooleanVar()
        self.retrain = tk.BooleanVar()
        self.scrollDown = tk.BooleanVar()
        
        self.isTraining = False        
        self.actualTraining = 0      
        
        f_padx = self.context['f_padx']
        f_pady = self.context['f_pady']
        padx_ = self.context['padx']
        pady_ = self.context['pady']        
                
        self.inputWidgets = []
        self.stop = False
        
        self.m = None
        self.d = None
        self.prevTrain = None
                
        # === Block 1: Selection panel
        
        self.f_t1_controls = ttk.Frame(self.master)
        
        self.f_t1_f1 = ttk.LabelFrame(self.f_t1_controls, relief=tk.SUNKEN, text=" Selection ")
        self.l_t1_f1_M = ttk.Label(self.f_t1_f1, text="Model:")
        self.l_t1_f1_D = ttk.Label(self.f_t1_f1, text="Dataset:")
                
        self.l_t1_f1_Mv = ttk.Label(self.f_t1_f1, text="               ")
        self.l_t1_f1_Dv = ttk.Label(self.f_t1_f1, text="               ")
        
        self.f_t1_f1_f1 = ttk.Frame(self.f_t1_f1)
        
        self.b_t1_f1_detailM = ttk.Button(self.f_t1_f1_f1, text="Previous training",  command=self.doDetailModel)
        self.b_t1_f1_detailM.grid(row=0, column=0, padx=padx_, pady=pady_)
                
        self.l_t1_f1_M.grid(row=0, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        self.l_t1_f1_Mv.grid(row=0, column=1, padx=padx_, pady=pady_,  sticky=tk.W)
        
        self.l_t1_f1_D.grid(row=1, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        self.l_t1_f1_Dv.grid(row=1, column=1, padx=padx_, pady=pady_,  sticky=tk.W)
        
        self.f_t1_f1_f1.grid(columnspan=2)
        
        # === Block 2: training parameters

        self.f_t1_f2 = ttk.LabelFrame(self.f_t1_controls, relief=tk.SUNKEN, text=" Parameters ")

        self.l_t1_f2_ADAM_A = ttk.Label(self.f_t1_f2, text="Adam: \u03B1")
        self.l_t1_f2_ADAM_B1 = ttk.Label(self.f_t1_f2, text="\u03B2\u2081:")
        self.l_t1_f2_ADAM_B2 = ttk.Label(self.f_t1_f2, text="\u03B2\u2082:")        
        self.l_t1_f2_Epochs = ttk.Label(self.f_t1_f2, text="Epochs:")        
        self.l_t1_f2_retrain = ttk.Label(self.f_t1_f2, text="Retrain:")
        self.l_t1_f2_greedy = ttk.Label(self.f_t1_f2, text="Greedy:")
                
        self.e_t1_f2_ADAM_A = ttk.Entry(self.f_t1_f2, width=10)
        self.e_t1_f2_ADAM_B1 = ttk.Entry(self.f_t1_f2, width=10)
        self.e_t1_f2_ADAM_B2 = ttk.Entry(self.f_t1_f2, width=10)                
        self.e_t1_f2_Epochs = ttk.Entry(self.f_t1_f2, width=10)                                  
        self.c_t1_f2_retrain = ttk.Checkbutton(self.f_t1_f2, variable=self.retrain, onvalue=True, offvalue=False)
        self.c_t1_f2_greedy = ttk.Checkbutton(self.f_t1_f2, variable=self.greedy, onvalue=True, offvalue=False)
                
        self.f_t1_f2_f1 = ttk.Frame(self.f_t1_f2)
        self.b_t1_f2_train = ttk.Button(self.f_t1_f2_f1, text="Train",  command=self.doTrain)
        self.b_t1_f2_stop = ttk.Button(self.f_t1_f2_f1, text="Stop",  command=self.doStop)
        
        self.inputWidgets.append(self.e_t1_f2_ADAM_A)
        self.inputWidgets.append(self.e_t1_f2_ADAM_B1)
        self.inputWidgets.append(self.e_t1_f2_ADAM_B2)
        self.inputWidgets.append(self.e_t1_f2_Epochs)        
        self.inputWidgets.append(self.c_t1_f2_greedy)
        self.inputWidgets.append(self.c_t1_f2_retrain)        

        self.l_t1_f2_ADAM_A.grid(row=0, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        self.e_t1_f2_ADAM_A.grid(row=0, column=1, padx=padx_, pady=pady_,  sticky=tk.W)        
        self.l_t1_f2_ADAM_B1.grid(row=0, column=2, padx=padx_, pady=pady_,  sticky=tk.E)
        self.e_t1_f2_ADAM_B1.grid(row=0, column=3, padx=padx_, pady=pady_,  sticky=tk.W)
        self.l_t1_f2_ADAM_B2.grid(row=0, column=4, padx=padx_, pady=pady_,  sticky=tk.E)
        self.e_t1_f2_ADAM_B2.grid(row=0, column=5, padx=padx_, pady=pady_,  sticky=tk.W)
        
        self.l_t1_f2_Epochs.grid(row=1, column=0, padx=padx_, pady=pady_,  sticky=tk.E)
        self.e_t1_f2_Epochs.grid(row=1, column=1, padx=padx_, pady=pady_,  sticky=tk.W)        
        self.l_t1_f2_retrain.grid(row=1, column=2, padx=padx_, pady=pady_,  sticky=tk.E)
        self.c_t1_f2_retrain.grid(row=1, column=3, padx=padx_, pady=pady_,  sticky=tk.W)
        self.l_t1_f2_greedy.grid(row=1, column=4, padx=padx_, pady=pady_,  sticky=tk.E)
        self.c_t1_f2_greedy.grid(row=1, column=5, padx=padx_, pady=pady_,  sticky=tk.W)
                                
        self.b_t1_f2_train.grid(row=1, column=0, padx=padx_, pady=pady_)
        self.b_t1_f2_stop.grid(row=1, column=1, padx=padx_, pady=pady_)
        self.f_t1_f2_f1.grid(columnspan=6)
        
        self.b_t1_f2_stop.configure(state=tk.DISABLED)
        
        # === Block 3: std output

        self.f_t1_output = ttk.Frame(self.master)
        self.f_t1_f3 = ttk.LabelFrame(self.f_t1_output, relief=tk.SUNKEN, text=" Output ")
        self.s_t1_f3_bar = ttk.Scrollbar(self.f_t1_f3)
        
        self.t_t1_f3_val = tk.Text(self.f_t1_f3, bg=self.consoleBg, fg=self.consoleFg)
        self.s_t1_f3_bar.pack(side=tk.RIGHT, fill=tk.Y)
        self.t_t1_f3_val.pack(side=tk.TOP, fill=tk.BOTH, expand="yes")
        self.t_t1_f3_val.config(yscrollcommand=self.s_t1_f3_bar.set)

        self.messenger = Message(self.t_t1_f3_val)
        self.redirect = ConsoleRedirect(self.t_t1_f3_val)
        
        # === WIDGETS FOR Design TAB 3 (training) - FRAME 4 (Output controls)
        
        self.f_t1_f4 = ttk.Frame(self.f_t1_output)
        self.c_t1_f4_scrolOut = ttk.Checkbutton(self.f_t1_f4, variable=self.scrollDown, text='Scroll down', onvalue=True, offvalue=False, command=self.doScrollDown)
        self.c_t1_f4_scrolOut.state(['selected'])
        self.scrollDown.set(True)
        self.b_t1_f4_clearOut = ttk.Button(self.f_t1_f4, text="Clear",  command=self.doOutputClear)
        self.c_t1_f4_scrolOut.pack(side=tk.LEFT, expand=0, fill=tk.X, anchor=tk.N)
        self.b_t1_f4_clearOut.pack(side=tk.LEFT, expand=0, fill=tk.X, anchor=tk.N)                        
        
        self.f_t1_f1.pack(side=tk.LEFT, expand=1, fill=tk.X, padx=0, pady=f_pady, anchor=tk.W)
        self.f_t1_f2.pack(side=tk.LEFT, expand=1, fill=tk.X, padx=0, pady=f_pady, anchor=tk.W)
        
        self.f_t1_controls.pack(side=tk.TOP, expand=0, fill=tk.X, padx=f_padx, pady=f_pady, anchor=tk.N)
        
        self.f_t1_f3.pack(side=tk.TOP, expand=1, fill=tk.BOTH, padx=1, pady=f_pady, anchor=tk.N)
        self.f_t1_f4.pack(side=tk.TOP, expand=0, fill=tk.X, padx=0, pady=f_pady, anchor=tk.N)
        
        self.f_t1_output.pack(side=tk.TOP, expand=1, fill=tk.BOTH, padx=f_padx, pady=f_pady, anchor=tk.N) 
        
        self.f_t1_f4.focus_set()
                
    def doDetailModel(self):
        
        mName = self.m['name']        
        if self.prevTrain is None:
            self.messenger.doInfo('The model \'{}\' has no previous training !'.format(mName))            
        else:                
            TrainingPlot(mName, self.prevTrain,self.context)
    
    def updateControls(self):        
        
        if not self.m is None:
            self.e_t1_f2_Epochs.delete(0, tk.END)        
            self.e_t1_f2_Epochs.insert(tk.END, self.m['epochs'])
            if self.prevTrain is None:
                mPath = self.m['modelpath']                
                if len(mPath) > 0:
                    self.prevTrain = self.ut.parseTrainingData(mPath)

        state_ = tk.NORMAL
        stStop_ = tk.DISABLED        
        stTrain_ = tk.NORMAL
        stDetais_ = tk.NORMAL
                
        if self.isTraining:
            stStop_ = tk.NORMAL
            stTrain_ = tk.DISABLED            
            state_ =  tk.DISABLED            
            
        if self.prevTrain is None:            
            stDetais_ = tk.DISABLED                                                       

        for widget in self.inputWidgets:
            widget.configure(state=state_)

        self.b_t1_f2_train.configure(state=stTrain_)                                    
        self.b_t1_f2_stop.configure(state=stStop_)                                    
        self.b_t1_f1_detailM.config(state=stDetais_)    
        self.redirect.setScroolDown(self.scrollDown.get())                     
        
    def doOutputClear(self):
        
        self.redirect.clear()    
        
    def doScrollDown(self):        
        
        self.redirect.setScroolDown(self.scrollDown.get())    
    
    def doStop(self):
        
        self.b_t1_f2_stop.configure(state=tk.DISABLED)
        self.stop = True        
        
    def validateForm(self):

        if len(self.d['data']) == 0:
            self.messenger.doWarning("The selected dataset is empty, please proceed to record some primitives in the tab Dataset")
            return False
        
        self.epochs = 0
        ADAM_A = self.ut.parseString(self.e_t1_f2_ADAM_A.get(), 'float', self.delimiter)    
        ADAM_B1 = self.ut.parseString(self.e_t1_f2_ADAM_B1.get(), 'float', self.delimiter)    
        ADAM_B2 = self.ut.parseString(self.e_t1_f2_ADAM_B2.get(), 'float', self.delimiter)    
        epochs = self.ut.parseString(self.e_t1_f2_Epochs.get(), 'int', self.delimiter)    
        len_list =[len(ADAM_A), len(ADAM_B1), len(ADAM_B2), len(epochs)]        
        if not (np.sum(np.array(len_list)) % 4 == 0) :
            self.messenger.doWarning("Please provide real numbers for the Adam optimization parameters and an integer number for the training epochs!")
            return False
                        
        if not self.retrain.get() and not self.prevTrain is None: 
            if not self.messenger.doYesNo("This operation cannot be undone. Previous training will be lost, do you want to proceed?'"):
                return False
            self.epochs = epochs[0]
        else:
            mEpochs = 0
            if len(self.m['epochs']) > 0:                
                mEpochs = int(self.m['epochs'])                            
            if mEpochs >= epochs[0]:                
                self.messenger.doWarning("The model has been already trained for {} epochs! Please set a higher value for the desired total number of epochs".format(mEpochs))
                return False
            self.epochs = epochs[0] - mEpochs   
                      
        absDatasetDir = self.cwp + os.sep + self.datasetDir.replace('/',os.sep) + os.sep + self.d['name']        
        
        self.m['datapath'] = absDatasetDir
        self.m['alpha'] = '{}'.format(ADAM_A[0])
        self.m['beta1'] = '{}'.format(ADAM_B1[0])
        self.m['beta2'] = '{}'.format(ADAM_B2[0])
        self.m['epochs'] = '{}'.format(epochs[0])        
        self.m['retrain'] = '{}'.format(self.retrain.get()).lower()
        self.m['greedy'] =  '{}'.format(self.greedy.get()).lower()
        nSamples = ''
        
            
        for d in self.d['data']:
            nSamples = nSamples + '1' + self.delimiter
        self.m['nsamples'] = nSamples[0:-1]
        
        mName = self.m['name']
        self.modelConfigPath = self.ut.modelPathString(self.modelConfigDir, mName)
          
            
        return True
            
    def updateModel(self):
        
        if not self.ut.saveModel(self.modelConfigPath, self.m, False):
            self.messenger.doWarning("Error, training could not be set up!")
            return False
        
    def enableTraining(self, _enable):
        
        self.isTraining = _enable
        if _enable:                        
            self.t_t1_f3_val.configure(cursor="watch")                
        else:
            self.t_t1_f3_val.configure(cursor="")                          
            
        self.updateModel()                          
        self.updateControls()               
        self.updateObserver()                 
            
        
    def doTrain(self):                
                        
        def starter():            
            
            sys.stdout = self.redirect                 
            print("\nTraining started!\n")
            self.nrl.t_init(False)
            train_buffer = np.zeros((7,), dtype=float);
            trainOut = (ctypes.c_float * 7)(*train_buffer)
            e_sum = 0
            epochs_ = self.epochs
            nTimes = int(epochs_ / self.refreshFactor)
            if epochs_ % self.refreshFactor > 0:
                nTimes = nTimes + 1    
            
            message = ''
            #message = message + '################################################################################################################\n'
            message = message + '-------------------\n'
            message = message + 'HEADER DESCRIPTION:\n'
            message = message + '-------------------\n'
            message = message + 'Epoch:         Number of training epochs modulus 100\n'
            message = message + 'RE(posterior): Reconstruction error from the posterior distribution\n'
            message = message + 'RE(prior):     Reconstruction error from the prior distribution\n'            
            message = message + 'Regulation:    Kullback-Leibler divergence between the prior and posterior distributions\n'
            message = message + 'N-ELBO:        Negative Variational Evidence Lower Bound\n'
            message = message + 'Saved:         Indicates whether the parameters are saved. It varies only if greedy training was selected\n\n'
            #message = message + '###############################################################################################################\n\n'
            message = message + '---------------------------------------------------------------------------------------------------------\n'            
            message = message + '{:<16}{:<16}{:<16}{:<16}{:<16}{:<16}{:<16}\n'.format('Epoch','Time(ms)','RE(posterior)','RE(prior)','Regulation','N-ELBO','Saved')            
            message = message + '---------------------------------------------------------------------------------------------------------'
            print(message)            
            
            trained = False
            for e in range(nTimes):                
                if self.stop == True:
                    print("\nStopped!")
                    break;
                nIt = self.refreshFactor
                if e == nTimes - 1:
                    nIt = min(nIt, epochs_-e_sum)                    
                self.nrl.t_loop(trainOut, nIt)
                e_sum = e_sum + self.refreshFactor
                tl = np.frombuffer(trainOut, np.float32).tolist()                                
                saved = 'No'
                if bool(int(tl[6]))> 0:
                    saved = 'Yes'
                    
                message = '{:<16}{:<16,.4f}{:<16,.4f}{:<16,.4f}{:<16,.4f}{:<16,.4f}{:<16}'.format(int(tl[0]),tl[1], tl[2], tl[3], tl[4], tl[5], saved)
                nT = int(tl[0])
                if nT > 0:
                    trained = True
                self.m['epochs'] = '{}'.format(nT)
                print(message)            
            
            if trained:
                self.m['dsname'] = self.d['name']
                self.m['train'] = trained
                
            self.nrl.t_end()
            self.enableTraining(False)
            print("\nTraining end!")            
            sys.stdout = sys.__stdout__
            
        #Updating the model training parameters        
        if not self.validateForm():
            return        
            
        if self.epochs > 0:
            self.enableTraining(True)       
            self.nrl.newModel(self.modelConfigPath.encode('ascii'))                
    
            ## Commence la Boucle de Tkinter, pour lire le stdout
            thread = threading.Thread(target=starter,args=[])
            thread.start()    
            

    # === Observer design pattern methods
    
    def updateObserver(self):
        
        self.context['t'] = self.isTraining
        self.context['main'].update(self.name)
        
        
    def notify(self):        
        
        if self.context['t'] or self.context['e']:
            return 

        self.m = self.context['m']
        self.d = self.context['d']            
            
        self.prevTrain = None
        self.actualTraining
        
        if not self.m is None:                
            
            if self.m['train']:                    
                mPath= self.m['modelpath']          
                self.prevTrain = self.ut.parseTrainingData(mPath)
            
            dsName = ''                                                
            if not self.d is None:                                        
                dsName = self.d['name']                    
            elif not self.prevTrain is None:                        
                self.d = self.ut.parseDataset(self.datasetDir + '/' + self.m['datapath'].split(os.sep)[-1])                    
                dsName = self.d['name']                        
            
                
            if len(dsName) > 0:
                                    
                epochs = self.m['epochs']             
                if len(epochs) > 0:
                    self.actualTraining = self.ut.parseString(epochs, 'int', self.delimiter)[0]                                                
            
                self.l_t1_f1_Mv.config(text=self.m['name'])
                self.l_t1_f1_Dv.config(text=dsName)
                                        
                self.e_t1_f2_ADAM_A.delete(0, tk.END)
                self.e_t1_f2_ADAM_B1.delete(0, tk.END)
                self.e_t1_f2_ADAM_B2.delete(0, tk.END)
                self.e_t1_f2_Epochs.delete(0, tk.END)        
                    
                self.e_t1_f2_ADAM_A.insert(tk.END, self.m['alpha'])
                self.e_t1_f2_ADAM_B1.insert(tk.END, self.m['beta1'])
                self.e_t1_f2_ADAM_B2.insert(tk.END, self.m['beta2'])
                self.e_t1_f2_Epochs.insert(tk.END, self.m['epochs'])
            
                if (self.m['retrain'] == 'true'):
                    self.c_t1_f2_retrain.state(['selected'])
                else:
                    self.c_t1_f2_retrain.state(['!selected'])
                if (self.m['greedy'] == 'true'):    
                    self.c_t1_f2_greedy.state(['selected'])
                else:
                    self.c_t1_f2_greedy.state(['!selected'])                            
                        
                self.updateControls()
