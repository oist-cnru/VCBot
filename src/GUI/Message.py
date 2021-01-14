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
import tkinter.messagebox 
from tkinter import filedialog

class Message():
    
    def __init__(self, _console=None, _bar=None):
        self.console = _console
        self.bar = _bar
        self.warningTitle = "Warning"
        self.infoTitle = "Information"
        
    def logConsole(self, _msg, _append=True, _scrollDown=True):
        if not (self.console == None):
            if not _append:
                self.console.delete(1.0,tk.END)   
            self.console.insert(tk.END, _msg + '\n')
            if _scrollDown:
                self.console.see(tk.END)
                
    def doWarning(self, _msg):    
        tk.messagebox.showwarning('Warning', _msg)
        self.logConsole(_msg)
        
    def doInfo(self, _msg):
        tk.messagebox.showinfo('Information', _msg)
        self.logConsole(_msg)
        
    def doYesNo(self, _msg):
        answer = tk.messagebox.askquestion('Caution', _msg, icon='warning')
        if not self.console is None:
            self.logConsole('\n{:<8} {}\n{:<8} {}\n'.format('QUESTION', _msg, 'ANSWER', answer))
        return answer == 'yes'        
        
    def doDirSelection(self):
        return filedialog.askdirectory()

