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
from PIL import ImageTk, Image
from sys import platform as _platform

class Help():
    
    class ScrollableImage(tk.Canvas):
        def __init__(self, master=None, **kw):
            self.image = kw.pop('image', None)
            super(Help.ScrollableImage, self).__init__(master=master, **kw)
            self['highlightthickness'] = 0
            self.propagate(0)  # wont let the scrollbars rule the size of Canvas
            self.create_image(0,0, anchor='nw', image=self.image)
            self.v_scroll = ttk.Scrollbar(self, orient='vertical')            
            self.v_scroll.pack(side='right', fill='y')            
            self.config(yscrollcommand=self.v_scroll.set)

            # Set canvas view to the scrollbars
            self.v_scroll.config(command=self.yview)

            # Assign the region to be scrolled 
            self.config(scrollregion=self.bbox('all'))
    
            self.focus_set()
            
            # with Windows OS
            self.bind("<MouseWheel>", self.mouse_scroll)
            
            # with Linux OS
            self.bind("<Button-4>", self.mouse_scroll)
            self.bind("<Button-5>", self.mouse_scroll)
            
        def mouse_scroll(self, _evt):            
            
            if _evt.num == 5 or _evt.delta == -120:
                if _platform == "darwin": # MAC OS X
                    self.yview_scroll(-1, 'units') 
                else:
                    self.yview_scroll(1, 'units') 
                    
            if _evt.num == 4 or _evt.delta == 120:
                if _platform == "darwin": # MAC OS X
                    self.yview_scroll(1, 'units') 
                else:
                    self.yview_scroll(-1, 'units') 
                

        
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
        self.documentDir = self.context['documentdir']      
    
        pageW = 835
        pageH = 1080
    
        # Create a Viewer frame
        self.frame = ttk.Frame(self.master)        
                        
        self.img = Image.open('data/document/guide.png')
        self.img = ImageTk.PhotoImage(self.img)
        self.scrImg = Help.ScrollableImage(self.master, image=self.img, width=pageW, height=760).pack()    
        #self.frame.pack()        
        
        
