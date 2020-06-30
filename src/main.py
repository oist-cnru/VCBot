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

import os
from tools.utils import Utils
from NRL import NRL
from GUI.Main import Main

def install(context):
    
    dataDir = "data"
    modelconfigdir = dataDir + '/config'
    modeldatadir = dataDir + '/model'
    datasetdir = dataDir + '/dataset'
    experimentdir = dataDir + '/experiment'
    documentdir = dataDir + '/document'
    
    success = True
    try:
        cwp = os.getcwd()
        ut = Utils(cwp)          
        
        checkList = [dataDir,\
                     modelconfigdir,
                     modeldatadir,
                     datasetdir,
                     experimentdir,
                     documentdir]
        
        createDir = []
        for d in checkList:
            if not ut.isDir('./'+d) : 
                createDir.append(d)
                
        for d in createDir:
            ut.createDir(d)
            
        absBasePath = os.path.abspath(".")
    
        for mName in ut.getModelList(modelconfigdir):
            
            mExpDir = experimentdir + os.sep + mName
            if not ut.isDir('./'+ mExpDir) : 
                ut.createDir(mExpDir)

            mFileName = ut.modelPathString(modelconfigdir, mName)
            m = ut.parseModel(mFileName)
            m['modelpath'] = absBasePath + os.sep + modeldatadir.replace('/',os.sep) + os.sep + mName
            m['datapath'] = absBasePath + os.sep + datasetdir.replace('/',os.sep) + os.sep + m['dsname']    
             
            if not ut.saveModel(mFileName, m, False):
                print("Warning: the model {} seem to be corrupted!".format(mName))
                break
             
    except:
        print("Error: The application could not be initialized!")   
        success = False
    
    if success:
                    
        context['cwd'] = cwp
        context['nrl'] = NRL()
        context['ut'] = ut
        context['m'] = None
        context['d'] = None
        context['t'] = None
        context['e'] = None
        context['istraining'] = False
        context['delimiter'] = ut.delimiter
        context['modelconfigdir'] = modelconfigdir        
        context['modeldatadir'] = modeldatadir
        context['datasetdir'] = datasetdir
        context['experimentdir'] = experimentdir      
        context['documentdir'] = documentdir        
    
        ## GUI global parameters
        context['appTitle'] = "VCBot"    
        context['controlBg'] = "#797979"
        context['controlFg'] = "#ffffff"
        context['rootBg'] = "#000000"
        context['consoleBg'] = "#0f0f0f"
        context['consoleFg'] = "#ffffff"                                      
        context['themeid'] = 'black'
        context['f_padx'] = 20    
        context['f_pady'] = 1    
        context['padx'] = 5
        context['pady'] = 1
        context['wentry'] = 20
        context['minH'] = 880
        context['minW'] = 1080
        
    return success
                                
print ("Starting NRL application ... ")            
context = {}
if install(context):
    Main(context)
print ("NRL application ended")

