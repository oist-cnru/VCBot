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
import time
import shutil
import numpy as np
from datetime import datetime
import csv 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

import parse

class Utils():
    
    def __init__(self, _pwd):
        
        self.pwd = _pwd
        self.modelFilePrefix = "properties_"
        self.trainingFileName = 'training.txt'
        self.modelFileSufix = ".d"
        self.datasetFileName = "dataset.d"
        self.experimentFileName = "experiment.d"
        self.primFilePrefix = "primitive_"
        self.primFileSufix = ".csv"
        self.delimiter = ','
        
        self.modelKeys = ['datapath',\
                              'modelpath',
                              'network',
                              'robot',
                              'activejoints',
                              'nsamples',
                              'w1',
                              'w',
                              'd',
                              'z',
                              't',
                              'epochs',
                              'alpha',
                              'beta1',
                              'beta2',
                              'shuffle',
                              'retrain',
                              'greedy',
                              'dsoft',
                              'sigma']

        self.datasetKeys = ['samplingperiod',\
                              'numbertimes',
                              'numberprims']                

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
                              'beta2',
                              'motorcompliance']

        self.experimentExportKeys = ['cur_pos',\
                              'tgt_pos',
                              'hum_pos',
                              'hum_int',
                              'states',                              
                              'elbo']

        
    def pcaFromList(self,_Xs, _nComp=2):
        
        nComp = _nComp
        lColumns = []
        for c in range(1, nComp+1):
            lColumns.append("PC{}".format(c))        
        
        VZ = np.vstack(_Xs)   
        
        AllHidden_std = StandardScaler().fit_transform(VZ)
        pca = PCA(n_components=nComp)
        pcaAnalysis = pca.fit_transform(AllHidden_std)
        principalDf = pd.DataFrame(data = pcaAnalysis, columns = lColumns)
        pcs = principalDf.loc[:,lColumns];
        pcsArrayZ = np.array(pcs);
            
        return pcsArrayZ


    def pcaFromDatabase(self,_Xs, _XdbPath, _nComp=2):
        
        sizeBuffer = []
        v_list = []
        for x in _Xs:
            sizeBuffer.append(x.shape)
            v_list.append(x)        
        
        for p in _XdbPath:                
            v = np.load(p)
            sizeBuffer.append(v.shape)
            v_list.append(v);
                            
        nComp = _nComp
        lColumns = []
        for c in range(1, nComp+1):
            lColumns.append("PC{}".format(c))        
        
        VZ = np.vstack(v_list)   
        
        AllHidden_std = StandardScaler().fit_transform(VZ)
        pca = PCA(n_components=nComp)
        pcaAnalysis = pca.fit_transform(AllHidden_std)
        principalDf = pd.DataFrame(data = pcaAnalysis, columns = lColumns)
        pcs = principalDf.loc[:,lColumns];
        pcsArrayZ = np.array(pcs);
        
        X = []
        idx = 0
        for i in range(len(_Xs)):
            X.append(pcsArrayZ[idx:sizeBuffer[i][0],:])
            idx += sizeBuffer[i][0]
    
        return X, sizeBuffer


    def parseTrainingData(self,_fname):
      
        fname = _fname +  '/' + self.trainingFileName        
        if not self.fileExists(fname):
            return None
        
        list_train_data = []    
        try:
            format_string = 'Epoch [{}] - Time [{}ms] - RE_Q [{}] - RE_P [{}] - Regulation [{}] - loss [{}]'
            with open(fname) as f:             
                for line in f:                 
                    parsed = parse.parse(format_string, line)                
                    if not parsed == None:
                        l_par = []
                        for i in range (6):
                            l_par.append(float(parsed[i]))                                        
                        list_train_data.append(np.array(l_par))
        except IOError:
            print("IOError file \'{}\'".format(_fname))        
            
        if len(list_train_data) > 0 :
            return np.vstack(list_train_data)
    
        return None
    
        
    def trimString(self, _str, _sep=' '):
        
        str_strip = _str.strip().split(_sep)
        str_rem =  ''
        for s in str_strip:
            str_rem = str_rem + s
        return str_rem
            
            
    def getModelList(self,_dir):
    
        mList = []    
        try:
            model_list = os.listdir(_dir)            
            format_string = 'properties_{}.d'        
            if len(model_list) > 0:            
                for s in model_list:
                    parsed = parse.parse(format_string, s)                
                    if not parsed == None:
                        mList.append(parsed[0])                
        except IOError:
            print("IOError dir \'{}\'".format(_dir))        
            return None
            
        return mList
    
    def parseModel(self,_fname):        
        m = self.modelFactory()    
        li = 0 
        
        fullName = self.pwd + '/' + _fname
        if not self.fileExists(fullName.replace('/',os.sep)):
            print('The model \'{}\' no longer exists !'.format(_fname.split('/')[-1]))
            return None
        
        try:        
            with open(_fname) as f:             
                for line in f:                    
                    if '#' in line or len(line) <= 1:
                        continue                    
                    k = self.modelKeys[li]
                    i = len(k)+1                           
                    if (line[:i]) == k + '=': 
                        m[k] = line[i:-1]
                        li = li + 1            
            if li < len(self.modelKeys):
                m = None
            else:
                m['name'] =  m['modelpath'].split(os.sep)[-1]
                m['dsname'] =  m['datapath'].split(os.sep)[-1]                
                m['nlayers'] =  len(m['d'].split(self.delimiter))  
                prevTrain = self.parseTrainingData(m['modelpath'])
                m['train'] = not prevTrain is None
                              
        except IOError:
            print("IOError file \'{}\'".format(_fname))        
            m = None
        except:
            print("Something went wrong with the file \'{}\'".format(_fname))        
            m = None
            
                
        return m
    
    def getCurrentTimeMS(self):
        
        return int(round(time.time() * 1000))
        
    def modelPathString(self,_bDir, _cName):
        
        return _bDir + '/' + self.modelFilePrefix +  _cName + self.modelFileSufix
        
    def fileExists(self,_fname):
        
        return os.path.isfile(_fname)
    
    def modelFactory(self):
        
        m = {}
        for k in self.modelKeys:
            m[k] = ''
        # default initialization
        m['activejoints'] = '1' + self.delimiter+ '1' +  self.delimiter + '0'                
        m['dsoft'] = '10'
        m['sigma'] = '0.2'    
        m['beta1'] = '0.9'
        m['beta2'] = '0.999'
        m['alpha'] = '0.001'
        m['retrain'] = 'false'
        m['greedy'] = 'false'
        m['name'] = ''        
        m['dsname'] = ''
        m['train'] = False      
        m['robot'] = 'cartesian'
        m['shuffle'] = 'false'
                
        return m    
    
    def saveModel(self,_fname, _m, _new=True):
        
        data = ''         
        mPath= _m['modelpath']        
        if _new:
            _m['datapath'] = self.pwd + os.sep + _m['datapath']                    
        
        for k in self.modelKeys:
            data = data + k + '=' + _m[k] + os.linesep
                              
        try:        
            f = open(_fname,"w") 
            f.write(data)
            f.close()            
            if not self.isDir(mPath):
                self.createDir(mPath)
            return True
        except IOError:
            print("IOError file \'{}\'".format(_fname))            
        return False

    def datasetFactory(self):
        
        d = {}
        for k in self.datasetKeys:
            d[k] = ''
        # default initialization
        d['name'] = ''                                
        d['data'] = []
        d['samplingpriod'] = '100'                                
        d['numbertimes'] = '0'                                
        d['numberprims'] = '0'                                
                
        return d    
        
    def getDirList(self,_dir):
        
        dList = []    
        try:
            dList = os.listdir(_dir+"/.")                        
        except IOError:
            print("IOError dir \'{}\'".format(_dir))        
            return None
            
        return dList

    def parseDataset(self,_fname):     
        
        d = self.datasetFactory()    
        li = 0 
        
        fname = _fname + '/' + self.datasetFileName
        fullName = self.pwd + '/' + fname 
        
        if not self.fileExists(fullName.replace('/',os.sep)):            
            print('The dataset \'{}\' no longer exists !'.format(_fname.split('/')[-1]))
            return None
            
        try:        
            with open(fname) as f:             
                for line in f:                    
                    if '#' in line or len(line) <= 1:
                        continue                    
                    k = self.datasetKeys[li]
                    i = len(k)+1                           
                    if (line[:i]) == k + '=': 
                        d[k] = line[i:-1]
                        li = li + 1            
            if li < len(self.datasetKeys):
                d = None
            else:                
                d['name'] =  _fname.split('/')[-1]                
                pList = []
                for i in range(int(d['numberprims'])):
                    pfname = '{}/{}{}_0{}'.format(_fname, self.primFilePrefix, i, self.primFileSufix)
                    vDat = self.readData(pfname, _delimiter=self.delimiter)
                    pList.append(vDat)
                d['data'] = pList
                                        
        except IOError:
            print("IOError file \'{}\'".format(fname))        
            d = None
        except:
            print("Something went wrong with the file \'{}\'".format(fname))        
            d = None
            
        return d


    def datasetPathString(self,_bDir, _cName):
        
        return _bDir + '/' + self.datasetFilePrefix +  _cName + self.datasetFileSufix

        
    def saveDataset(self,_dir, _d):
        
        flag = True
        dsName = _d['name']
        dsDir = _dir + '/' + dsName
        if not self.isDir('./'+dsDir):
            flag = self.createDir(dsDir)        
        else:
            #clear previous files
            for f in os.listdir(dsDir):
                self.removeFile(f)
                
        data = ''         
        
        for k in self.datasetKeys:
            data = data + k + '=' + _d[k] + os.linesep
        try:        
            fName = dsDir + '/' + self.datasetFileName
            f = open(fName,"w") 
            f.write(data)
            f.close()            
            dList = _d['data']                 
            
            for i in range(len(dList)):        
                pName = '{}/{}{}_0{}'.format(dsDir, self.primFilePrefix, i, self.primFileSufix)                
                self.saveData(pName, dList[i])
        except IOError:
            self.removeDir(dsDir)
            print("IOError file \'{}\'".format(dsName))            
            flag = False
        
        return flag
    
    def saveExperiment(self,_dir, _e):
        
        flag = True
        exDir = ''
        try:        
            now = datetime.now()        
            dt_string = now.strftime("date[%Y_%m_%d]_time[%H_%M_%S]")            
            exDir = _dir + '/' + dt_string
            if self.createDir(exDir):
                data = ''         
                _e['datetime']=dt_string        
                for k in self.experimentKeys:
                    data = data + k + '=' + _e[k] + os.linesep
                
                fName = exDir + '/' + self.experimentFileName
                f = open(fName,"w") 
                f.write(data)
                f.close()            
                
                np.save(exDir + '/cur_pos.npy', _e['cur_pos'])
                np.save(exDir + '/tgt_pos.npy', _e['tgt_pos'])
                np.save(exDir + '/hum_pos.npy', _e['hum_pos'])
                np.save(exDir + '/hum_int.npy', _e['hum_int'])
                np.save(exDir + '/states.npy', _e['states'])
                np.save(exDir + '/elbo.npy', _e['elbo'])
            else:
                flag = False
        except IOError:
            self.removeDir(exDir)
            print("IOError file \'{}\'".format(exDir))            
            flag = False
    
        return flag

    def parseExperiment(self,_path):
        
        e_ = {}
        li = 0 
        fname = _path + '/' + self.experimentFileName
        #print(fname)
        try:        
            with open(fname) as f:             
                for line in f:                                        
                    if '#' in line or len(line) <= 1:
                        continue                    
                    k = self.experimentKeys[li]
                    i = len(k)+1                           
                    if (line[:i]) == k + '=': 
                        e_[k] = line[i:-1]
                        li = li + 1            
            if li < len(self.experimentKeys):
                e_ = None
            else:                                
                e_['cur_pos'] = np.load(_path + '/cur_pos.npy')
                e_['tgt_pos'] = np.load(_path  + '/tgt_pos.npy')
                e_['hum_pos'] = np.load(_path + '/hum_pos.npy')
                e_['hum_int'] = np.load(_path + '/hum_int.npy')
                e_['states'] = np.load(_path  + '/states.npy')
                e_['elbo'] = np.load(_path  + '/elbo.npy')                
        except IOError:
            print("IOError dir \'{}\'".format(_path))            
            return None        
        return e_

    def exportExperimentCsv(self,_expPath, _savePath):
                
        if not self.isDir(_savePath):
            return False
        
        e_ = self.parseExperiment(_expPath)
        
        if not e_ is None: 
            for k in self.experimentExportKeys:                
                if not self.saveData(_savePath + '/{}.csv'.format(k), e_[k]):
                    return False
        else:
            return False
        return True
        
    def datasetExists(self, _dir, _dname):   
        
        return self.isDir('./'+ _dir + '/' + _dname)
        
            
    def removeFile(self,_fname):
        
        try:
            if self.fileExists(_fname):
                os.remove(_fname)        
            return True
        except IOError:
            print("IOError file \'{}\'".format(_fname))                        
        return False
               
    def isDir(self,_dir):
        
        return os.path.isdir(_dir)
    
    def createDir(self,_dir):
        
        flag = True
        try:
            if not self.isDir(_dir):
                os.mkdir(_dir)            
        except OSError:
            print ("Dir \'%s\' creation failed" % _dir)
            flag = False        
        return flag


    def removeDir(self,_dir): 
        
        flag = True        
        try:        
            if self.isDir(_dir):
                shutil.rmtree(_dir)            
        except IOError:
            print("IOError dir \'{}\'".format(_dir))                        
            flag = False            
        return flag
    
    
    def removeDataset(self,_cDir):
        
        return self.removeDir(_cDir)
    
    def saveData(self,_fname, _X): 
               
        flag = True
        try:        
            with open(_fname, 'w') as writeFile:
                writer = csv.writer(writeFile, delimiter=self.delimiter)
                for i in range(_X.shape[0]):       
                    line = []
                    x_i = _X[i,:]
                    for j in range(_X.shape[1]):
                        line.append(str(x_i[j]))                            
                    writer.writerow(line)
                
            writeFile.close()
        except IOError:
            flag = False
            print("IOError file \'{}\'".format(_fname))
        return flag
    
    def parseString(self, _string, _dtype='float', _delim=' '):  
        
        list_ = []
        try:
            if (_dtype=='float'):
                list_ = [float(i) for i in _string.split(_delim)]        
            elif (_dtype=='bool'):
                list_ = [bool(i) for i in _string.split(_delim)]        
            elif (_dtype=='int'):
                list_ = [int(i) for i in _string.split(_delim)]        
            else:
                print("Error unknown data type \'{}\' for parsing".format(_dtype))                
        except ValueError:
            print("Cannot parse string=\'{}\', dtype=\'{}\', delimiter=\'{}\'!".format(_string,_dtype, _delim))                
        return list_ 
    
        
    def readData(self,_fname, _type="float", _delimiter=' '):
            
        def toFloat(_v):
            return float(_v)
    
        def toBool(_v):
            return _v == "1"
        
        try:
            ifile = open(_fname, "r")        
            reader = csv.reader(ifile, delimiter=_delimiter)
            data = []    
            func = toFloat
            if _type == "bool":
                func = toBool
            for row in reader:
                coord = []     
                for col in row:
                    coord.append(func(col))        
                data.append(coord)
            ifile.close()
            X = np.array(data)
            return X
        
        except IOError:
            print("IOError file \'{}\'".format(_fname))
            
        return None
    
    def str2Int(self, _str):
        
        i = None
        try:
            i = int(_str)            
        except ValueError:
            print("ValueError converting to int \'{}\'".format(_str))
        return i
                        
    def readDataRobust(self,_fname, _type="float", _delimiter=' '):
            
        def toFloat(_v):
            return float(_v)
    
        def toBool(_v):
            return _v == "1"
        
        try:
            ifile = open(_fname, "r")        
            reader = csv.reader(ifile, delimiter=_delimiter)
            data = []    
            func = toFloat
            if _type == "bool":
                func = toBool
            count = 0    
            for row in reader:
                if count == 0:
                    count = 1
                    continue
                
                coord = []     
                for col in row:
                    try:
                        coord.append(func(col))  
                    except ValueError:
                        pass
                    #    print('NaN')
                    #coord.append(float(col))        
                data.append(coord)
            ifile.close()
            X = np.array(data)
            return X
        except IOError:
            print("IOError file \'{}\'".format(_fname))
            
        return None

