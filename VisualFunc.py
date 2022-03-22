"""
This python file is for single prediction
Visual the following plots:
- original spectrum
- segmentation map
- SGS gather input map
"""


##########################
# Import Settings 
##########################
import argparse
import os
import warnings

import h5py
import numpy as np
import pandas as pd
import segyio

from utils.PastProcess import NMOCorr, interpolation2
from utils.PlotTools import *

warnings.filterwarnings("ignore")

#######################
# Base Setting
#######################
def BaseSetting():
    parser = argparse.ArgumentParser()
    parser.add_argument('--DataSetRoot', type=str, help='Dataset Root Path')
    parser.add_argument('--LoadModel', type=str, help='Model Path')
    parser.add_argument('--Predthre', type=float, default=0.3)
    parser.add_argument('--OutputPath', type=float, default=0.3)
    parser.add_argument('--GPUNO', type=int, default=0)
    parser.add_argument('--line', type=int, default=0)
    parser.add_argument('--cdp', type=int, default=0)
    opt = parser.parse_args(args=[])
    return opt


########################
# Show the test index 
########################
def Visual(opt):
    # setting model parameters
    ParaDict = pd.read_csv(os.path.join(opt.LoadModel, 'TrainPara.csv')).to_dict()
    DataSet = ParaDict['DataSet'][0]
    Datapath = os.path.join(opt.DataSetRoot, DataSet)
    # load segy data
    SegyName = {'pwr': 'vel.pwr.sgy',
                'stk': 'vel.stk.sgy',
                'gth': 'vel.gth.sgy'}
    SegyDict = {}
    for name, path in SegyName.items():
        SegyDict.setdefault(name, segyio.open(os.path.join(Datapath, 'segy', path), "r", strict=False))
    # load h5 file
    H5Name = {'pwr': 'SpecInfo.h5',
              'stk': 'StkInfo.h5',
              'gth': 'GatherInfo.h5'}
    H5Dict = {}
    for name, path in H5Name.items():
        H5Dict.setdefault(name, h5py.File(os.path.join(Datapath, 'h5File', path), 'r'))


    ############################################
    # load picking result dict
    ############################################
    PickDict = np.load(os.path.join(opt.LoadModel, 'test', DataSet, '0-PickDict.npy'), allow_pickle=True).item()


    #######################
    # get single one data
    #######################
    index = '%d_%d' % (opt.line, opt.cdp)
    ResultDict = PickDict[index]
    print(index, 'VMAE: %.3f' % ResultDict['VMAE'])

    ###########################################
    # load original data
    ###########################################
    # load gather data
    GthIndex = np.array(H5Dict['gth'][index]['GatherIndex'])
    Gth = np.array(SegyDict['gth'].trace.raw[GthIndex[0]: GthIndex[1]].T)
    OVec = np.array(SegyDict['gth'].attributes(segyio.TraceField.offset)[GthIndex[0]: GthIndex[1]])
    tInd = np.array(SegyDict['gth'].samples)

    ############################################
    # visual result
    ############################################
    # create save folder
    BaseName = '%s-%d-%d' % (DataSet, opt.line, opt.cdp)
    SavePath = os.path.join(opt.OutputPath, BaseName)
    if not os.path.exists(SavePath):
        os.makedirs(SavePath)
    FileType = 'png'
    # 1.1 original spectrum
    plot_spectrum(ResultDict['Pwr'], ResultDict['Tint'], ResultDict['VInt'],
                  save_path=os.path.join(SavePath, '%s-1-1-Pwr.%s' % (BaseName, FileType)), ShowY=True)
    # 1.2 seg map 
    plot_spectrum(ResultDict['Seg'], ResultDict['Tint'], ResultDict['VInt'],
                  save_path=os.path.join(SavePath, '%s-1-2-Seg.%s' % (BaseName, FileType)), ShowY=False)
    # 1.3 SGS feature map
    plot_spectrum(ResultDict['Feature']['Stk'], ResultDict['Tint'], 
                  ResultDict['VInt'], save_path=os.path.join(SavePath, '%s-1-3-Seg.%s' % (BaseName, FileType)), ShowY=False)
    # 1.4 velocity curve
    InterpResult(ResultDict['APPeaks'], ResultDict['MP'], ResultDict['Tint'], ResultDict['VInt'], SavePath=os.path.join(SavePath, '%s-1-4-StackVelocity.%s' % (BaseName, FileType)), ShowY=False)
    # 1.5 original CMP gather
    plot_cmp(Gth, tInd, OVec, save_path=os.path.join(SavePath, '%s-1-5-CMP.%s' % (BaseName, FileType)), if_add=0, ShowY=True)
    # 1.6 NMO CMP
    AP = interpolation2(ResultDict['AP'], np.array(SegyDict['gth'].samples), ResultDict['VInt'], RefRange=300)
    NMOGth = NMOCorr(Gth, np.array(SegyDict['gth'].samples), OVec, AP[:, 1], CutC=1.2)
    plot_cmp(NMOGth, tInd, OVec, save_path=os.path.join(SavePath, '%s-1-6-NMOCMP.%s' % (BaseName, FileType)), if_add=1, ShowY=False)
