"""
Predict all sample of a data field
---
Author: Hongtao Wang | stolzpi@163.com
"""
import argparse
import os
import shutil
import warnings

import h5py
import numpy as np
import pandas as pd
import segyio
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from net.MIFNet import MultiInfoNet
from utils.evaluate import GetResult
from utils.LoadData import DLSpec
from utils.PastProcess import NMOCorr, interpolation2
from utils.PlotTools import *

warnings.filterwarnings("ignore")


def CheckSavePath(opt):
    if opt.Resave:
        if os.path.exists(opt.OutputPath):
            shutil.rmtree(opt.OutputPath)
    if not os.path.exists(opt.OutputPath):
        os.makedirs(opt.OutputPath)


###########################################################
# get predict parameters
###########################################################
def GetPredictPara():
    parser = argparse.ArgumentParser()
    parser.add_argument('--DataSetRoot', type=str, default='E:\\Spectrum', help='Dataset Root Path')
    parser.add_argument('--LoadModel', type=str, help='Model Path')
    parser.add_argument('--OutputPath', type=str, help='Path of Output')
    parser.add_argument('--GPUNO', type=int, default=0)
    parser.add_argument('--PredBS', type=int, default=64, help='The batchsize of Predict')
    opt = parser.parse_args()
    return opt


################################################################
# main function for prediction
################################################################
def predict(opt):
    # setting model parameters
    ParaDict = pd.read_csv(os.path.join(opt.LoadModel, 'TrainPara.csv')).to_dict()
    # check output folder
    DataSet = ParaDict['DataSet'][0]
    OutputPath = os.path.join(opt.LoadModel, 'predict', DataSet)
    DataSetPath = os.path.join(opt.DataSetRoot, DataSet)
    CheckSavePath(opt)
    
    Resize = [int(ParaDict['SizeH'][0]), int(ParaDict['SizeW'][0])]
    ModelPath = os.path.join(opt.LoadModel, 'model', 'Best.pth')

    # check gpu is available
    use_gpu = torch.cuda.device_count() > 0

    # load segy data
    SegyName = {'pwr': 'vel.pwr.sgy',
                'stk': 'vel.stk.sgy',
                'gth': 'vel.gth.sgy'}
    SegyDict = {}
    for name, path in SegyName.items():
        SegyDict.setdefault(name, segyio.open(os.path.join(DataSetPath, 'segy', path), "r", strict=False))
    # load h5 file
    H5Name = {'pwr': 'SpecInfo.h5',
              'stk': 'StkInfo.h5',
              'gth': 'GatherInfo.h5'}
    H5Dict = {}
    for name, path in H5Name.items():
        H5Dict.setdefault(name, h5py.File(os.path.join(DataSetPath, 'h5File', path), 'r'))
    t0Int = np.array(SegyDict['pwr'].samples)

    # load label.npy
    LabelDict = np.load(os.path.join(DataSetPath, 't_v_labels.npy'), allow_pickle=True).item()
    HaveLabelIndex = []
    for lineN in LabelDict.keys():
        for cdpN in LabelDict[lineN].keys():
            HaveLabelIndex.append('%s_%s' % (lineN, cdpN))
    pwr_index = set(H5Dict['pwr'].keys())
    stk_index = set(H5Dict['stk'].keys())
    gth_index = set(H5Dict['gth'].keys())

    # find common index
    Index = sorted(list((pwr_index & stk_index) & (gth_index & set(HaveLabelIndex))))
    
    # build DataLoader
    dsPred = DLSpec(SegyDict, H5Dict, LabelDict, Index, t0Int,
                    mode='predict', GatherLen=int(ParaDict['SGSL']), resize=Resize)
    dlPred = DataLoader(dsPred,
                        batch_size=opt.PredBS,
                        shuffle=False,
                        num_workers=0,
                        drop_last=False)

    # Load Predict Network
    net = MultiInfoNet(t0Int, mode=ParaDict['SGSMode'][0], in_channels=11, resize=Resize)
    if use_gpu:
        net = net.cuda(opt.GPUNO)
    net.eval()
    # Load the weights of network
    if os.path.exists(ModelPath):
        print("Load Model Successfully! \n(%s)" % ModelPath)
        net.load_state_dict(torch.load(ModelPath)['Weights'])
    else:
        raise print("There is no such model file!")
    bar = tqdm(total=len(dlPred))
    # create dict to save auto picking results
    PickDict = {}   
    # predict all of the dataset
    with torch.no_grad():
        for pwr, stkG, stkC, _, VMM, _, name in dlPred:
            if use_gpu:
                pwr = pwr.cuda(opt.GPUNO)
                stkG = stkG.cuda(opt.GPUNO)
                stkC = stkC.cuda(opt.GPUNO)

            out, _ = net(pwr, stkG, stkC, VMM, True)
            PredSeg = out.squeeze()

            for ind, name_single in enumerate(name):
                PickDict.setdefault(name_single, PredSeg[ind].cpu().detach().numpy())
            bar.update(1)

    # Save result
    np.save(os.path.join(OutputPath, '0-PickDict.npy'), PickDict)


if __name__ == '__main__':
    # get parameters
    optN = GetPredictPara()
    # start to test
    predict(optN)
