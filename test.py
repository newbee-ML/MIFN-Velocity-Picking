import argparse
import os
import random
import shutil
import time
import warnings

import h5py
import numpy as np
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


def test():
    # check output folder
    opt.OutputPath = os.path.join(opt.LoadModel, 'test', opt.DataSet)
    CheckSavePath(opt)
    # setting model parameters
    ParaDict = {para.split('_')[0]: para.split('_')[1] for para in os.path.split(opt.LoadModel)[-1].split('-')}
    Resize = [int(ParaDict['SH']), int(ParaDict['SW'])]

    ModelPath = os.path.join(opt.LoadModel, 'model', 'Best.pth')

    # check gpu is available
    use_gpu = torch.cuda.device_count() > 0

    # load segy data
    SegyName = {'pwr': 'vel.pwr.sgy',
                'stk': 'vel.stk.sgy',
                'gth': 'vel.gth.sgy'}
    SegyDict = {}
    for name, path in SegyName.items():
        SegyDict.setdefault(name, segyio.open(os.path.join(opt.DataSetRoot, 'segy', path), "r", strict=False))
    # load h5 file
    H5Name = {'pwr': 'SpecInfo.h5',
              'stk': 'StkInfo.h5',
              'gth': 'GatherInfo.h5'}
    H5Dict = {}
    for name, path in H5Name.items():
        H5Dict.setdefault(name, h5py.File(os.path.join(opt.DataSetRoot, 'h5File', path), 'r'))
    opt.t0Int = np.array(SegyDict['pwr'].samples)

    # load label.npy
    LabelDict = np.load(os.path.join(opt.DataSetRoot, 't_v_labels.npy'), allow_pickle=True).item()
    HaveLabelIndex = []
    for lineN in LabelDict.keys():
        for cdpN in LabelDict[lineN].keys():
            HaveLabelIndex.append('%s_%s' % (lineN, cdpN))
    pwr_index = set(H5Dict['pwr'].keys())
    stk_index = set(H5Dict['stk'].keys())
    gth_index = set(H5Dict['gth'].keys())

    # find common index
    Index = sorted(list((pwr_index & stk_index) & (gth_index & set(HaveLabelIndex))))
    _, testIndex = train_test_split(Index, test_size=0.2, random_state=123)
    random.seed(123)
    # VisualSample = random.sample(testIndex, 16)
    
    # build DataLoader
    dsPred = DLSpec(SegyDict, H5Dict, LabelDict, testIndex, opt.t0Int,
                    mode='train', GatherLen=int(ParaDict['SGSL']), resize=Resize)
    dlPred = DataLoader(dsPred,
                        batch_size=opt.PredBS,
                        shuffle=False,
                        num_workers=0,
                        drop_last=False)

    startP = time.time()  # start clock

    # if have predicted then compute the VMAE result
    if not os.path.exists(os.path.join(opt.OutputPath, '0-PickDict.npy')):
        # Load Predict Network
        net = MultiInfoNet(opt.t0Int, opt, in_channels=11, resize=Resize)
        if use_gpu:
            net = net.cuda(opt.GPUNO)
        net.eval()
        # Load the weights of network
        if os.path.exists(ModelPath):
            print("Load Model Successfully! \n(%s)" % ModelPath)
            net.load_state_dict(torch.load(ModelPath)['Weights'])
        else:
            print("There is no such model file, start a new training!")
        bar = tqdm(total=len(dlPred))
        # create dict to save auto picking results
        PickDict = {}   
        # predict all of the dataset
        with torch.no_grad():
            for pwr, stkG, stkC, _, VMM, MP, name in dlPred:
                if use_gpu:
                    pwr = pwr.cuda(opt.GPUNO)
                    stkG = stkG.cuda(opt.GPUNO)
                    stkC = stkC.cuda(opt.GPUNO)

                out, STKInfo = net(pwr, stkG, stkC, VMM)
                PredSeg, STKInfo = out.squeeze(), STKInfo.squeeze()
                RawPwr, VInt, NameList = [], [], []
                for n in name:
                    PwrIndex = np.array(H5Dict['pwr'][n]['SpecIndex'])
                    RawPwr.append(np.array(SegyDict['pwr'].trace.raw[PwrIndex[0]: PwrIndex[1]].T))
                    VInt.append(np.array(SegyDict['pwr'].attributes(segyio.TraceField.offset)[PwrIndex[0]: PwrIndex[1]]))
                    NameList.append(n.split('_'))

                # get velocity Picking
                AP, APPeaks = GetResult(PredSeg.cpu().numpy(), opt.t0Int, VInt, threshold=opt.Predthre)
                # VMAE  
                for ind, name_single in enumerate(name):
                    PickDict.setdefault(name_single, {'AP': AP[ind], 'APPeaks': APPeaks[ind], 'MP': MP[ind].numpy(), 'Pwr': pwr[ind, 0, :].cpu().detach().numpy(), 'Seg': out[ind].cpu().detach().numpy(), 'VInt': VInt[ind]})
                bar.update(1)
    else:
        PickDict = np.load(os.path.join(opt.OutputPath, '0-PickDict.npy'), allow_pickle=True).item()
        for name, ResultDict in PickDict.items():
            AP, APPeaks = GetResult(PickDict[name]['Seg'], opt.t0Int, [PickDict[name]['VInt']], threshold=opt.Predthre)
            PickDict[name]['APPeaks'] = APPeaks[0]
            PickDict[name]['AP'] = AP[0]

    PredCost = time.time() - startP 

    # Compute FPS
    print('Predict %d Samples, Cost %.2fs, FPS %.3f' % (len(testIndex), PredCost, len(testIndex)/PredCost))
    # Compute VMAE
    for name, ResultDict in PickDict.items():
        vmae = np.mean(np.abs(ResultDict['AP'][:, 1] - ResultDict['MP'][:, 1]))
        PickDict[name]['VMAE'] = vmae
    # Save result
    np.save(os.path.join(opt.OutputPath, '0-PickDict.npy'), PickDict)

    ## visual result
    # 1 hist of VMAE results
    VMAEList = [RDict['VMAE'] for RDict in PickDict.values()]
    print('test mean VMAE: %.3f' % (np.mean(VMAEList)))
    VMAEHist(VMAEList, SavePath=os.path.join(opt.OutputPath, '1-VMAEHist'))
    # get the bad sample index
    BadSample = [name for name in PickDict.keys() if PickDict[name]['VMAE'] > 50]

    # 2 Visual Part Bad Results
    for name in BadSample:
        ResultDict = PickDict[name]
        print(name, 'VMAE: %.3f' % ResultDict['VMAE'])
        # 2.1 Pwr and Seg Map
        PwrASeg(ResultDict['Pwr'], ResultDict['Seg'], SavePath=os.path.join(opt.OutputPath, '2-1-PwrASeg %s' % name))
        # 2.2.1 AP peaks and MP curve
        InterpResult(ResultDict['APPeaks'], ResultDict['MP'], SavePath=os.path.join(opt.OutputPath, '2-2-1-InterpResult %s' % name))
        # 2.2 Seg with AP and MP 
        PwrIndex = np.array(H5Dict['pwr'][name]['SpecIndex'])
        vVec = np.array(SegyDict['pwr'].attributes(segyio.TraceField.offset)[PwrIndex[0]: PwrIndex[1]])
        SegPick(ResultDict['Seg'], opt.t0Int, vVec, ResultDict['AP'], ResultDict['MP'], SavePath=os.path.join(opt.OutputPath, '2-2-SegPick %s' % name))
        # 2.3 CMP gather and NMO result
        GthIndex = np.array(H5Dict['gth'][name]['GatherIndex'])
        Gth = np.array(SegyDict['gth'].trace.raw[GthIndex[0]: GthIndex[1]].T)
        OVec = np.array(SegyDict['gth'].attributes(segyio.TraceField.offset)[GthIndex[0]: GthIndex[1]])
        AP = interpolation2(ResultDict['AP'], np.array(SegyDict['gth'].samples), vVec, RefRange=300)
        NMOGth = NMOCorr(Gth, np.array(SegyDict['gth'].samples), OVec, AP[:, 1], CutC=1.2)
        CMPNMO(Gth, NMOGth, SavePath=os.path.join(opt.OutputPath, '2-3-CMPNMO %s' % name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DataSet', type=str, default='hade', help='Dataset Root Path')
    parser.add_argument('--DataSetRoot', type=str, default='E:\\Spectrum\\hade', help='Dataset Root Path')
    parser.add_argument('--LoadModel', type=str, default=r'F:\\VSP-MIFN\\0Ablation\DS_hade-SGSL_15-SR_0.80-LR_0.0100-BS_32-SH_256-SW_256-PT_0.10-SGSM_mute-RT_0', help='Model Path')
    parser.add_argument('--OutputPath', type=str, default='result', help='Path of Output')
    parser.add_argument('--SGSMode', type=str, default='all')
    parser.add_argument('--Predthre', type=float, default=0.15)
    parser.add_argument('--Resave', type=int, default=0)
    parser.add_argument('--GPUNO', type=int, default=0)
    parser.add_argument('--PredBS', type=int, default=64, help='The batchsize of Predict')
    parser.add_argument('--t0Int', type=list, default=[])
    opt = parser.parse_args()

    # start to test
    test()
