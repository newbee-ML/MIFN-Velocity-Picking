import argparse
import os
import random
import shutil
import time
import warnings

import h5py
import pandas as pd
import numpy as np
import segyio
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from net.AblationNet import MixNet
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
    FileList = ['FigResults', 'FeatureMaps']
    for file in FileList:
        if not os.path.exists(os.path.join(opt.OutputPath, file)):
            os.makedirs(os.path.join(opt.OutputPath, file))


def GetTestPara():
    parser = argparse.ArgumentParser()
    parser.add_argument('--DataSet', type=str, help='Dataset Root Path')
    parser.add_argument('--DataSetRoot', type=str, default='E:\\Spectrum', help='Dataset Root Path')
    parser.add_argument('--EpName', type=str, help='Dataset Root Path')
    parser.add_argument('--LoadModel', type=str, help='Model Path')
    parser.add_argument('--OutputPath', type=str, help='Path of Output')
    parser.add_argument('--PostProcessing', type=int, default=1, help='post-processing method')
    parser.add_argument('--Predthre', type=float, default=0.25)
    parser.add_argument('--Resave', type=int, default=0)
    parser.add_argument('--GPUNO', type=int, default=0)
    parser.add_argument('--PredBS', type=int, default=32, help='The batchsize of Predict')
    parser.add_argument('--TransferL', type=int, default=0, help='Whether transfer learning')
    opt = parser.parse_args()
    return opt


def test(opt):
    # setting model parameters
    ParaDict = pd.read_csv(os.path.join(opt.LoadModel, 'TrainPara.csv')).to_dict()
    # check output folder
    if opt.DataSet is None:
        opt.DataSet = ParaDict['DataSet'][0]
    if opt.TransferL:
        opt.OutputPath = os.path.join(opt.LoadModel, 'test', 'TL-'+opt.DataSet)
    else:
        opt.OutputPath = os.path.join(opt.LoadModel, 'test', opt.DataSet)
    DataSetPath = os.path.join(opt.DataSetRoot, opt.DataSet)
    CheckSavePath(opt)
    Resize = [int(ParaDict['SizeH'][0]), int(ParaDict['SizeW'][0])]
    BadThre = int(200/ParaDict['SeedRate'][0])

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
    IndexDict = {}
    for index in Index:
        line, cdp = index.split('_')
        IndexDict.setdefault(int(line), [])
        IndexDict[int(line)].append(int(cdp))
    LineIndex = sorted(list(IndexDict.keys()))
    # use the last 20% for test set
    LastSplit2 = int(len(LineIndex)*0.8)
    testLine = LineIndex[LastSplit2:]
    testIndex = []
    for line in testLine:
        for cdp in IndexDict[line]:
            testIndex.append('%d_%d' % (line, cdp))
    print(opt.EpName, 'Test Line: ', testLine, '\nTest Number: %d' % len(testIndex))
    
    # build DataLoader
    dsPred = DLSpec(SegyDict, H5Dict, LabelDict, testIndex, t0Int,
                    mode='train', GatherLen=int(ParaDict['GatherLen'][0]), resize=Resize)
    dlPred = DataLoader(dsPred,
                        batch_size=opt.PredBS,
                        shuffle=False,
                        num_workers=0,
                        drop_last=False)

    startP = time.time()  # start clock

    # if have predicted then compute the VMAE result
    if not os.path.exists(os.path.join(opt.OutputPath, '0-PickDict.npy')):
        # Load Predict Network
        try:
            net = MixNet(t0Int, NetType=ParaDict['NetType'][0], mode=ParaDict['SGSMode'][0], resize=Resize)
        except:
            net = MultiInfoNet(t0Int, mode=ParaDict['SGSMode'][0], in_channels=11, resize=Resize)
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
                
                out, feature = net(pwr, stkG, stkC, VMM, True)
                PredSeg = out.squeeze()

                # save the Vint information
                VInt = []
                for n in name:
                    PwrIndex = np.array(H5Dict['pwr'][n]['SpecIndex'])
                    VInt.append(np.array(SegyDict['pwr'].attributes(segyio.TraceField.offset)[PwrIndex[0]: PwrIndex[1]]))

                # get velocity Picking
                AP, APPeaks = GetResult(PredSeg.cpu().numpy(), t0Int, VInt, threshold=opt.Predthre, PostProcessing=opt.PostProcessing)
                # save the picking result
                for ind, name_single in enumerate(name):
                    FeatureN = {}
                    for key, featuremaps in feature.items():
                        FeatureN.setdefault(key, featuremaps[ind].squeeze().cpu().numpy())
                    PickDict.setdefault(name_single, {'AP': AP[ind], 'APPeaks': APPeaks[ind], 'MP': MP[ind].numpy(), 'Pwr': pwr[ind, 0, :].cpu().detach().numpy(), 'Seg': out[ind].cpu().detach().numpy(), 'Tint': t0Int, 'VInt': VInt[ind], 'Feature': FeatureN})
                bar.update(1)
    else:
        PickDict = np.load(os.path.join(opt.OutputPath, '0-PickDict.npy'), allow_pickle=True).item()
        for name, ResultDict in PickDict.items():
            AP, APPeaks = GetResult(PickDict[name]['Seg'], t0Int, [PickDict[name]['VInt']], threshold=opt.Predthre, PostProcessing=opt.PostProcessing)
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
    BadSample = [name for name in PickDict.keys() if PickDict[name]['VMAE'] > BadThre]

    # 2 Visual Part Bad Results
    for ind, name in enumerate(BadSample):
        break
        ResultDict = PickDict[name]
        print(name, 'VMAE: %.3f' % ResultDict['VMAE'])
        # 2.1 Pwr and Seg Map
        PwrASeg(ResultDict['Pwr'], ResultDict['Seg'], SavePath=os.path.join(opt.OutputPath, 'FigResults', '2-1-PwrASeg %s' % name))
        # 2.2.1 AP peaks and MP curve
        InterpResult(ResultDict['APPeaks'], ResultDict['MP'], SavePath=os.path.join(opt.OutputPath, 'FigResults', '2-2-1-InterpResult %s' % name))
        # 2.2 Seg with AP and MP 
        PwrIndex = np.array(H5Dict['pwr'][name]['SpecIndex'])
        vVec = np.array(SegyDict['pwr'].attributes(segyio.TraceField.offset)[PwrIndex[0]: PwrIndex[1]])
        SegPick(ResultDict['Seg'], t0Int, vVec, ResultDict['AP'], ResultDict['MP'], SavePath=os.path.join(opt.OutputPath, 'FigResults', '2-2-SegPick %s' % name))
        # 2.3 CMP gather and NMO result
        GthIndex = np.array(H5Dict['gth'][name]['GatherIndex'])
        Gth = np.array(SegyDict['gth'].trace.raw[GthIndex[0]: GthIndex[1]].T)
        OVec = np.array(SegyDict['gth'].attributes(segyio.TraceField.offset)[GthIndex[0]: GthIndex[1]])
        AP = interpolation2(ResultDict['AP'], np.array(SegyDict['gth'].samples), vVec, RefRange=300)
        NMOGth = NMOCorr(Gth, np.array(SegyDict['gth'].samples), OVec, AP[:, 1], CutC=1.2)
        CMPNMO(Gth, NMOGth, SavePath=os.path.join(opt.OutputPath, 'FigResults', '2-3-CMPNMO %s' % name))
        # 3 visual feature maps
        for key, feature in ResultDict['Feature'].items():
            NetFeatureMap(feature, 'seismic', os.path.join(opt.OutputPath, 'FeatureMaps', '3-%s-%s' % (name, key)))
    


if __name__ == '__main__':
    # get parameters
    optN = GetTestPara()
    optN.__dict__['LoadModel'] = 'F:\VelocitySpectrum\MIFN\\2GeneraTest\Ep-31'
    # start to test
    test(optN)
