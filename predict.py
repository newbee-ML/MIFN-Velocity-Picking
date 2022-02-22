import argparse
import os
import time
import warnings

import h5py
import numpy as np
import pandas as pd
import segyio
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import shutil
from utils.evaluate import GetResult
from net.MIFNet import MultiInfoNet
from utils.LoadData import DLSpec

warnings.filterwarnings("ignore")



def CheckSavePath(opt):
    if opt.Resave:
        if os.path.exists(os.path.join(opt.OutputPath, 'PredictResults')):
            shutil.rmtree(os.path.join(opt.OutputPath, 'PredictResults'))
    filePath = os.path.join(opt.OutputPath, 'PredictResults')
    if not os.path.exists(filePath):
        os.makedirs(filePath)


def predict():
    print('==== Begin to Predict ====')
    # check output folder
    opt.OutputPath = os.path.join(opt.OutputPath, opt.DataSet)
    CheckSavePath(opt)
    opt.Resize = [int(opt.SizeH), int(opt.SizeH/2)]

    BestPath = opt.LoadModel

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

    # load label.npy
    LabelDict = np.load(os.path.join(opt.DataSetRoot, 't_v_labels.npy'), allow_pickle=True).item()
    HaveLabelIndex = []
    for lineN in LabelDict.keys():
        for cdpN in LabelDict[lineN].keys():
            HaveLabelIndex.append('%s_%s' % (lineN, cdpN))

    # find common index
    ALLIndex = (set(H5Dict['pwr'].keys()) & set(H5Dict['stk'].keys())) & set(H5Dict['gth'].keys())
    ALLIndex = sorted(list(ALLIndex))
    T0Ind = np.array(SegyDict['pwr'].samples)
    opt.t0Int = T0Ind

    # build DataLoader
    dsPred = DLSpec(SegyDict, H5Dict, LabelDict, ALLIndex, opt.t0Int,
                    mode='test', GatherLen=opt.GatherLen, resize=opt.Resize)
    dlPred = DataLoader(dsPred,
                        batch_size=opt.PredBS,
                        shuffle=False,
                        num_workers=0,
                        drop_last=True)

    # Load Predict Network
    net = MultiInfoNet(T0Ind, opt, in_channels=11, resize=opt.Resize)
    if use_gpu:
        net = net.cuda(opt.GPUNO)
    net.eval()
    # Load the weights of network
    if os.path.exists(BestPath):
        print("Load Model Successfully! \n(%s)" % BestPath)
        net.load_state_dict(torch.load(BestPath)['Weights'])
    else:
        print("There is no such model file, start a new training!")

    # create dict to save auto picking results
    CSVPath = os.path.join(opt.OutputPath, 'AutoPicking_%s.csv' % os.path.basename(BestPath).strip('.pth'))
    PickDict = {'line': [], 'trace': [], 'time': [], 'velocity': []}

    startP = time.time()  # start clock
    bar = tqdm(total=len(dlPred))

    # predict all of the dataset
    with torch.no_grad():
        for iBatch, (pwr, stkG, stkC, _, VMM, _, name) in enumerate(dlPred):
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
                VInt.append(np.array(SegyDict['pwr'].attributes(segyio.TraceField.offset)
                                     [PwrIndex[0]: PwrIndex[1]]))
                NameList.append(n.split('_'))

            # get velocity Picking
            _, AutoPick = GetResult(PredSeg, opt.t0Int, VInt, threshold=opt.Predthre)
            for ind, (lineN, cdpN) in enumerate(NameList):
                Picking = AutoPick[ind]
                if Picking.shape[0] > 0:
                    PickDict['line'] += Picking.shape[0] * [int(lineN)]
                    PickDict['trace'] += Picking.shape[0] * [int(cdpN)]
                    PickDict['time'] += Picking[:, 0].astype(np.int).tolist()
                    PickDict['velocity'] += Picking[:, 1].astype(np.int).tolist()
                else:
                    print(Picking)
        
            bar.update(1)

    PredCost = time.time() - startP 

    # Compute FPS
    print('Predict %d Samples, Cost %.3f, FPS %.3f' % (len(ALLIndex), PredCost, len(ALLIndex)/PredCost))

    # save the predict results
    PickDF = pd.DataFrame(PickDict)
    PickDF.to_csv(CSVPath, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DataSet', type=str, default='hade', help='Dataset Root Path')
    parser.add_argument('--DataSetRoot', type=str, help='Dataset Root Path')
    parser.add_argument('--GatherLen', type=int, default=21)
    parser.add_argument('--LoadModel', type=str, help='Model Path')
    parser.add_argument('--OutputPath', type=str, default='result', help='Path of Output')
    parser.add_argument('--SGSMode', type=str, default='all')
    parser.add_argument('--Predthre', type=float, default=0.3)
    parser.add_argument('--Resave', type=int, default=0)
    parser.add_argument('--GPUNO', type=int, default=0)
    parser.add_argument('--Resize', type=list, help='CropSize')
    parser.add_argument('--SizeH', type=int, default=256, help='Size Height')
    parser.add_argument('--PredBS', type=int, default=32, help='The batchsize of Predict')
    parser.add_argument('--t0Int', type=list, default=[])
    opt = parser.parse_args()

    # start to predict
    predict()
