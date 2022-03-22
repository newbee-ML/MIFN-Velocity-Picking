import os
import sys

import h5py
import segyio
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.evaluate import GetResult
from utils.PlotTools import *
from utils.LoadData import interpolation
from utils.PastProcess import NMOCorr, interpolation2


class StkCMP:
    def __init__(self, EpName, line, RootPath, DataRoot):
        self.PickPath = os.path.join(RootPath, EpName)
        ParaDict = pd.read_csv(os.path.join(self.PickPath, 'TrainPara.csv')).to_dict()
        self.DataSet = ParaDict['DataSet'][0]
        self.line = line
        self.PickDict = np.load(os.path.join(self.PickPath, 'predict', ParaDict['DataSet'][0], '0-PickDict.npy'), allow_pickle=True).item()
        self.DataPath = os.path.join(DataRoot, ParaDict['DataSet'][0])
        self.LabelDict = np.load(os.path.join(self.DataPath, 't_v_labels.npy'), allow_pickle=True).item()
        self.SegyDict, self.H5Dict = self.LoadData()
        APDict = self.GetAP(self.PickDict)
        self.GetStkNMO(APDict)

    ################################################################
    # get the AP with different predict threshold
    ################################################################
    def GetAP(self, SegDict):
        APDict = {}
        PtList = [0.2, 0.25]
        bar = tqdm(total=len(list(SegDict.keys())), file=sys.stdout)
        for name, PickDict in SegDict.items():
            line, cdp = map(int, name.split('_'))
            if line != int(self.line):
                bar.set_description(name)
                bar.update(1)
                continue
            # get the picking results under different prediction threshold
            for pt in PtList:
                AP, _ = GetResult(PickDict['Seg'][np.newaxis, ...], PickDict['TInt'], [PickDict['VInt']], threshold=pt)
                APDict.setdefault(line, {})
                APDict[line].setdefault(pt, {})
                APDict[line][pt].setdefault(cdp, AP)
            # load manual picking result
            if type(list(self.LabelDict.keys())[0]) == str:
                lineM, cdpM = str(line), str(cdp)
            else:
                lineM, cdpM = line, cdp
            if lineM in list(self.LabelDict.keys()):
                if cdpM in list(self.LabelDict[lineM].keys()):
                    APDict[line].setdefault('MP', {})
                    APDict[line]['MP'].setdefault(cdp, interpolation(self.LabelDict[lineM][cdpM], PickDict['TInt'], PickDict['VInt']))
            bar.set_description(name)
            bar.update(1)
        bar.close()
        return APDict

    ################################################################
    # load data 
    ################################################################
    def LoadData(self):
        # load segy data
        SegyName = {'pwr': 'vel.pwr.sgy',
                    'stk': 'vel.stk.sgy',
                    'gth': 'vel.gth.sgy'}
        SegyDict = {}
        for name, path in SegyName.items():
            SegyDict.setdefault(name, segyio.open(os.path.join(self.DataPath, 'segy', path), "r", strict=False))
        # load h5 file
        H5Name = {'pwr': 'SpecInfo.h5',
                    'stk': 'StkInfo.h5',
                    'gth': 'GatherInfo.h5'}
        H5Dict = {}
        for name, path in H5Name.items():
            H5Dict.setdefault(name, h5py.File(os.path.join(self.DataPath, 'h5File', path), 'r'))
        return SegyDict, H5Dict

    ################################################################
    # plot the velocity field
    ################################################################
    def GetStkNMO(self, APDict):
        tInd = np.array(self.SegyDict['gth'].samples)
        for line, ResultDict in APDict.items():
            SavePath = os.path.join(self.PickPath, 'predict', self.DataSet, 'StkNMO', str(line))
            if not os.path.exists(SavePath):
                os.makedirs(SavePath)
            for pt, APResult in ResultDict.items():
                cdpList = sorted(list(APResult.keys()))
                NMOCMP = []
                for cdp in cdpList:
                    index = '%d_%d'%(line, cdp)
                    SegDict = self.PickDict[index]
                    VelA = interpolation2(np.squeeze(APResult[cdp]), tInd, SegDict['VInt'], RefRange=300)
                    GthIndex = np.array(self.H5Dict['gth'][index]['GatherIndex'])
                    Gth = np.array(self.SegyDict['gth'].trace.raw[GthIndex[0]: GthIndex[1]].T)
                    OVec = np.array(self.SegyDict['gth'].attributes(segyio.TraceField.offset)[GthIndex[0]: GthIndex[1]])
                    NMOGth = NMOCorr(Gth, np.array(self.SegyDict['gth'].samples), OVec, VelA[:, 1], CutC=1.2)
                    NMOCMP.append(np.sum(NMOGth, axis=1))
                NMOCMP = np.array(NMOCMP).T
                # PlotStkGather(NMOCMP, tInd, cdpList, save_path=os.path.join(SavePath, 'pt-%.2f-Wplot.pdf'% float(pt)))
                try:
                    StkNMOCMP(NMOCMP, tInd, np.array(cdpList), cmap='gray', save_path=os.path.join(SavePath, 'pt-%.2f-gray.pdf'% float(pt)))
                except:
                    StkNMOCMP(NMOCMP, tInd, np.array(cdpList), cmap='gray', save_path=os.path.join(SavePath, 'pt-%s-gray.pdf'% pt))


if __name__ == '__main__':
    ################################################################
    # predict experiment list
    ################################################################
    RootPath = 'F:\\VelocitySpectrum\\MIFN\\2GeneraTest'
    DataRoot = 'E:\\Spectrum'
    EpList = {## 'A': {'Index': [203], 'Line': 3240},
              'B': {'Index': [213], 'Line': 940}}
    for data, InfoDict in EpList.items():
        for EpNum in InfoDict['Index']:
            StkCMP('Ep-%d' % EpNum, InfoDict['Line'], RootPath, DataRoot)

