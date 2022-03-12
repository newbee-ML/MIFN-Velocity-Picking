import segyio
import argparse
import os
import pandas as pd
import numpy as np
from LoadData import interpolation
from PlotTools import PlotVelField, PlotStkGather
import h5py
import multiprocessing as mp
from PastProcess import NMOCorr


# Reindex the Velocity Picking Results
def ReindexDict():
    PickDict = {}
    VP = pd.read_csv(opt.CSVPath)
    A = VP.groupby(['line', 'trace'])
    for LineCdp, pick in A:
        PickDict.setdefault(LineCdp[0], {})
        PickDict[LineCdp[0]].setdefault(LineCdp[1], pick.to_numpy()[:, 2:])
    return PickDict


class SingleNMO:
    def __init__(self, line, cdp, pick, Gather, vInd, oInd):
        self.line = line
        self.cdp = cdp
        self.pick = pick
        self.Gather = Gather
        self.vInd = vInd
        self.oInd = oInd

    def __call__(self, idx):
        SavePath = os.path.join(opt.OutputPath, 'NMOResults')
        if not os.path.exists(SavePath):
            os.makedirs(SavePath)
        if not os.path.exists(os.path.join(SavePath, '%d_%d.npy' % (self.line, self.cdp[idx]))):
            VelCurve = interpolation(self.pick[idx], opt.tInt, self.vInd[idx])
            NMOGather = NMOCorr(self.Gather[idx], opt.tInt, self.oInd[idx], VelCurve[:, 1])

            np.save(os.path.join(SavePath, '%d_%d.npy' % (self.line, self.cdp[idx])), NMOGather)
            print('%d_%d' % (self.line, self.cdp[idx]))
        else:
            print('%d_%d exists!' % (self.line, self.cdp[idx]))


def MultiComNMOResult():
    # check a few path and output folder
    opt.CSVPath = os.path.join(opt.OutputPath, opt.CSVFile)
    if not os.path.exists(os.path.join(opt.OutputPath, 'VisualResults')):
        os.makedirs(os.path.join(opt.OutputPath, 'VisualResults'))

    # load segy data and h5 file
    PwrSegy = segyio.open(os.path.join(opt.DataSetRoot, 'segy', 'vel.pwr.sgy'), "r", strict=False)
    PwrH5 = h5py.File(os.path.join(opt.DataSetRoot, 'h5File', 'SpecInfo.h5'), 'r')
    GthSegy = segyio.open(os.path.join(opt.DataSetRoot, 'segy', 'vel.gth.sgy'), "r", strict=False)
    GthH5 = h5py.File(os.path.join(opt.DataSetRoot, 'h5File', 'GatherInfo.h5'), 'r')

    # load t0 ind and t ind
    T0Ind, TInd = np.array(PwrSegy.samples), np.array(GthSegy.samples)
    opt.t0Int, opt.tInt = T0Ind, TInd

    # Load Picking Results and Reindex to Dict
    VelPickDict = ReindexDict()

    for line in VelPickDict.keys():
        # load basic parameters
        cdpList, pickList, GatherList, vIndList, oIndList = [], [], [], [], []
        cdpAll = list(VelPickDict[line].keys())
        for cdp in cdpAll:
            if os.path.exists(os.path.join(os.path.join(opt.OutputPath, 'NMOResults'),
                                           '%d_%d.npy' % (line, cdp))):
                print('%d_%d.npy exists' % (line, cdp))
                continue
            PwrIndex = np.array(PwrH5['%d_%d' % (line, cdp)]['SpecIndex'])
            GthIndex = np.array(GthH5['%d_%d' % (line, cdp)]['GatherIndex'])
            Gather = np.array(GthSegy.trace.raw[GthIndex[0]: GthIndex[1]].T)
            vInd = np.array(PwrSegy.attributes(segyio.TraceField.offset)[PwrIndex[0]: PwrIndex[1]])
            oInd = np.array(GthSegy.attributes(segyio.TraceField.offset)[GthIndex[0]: GthIndex[1]])
            pickList.append(VelPickDict[line][cdp])
            cdpList.append(cdp)
            GatherList.append(Gather)
            vIndList.append(vInd)
            oIndList.append(oInd)
            if len(cdpList) == 64 or cdp == cdpAll[-1]:
                with mp.Pool(8) as p:
                    p.map(SingleNMO(line, cdpList, pickList, GatherList, vIndList, oIndList), range(len(cdpList)))
                cdpList, pickList, GatherList, vIndList, oIndList = [], [], [], [], []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    RootPath = r'/home/colin/data/Spectrum/1011'
    Output = os.path.join('result', os.path.basename(RootPath))
    CSVPath = '/home/colin/project/spectrum/MultiInfoMethod/result/1011/AutoPicking_all_1011_256_0.6.csv'
    # r'G:\Dataset\Spectrum\dq8'
    parser.add_argument('--DataSetRoot', type=str, default=RootPath, help='Dataset Root Path')
    parser.add_argument('--OutputPath', type=str, default=Output, help='Path of Output')
    parser.add_argument('--CSVFile', type=str, default=CSVPath, help='Path of PickingResult')
    parser.add_argument('--ComNMO', type=int, default=0)
    parser.add_argument('--t0Int', type=list, default=[])
    parser.add_argument('--tInt', type=list, default=[])
    parser.add_argument('--CSVPath', type=str, default='')
    opt = parser.parse_args()
    MultiComNMOResult()
