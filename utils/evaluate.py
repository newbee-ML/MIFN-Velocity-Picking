import segyio
import os

from utils.PastProcess import GetSingleCurve
from utils.PlotTools import *
from utils.metrics import VMAE
import numpy as np

"""
Evaluate Processing for Network Training
"""

def GetResult(SegMat, t0Ind, vInd, threshold=0.1):
    VelCurve = []
    VelPick = []

    # resize the pred
    PredNew = []
    for i in range(SegMat.shape[0]):
        PredNew.append(resize_spectrum(SegMat[i].float(), (len(t0Ind), len(vInd[i]))))

    # get the picking velocity curve
    for i in range(len(PredNew)):
        try:
            VelCurveI, VelPickI = GetSingleCurve(PredNew[i], t0Ind, t0Ind, vInd[i], threshold=threshold)
            VelCurve.append(VelCurveI)
            VelPick.append(VelPickI)
        except RecursionError:
            VelCurve.append(np.array([]))
            VelPick.append(np.array([]))

    VelCurve = np.array(VelCurve)
    return VelCurve, VelPick


def GetPick(SegMat, RawPwr, t0Ind, vInd, threshold=0.1):
    VelPick, PredNew = [], []

    # resize the pred
    for i in range(SegMat.shape[0]):
        PredNew.append(resize_spectrum(SegMat[i].float(), (len(t0Ind), len(vInd[i]))))

    # get the picking velocity curve
    for i in range(len(PredNew)):
        _, VelPickI = GetSingleCurve(PredNew[i], RawPwr[i], t0Ind, t0Ind, vInd[i], threshold=threshold)
        VelPick.append(VelPickI)

    VelPick = np.array(VelPick)
    return VelPick


def EvaluateValid(net, DataLoader, criterion, opt, SegyDict, H5Dict, use_gpu=1):
    LossAvg = []
    VMAEAvg = []
    
    for i, (pwr, stkG, stkC, label, VMM, MC, name) in enumerate(DataLoader):
        if use_gpu:
            pwr = pwr.cuda(opt.GPUNO)
            label = label.cuda(opt.GPUNO)
            stkG = stkG.cuda(opt.GPUNO)
            stkC = stkC.cuda(opt.GPUNO)
        out, STKInfo = net(pwr, stkG, stkC, VMM)
        out, STKInfo = out.squeeze(), STKInfo.squeeze()
        PredSeg = out

        # compute loss
        loss = criterion(out.squeeze(), label)

        # load raw spectrum and v intervals
        RawPwr, VInt = [], []
        for n in name:
            PwrIndex = np.array(H5Dict['pwr'][n]['SpecIndex'])
            RawPwr.append(np.array(SegyDict['pwr'].trace.raw[PwrIndex[0]: PwrIndex[1]].T))
            VInt.append(np.array(SegyDict['pwr'].attributes(segyio.TraceField.offset)
                                 [PwrIndex[0]: PwrIndex[1]]))
        # get velocity curve
        AutoCurve, _ = GetResult(PredSeg, opt.t0Int, VInt, threshold=opt.Predthre)
        LossAvg.append(loss.item())
        MC = MC.numpy()
        mVMAE, _ = VMAE(AutoCurve, MC)
        VMAEAvg.append(mVMAE)
    try:
        return sum(LossAvg) / len(LossAvg), sum(VMAEAvg) / len(VMAEAvg)
    except ZeroDivisionError:
        return 1000, 1000
