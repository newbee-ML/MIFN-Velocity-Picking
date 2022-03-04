import segyio
import os

from utils.PastProcess import GetResult
from utils.PlotTools import *
from utils.metrics import VMAE
import numpy as np

"""
Evaluate Processing for Network Training
"""


def EvaluateValid(net, DataLoader, criterion, SegyDict, H5Dict, t0Int, Predthre, device=0):
    LossAvg = []
    VMAEAvg = []
    
    for i, (pwr, stkG, stkC, label, VMM, MC, name) in enumerate(DataLoader):
        if device is not 'cpu':
            pwr = pwr.cuda(device)
            label = label.cuda(device)
            stkG = stkG.cuda(device)
            stkC = stkC.cuda(device)
        out, _ = net(pwr, stkG, stkC, VMM)
        PredSeg = out.squeeze()

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
        AutoCurve, _ = GetResult(PredSeg.cpu().numpy(), t0Int, VInt, threshold=Predthre)
        LossAvg.append(loss.item())
        MC = MC.numpy()
        mVMAE, _ = VMAE(AutoCurve, MC)
        VMAEAvg.append(mVMAE)
    try:
        return sum(LossAvg) / len(LossAvg), sum(VMAEAvg) / len(VMAEAvg)
    except ZeroDivisionError:
        return 1000, 1000
