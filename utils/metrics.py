import numpy as np


# ===== compute the VMAE ======
def VMAE(AutoCurve, ManualCurve):
    VMAEList = []
    for i in range(len(AutoCurve)):
        if AutoCurve[i].shape[0] == ManualCurve[i].shape[0]:
            VMAEList.append(np.mean(np.abs(AutoCurve[i][:, 1] - ManualCurve[i][:, 1])))
        else:
            VMAEList.append(np.nan)
    try:
        return np.nanmean(VMAEList), VMAEList
    except ZeroDivisionError:
        return 1000, [1000]*len(AutoCurve)
