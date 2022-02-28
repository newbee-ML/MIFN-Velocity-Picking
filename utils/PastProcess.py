import numpy as np
from utils.LoadData import interpolation
import random
import copy

"""
Past-processing for segmentation map

 - Get the picking result from the segmentation map
"""

# ------------- scale data points -------------------------
# scale the data axis to bigger or smaller
def scale_change(x, y, x_ind, y_ind, method):

    if method == 'down':
        x_result = (x - x_ind[0]) / (x_ind[1] - x_ind[0])
        y_result = (y - y_ind[0]) / (y_ind[1] - y_ind[0])
        return x_result, y_result

    elif method == 'up':
        x_result = x_ind[0] + x * (x_ind[1] - x_ind[0])
        y_result = y_ind[0] + y * (y_ind[1] - y_ind[0])
        return x_result, y_result

    else:
        print('error type input!')


# ---------- Pick the trend peaks ------------
# get the velocity curve from the segmentation matrix
def GetSingleCurve(seg_mat, t_ind, t0_ind, v_ind, threshold=0.1, k=0):
    # scale SegMat and find the index up threshold
    SegMat = copy.deepcopy(seg_mat).astype(np.float)
    SegMat = (SegMat-np.min(SegMat))/np.ptp(SegMat)
    stack_energy = np.max(SegMat, axis=1)
    # EnergyHist(stack_energy)  # plot the distribution

    SelectInd = np.where(stack_energy > threshold)[0]
    if k > 10:
        return np.array([]), np.array([])
    if len(SelectInd) < 10:
        return GetSingleCurve(seg_mat, t_ind, t0_ind, v_ind, threshold/2, k=k+1)
    SelectInd = np.array(sorted(SelectInd))
    SelectSeg = SegMat[SelectInd, :]
    # find the maximum of each row
    MaxValue = np.max(SelectSeg, axis=1)
    SelectVel = []
    for ind, max_val in enumerate(MaxValue):
        MaxIndex = np.where(SelectSeg[ind, :]==max_val)[0]
        if len(MaxIndex) > 1:
            SelectVel.append(int(np.mean(MaxIndex)))
        else:
            SelectVel.append(int(MaxIndex))
    SelectVel = np.array(SelectVel)
    trend_peaks = np.hstack((SelectInd.reshape(-1, 1), SelectVel.reshape(-1, 1)))
        
    if np.nan in seg_mat:
        return np.array([]), np.array([])

    if len(trend_peaks) < 10:
        return GetSingleCurve(seg_mat, t_ind, t0_ind, v_ind, threshold/2)

    # scale the t-v points to original scale
    t_real, v_real = scale_change(trend_peaks[:, 0], trend_peaks[:, 1], t0_ind, v_ind, 'up')
    auto_peaks = np.array((t_real, v_real)).T

    # interpolate the peaks to velocity curve
    auto_curve = interpolation(auto_peaks, t_ind, v_ind)

    return auto_curve, auto_peaks


# ---------- NMO Correction ------------
# Calculate travel time
def TravelTime(t0, x, vNMO):
    return np.sqrt(t0 ** 2 + (x / vNMO * 1000) ** 2)


# main function 
def NMOCorr(CMPGather, tVec, OffsetVec, VNMO, CutC=1.2):
    nmo = np.zeros_like(CMPGather)
    tMin, dt = tVec[0], tVec[1] - tVec[0]
    for j, x in enumerate(OffsetVec):
        TravelT = TravelTime(tVec, x, VNMO)
        ChangeS = TravelT / (tVec + 1)
        # invert to the t0 index
        t0Index = ((TravelT - tMin) / dt).astype(np.int32)
        RevertIndex = np.where(t0Index < len(tVec))[0]
        SaveIndex = np.where(ChangeS < CutC)[0]
        SaveIndex = list(set(SaveIndex) & set(RevertIndex))
        nmo[SaveIndex, j] = CMPGather[t0Index[SaveIndex], j]
    return nmo
