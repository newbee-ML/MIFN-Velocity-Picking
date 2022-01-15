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
def GetSingleCurve(seg_mat, t_ind, t0_ind, v_ind, threshold=0.5, SampleRate=0.3, k=0):
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
    SelectInd = random.sample(list(SelectInd), int(len(SelectInd)*SampleRate))
    SelectInd = np.array(sorted(SelectInd))
    SelectSeg = SegMat[SelectInd, :]
    SelectVel = np.argmax(SelectSeg, axis=1)
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