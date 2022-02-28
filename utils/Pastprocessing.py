"""
Past-processing for spectrum picking results

Author: Hongtao Wang | stolzpi@163.com
"""


import numpy as np
from scipy import interpolate
from sklearn import linear_model



################################################################
# Tool A: Interpolation for velocity points
################################################################
def interpolation(label_point, t_interval, v_interval=None):
    # sort the label points
    label_point = np.array(sorted(label_point, key=lambda t_v: t_v[0]))

    # ensure the input is int
    t0_vec = np.array(t_interval).astype(int)

    # get the ground truth curve using interpolation
    peaks_selected = np.array(label_point)
    func = interpolate.interp1d(peaks_selected[:, 0], peaks_selected[:, 1], kind='linear', fill_value="extrapolate")
    y = func(t0_vec)
    if v_interval is not None:
        v_vec = np.array(v_interval).astype(int) 
        y = np.clip(y, v_vec[0], v_vec[-1])

    return np.hstack((t0_vec.reshape((-1, 1)), y.reshape((-1, 1))))

def interpolation2(LabelPoints, tVec, vVec=None, VelRef=None):
    # sort the label points
    LabelPoints = np.array(sorted(LabelPoints, key=lambda t_v: t_v[0]))

    # clip the tVec
    LabelTmin, LabelTmax = np.min(LabelPoints[:, 0]), np.max(LabelPoints[:, 0])
    UpIndex, DownIndex = np.where(tVec <= LabelTmin)[0], np.where(tVec >= LabelTmax)[0]
    MedianIndex = sorted(list((set(np.arange(len(tVec))) - set(UpIndex)) - set(DownIndex)))

    # ensure the input is int
    tVec = np.array(tVec).astype(int)
    tVecUp, tVecDown, tVecMed = tVec[UpIndex], tVec[DownIndex], tVec[MedianIndex]

    # get the ground truth curve using interpolation
    PeakSelected = np.array(LabelPoints)
    func = interpolate.interp1d(PeakSelected[:, 0], PeakSelected[:, 1], kind='linear')
    vMed = func(tVecMed)

    if VelRef is not None:
        # up part
        RefUpPart = VelRef[VelRef[:, 0] <= LabelTmin, :]
        modelUp = linear_model.LinearRegression()
        modelUp.fit(RefUpPart[:, 0].reshape(-1, 1), RefUpPart[:, 1]) 
        modelUp.intercept_ = vMed[0] - modelUp.coef_.item()*tVecMed[0]
        vUp = modelUp.predict(tVecUp.reshape(-1, 1))
        # down part
        RefDownPart = VelRef[VelRef[:, 0] >= LabelTmax, :]
        modelDown = linear_model.LinearRegression()
        modelDown.fit(RefDownPart[:, 0].reshape(-1, 1), RefDownPart[:, 1]) 
        modelDown.intercept_ = vMed[-1] - modelDown.coef_.item()*tVecMed[-1]
        vDown = modelDown.predict(tVecDown.reshape(-1, 1))
    else:
        vUp = np.ones_like(tVecUp) * tVecMed[0]
        vDown = np.ones_like(tVecDown) * tVecMed[-1]
    
    vPred = np.concatenate((vUp, vMed, vDown))

    
    if vVec is not None:
        vVec = np.array(vVec).astype(int) 
        vPred = np.clip(vPred, vVec[0], vVec[-1])

    return np.hstack((tVec.reshape(-1, 1), vPred.reshape(-1, 1)))


################################################################
# Tool B: NMO correction
################################################################
# Calculate travel time
def TravelTime(t0, x, vNMO):
    return np.sqrt(t0 ** 2 + (x / vNMO * 1000) ** 2)


# main function 
def NMOCorr(CMPGather, tVec, OffsetVec, VNMO, CutC=120):
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
