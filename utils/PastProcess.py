import numpy as np
from scipy import interpolate
from sklearn import linear_model
from utils.LoadData import interpolation

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

def GetResult(SegMat, t0Ind, vInd, threshold=0.1, PostProcessing=1):
    # 1 get the high probability points 
    Peaks = np.array([GetHighProb(SegMap, threshold) for SegMap in SegMat])

    # 2 transfer the scale
    ScaledPeaks = []  
    for ind, Peak in enumerate(Peaks):
        _, H, W = SegMat.shape
        t0IndN, vIndN = np.linspace(t0Ind[0], t0Ind[-1], H), np.linspace(vInd[ind][0], vInd[ind][-1], W)
        ScaledT, ScaledV = scale_change(Peak[:, 0], Peak[:, 1], t0IndN, vIndN, 'up')
        ScaledPeaks.append(np.array([ScaledT, ScaledV]).T)
    ScaledPeaks = np.array(ScaledPeaks)

    # 3 remove the outliers
    FinalPeaks = ScaledPeaks

    # 4 interpolate & regression
    if PostProcessing:
        Curve = np.array([interpolation2(FinalPeak, t0Ind, vInd[ind], RefRange=300) for ind, FinalPeak in enumerate(FinalPeaks)])
    else:
        Curve = np.array([interpolation(FinalPeak, t0Ind, vInd[ind]) for ind, FinalPeak in enumerate(FinalPeaks)])

    # # 5 Get the key points
    # ResModel = KeyPoints(Curve)

    return Curve, FinalPeaks


# # extract the key points
# def KeyPoints(Curve):
#     my_pwlf = pwlf.PiecewiseLinFit(Curve[:, 0], Curve[:, 1])
#     # fit the data for four line segments
#     res = my_pwlf.fit(6)
#     return res


# get the points with high probability on the segmentation map
def GetHighProb(SegMat, threshold=0.1):
    # scale the seg map to [0, 1]
    SegMat = (SegMat-np.min(SegMat))/np.ptp(SegMat)
    # save the points with high probability
    stack_energy = np.max(SegMat, axis=1)
    SelectInd = np.where(stack_energy > threshold)[0]
    SelectInd = np.array(sorted(SelectInd))
    SelectSeg = SegMat[SelectInd, :]
    # find the maximum of each row
    MaxValue = np.max(SelectSeg, axis=1)
    SelectVel = []
    for ind, max_val in enumerate(MaxValue):
        MaxIndex = np.where(SelectSeg[ind, :]==max_val)[0]
        if len(MaxIndex) > 1:
            SelectVel.append(np.mean(MaxIndex))
        else:
            SelectVel.append(MaxIndex.item())
    SelectVel = np.array(SelectVel)
    Peaks = np.hstack((SelectInd.reshape(-1, 1), SelectVel.reshape(-1, 1)))
    return Peaks


def interpolation2(LabelPoints, tVec, vVec=None, RefRange=300):
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

    # up part
    UpRefPoints = LabelPoints[LabelPoints[:, 0] < LabelPoints[0, 0] + RefRange, :]
    modelUp = linear_model.LinearRegression()
    modelUp.fit(UpRefPoints[:, 0].reshape(-1, 1), UpRefPoints[:, 1]) 
    modelUp.intercept_ = vMed[0] - modelUp.coef_.item()*tVecMed[0]
    vUp = modelUp.predict(tVecUp.reshape(-1, 1))

    # down part
    DownRefPoints = LabelPoints[LabelPoints[:, 0] > LabelPoints[-1, 0] - RefRange, :]
    modelDown = linear_model.LinearRegression()
    modelDown.fit(DownRefPoints[:, 0].reshape(-1, 1), DownRefPoints[:, 1]) 
    modelDown.intercept_ = vMed[-1] - modelDown.coef_.item()*tVecMed[-1]
    vDown = modelDown.predict(tVecDown.reshape(-1, 1))
    
    vPred = np.concatenate((vUp, vMed, vDown))

    
    if vVec is not None:
        vVec = np.array(vVec).astype(int) 
        vPred = np.clip(vPred, vVec[0], vVec[-1])

    return np.hstack((tVec.reshape(-1, 1), vPred.reshape(-1, 1)))


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
