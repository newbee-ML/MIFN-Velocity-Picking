import os

import numpy as np
import segyio
import h5py
import torch
import torch.utils.data as data
from scipy import interpolate
from sklearn.linear_model import LinearRegression
from torchvision import transforms

from utils.SpecEnhanced import spec_enhance

"""
Loading Sample Data from segy, h5, npy file
"""

# -------- build ground truth figure ------------------------------
# interpolation function
def linear_predict(x, y, predict_x):
    x_in = np.array(x).reshape(-1, 1)
    y_in = np.array(y).reshape(-1, 1)
    l_reg = LinearRegression()
    l_reg.fit(x_in, y_in)
    y_prd = l_reg.predict(predict_x)
    return y_prd


# make ground truth curve
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


# make the ground truth matrix from curve
def curve2matrix(curve, t0_vec, v_vec, type_generator='hard'):
    # init the matrix
    mat_lab = np.zeros((len(t0_vec), len(v_vec))).astype(np.float32)

    # type 1 hard label
    if type_generator == 'hard':
        for t, v in curve:
            v_ind = np.argmin(np.abs(v_vec - v))
            mat_lab[np.argmin(np.abs(t0_vec - t)), (v_ind - 1): (v_ind + 2)] = 1.0

    # type 2 soft label
    else:
        for t, v in curve:
            v_ind = np.argmin(np.abs(v_vec - v))
            mat_lab[np.argmin(np.abs(t0_vec - t)), v_ind] = 1.0
            for i in range(2):
                try:
                    mat_lab[np.argmin(np.abs(t0_vec - t)), v_ind - (i + 1)] = 1 / (i + 2)
                except IndexError:
                    pass
                try:
                    mat_lab[np.argmin(np.abs(t0_vec - t)), v_ind + (i + 1)] = 1 / (i + 2)
                except IndexError:
                    pass

    return mat_lab


# the main func of making the ground truth matrix
def make_label_main(label_vector, spec, t0_vec, v_vec, type_generator='hard'):
    curve_lab = interpolation(label_vector, t0_vec, v_vec)
    mat_lab = curve2matrix(curve_lab, t0_vec, v_vec, type_generator)
    spec_stack_t = np.sum(spec, axis=1)
    if np.max(spec_stack_t) > 0:
        spec_stack_t /= np.max(spec_stack_t)
    mat_lab[spec_stack_t < 0.01, :] = 0
    return mat_lab, curve_lab


# -------- scale the image function -------------------------------
# scale the image (ToPILImage -> Resize -> ToTensor)
def scale_img(spectrum, resize_n):
    spectrum = transforms.ToPILImage()(spectrum)
    resize_spec = transforms.Resize(size=resize_n)(spectrum)
    tensor_spec = transforms.ToTensor()(resize_spec)
    return tensor_spec


# -------- split stk data ---------------------------------------
def SplitGatherCV(StkData):
    gather = []
    cv = []
    interval, k = 0, 0
    # find the interval
    for i in range(50):
        RowN = StkData[:, i]
        UpZero = np.where(RowN < 0)[0]
        if len(UpZero) == 0:
            interval = i
            break

    while k < StkData.shape[1]:
        Gstart, Gend = k, k + interval
        gather.append(StkData[:, Gstart: Gend])
        k += interval
        cv.append(StkData[np.newaxis, StkData[:, k+1] > 0, k: (k + 2)])
        k += 2
    gather = np.array(gather).astype(np.float32)
    cv = np.vstack(cv).astype(np.float32)

    return gather, cv


# -------- load data from segy, h5 and label.npy ------------------
def LoadSingleData(SegyDict, H5Dict, LabelDict, index, mode='train', LoadG=True):
    # data dict
    DataDict = {}
    PwrIndex = np.array(H5Dict['pwr'][index]['SpecIndex'])
    StkIndex = np.array(H5Dict['stk'][index]['StkIndex'])
    line, cdp = index.split('_')

    DataDict.setdefault('spectrum', np.array(SegyDict['pwr'].trace.raw[PwrIndex[0]: PwrIndex[1]].T))
    DataDict.setdefault('vInt', np.array(SegyDict['pwr'].attributes(segyio.TraceField.offset)[PwrIndex[0]: PwrIndex[1]]))
    if LoadG:
        StkData = np.array(SegyDict['stk'].trace.raw[StkIndex[0]: StkIndex[1]].T)
        gather, cv = SplitGatherCV(StkData)
        if cv.shape[1] < 200:
            cvN = np.zeros((cv.shape[0], 200, 2))
            for k in range(cv.shape[0]):
                cvN[k, 0: cv.shape[1], :] = cv[k]
            cv = cvN
        DataDict.setdefault('stkG', gather)
        DataDict.setdefault('stkC', cv)

    if mode == 'train':
        try:
            DataDict.setdefault('label', np.array(LabelDict[int(line)][int(cdp)]))
        except KeyError:
            DataDict.setdefault('label', np.array(LabelDict[str(line)][str(cdp)]))

    return DataDict


# ------------ generate feature map -----------------------
def generate_feature_map(ori_spec, resize, visual=0):
    ParaDict = {
        'ws': [5, 5, 5, 10, 10, 10, 15, 15, 15],
        'st': [1, 2, 3, 1, 2, 3, 1, 2, 3],
        'eec': [1, 1.5, 1.8, 1, 1.5, 1.8, 1, 1.5, 1.8],
        'ln': [5, 8, 12, 5, 8, 12, 5, 8, 12]
    }
    feature_map = [scale_img(ori_spec, resize_n=resize)]
    for i in range(len(ParaDict['ws'])):
        if visual and i == 1:
            _, EnDict = spec_enhance(ori_spec, smooth_ws=ParaDict['ws'][i],
                                             smooth_num=ParaDict['st'][i],
                                             norm_num=ParaDict['ln'][i],
                                             exp_num=ParaDict['eec'][i], save=1)
            return EnDict
        else:
            fea_map_n = spec_enhance(ori_spec, smooth_ws=ParaDict['ws'][i],
                                    smooth_num=ParaDict['st'][i],
                                    norm_num=ParaDict['ln'][i],
                                    exp_num=ParaDict['eec'][i])
        fea_map_scaled = scale_img(fea_map_n, resize_n=resize)
        feature_map.append(fea_map_scaled)
    feature_map = torch.cat(feature_map, dim=0)
    return feature_map


# Padding for stk gather 
def STKPadding(StkGather, GthLen):
    K, H, Olen = StkGather.shape
    PaddingNew = np.zeros((K, H, GthLen), dtype=np.float32)
    PadNum = int((GthLen - Olen)/2)
    for i in range(K):
        if (GthLen - Olen) % 2 != 0:
            PaddingNew[i][:, PadNum: -(PadNum+1)] = StkGather[i]
        else:
            PaddingNew[i][:, PadNum: -PadNum] = StkGather[i]
    return PaddingNew


# -------- pytorch dataset iterator --------------------------------
"""
Load Data: 
    FM: multi-scale observe feature maps of velocity spectrum      | shape = 10 * H * W
    stkG: stk data -- gather                                       | shape =  K * t * W'
    stkC: stk data -- t-v points (reference velocity)              | shape =  K * N * 2
    mask: segmentation ground truth                                | shape =  H * W
    VMM:  minimum and maximum of velocity                          | shape = BS * 2
    ManualCurve: manual picking t-v points results                 | shape =  N'* 2
    self.index[index]: sample index                                | string
"""
class DLSpec(data.Dataset):
    def __init__(self, SegyDict, H5Dict, LabelDict, index, t0Ind,
                 label_type='soft', resize=(432, 256),
                 GatherLen=21, mode='train'):
        self.SegyDict = SegyDict
        self.LabelDict = LabelDict
        self.H5Dict = H5Dict
        self.index = index
        self.t0Ind = t0Ind
        self.label_type = label_type
        self.resize = resize
        self.index_list = self.index
        self.GatherLen = GatherLen
        self.mode = mode

    def __getitem__(self, index):
        LoadStk, LoadMS = True, True 

        # load the data from segy and h5 file
        DataDict = LoadSingleData(self.SegyDict, self.H5Dict, self.LabelDict,
                                  self.index[index], self.mode, LoadG=LoadStk)
        VMM = np.array([DataDict['vInt'][0], DataDict['vInt'][-1]])

        # make mask
        if 'label' in DataDict.keys():
            mask, ManualCurve = make_label_main(DataDict['label'], DataDict['spectrum'], self.t0Ind, DataDict['vInt'], self.label_type)
            mask = scale_img(mask, resize_n=self.resize).squeeze()
        else:
            mask, ManualCurve = 0, 0

        if LoadStk:  # loading stk data
            # padding the gather
            if DataDict['stkG'].shape[-1] < self.GatherLen:
                DataDict['stkG'] = STKPadding(DataDict['stkG'], self.GatherLen)
            stkG, stkC = DataDict['stkG'], DataDict['stkC']
        else:
            stkG, stkC = 0, 0

        if LoadMS:
            FM = generate_feature_map(DataDict['spectrum'], self.resize)
        else:
            FM = scale_img(DataDict['spectrum'], resize_n=self.resize)

        return FM, stkG, stkC, mask, VMM, ManualCurve, self.index[index]

    def __len__(self):
        return len(self.index)


def PredictSingleLoad(DataDict, t0Ind, resize, GatherLen):
    VMM = np.array([DataDict['vInt'][0], DataDict['vInt'][-1]])[np.newaxis, ...]
    # make mask
    mask, ManualCurve = make_label_main(DataDict['label'], DataDict['spectrum'], t0Ind, DataDict['vInt'], 'soft')
    mask = scale_img(mask, resize_n=resize).squeeze()

    # padding the gather
    if DataDict['stkG'].shape[-1] < GatherLen:
        DataDict['stkG'] = STKPadding(DataDict['stkG'], GatherLen)
    stkG, stkC = DataDict['stkG'], DataDict['stkC']
    FM = generate_feature_map(DataDict['spectrum'], resize)[np.newaxis, ...]
    stkG, stkC = torch.tensor(stkG[np.newaxis, ...]), torch.tensor(stkC[np.newaxis, ...])
    return FM, stkG, stkC, mask, VMM, ManualCurve


# --------- get data dict ------------------
def GetDataDict(DataSetPath):
    # load segy data
    SegyName = {'pwr': 'vel.pwr.sgy',
                'stk': 'vel.stk.sgy',
                'gth': 'vel.gth.sgy'}
    SegyDict = {}
    for name, path in SegyName.items():
        SegyDict.setdefault(name, segyio.open(os.path.join(DataSetPath, 'segy', path), "r", strict=False))
    # load h5 file
    H5Name = {'pwr': 'SpecInfo.h5',
              'stk': 'StkInfo.h5',
              'gth': 'GatherInfo.h5'}
    H5Dict = {}
    for name, path in H5Name.items():
        H5Dict.setdefault(name, h5py.File(os.path.join(DataSetPath, 'h5File', path), 'r'))
    t0Int = np.array(SegyDict['pwr'].samples)

    # load label.npy
    LabelDict = np.load(os.path.join(DataSetPath, 't_v_labels.npy'), allow_pickle=True).item()

    return SegyDict, H5Dict, LabelDict
