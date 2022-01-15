import copy
import os
import sys

import h5py
import numpy as np
import segyio
from tqdm import tqdm

"""
Build H5 index file for segy dataset

from /segy/'vel.stk.sgy', 'vel.pwr.sgy', 'vel.gth.sgy' load sample index information
resave the index information to /h5File/'StkInfo.h5', 'SpecInfo.h5', 'GatherInfo.h5'
"""

def MKStkH5file(SegyPath, SavePath, MaxLength=500, minLength=160):
    # check save path
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)
    SavePath = os.path.join(SavePath, 'StkInfo.h5')

    # open h5 file and segy file
    h5File = h5py.File(SavePath, 'w')
    SegyFile = segyio.open(SegyPath, "r", strict=False)
    bar = tqdm(total=int(SegyFile.tracecount))
    start, end, length = 0, 0, 0
    while end < SegyFile.tracecount:
        start = copy.deepcopy(end)
        CDPIndex = np.array(SegyFile.attributes(segyio.TraceField.CDP)[start: (start+MaxLength)])
        LineIndex = np.array(SegyFile.attributes(segyio.TraceField.INLINE_3D)[start: (start+MaxLength)])
        if len(CDPIndex) < length:
            bar.update(len(CDPIndex))
            break
        elif len(CDPIndex) == length:
            length = len(CDPIndex)
        else:
            DiffCDP = np.where(CDPIndex[1:] - CDPIndex[:-1] != 0)[0]
            DiffLine = np.where(LineIndex[1:] - LineIndex[:-1] != 0)[0]
            length = DiffCDP[0] + 1 if len(DiffCDP) > 0 else DiffLine[0] + 1
        end = copy.deepcopy(start + length)
        if length > minLength:
            row_n = h5File.create_group('%d_%d' % (LineIndex[0], CDPIndex[0]))
            row_n.create_dataset('StkIndex', data=np.array([start, end]))
        bar.update(int(length))
    SegyFile.close()
    h5File.close()


def MKSpecH5file(SegyPath, SavePath, MaxLength=500):
    # check save path
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)
    SavePath = os.path.join(SavePath, 'SpecInfo.h5')
    # open h5 file and segy file
    h5File = h5py.File(SavePath, 'w')
    SegyFile = segyio.open(SegyPath, "r", strict=False)
    start, end, length = 0, 0, 0
    bar = tqdm(total=SegyFile.tracecount)
    while end < SegyFile.tracecount:
        start = copy.deepcopy(end)
        VelIndex = np.array(SegyFile.attributes(segyio.TraceField.offset)[start: (start+MaxLength)])
        if len(VelIndex) < length:
            bar.update(len(VelIndex))
            break
        elif len(VelIndex) == length:
            length = len(VelIndex)
        else:
            length = np.where(VelIndex[1:] - VelIndex[:-1] < 0)[0][0] + 1
        end = copy.deepcopy(start + length)
        cdp = SegyFile.attributes(segyio.TraceField.CDP)[start]
        line = SegyFile.attributes(segyio.TraceField.INLINE_3D)[start]
        row_n = h5File.create_group('%d_%d' % (line, cdp))
        row_n.create_dataset('SpecIndex', data=np.array([start, end]))
        bar.update(int(length))
    SegyFile.close()
    h5File.close()


def MKGatherH5file(SegyPath, SavePath, MaxLength=1000):
    # check save path
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)
    SavePath = os.path.join(SavePath, 'GatherInfo.h5')

    # open h5 file and segy file
    h5File = h5py.File(SavePath, 'w')
    SegyFile = segyio.open(SegyPath, "r", strict=False)
    start, end, length = 0, 0, 0
    bar = tqdm(total=SegyFile.tracecount)
    while end < SegyFile.tracecount:
        start = copy.deepcopy(end)
        OSIndex = np.array(SegyFile.attributes(segyio.TraceField.offset)[start: (start+MaxLength)])
        if len(OSIndex) < length:
            bar.update(len(OSIndex))
            break
        length = np.where(OSIndex[1:] - OSIndex[:-1] < 0)[0][0] + 1
        end = copy.deepcopy(start + length)
        cdp = SegyFile.attributes(segyio.TraceField.CDP)[start]
        line = SegyFile.attributes(segyio.TraceField.INLINE_3D)[start]
        row_n = h5File.create_group('%d_%d' % (line, cdp))
        row_n.create_dataset('GatherIndex', data=np.array([start, end]))
        bar.update(int(length))
    SegyFile.close()
    h5File.close()


if __name__ == '__main__':
    root_path = sys.argv[1]
    SegyList = ['vel.stk.sgy', 'vel.pwr.sgy', 'vel.gth.sgy']
    segy_path = [os.path.join(root_path, 'segy', path) for path in SegyList]
    h5_path = os.path.join(root_path, 'h5File')

    MKStkH5file(SegyPath=segy_path[0], SavePath=h5_path, minLength=50)
    MKSpecH5file(SegyPath=segy_path[1], SavePath=h5_path)
    MKGatherH5file(SegyPath=segy_path[2], SavePath=h5_path)

    # compare three data sets
    h5List = ['StkInfo.h5', 'SpecInfo.h5', 'GatherInfo.h5']
    h5file_path = [os.path.join(h5_path, path) for path in h5List]
    stk_h5_data = h5py.File(h5file_path[0], 'r')
    spec_h5_data = h5py.File(h5file_path[1], 'r')
    gather_h5_data = h5py.File(h5file_path[2], 'r')
    index_spec = set(spec_h5_data.keys())
    index_stk = set(stk_h5_data.keys())
    index_gather = set(gather_h5_data.keys())

    common_part = list((index_spec & index_stk) & index_gather)
    print('Common Part Number in Three Dataset: %d' % len(common_part))
    lengthSet = set()
    for key in stk_h5_data.keys():
        lengthSet.add(np.ptp(np.array(stk_h5_data[key]['StkIndex'])))
    print(lengthSet)

    LabelDict = np.load(os.path.join(root_path, 't_v_labels.npy'), allow_pickle=True).item()
    HaveLabelIndex = []
    for lineN in LabelDict.keys():
        for cdpN in LabelDict[lineN].keys():
            HaveLabelIndex.append('%s_%s' % (lineN, cdpN))

    # find common index
    Index = list((set(gather_h5_data.keys()) & set(stk_h5_data.keys())) &
                 (set(spec_h5_data.keys()) & set(HaveLabelIndex)))
    print(len(Index))
