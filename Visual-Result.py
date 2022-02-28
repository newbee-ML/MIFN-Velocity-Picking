"""
This python file is for single prediction
Visual the following plots:
- original spectrum
- segmentation map
- SGS gather input map
"""


##########################
# Import Settings 
##########################
import argparse
import os
import warnings

import h5py
import numpy as np
import pandas as pd
import segyio
import torch

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils.evaluate import GetResult
from net.MIFNet import MultiInfoNet
from utils.LoadData import LoadSingleData, PredictSingleLoad
from utils.PlotTools import *

warnings.filterwarnings("ignore")

#######################
# Base Setting
#######################
parser = argparse.ArgumentParser()
parser.add_argument('--DataSet', type=str, default='hade', help='Dataset Root Path')
parser.add_argument('--DataSetRoot', type=str, help='Dataset Root Path')
parser.add_argument('--GatherLen', type=int, default=21)
parser.add_argument('--LoadModel', type=str, help='Model Path')
parser.add_argument('--SGSMode', type=str, default='all')
parser.add_argument('--Predthre', type=float, default=0.3)
parser.add_argument('--SeedRate', type=float, default=0.5)
parser.add_argument('--Resave', type=int, default=0)
parser.add_argument('--GPUNO', type=int, default=0)
parser.add_argument('--Resize', type=list, help='CropSize')
parser.add_argument('--SizeH', type=int, default=256, help='Size Height')
parser.add_argument('--t0Int', type=list, default=[])
opt = parser.parse_args(args=[])
# setting
opt.DataSetRoot = 'E:\\Spectrum\\hade'
opt.LoadModel = 'F:\\VSP-MIFN\\0Ablation\\hade\\model\\DS_hade-SR_0.80-LR_0.0010-BS_8-SH_256-PT_0.10-SGSM_mute.pth'
opt.Predthre = 0.3
opt.SeedRate = 0.5
RootSave = os.path.join('F:/', 'VSP-MIFN', '1VisualResult')

########################
# Show the test index 
########################
# load segy data
SegyName = {'pwr': 'vel.pwr.sgy',
            'stk': 'vel.stk.sgy',
            'gth': 'vel.gth.sgy'}
SegyDict = {}
for name, path in SegyName.items():
    SegyDict.setdefault(name, segyio.open(os.path.join(opt.DataSetRoot, 'segy', path), "r", strict=False))
# load h5 file
H5Name = {'pwr': 'SpecInfo.h5',
            'stk': 'StkInfo.h5',
            'gth': 'GatherInfo.h5'}
H5Dict = {}
for name, path in H5Name.items():
    H5Dict.setdefault(name, h5py.File(os.path.join(opt.DataSetRoot, 'h5File', path), 'r'))

# load label.npy
LabelDict = np.load(os.path.join(opt.DataSetRoot, 't_v_labels.npy'), allow_pickle=True).item()
HaveLabelIndex = []
for lineN in LabelDict.keys():
    for cdpN in LabelDict[lineN].keys():
        HaveLabelIndex.append('%s_%s' % (lineN, cdpN))
pwr_index = set(H5Dict['pwr'].keys())
stk_index = set(H5Dict['stk'].keys())
gth_index = set(H5Dict['gth'].keys())

# find common index
Index = sorted(list((pwr_index & stk_index) & (gth_index & set(HaveLabelIndex))))
trainIndex, testIndex = train_test_split(Index, test_size=0.2, random_state=123)
trainIndex, _ = train_test_split(trainIndex, test_size=1-opt.SeedRate, random_state=123)

# Print test list
print('Predict Number', len(testIndex), '\n', testIndex)

########################
# Load Model 
########################
# Load MIFN
opt.t0Int = np.array(SegyDict['pwr'].samples)
opt.Resize = [int(opt.SizeH), int(opt.SizeH/2)]
net = MultiInfoNet(opt.t0Int, opt, in_channels=11, resize=opt.Resize)
use_gpu = torch.cuda.device_count() > 0
if use_gpu:
    net = net.cuda(opt.GPUNO)
net.eval()
# Load Model
net.load_state_dict(torch.load(opt.LoadModel)['Weights'])

#######################
# Predict Single One 
#######################
index = '2800_2040'
# Load data
DataDict = LoadSingleData(SegyDict, H5Dict, LabelDict, index, mode='train', LoadG=True)
VInt = DataDict['vInt']
FM, stkG, stkC, mask, VMM, ManualCurve = PredictSingleLoad(DataDict, opt.t0Int, opt.Resize, opt.GatherLen)
if use_gpu:
    pwr = FM.cuda(opt.GPUNO)
    stkG = stkG.cuda(opt.GPUNO)
    stkC = stkC.cuda(opt.GPUNO)

out, STKInfo = net(pwr, stkG, stkC, VMM)
PredSeg, STKInfo = out.squeeze(), STKInfo.squeeze()
print('the shape of output probability map is', list(PredSeg.shape))

# plot original spectrum
FeatureMap(FM[:, 0, ...].cpu().detach().numpy(), SavePath=os.path.join(RootSave, '%s-OriginPwr' % index))
# plot prediction
FeatureMap(PredSeg.cpu().detach().numpy(), SavePath=os.path.join(RootSave, '%s-SegMap' % index))
# plot stk gather
FeatureMap(stkG[:, 1, ...].cpu().detach().numpy(), cmap='Gray', SavePath=os.path.join(RootSave, '%s-StkGather' % index))