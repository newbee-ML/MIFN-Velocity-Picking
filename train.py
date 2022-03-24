"""
The main file for train MIFN
Author: Hongtao Wang | stolzpi@163.com
"""
from ast import Raise
import sys
import h5py
import numpy as np
import pandas as pd
import torch
import segyio
import os
import copy
import random
import shutil
import warnings
import argparse
import torch.nn as nn
from net.AblationNet import MixNet
from torch.utils.data import DataLoader
from utils.LoadData import DLSpec
from utils.logger import MyLog
from torch.optim.lr_scheduler import MultiStepLR
from utils.evaluate import EvaluateValid
from tensorboardX import SummaryWriter


sys.path.append('..')
warnings.filterwarnings("ignore")


"""
Initialize the folder
"""

def CheckSavePath(opt, BaseName):
    basicFile = ['log', 'model', 'TBLog']
    SavePath = os.path.join(opt.OutputPath, BaseName)
    if opt.ReTrain:
        if os.path.exists(SavePath):
            shutil.rmtree(SavePath)
        if not os.path.exists(SavePath):
            for file in basicFile:
                Path = os.path.join(SavePath, file)
                os.makedirs(Path)

"""
Save the training parameters
"""
def SaveParameters(opt, BaseName):
    ParaDict = opt.__dict__
    ParaDict = {key: [value] for key, value in ParaDict.items()}
    ParaDF = pd.DataFrame(ParaDict)
    ParaDF.to_csv(os.path.join(opt.OutputPath, BaseName, 'TrainPara.csv'))


"""
Get the hyper parameters
"""
def GetTrainPara():
    parser = argparse.ArgumentParser()
    parser.add_argument('--DataSetRoot', type=str, default='E:\\Spectrum', help='Dataset Root Path')
    parser.add_argument('--DataSet', type=str, default='hade', help='Dataset Root Path')
    parser.add_argument('--EpName', type=str, default='Ep-100', help='The index of the experiment')
    parser.add_argument('--OutputPath', type=str, default='F:\\VelocitySpectrum\\MIFN\\2GeneraTest', help='Path of Output')
    parser.add_argument('--SGSMode', type=str, default='mute')
    parser.add_argument('--NetType', type=str, default='all')
    parser.add_argument('--GatherLen', type=int, default=15)
    parser.add_argument('--RepeatTime', type=int, default=0)
    parser.add_argument('--SeedRate', type=float, default=1)
    parser.add_argument('--ReTrain', type=int, default=1)
    parser.add_argument('--GPUNO', type=int, default=0)
    parser.add_argument('--SizeH', type=int, default=256, help='Size Height')
    parser.add_argument('--SizeW', type=int, default=128, help='Size Width')
    parser.add_argument('--Predthre', type=float, default=0.1)
    parser.add_argument('--MaxIter', type=int, default=10000, help='max iteration')
    parser.add_argument('--SaveIter', type=int, default=30, help='checkpoint each SaveIter')
    parser.add_argument('--MsgIter', type=int, default=2, help='log the loss each MsgIter')
    parser.add_argument('--lrStart', type=float, default=0.001, help='the beginning learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', help=r"the optimizer of training, 'adam' or 'sgd'")
    parser.add_argument('--PretrainModel', type=str, help='The path of pretrain model to train (Path)')
    parser.add_argument('--trainBS', type=int, default=32, help='The batchsize of train')
    parser.add_argument('--valBS', type=int, default=16, help='The batchsize of valid')
    opt = parser.parse_args()
    return opt
    

"""
Main train function
"""

def train(opt):
    ####################
    # base setting
    ####################
    BaseName = opt.EpName
    # data set path setting
    DataSetPath = os.path.join(opt.DataSetRoot, opt.DataSet)
    # check output folder and check path
    CheckSavePath(opt, BaseName)
    TBPath = os.path.join(opt.OutputPath, BaseName, 'TBLog')
    writer = SummaryWriter(TBPath)
    BestPath = os.path.join(opt.OutputPath, BaseName, 'model', 'Best.pth')
    LogPath = os.path.join(opt.OutputPath, BaseName, 'log')
    logger = MyLog(BaseName, LogPath)
    logger.info('%s start to train ...' % BaseName)
    # save the train parameters to csv
    SaveParameters(opt, BaseName)

    #######################################
    # load data from segy, H5file and npy
    #######################################

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

    # load label.npy
    LabelDict = np.load(os.path.join(DataSetPath, 't_v_labels.npy'), allow_pickle=True).item()
    HaveLabelIndex = []
    for lineN in LabelDict.keys():
        for cdpN in LabelDict[lineN].keys():
            HaveLabelIndex.append('%s_%s' % (lineN, cdpN))
    pwr_index = set(H5Dict['pwr'].keys())
    stk_index = set(H5Dict['stk'].keys())
    gth_index = set(H5Dict['gth'].keys())

    #########################################
    # split the train, valid and test set
    #########################################
    Index = sorted(list((pwr_index & stk_index) & (gth_index & set(HaveLabelIndex))))
    IndexDict = {}
    for index in Index:
        line, cdp = index.split('_')
        IndexDict.setdefault(int(line), [])
        IndexDict[int(line)].append(int(cdp))
    LineIndex = sorted(list(IndexDict.keys()))
    # use the last 20% for test set
    LastSplit1, LastSplit2 = int(len(LineIndex)*0.6), int(len(LineIndex)*0.8)
    # use the first sr% (seed rate) for train set and the other for valid set
    MedSplit = int(LastSplit1*opt.SeedRate)
    trainLine, validLine, testLine = LineIndex[:MedSplit], LineIndex[LastSplit1: LastSplit2], LineIndex[LastSplit2:]
    logger.info('There are %d lines, using for training: ' % len(trainLine) + ','.join(map(str, trainLine)))
    logger.info('There are %d lines, using for valid: ' % len(validLine) + ','.join(map(str, validLine)))
    logger.info('There are %d lines, using for test: ' % len(testLine) + ','.join(map(str, testLine)))
    trainIndex, validIndex = [], []
    for line in trainLine:
        for cdp in IndexDict[line]:
            trainIndex.append('%d_%d' % (line, cdp))
    for line in validLine:
        for cdp in IndexDict[line]:
            validIndex.append('%d_%d' % (line, cdp))
    random.seed(123)
    VisualSample = random.sample(trainIndex, 16)
    print('Train Num %d, Valid Num %d' % (len(trainIndex), len(validIndex)))
    ##################################
    # build the data loader
    ##################################
    # load t0 ind
    t0Int = np.array(SegyDict['pwr'].samples)
    resize = [opt.SizeH, opt.SizeW]
    # build data loader
    ds = DLSpec(SegyDict, H5Dict, LabelDict, trainIndex, t0Int, resize=resize, GatherLen=opt.GatherLen)
    dsval = DLSpec(SegyDict, H5Dict, LabelDict, validIndex, t0Int, resize=resize, GatherLen=opt.GatherLen)
    dl = DataLoader(ds,
                    batch_size=opt.trainBS,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=False,
                    drop_last=False)
    dlval = DataLoader(dsval,
                       batch_size=opt.valBS,
                       shuffle=False,
                       num_workers=0,
                       drop_last=False)

    ###################################
    # load the network
    ###################################

    # check gpu is available
    if torch.cuda.device_count() > 0:
        device = opt.GPUNO
    else:
        device = 'cpu'

    # load network
    net = MixNet(t0Int, NetType=opt.NetType, resize=resize, mode=opt.SGSMode, device=device)
    if device is not 'cpu':
        net = net.cuda(device)
    net.train()

    # load pretrain model or last model
    if opt.PretrainModel is None:
        if os.path.exists(BestPath):
            print("Load Last Model Successfully!")
            LoadModelDict = torch.load(BestPath)
            net.load_state_dict(LoadModelDict['Weights'])
            TrainParaDict = LoadModelDict['TrainParas']
            countIter, epoch = TrainParaDict['it'], TrainParaDict['epoch']
            BestValidLoss, lrStart = TrainParaDict['bestLoss'], TrainParaDict['lr']
        else:
            print("Start a new training!")
            countIter, epoch, lrStart, BestValidLoss = 0, 1, opt.lrStart, 1e10
    else:
        print("Load PretrainModel Successfully!")
        LoadModelDict = torch.load(opt.PretrainModel)
        net.load_state_dict(LoadModelDict['Weights'])
        countIter, epoch, lrStart, BestValidLoss = 0, 1, opt.lrStart, 1e10
    
    # loss setting
    criterion = nn.BCELoss()

    # define the optimizer
    if opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lrStart)
    elif opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=lrStart, momentum=0.9)
    else:
        Raise("Error: invalid optimizer") 

    # define the lr_scheduler of the optimizer
    scheduler = MultiStepLR(optimizer, [1000000], 0.1)
    
    ####################################
    # training iteration 
    ####################################

    # initialize
    LossList, BestValidVMAE, EarlyStopCount = [], 1e8, 0

    # start the iteration
    diter = iter(dl)
    for _ in range(opt.MaxIter):
        if countIter % len(dl) == 0 and countIter > 0:
            epoch += 1
            scheduler.step()
        countIter += 1
        try:
            pwr, stkG, stkC, label, VMM, _, name = next(diter)
        except StopIteration:
            diter = iter(dl)
            pwr, stkG, stkC, label, VMM, _, name = next(diter)
        if device is not 'cpu':
            pwr = pwr.cuda(device)
            label = label.cuda(device)
            stkG = stkG.cuda(device)
            stkC = stkC.cuda(device)
        optimizer.zero_grad()
        out, _ = net(pwr, stkG, stkC, VMM)

        # compute loss
        loss = criterion(out.squeeze(), label)

        # update parameters
        loss.backward()
        optimizer.step()
        LossList.append(loss.item())

        # save loss lr & seg map
        writer.add_scalar('Train-Loss', loss.item(), global_step=countIter)
        writer.add_scalar('Train-Lr', optimizer.param_groups[0]['lr'], global_step=countIter)
        for ind, name_ind in enumerate(name):
            if name_ind in VisualSample:
                writer.add_image('SegProbMap-%s' % name_ind, out[ind].squeeze(), global_step=epoch, dataformats='HW')

        # print the log per opt.MsgIter
        if countIter % opt.MsgIter == 0:
            lr = optimizer.param_groups[0]['lr']
            msg = 'it: %d/%d, epoch: %d, lr: %.6f, train-loss: %.7f' % (countIter, opt.MaxIter, epoch, lr, sum(LossList) / len(LossList))
            logger.info(msg)

        
        # check points
        if countIter % opt.SaveIter == 0:  
            net.eval()
            # evaluator
            with torch.no_grad():
                LossValid, VMAEValid = EvaluateValid(net, dlval, criterion, 
                                                     SegyDict, H5Dict, t0Int, opt.Predthre, device=device)
                if VMAEValid < BestValidVMAE:
                    BestValidVMAE = VMAEValid
                writer.add_scalar('Valid-Loss', LossValid, global_step=countIter)
                writer.add_scalar('Valid-VMAE', VMAEValid, global_step=countIter)

            if LossValid < BestValidLoss:
                BestValidLoss = LossValid
                state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
                StateDict = {
                    'TrainParas': {'lr': optimizer.param_groups[0]['lr'], 
                                   'it': countIter,
                                   'epoch': epoch,
                                   'bestLoss': BestValidLoss},
                    'Weights': state}

                torch.save(StateDict, BestPath)
                EarlyStopCount = 0
            else:
                # count 1 time
                EarlyStopCount += 1
                # reload checkpoint pth
                if os.path.exists(BestPath):
                    net.load_state_dict(torch.load(BestPath)['Weights'])
                # if do not decreate for 10 times then early stop
                if EarlyStopCount > 10:
                    break
            
            # write the valid log
            try:
                logger.info('it: %d/%d, epoch: %d, Loss: %.6f, VMAE: %.4f, best valid-Loss: %.6f, best valid-VMAE: %.4f' % (countIter, opt.MaxIter, epoch, LossValid, VMAEValid, BestValidLoss, BestValidVMAE))
            except TypeError:
                logger.info('it: %d/%d, epoch: %d, TypeError')
            net.train()
    # save the finish csv
    ResultDF = pd.DataFrame({'BestValidLoss': [BestValidLoss], 'BestValidVMAE': [BestValidVMAE]})
    ResultDF.to_csv(os.path.join(opt.OutputPath, BaseName, 'Result.csv'))
    return BestValidLoss, BestValidVMAE

if __name__ == '__main__':
    # get hyper parameters
    OptN = GetTrainPara()
    # start to train
    train(OptN)
