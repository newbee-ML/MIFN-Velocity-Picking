import sys
import h5py
import logging
import numpy as np
import torch
import copy
import segyio
import time
import os
import shutil
import warnings
import datetime
import argparse
import torch.nn as nn
from net.MIFNet import MultiInfoNet
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils.LoadData import DLSpec
from utils.logger import setup_logger
from torch.optim.lr_scheduler import MultiStepLR
from utils.evaluate import EvaluateValid
from tensorboardX import SummaryWriter


sys.path.append('..')
warnings.filterwarnings("ignore")


def CheckSavePath():
    basicFile = ['log', 'model', 'TBLog']
    BaseName = '%s_%d_%.1f_%s' % (opt.DataSet, opt.Resize[0], opt.SeedRate, opt.SGSMode)
    for ind, file in enumerate(basicFile):
        if ind == 3:
            Path = os.path.join(opt.OutputPath, file, BaseName)
        else:
            Path = os.path.join(opt.OutputPath, opt.DataSet, file, BaseName)
        if opt.ReTrain:
            if os.path.exists(Path):
                shutil.rmtree(Path)
        if not os.path.exists(Path):
            os.makedirs(Path)


def train():
    print('==== Begin to train ====')
    # check output folder and check path
    opt.Resize = [int(opt.SizeH), int(opt.SizeH/2)]
    TBPath = os.path.join(opt.OutputPath, 'TBLog', '%s_%d_%.1f_%s' % (opt.DataSet, opt.Resize[0], opt.SeedRate, opt.SGSMode))
    CheckSavePath()

    opt.OutputPath = os.path.join(opt.OutputPath, opt.DataSet)
    OutputPath = opt.OutputPath
    BestPath = os.path.join(OutputPath, 'model', '%s_%d_%.1f_%s.pth' % (opt.DataSet, opt.Resize[0], opt.SeedRate, opt.SGSMode))

    # setup the logger
    LogPath = os.path.join(OutputPath, 'log', '%s_%d_%.1f_%s' % (opt.DataSet, opt.Resize[0], opt.SeedRate, opt.SGSMode))
    setup_logger(LogPath)
    writer = SummaryWriter(TBPath)
    logger = logging.getLogger()
    logger.info('Training --- DataSet=%s  SizeH=%d SeedRate=%.1f SGSMode=%s' % (opt.DataSet, opt.Resize[0], opt.SeedRate, opt.SGSMode))

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
    trainIndex, validIndex = train_test_split(Index, test_size=1-opt.SeedRate, random_state=2333)

    # load t0 ind and v ind
    T0Ind = np.array(SegyDict['pwr'].samples)
    opt.t0Int = T0Ind

    # build data loader
    ds = DLSpec(SegyDict, H5Dict, LabelDict, trainIndex, T0Ind, resize=opt.Resize, GatherLen=opt.GatherLen)
    dsval = DLSpec(SegyDict, H5Dict, LabelDict, validIndex, T0Ind, resize=opt.Resize, GatherLen=opt.GatherLen)
    dl = DataLoader(ds,
                    batch_size=opt.trainBS,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=False,
                    drop_last=True)
    dlval = DataLoader(dsval,
                       batch_size=opt.valBS,
                       shuffle=False,
                       num_workers=0,
                       drop_last=True)

    # check gpu is available
    use_gpu = torch.cuda.device_count() > 0

    # load network
    net = MultiInfoNet(T0Ind, opt, in_channels=11, resize=opt.Resize, mode=opt.SGSMode)
    if use_gpu:
        net = net.cuda(opt.GPUNO)
    net.train()
    if not opt.ReTrain:
        opt.LoadModel = BestPath
        if os.path.exists(opt.LoadModel):
            print("Load Last Model Successfully!")
            LoadModelDict = torch.load(opt.LoadModel)
            net.load_state_dict(LoadModelDict['Weights'])
            TrainParaDict = LoadModelDict['TrainParas']
            countIter, epoch = TrainParaDict['it'], TrainParaDict['epoch']
            bestVloss, lrStart = TrainParaDict['bestLoss'], TrainParaDict['lr']
        else:
            raise "There is no such model file, start a new training!"
    else:
        countIter, epoch, lrStart, bestVloss = 0, 1, opt.lrStart, 1e10

    criterion = nn.BCELoss()
    # define the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lrStart)

    # define the lr_scheduler of the optimizer
    scheduler = MultiStepLR(optimizer, [10, 30, 100], 0.1)
    
    """ ==== Start to Train ==== """
    # initialize
    loss_avg = []
    st = glob_st = time.time()

    # start the iteration
    diter = iter(dl)
    for it in range(opt.MaxIter):
        if countIter % len(dl) == 0 and countIter > 0:
            epoch += 1
            scheduler.step()
        countIter += 1
        opt.iter = countIter
        try:
            pwr, stkG, stkC, label, VMM, _, name = next(diter)
        except StopIteration:
            diter = iter(dl)
            pwr, stkG, stkC, label, VMM, _, name = next(diter)

        if use_gpu:
            pwr = pwr.cuda(opt.GPUNO)
            label = label.cuda(opt.GPUNO)
            stkG = stkG.cuda(opt.GPUNO)
            stkC = stkC.cuda(opt.GPUNO)

        optimizer.zero_grad()
        out, _ = net(pwr, stkG, stkC, VMM)

        # compute loss
        loss = criterion(out.squeeze(), label)

        # update parameters
        loss.backward()
        optimizer.step()
        loss_avg.append(loss.item())
        writer.add_scalar('Train-Loss', loss.item(), global_step=countIter)
        writer.add_scalar('Train-Lr', optimizer.param_groups[0]['lr'], global_step=countIter)
        if countIter % opt.MsgIter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            lr = optimizer.param_groups[0]['lr']
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((opt.MaxIter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))

            msg = ', '.join([
                'it: {it}/{max_it}',
                'epoch: {epoch}',
                'lr: {lr:.8f}',
                'loss: {loss:.7f}',
                'eta: {eta}',
                'time: {time:.2f}s/{Alltime:.2f}s',
            ]).format(
                it=countIter,
                epoch=epoch,
                max_it=opt.MaxIter,
                lr=lr,
                loss=loss_avg,
                time=t_intv,
                eta=eta,
                Alltime=t_intv * opt.MaxIter / opt.MsgIter,
            )
            logger.info(msg)
            loss_avg = []
            st = ed
        
        # check points
        if countIter % opt.SaveIter == 0:  
            net.eval()

            # evaluator
            with torch.no_grad():
                LossValid, VMAEValid = EvaluateValid(net, dlval, criterion, opt,
                                                     SegyDict, H5Dict, use_gpu=use_gpu)
                writer.add_scalar('Valid-Loss', LossValid, global_step=countIter)
                writer.add_scalar('Valid-VMAE', VMAEValid, global_step=countIter)
            if LossValid < bestVloss:
                bestVloss = copy.deepcopy(LossValid)
                state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
                StateDict = {
                    'TrainParas': {'lr': optimizer.param_groups[0]['lr'], 
                                   'it': countIter,
                                   'epoch': epoch,
                                   'bestLoss': bestVloss},
                    'Weights': state}

                torch.save(StateDict, BestPath)
            else:
                # reload checkpoint pth
                if os.path.exists(BestPath):
                    net.load_state_dict(torch.load(BestPath)['Weights'])
            try:
                logger.info('it: %d/%d, epoch: %d, Loss: %.6f, VMAE: %.6f, BestLoss: %.6f' %
                            (countIter, opt.MaxIter, epoch, LossValid, VMAEValid, bestVloss))
            except TypeError:
                logger.info('it: %d/%d, epoch: %d, TypeError')
                
            net.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DataSetRoot', type=str, help='Dataset Root Path')
    parser.add_argument('--DataSet', type=str, default='hade1', help='Dataset Root Path')
    parser.add_argument('--OutputPath', type=str, default='result', help='Path of Output')
    parser.add_argument('--SGSMode', type=str, default='all')
    parser.add_argument('--GatherLen', type=int, default=21)
    parser.add_argument('--SeedRate', type=float, default=0.5)
    parser.add_argument('--ReTrain', type=int, default=1)
    parser.add_argument('--GPUNO', type=int, default=0)
    parser.add_argument('--Resize', type=list, help='Reset Image Size')
    parser.add_argument('--SizeH', type=int, default=256, help='Size Height')
    parser.add_argument('--Predthre', type=float, default=0.3)
    parser.add_argument('--MaxIter', type=int, default=15000, help='max iteration')
    parser.add_argument('--SaveIter', type=int, default=100, help='checkpoint each SaveIter')
    parser.add_argument('--MsgIter', type=int, default=20, help='log the loss each MsgIter')
    parser.add_argument('--lrStart', type=float, default=0.01, help='the beginning learning rate')
    parser.add_argument('--LoadModel', type=str, help='Load Old to train (Path)')
    parser.add_argument('--trainBS', type=int, default=16, help='The batchsize of train')
    parser.add_argument('--valBS', type=int, default=16, help='The batchsize of valid')
    parser.add_argument('--t0Int', type=list, default=[])
    parser.add_argument('--vInt', type=list, default=[])
    parser.add_argument('--iter', type=int, default=0)
    opt = parser.parse_args()

    # start to train
    train()
