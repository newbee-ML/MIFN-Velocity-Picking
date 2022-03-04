from pandas.core.indexes.base import Index
from multiprocessing import Pool
from HyperParas import ParaDict
from train import GetPara, train
import pandas as pd
import os
import time
import shutil

# get all experiment parameters combination
def ListPara(ParaDict):
    AllKeys = list(ParaDict.keys())

    # get recursion tree
    def GridTree(att):
        NewNode = list(map(str, ParaDict[att][1]))
        if att == AllKeys[-1]:
            return NewNode
        else:
            NewAtt = AllKeys[AllKeys.index(att) + 1]
            NewTree = []
            for node in NewNode:
                for NextTree in GridTree(NewAtt):
                    NewTree.append('%s-%s' % (node, NextTree))
            return NewTree

    AllCom = GridTree(AllKeys[0])
    return AllCom


# query the finished experiments
def QueryFinish(root=str, TestNum=25):
    ParaKeys = list(ParaDict.keys())
    FinishedSet = set()
    UnFinish = []
    # get history experiments parameters
    for folder in os.listdir(root):
        LogTxt = os.path.join(root, folder, 'logs', 'VMAE.log')
        try:
            with open(LogTxt, 'r', encoding='utf-8') as file:
                lows = file.readlines()
        except FileNotFoundError:
            UnFinish.append(os.path.join(root, folder))
            continue
        if len(lows) < TestNum:
            UnFinish.append(os.path.join(root, folder))
            continue
        ParaCSV = os.path.join(root, folder, 'logs', 'Parameters.csv')
        ParaDf = pd.read_csv(ParaCSV)
        ParaList = list(map(str, list(ParaDf[ParaKeys].values[0])))
        FinishedSet.add('-'.join(ParaList))
    # delete the unfinish folder
    if len(UnFinish) > 0:
        for path in UnFinish:
            shutil.rmtree(path)
    return list(FinishedSet)

# Get the undo parameter combination list
def UndoList(root=str):
    # get all experiment parameters combination
    AllParaSet = set(ListPara(ParaDict))
    # get history experiments parameters
    FinishedSet = set(QueryFinish(root, ParaDict['TestNum'][-1][0]))
    # get undo comb list
    UndoList = list(AllParaSet - FinishedSet)
    return UndoList

# tran para str to para dict
def ParaStr2Dict(ParaStr):
    ParaList = ParaStr.split('-')
    Paradict = {}
    for ind, key in enumerate(list(ParaDict.keys())):
        Ttype = ParaDict[key][0]
        if Ttype == 'str':
            Paradict.setdefault(key, str(ParaList[ind]))
        elif Ttype == 'int':
            Paradict.setdefault(key, int(ParaList[ind]))
        else:
            Paradict.setdefault(key, float(ParaList[ind]))
    return Paradict

def MainTuning():
    UndoEp = UndoList('result')
    for parastr in UndoEp:
        Paradict = ParaStr2Dict(parastr)
        keys = list(Paradict.keys())
        timeNow = time.strftime('%y%m%d-%H%M%S', time.localtime())
        NewEpName = os.path.split(Paradict['DataSetRoot'])[-1] + '-' + timeNow
        ParaOpt = GetPara()
        ParaOpt.__dict__['EpName'] = NewEpName
        for key in keys:
            ParaOpt.__dict__[key] = Paradict[key]
        train(ParaOpt)


if __name__ == "__main__":
    UndoEp = UndoList('result')
    p = Pool(8)
    for parastr in UndoEp:
        # ForkNew(parastr)
        p.apply_async(ForkNew, args=(parastr,))
        time.sleep(1.5)
    p.close()
    p.join()

