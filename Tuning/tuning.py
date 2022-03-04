"""
Hyper-parameters Tuning Tool

Author: Hongtao Wang | stolzpi@163.com
---
Input: 
    1. Setting Json 
    2. Main Flow Function
Output: 
    csv file: parameter + test result
"""

import os
import time
import pandas as pd


# get all experiment parameters combination
def ListPara(ParaDict):
    AllKeys = list(ParaDict.keys())

    # get recursion tree
    def GridTree(att):
        NewNode = list(map(str, ParaDict[att][1]))
        if att == AllKeys[-1]:
            return [att + '_' + NewNode[0]]
        else:
            NewAtt = AllKeys[AllKeys.index(att) + 1]
            NewTree = []
            for node in NewNode:
                for NextTree in GridTree(NewAtt):
                    NewTree.append('%s_%s-%s' % (att, node, NextTree))
            return NewTree

    AllCom = GridTree(AllKeys[0])
    return AllCom


# transfer para string to para dict
def ParaStr2Dict(ParaStr, ParaDict):
    ParaList = ParaStr.split('-')
    EpDict = {}
    for elm in ParaList:
        name, value = elm.split('_')
        Ttype = ParaDict[name][0]
        if Ttype == 'str':
            EpDict.setdefault(name, str(value))
        elif Ttype == 'int':
            EpDict.setdefault(name, int(value))
        else:
            EpDict.setdefault(name, float(value))
    return EpDict


# update the parameters in opt
def UpdateOpt(ParaDict, opt):
    for key in list(ParaDict.keys()):
        opt.__dict__[key] = ParaDict[key]
    return opt

# main function for tuning processing
def MainTuning(ParaDict, GetPara, Sample, OutPath):
    ###########################################
    # get the hyper-parameter list from Json
    ###########################################
    EpList = ListPara(ParaDict)

    ###########################################
    # get the undo list
    ###########################################
    FileList = [os.path.join(OutPath, file) for file in os.listdir(OutPath) if file.split('.')[-1] == 'csv']
    DoneList = []
    for file in FileList:
        DFCSV = pd.read_csv(file)
        ParaStr = ['%s_%s' % (name, str(DFCSV[name])) for name in list(ParaDict.keys())]
        DoneList.append('-'.join(ParaStr))
    print('There are %d tasks, and %d are done!' % (len(EpList), len(DoneList)))
    EpList = sorted(list(set(EpList)-set(DoneList)))

    ###########################################
    # grid searching function
    ###########################################
    for Ep in EpList:
        Paradict = ParaStr2Dict(Ep)
        keys = list(Paradict.keys())
        ParaOpt = GetPara()
        for key in keys:
            ParaOpt.__dict__[key] = Paradict[key]
        TestResult = Sample(ParaOpt)
        for key, elm in ParaOpt.__dict__.items():
            TestResult.setdefault(key, elm)
        timeNow = time.strftime('%y%m%d-%H%M%S', time.localtime())
        ResultCSV = pd.DataFrame(TestResult)
        ResultCSV.to_csv(os.path.join(OutPath, 'Ep-%s.csv' % timeNow))
        



