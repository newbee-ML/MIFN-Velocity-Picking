from train import train, GetTrainPara
from Tuning.tuning import ListPara, ParaStr2Dict, UpdateOpt
import os

#################################
# experiment setting
#################################
ParaDict = {'DataSet': ['str', ['hade', 'dq8']], 
            'OutputPath': ['str', ['F:\\VelocitySpectrum\\MIFN\\2GeneraTest']], 
            'SeedRate': ['float', [0.2, 0.3, 0.5, 0.8, 1, ]], 
            'SizeW': ['int', [256, 128]], 
            'trainBS': ['int', [32]], 
            'lrStart': ['float', [0.01]]}


#################################
# training
#################################
# get the experiment (ep) list
EpList = ListPara(ParaDict)
# get default training parameters
OptDefault = GetTrainPara()
for ind, EpName in enumerate(EpList):
    try:
        # get the ep para dict
        EpDict = ParaStr2Dict(EpName, ParaDict)
        EpDict.setdefault('EpName', 'Ep-%d' % (ind+1))
        # judge whether done before
        if os.path.exists(os.path.join(EpDict['OutputPath'], 'Ep-%d' % (ind+1), 'Result.csv')):
            continue
        if os.path.exists(os.path.join(EpDict['OutputPath'], 'Ep-%d' % (ind+1))):
            EpDict.setdefault('ReTrain', 0)
        else:
            EpDict.setdefault('ReTrain', 1)
        # update the para
        EpOpt = UpdateOpt(EpDict, OptDefault)
        # start this experiment
        train(EpOpt)
    except:
        print(EpName)
        continue
