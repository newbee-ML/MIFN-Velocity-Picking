import pandas as pd
import numpy as np
import argparse
import os
import psutil
from utils.PlotTools import *
from utils.evaluate import GetResult
from utils.LoadData import GetDataDict, LoadSingleData
import sys
from tqdm import tqdm


###############################################
# Summary test results
###############################################
def SummaryTestResults(RootPath):
    FolderList = [file for file in os.listdir(RootPath) if int(file.split('-')[0]) in np.arange(10, 20)]
    Result, ColName = [], None
    for ind, file in enumerate(FolderList):
        FolderPath = os.path.join(RootPath, file)
        TrainParaCSV = pd.read_csv(os.path.join(FolderPath, 'TrainPara.csv'))
        ParaList = TrainParaCSV.values.tolist()

        ResultPath = os.path.join(FolderPath, 'test', TrainParaCSV['DataSet'][0], '0-PickDict.npy')
        if os.path.exists(ResultPath):
            PickDict = np.load(ResultPath, allow_pickle=True).item()
            VMAE = np.mean([RDict['VMAE'] for RDict in PickDict.values()])
            ParaList[0] += [VMAE]
            print(1, ResultPath)
        else:
            print(0, ResultPath)
            ParaList[0] += [np.nan]
        Result.append(ParaList[0][1:])
        if ind == 0:
            ColName = list(TrainParaCSV.columns)[1:] + ['TestVMAE']
    SumDF = pd.DataFrame(Result, columns=ColName)
    SumDF.to_csv(os.path.join(RootPath, 'TestSummary.csv'))


if __name__ == '__main__':
    Root = 'F:\VelocitySpectrum\MIFN\\1TransferL'
    SummaryTestResults(Root)