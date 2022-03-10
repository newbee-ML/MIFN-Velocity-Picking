import pandas as pd
import numpy as np
import argparse
import os


############################################
# summary validation results
############################################
def LoadResult(opt):
    FolderList = [file for file in os.listdir(opt.LoadRoot) if file.split('-')[0] == 'Ep']
    Result, ColName = [], None
    for ind, file in enumerate(FolderList):
        FolderPath = os.path.join(opt.LoadRoot, file)
        TrainParaCSV = pd.read_csv(os.path.join(FolderPath, 'TrainPara.csv'))
        ParaList = TrainParaCSV.values.tolist()
        if os.path.exists(os.path.join(FolderPath, 'Result.csv')):
            ResultCSV = pd.read_csv(os.path.join(FolderPath, 'Result.csv'))
            ParaList[0] += ResultCSV.values.tolist()[0][1:]
        else:
            ParaList[0] += [np.nan, np.nan]
        Result.append(ParaList[0][1:])
        if ind == 0:
            ColName = list(TrainParaCSV.columns)[1:] + ['BestValidLoss', 'BestValidVMAE']
            
    SumDF = pd.DataFrame(Result, columns=ColName)
    SumDF.to_csv(os.path.join(opt.LoadRoot, 'ValidSummary.csv'))


###############################################
# Summary test results
###############################################
def SummaryTestResults(opt):
    FolderList = [file for file in os.listdir(opt.LoadRoot) if file.split('-')[0] == 'Ep']
    Result, ColName = [], None
    for ind, file in enumerate(FolderList):
        FolderPath = os.path.join(opt.LoadRoot, file)
        TrainParaCSV = pd.read_csv(os.path.join(FolderPath, 'TrainPara.csv'))
        ParaList = TrainParaCSV.values.tolist()
        ResultPath = os.path.join(FolderPath, 'test', TrainParaCSV['DataSet'][0], '0-PickDict.npy')
        if os.path.exists(ResultPath):
            PickDict = np.load(ResultPath, allow_pickle=True).item()
            VMAE = np.mean([RDict['VMAE'] for RDict in PickDict.values()])
            ParaList[0] += [VMAE]
        else:
            ParaList[0] += [np.nan]
        Result.append(ParaList[0][1:])
        if ind == 0:
            ColName = list(TrainParaCSV.columns)[1:] + ['TestVMAE']
    SumDF = pd.DataFrame(Result, columns=ColName)
    SumDF.to_csv(os.path.join(opt.LoadRoot, 'TestSummary.csv'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--LoadRoot', type=str, default='F:\\VelocitySpectrum\\MIFN\\2GeneraTest', help='EP path')
    optN = parser.parse_args()
    LoadResult(optN)
    SummaryTestResults(optN)
