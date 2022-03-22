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

############################################
# summary validation results
############################################
def SummaryValidResults(opt):
    FolderList = [file for file in os.listdir(opt.LoadRoot) if file.split('-')[0] == 'Ep']
    FolderList = [file for file in FolderList if int(file.split('-')[1])>=StratEp]
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
    FolderList = [file for file in FolderList if int(file.split('-')[1])>=StratEp]
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
            print(1, ResultPath)
        else:
            print(0, ResultPath)
            ParaList[0] += [np.nan]
        Result.append(ParaList[0][1:])
        if ind == 0:
            ColName = list(TrainParaCSV.columns)[1:] + ['TestVMAE']
    SumDF = pd.DataFrame(Result, columns=ColName)
    SumDF.to_csv(os.path.join(opt.LoadRoot, 'TestSummary.csv'))


##########################################################
# Visual the picking results with different seed rate 
##########################################################
def VisualResults(opt):
    # -------- load the model path ----------
    RootPath = 'F:\\VelocitySpectrum\\MIFN\\2GeneraTest'
    SaveRoot = os.path.join(RootPath, 'VisualResult')
    if not os.path.exists(SaveRoot):
        os.makedirs(SaveRoot)
    # get the best model in 
    def HelpMeFindBset():
        TestCSV = pd.read_csv(os.path.join(RootPath, 'TestSummary.csv'))
        GroupResults = TestCSV[TestCSV['SizeW']==128].groupby(['DataSet', 'SeedRate'])
        BestPath = []
        for Set in GroupResults:
            data, sr = Set[0]
            DFResult = Set[1]
            BestName = DFResult[DFResult['TestVMAE']==np.min(DFResult['TestVMAE'].values)]['EpName'].values[0]
            BestPath.append([data, sr, os.path.join(RootPath, BestName)])
        return BestPath
    BestPath = HelpMeFindBset()
    
    # ------------- get the best picking results  ------------------
    ResultDict = {}
    for Ep in BestPath:
        data, sr, folder = Ep
        TestPick = np.load(os.path.join(folder, 'test', data, '0-PickDict.npy'), allow_pickle=True).item()
        PredThreList = np.linspace(0.1, 0.3, 10)
        bar = tqdm(total=10, file=sys.stdout)
        for ind, (name, dictN) in enumerate(TestPick.items()):
            if ind > 10:
                break
            NResults = []
            for pt in PredThreList:
                AP, APPeaks = GetResult(dictN['Seg'], dictN['Tint'], [dictN['VInt']], threshold=pt)
                vmae = np.mean(np.abs(AP[0][:, 1] - dictN['MP'][:, 1]))
                NResults.append([vmae, AP, APPeaks, dictN['MP']])
            BestList = NResults[np.argmin([row[0] for row in NResults])]
            BestResult = {'VMAE': BestList[0], 'AP': BestList[1], 'APPeaks': BestList[2], 'MP': BestList[3], 'Seg': TestPick[name]['Seg'], 'Pwr': dictN['Pwr'], 'VInt': dictN['VInt'], 'TInt': dictN['Tint']}
            ResultDict.setdefault(data, {})
            ResultDict[data].setdefault(name, {})
            ResultDict[data][name].setdefault(sr, BestResult)
            bar.set_description(name)
            bar.update(1)
        bar.close()

    # ------------- load original data & visual ------------------
    DataSetRoot = 'E:\Spectrum'
    for data in ['hade', 'dq8']:
        DataSetPath = os.path.join(DataSetRoot, data)
        SegyDict, H5Dict, LabelDict = GetDataDict(DataSetPath)
        bar = tqdm(total=10*len(list(ResultDict[data].keys())), file=sys.stdout)
        for name, Results in ResultDict[data].items():
            SaveFolder = os.path.join(SaveRoot, data, name)
            if not os.path.exists(SaveFolder):
                os.makedirs(SaveFolder)
            # load original data for single sample
            DataDict = LoadSingleData(SegyDict, H5Dict, LabelDict, name, mode='train', LoadG=True)
            # plot the results with different seed rate
            for sr, srDict in Results.items():
                # 1 Pwr and Seg Map
                PwrASeg(srDict['Pwr'], srDict['Seg'], SavePath=os.path.join(SaveFolder, '1-PwrASeg-%s-%.1f-%.3f.pdf' % (name, sr, srDict['VMAE'])))
                # 2 picking result (APPeaks + MP)
                InterpResult(np.squeeze(srDict['APPeaks']), srDict['MP'], SavePath=os.path.join(SaveFolder, '2-PickCurve-%s-%.1f-%.3f.pdf' % (name, sr, srDict['VMAE'])))
                # 3 Seg with AP and MP 
                SegPick(srDict['Seg'], srDict['TInt'], srDict['VInt'], np.squeeze(srDict['AP']), srDict['MP'], SavePath=os.path.join(SaveFolder, '3-APAMP-%s-%.1f-%.3f.pdf' % (name, sr, srDict['VMAE'])))
                mem = psutil.virtual_memory().free / 1e9
                print(f'memory used: {mem} [GB]')
                bar.set_description('%s SR=%.1f' % (name, sr))
                bar.update(1)
        bar.close()


################################################################
# Plot train and test loss curve
################################################################
def PlotLossCurve(EpList, RootPath):
    LossDict = {}
    # --------- load data from log file ----------
    for Ep in EpList:
        # load model parameters
        ParaDict = pd.read_csv(os.path.join(RootPath, Ep, 'TrainPara.csv')).to_dict()
        DataSet, SR = ParaDict['DataSet'][0], ParaDict['SeedRate'][0]
        LossDict.setdefault(DataSet, {})
        LossDict[DataSet].setdefault(SR, {})
        # load log file
        LogRootPath = os.path.join(RootPath, Ep, 'log')
        LogPath = os.path.join(LogRootPath, os.listdir(LogRootPath)[0])
        LogFile = open(LogPath, 'r')
        LogRows =  LogFile.readlines()
        # split the details
        for row in LogRows:
            if 'train-loss' in row:
                it = int(row.split('it: ')[1].split('/')[0])
                TrainLoss = float(row.split('train-loss: ')[1].strip('\n'))
                LossDict[DataSet][SR].setdefault('train', [])
                LossDict[DataSet][SR]['train'].append([it, TrainLoss])
            elif 'valid-Loss' in row:
                it = int(row.split('it: ')[1].split('/')[0])
                ValidLoss = float(row.split('Loss: ')[1].split(',')[0])
                LossDict[DataSet][SR].setdefault('valid', [])
                LossDict[DataSet][SR]['valid'].append([it, ValidLoss])
    # -------- plot the loss curve --------
    for data, loss_dict in LossDict.items():
        PlotLoss(loss_dict, os.path.join(RootPath, 'LossCurve-%s.pdf' % data))


########################################################################
# plot the VMAE of test results
########################################################################
def PlotTestVMAE(RootPath):
    TestCsv = pd.read_excel(os.path.join(RootPath, 'TestResult.xlsx'))
    TestData = TestCsv.groupby('DataSet')
    TestDict = {}
    for data, df in TestData:
        TestDict.setdefault(data, df[['SeedRate', 'TestVMAE']].values)
    VMAETest(TestDict, os.path.join(RootPath, 'VMAETest.pdf'))
    



if __name__ == "__main__":
    StratEp = 140
    EpList = ['Ep-%d'%ind for ind in [203, 146, 223, 144, 213, 142, 141, 140]]
    parser = argparse.ArgumentParser()
    parser.add_argument('--LoadRoot', type=str, default='F:\\VelocitySpectrum\\MIFN\\2GeneraTest', help='EP path')
    optN = parser.parse_args()
    # visual sample result
    # SampleLines(optN.LoadRoot)
    # SummaryValidResults(optN)
    # SummaryTestResults(optN)
    # VisualResults(optN)
    PlotLossCurve(EpList, optN.LoadRoot)
    PlotTestVMAE(optN.LoadRoot)
    
