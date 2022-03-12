"""
Predict the test data set for different seed rates

Author: Hongtao Wang
"""
import os
import pandas as pd
from test import test, GetTestPara

"""
Get all model path and para path
"""
def GetModelAPara(RootPath):
    Folder = [file for file in os.listdir(RootPath) if file.split('-')[0]=='Ep']
    InfoDict = {}
    for file in Folder:
        ModelPath = os.path.join(RootPath, file, 'model', 'Best.pth')
        ParaPath = os.path.join(RootPath, file, 'TrainPara.csv')
        if os.path.exists(ModelPath) and os.path.exists(ParaPath):
            if not os.path.exists(os.path.join(RootPath, file, 'test')):
                InfoDict.setdefault(file, os.path.join(RootPath, file))
    return InfoDict

"""
Test function
"""

if __name__ == '__main__':
    RootPath = 'F:\VelocitySpectrum\MIFN\\2GeneraTest'
    TestOpt = GetTestPara()
    FileList = GetModelAPara(RootPath)
    for name, model in FileList.items():
        # ForkNew(parastr)
        TestOpt.EpName = name
        TestOpt.LoadModel = model
        test(TestOpt)
