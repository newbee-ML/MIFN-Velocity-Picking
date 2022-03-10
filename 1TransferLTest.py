"""
Predict the test data set for transfer learning

Author: Hongtao Wang
"""
import os
import pandas as pd
from test import test, GetTestPara
from Tuning.tuning import UpdateOpt


##############################################
# Experiment settings
##############################################
# directly predict
Ep1 = {'DataSet': 'hade', 
       'Predthre': 0.1,
       'TransferL': 1, 
       'LoadModel': 'F:\\VelocitySpectrum\\MIFN\\2GeneraTest\\Ep-20'}
Ep2 = {'DataSet': 'dq8', 
       'Predthre': 0.1,
       'LoadModel': 'F:\\VelocitySpectrum\\MIFN\\2GeneraTest\\Ep-10'}

# after fine tuning to predict
Ep3 = {'DataSet': 'hade', 
       'Predthre': 0.1,
       'TransferL': 1, 
       'LoadModel': 'F:\VelocitySpectrum\MIFN\\1TransferL\FineTune-hade'}
Ep4 = {'DataSet': 'dq8', 
       'Predthre': 0.1,
       'TransferL': 1, 
       'LoadModel': 'F:\VelocitySpectrum\MIFN\\1TransferL\FineTune-dq8'}


##################################################
# test function
##################################################
def TransferTest(Setting):
    OptPara = GetTestPara()
    EpOpt = UpdateOpt(Setting, OptPara)
    test(EpOpt)


#################################################
# test main
#################################################
if __name__ == '__main__':
    for Ep in [Ep1, Ep2, Ep3, Ep4]:
        TransferTest(Ep)