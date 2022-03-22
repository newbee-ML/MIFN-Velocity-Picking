"""
Predict the test data set for transfer learning

Author: Hongtao Wang
"""
from test import test, GetTestPara
from Tuning.tuning import UpdateOpt



##############################################
# Experiment settings
##############################################
# directly predict
Ep1 = {'DataSet': 'hade', 
       'Predthre': 0.2,
       'TransferL': 0, 
       'LoadModel': 'F:\\VelocitySpectrum\\MIFN\\2GeneraTest\\Ep-213'}
Ep2 = {'DataSet': 'dq8', 
       'Predthre': 0.2,
       'TransferL': 0, 
       'LoadModel': 'F:\\VelocitySpectrum\\MIFN\\2GeneraTest\\Ep-203'}

# after fine tuning to predict
Ep3 = {'DataSet': 'hade', 
       'Predthre': 0.2,
       'TransferL': 1, 
       'LoadModel': 'F:\VelocitySpectrum\MIFN\\1TransferL\FineTune-hade'}
Ep4 = {'DataSet': 'dq8', 
       'Predthre': 0.2,
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
    for Ep in [Ep1, Ep2]:
       TransferTest(Ep)
    for data in ['dq8', 'hade']: 
       for sr in ['0.2', '0.5']:
           for ind in range(10, 20):
              Ep['LoadModel'] = 'F:\VelocitySpectrum\MIFN\\1TransferL\%d-FineTune-%s-SR-%s' % (ind, data, sr)
              Ep['DataSet'] = data
              TransferTest(Ep)