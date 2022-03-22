from train import GetTrainPara, train
import copy
from Tuning.tuning import UpdateOpt
import os

#################################
# experiment setting
#################################
Ep1 = {'DataSet': 'hade', 
       'SeedRate': 0.2, 
       'SizeW': 128, 
       'trainBS': 32, 
       'lrStart': 0.01,
       'PretrainModel': 'F:\\VelocitySpectrum\\MIFN\\2GeneraTest\\Ep-213\\model\\Best.pth'}
Ep2 = {'DataSet': 'dq8', 
       'SeedRate': 0.2, 
       'SizeW': 128, 
       'trainBS': 32, 
       'lrStart': 0.01,
       'PretrainModel': 'F:\\VelocitySpectrum\\MIFN\\2GeneraTest\\Ep-203\\model\\Best.pth'}
Ep3 = {'DataSet': 'hade', 
       'SeedRate': 0.5, 
       'SizeW': 128, 
       'trainBS': 32, 
       'lrStart': 0.01,
       'PretrainModel': 'F:\\VelocitySpectrum\\MIFN\\2GeneraTest\\Ep-213\\model\\Best.pth'}
Ep4 = {'DataSet': 'dq8', 
       'SeedRate': 0.5, 
       'SizeW': 128, 
       'trainBS': 32, 
       'lrStart': 0.01,
       'PretrainModel': 'F:\\VelocitySpectrum\\MIFN\\2GeneraTest\\Ep-203\\model\\Best.pth'}


#################################
# training
#################################
def FineTune(EpSetting, ind=0):
    NewSetting = copy.deepcopy(EpSetting)
    BaseName = '%d-FineTune-%s-SR-%.1f'%(ind, NewSetting['DataSet'], NewSetting['SeedRate'])
    NewSetting.setdefault('OutputPath', 'F:\\VelocitySpectrum\\MIFN\\1TransferL')
    NewSetting.setdefault('EpName', BaseName)
    # judge whether done before
    if os.path.exists(os.path.join(NewSetting['OutputPath'], BaseName, 'Result.csv')):
        return 0
    OptDefault = GetTrainPara()
    EpOpt = UpdateOpt(NewSetting, OptDefault)
    train(EpOpt)


if __name__ == '__main__':
    for times in range(10):
        FineTune(Ep4, times)