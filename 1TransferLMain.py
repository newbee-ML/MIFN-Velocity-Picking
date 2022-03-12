from train import GetTrainPara, train

from Tuning.tuning import UpdateOpt
import os

#################################
# experiment setting
#################################
Ep1 = {'DataSet': 'hade', 
        'SeedRate': 0.4, 
        'SizeW': 128, 
        'trainBS': 8, 
        'lrStart': 0.01,
        'PretrainModel': 'F:\\VelocitySpectrum\\MIFN\\2GeneraTest\\Ep-20\\model\\Best.pth'}
Ep2 = {'DataSet': 'dq8', 
        'SeedRate': 0.4, 
        'SizeW': 128, 
        'trainBS': 8, 
        'lrStart': 0.01,
        'PretrainModel': 'F:\\VelocitySpectrum\\MIFN\\2GeneraTest\\Ep-10\\model\\Best.pth'}

#################################
# training
#################################
def FineTune(EpSetting):
    EpSetting.setdefault('OutputPath', 'F:\\VelocitySpectrum\\MIFN\\1TransferL')
    EpSetting.setdefault('EpName', 'FineTune-%s-SR0.4'%EpSetting['DataSet'])
    # judge whether done before
    if os.path.exists(os.path.join(EpSetting['OutputPath'], 'FineTune-%s-SR0.4'%EpSetting['DataSet'], 'Result.csv')):
        return 0
    OptDefault = GetTrainPara()
    EpOpt = UpdateOpt(EpSetting, OptDefault)
    train(EpOpt)


if __name__ == '__main__':
    for Ep in [Ep1, Ep2]:
        FineTune(Ep)