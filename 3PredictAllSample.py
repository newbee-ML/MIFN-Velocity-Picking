import os

from predict import GetPredictPara, predict

EpList = {'A': {'Index': [144, 223, 146, 203], 'Line': 3240},
          'B': {'Index': [140, 141, 142, 213], 'Line': 940}}
if __name__ == '__main__':
    Lines = []
    ResultRoot = 'F:\\VelocitySpectrum\\MIFN\\2GeneraTest'
    optN = GetPredictPara()
    for data, InfoDict in EpList.items():
        for EpNum in InfoDict['Index']:
            optN.LoadModel = os.path.join(ResultRoot, 'Ep-%d' % EpNum)
            optN.PredLine = InfoDict['Line']
            predict(optN)

