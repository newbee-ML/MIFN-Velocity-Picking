from VisualFunc import BaseSetting, Visual
from Tuning.tuning import UpdateOpt


VisualEp1 = {
    'DataSetRoot': 'E:\\Spectrum', 
    'LoadModel': 'F:\\VelocitySpectrum\\MIFN\\2GeneraTest\\Ep-203', 
    'OutputPath': 'F:\\VelocitySpectrum\\MIFN\\4Visual', 
    'Predthre': 0.2,
    'line': 3240,
    'cdp': 1000
}
VisualEp2 = {
    'DataSetRoot': 'E:\\Spectrum', 
    'LoadModel': 'F:\\VelocitySpectrum\\MIFN\\2GeneraTest\\Ep-213', 
    'OutputPath': 'F:\\VelocitySpectrum\\MIFN\\4Visual', 
    'Predthre': 0.2,
    'line': 940,
    'cdp': 2150
}

if __name__ == '__main__':
    OptN = BaseSetting()
    for Ep in [VisualEp1, VisualEp2]:
        OptN = UpdateOpt(Ep, OptN)
        Visual(OptN)