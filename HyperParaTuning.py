from Tuning.tuning import MainTuning


# Hyper-parameter setting
ParaDict = {'DataSet': ['str', ['hade', 'dq8']], 
            'OutputPath': ['str', ['F:\\VSP-MIFN\\2GeneralTest']], 
            'SeedRate': ['float', [0.2, 0.3, 0.5, 0.8, 1]], 
            'SizeW': ['int', [256, 128]], 
            'trainBS': ['int', [16, 32]], 
            'lrStart': ['float', [0.01, 0.001]]}


# main 
class ValidFunc:
    
    print(1223)
def ValidFunc():
    pass

# Main tuning part
MainTuning(ParaDict, None, None)