import numpy as np
import copy

"""
Feature Extractor for Velocity Spectrum

    Consists of 4 steps:
    - smooth the spec
    - exponential the original spectrum
    - layer-wise normalization
    - clip the amplitude to (min_thre, max_thre)
"""

# ---- smooth function ----------------------------
def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')


# ---- normalization function ----------------------
def normal_mat(mat):
    mat_cp = mat.copy()
    if np.ptp(mat_cp) > 0:
        return (mat_cp - np.min(mat_cp)) / np.ptp(mat_cp)
    else:
        return mat_cp


# ---- enhance main function ---------------------
def spec_enhance(spec_mat, smooth_ws=5, smooth_num=1, norm_num=8, exp_num=1, clip_thres=(0.2, 0.8), save=0):
    cp_spec = spec_mat.copy()
    if save:
        ProcessDict = {}
    # step 1 smooth the spec
    for time in range(smooth_num):
        for col in range(cp_spec.shape[1]):
            cp_spec[:, col] = moving_average(cp_spec[:, col], window_size=smooth_ws)
    if save:
        ProcessDict.setdefault('1', copy.deepcopy(cp_spec))

    # step 2 exponential the original spectrum
    cp_spec = cp_spec ** exp_num
    if save:
        ProcessDict.setdefault('2', copy.deepcopy(cp_spec))

    # step 3 layer-wise normalization
    index_cp = np.linspace(0, cp_spec.shape[0]-1, num=norm_num+1).astype(np.int)
    for i in range(norm_num):
        cp_spec[index_cp[i]:index_cp[i+1], :] = normal_mat(cp_spec[index_cp[i]:index_cp[i+1], :])
    if save:
        ProcessDict.setdefault('3', copy.deepcopy(cp_spec))

    # step 4 clip the amplitude to (min_thre, max_thre)
    cp_spec[cp_spec < clip_thres[0]] = 0
    cp_spec[cp_spec > clip_thres[1]] = clip_thres[1]
    if save:
        ProcessDict.setdefault('4', copy.deepcopy(cp_spec))

    if save:
        return cp_spec, ProcessDict
    else:
        return cp_spec
