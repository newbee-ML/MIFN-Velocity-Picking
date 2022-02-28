import copy
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from torchvision import transforms

"""
A few functions to Plot results
"""


# Basic plot setting
def plot_basic_set():
    plt.rcParams['font.sans-serif'] = 'Times New Roman'


# change axis set
def set_axis(x_dim, y_dim, interval=50):
    x_dim, y_dim = x_dim.astype(int), y_dim.astype(int)
    show_x = range(0, len(x_dim) - 1, interval)
    show_y = range(0, len(y_dim) - 1, interval)
    ori_x = range(x_dim[0], x_dim[-1], (x_dim[1] - x_dim[0]) * interval)
    ori_y = range(y_dim[0], y_dim[-1], (y_dim[1] - y_dim[0]) * interval)
    plt.xticks(show_x, ori_x)
    plt.yticks(show_y, ori_y)


# Plot CMP gather and NMO CMP gather
def plot_cmp(cmp_data, t_vec, o_vec, save_path='xxx', if_add=1):
    center_single = np.linspace(np.min(o_vec), np.max(o_vec), cmp_data.shape[1])
    o_d = center_single[1] - center_single[0]
    cmp_data_copy = copy.deepcopy(cmp_data)

    # compute the stacked amplitude
    if if_add:
        stack_amp = []
        for t_bar in cmp_data_copy:
            t_bar_real = t_bar[t_bar != -10000]
            if len(t_bar_real) > 5:
                stack_amp.append(np.mean(t_bar_real))
            else:
                stack_amp.append(0)
        stack_amp = np.array(stack_amp)
        stack_amp = (stack_amp - np.min(stack_amp)) / stack_amp.ptp()

    # scale the amplitude of cmp_data to range (-0.5*d, 0.5*d)
    cmp_data_copy[cmp_data_copy < 0] = 0
    cmp_data_copy = (cmp_data_copy - np.min(cmp_data_copy)) / np.ptp(cmp_data_copy) * o_d

    if if_add:
        fig = plt.figure(figsize=(3, 10), dpi=300)
        ax1 = plt.subplot2grid((10, 3), (0, 0), colspan=2, rowspan=10)
        ax2 = plt.subplot2grid((10, 3), (0, 2), colspan=1, rowspan=10)
        for i in range(cmp_data.shape[1]):
            ax1.fill_betweenx(t_vec, center_single[i], center_single[i] + 2 * cmp_data_copy[:, i], facecolor='black')
        # ax1.set_title(title)
        ax1.set_xlabel('Offset (m)')
        ax1.set_ylabel('Time (ms)')
        ax1.set_xlim(min(o_vec), max(o_vec))
        ax1.set_ylim(min(t_vec), max(t_vec))
        ax1.invert_yaxis()
        ax2.plot(stack_amp, t_vec, c='red', linewidth=0.5)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_ylim(min(t_vec), max(t_vec))
        ax2.invert_yaxis()
        # plt.tight_layout()
    else:
        _, ax = plt.subplots(figsize=(2, 10), dpi=90)
        for i in range(cmp_data.shape[1]):
            ax.fill_betweenx(t_vec, center_single[i], center_single[i] + 2 * cmp_data_copy[:, i], facecolor='black')
        # ax.set_title(title)
        ax.set_xlabel('Offset (m)')
        ax.set_ylabel('Time (ms)')
        ax.set_xlim(min(o_vec), max(o_vec))
        ax.set_ylim(min(t_vec), max(t_vec))
        ax.invert_yaxis()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# Plot spectrum
def plot_spectrum(spectrum, t0_vec, v_vec, save_path=None, VelCurve=None):
    if len(t0_vec) != spectrum.shape[0]:
        t0_vec = np.linspace(t0_vec[0], t0_vec[-1], spectrum.shape[0])
    if len(v_vec) != spectrum.shape[1]:
        v_vec = np.linspace(v_vec[0], v_vec[-1], spectrum.shape[1])
    origin_pwr = 255 - (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum)) * 255
    data_plot = origin_pwr.astype(np.uint8).squeeze()
    data_plot_hot = cv2.applyColorMap(data_plot, cv2.COLORMAP_JET)  # COLORMAP_JET COLORMAP_HOT
    plt.figure(figsize=(2, 10), dpi=300)
    if VelCurve is not None:
        label = ['Auto Velocity', 'Manual Velocity']
        col = ['r', 'darkorange']
        for ind, VelC in enumerate(VelCurve):
            VCCP = copy.deepcopy(VelC)
            VCCP[:, 0] = (VCCP[:, 0]-t0_vec[0]) / (t0_vec[1]-t0_vec[0])
            VCCP[:, 1] = (VCCP[:, 1]-v_vec[0]) / (v_vec[1]-v_vec[0])
            plot_curve(VCCP, col[ind], label[ind])   
        plt.legend()
    plt.imshow(data_plot_hot, aspect='auto')
    plt.xlabel('Velocity (m/s)')
    set_axis(v_vec, t0_vec)
    plt.ylabel('Time (ms)')
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close('all')


# plot original spectrum
def OriSpec(spectrum, save_path=None):
    origin_pwr = 255 - (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum)) * 255
    data_plot = origin_pwr.astype(np.uint8).squeeze()
    data_plot_hot = cv2.applyColorMap(data_plot, cv2.COLORMAP_JET)  # COLORMAP_JET COLORMAP_HOT
    plt.figure(figsize=(2, 10), dpi=300)
    plt.imshow(data_plot_hot, aspect='auto')
    plt.axis('off')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close('all')


# plot the energy peaks
def plot_peaks(peaks, label, if_show=1, if_legend=1, col_index=0,
               marker='x', if_save=0, save_path='xxx'):
    col_list = ['blue', 'black']
    plt.scatter(
        x=peaks[:, 1],
        y=peaks[:, 0],
        c=col_list[col_index],
        marker=marker,
        label=label,
        edgecolors='white'
    )
    if if_legend:
        plt.legend()
    if if_show:
        plt.show()
    if if_save:
        plt.savefig(save_path)
    plt.close('all')


# plot the line curve
def plot_curve(curve, c, label='interpolate curve'):
    plt.plot(curve[:, 1], curve[:, 0],
             c=c,
             label=label, linewidth=1)


# plot the velocity curve of auto picking and manual picking
def plot_vel_curve(auto_curve, label_curve, t0_ind, v_ind, auto_peaks=None, save_path='xxx'):
    plt.figure(figsize=(2, 10), dpi=300)
    ax = plt.gca()
    plt.plot(auto_curve[:, 1], auto_curve[:, 0], c='#ef233c',
             label='Auto Curve', linewidth=1)
    plt.plot(label_curve[:, 1], label_curve[:, 0], c='#8d99ae',
             label='Manual Curve', linewidth=1)
    if auto_peaks is not None:
        plt.scatter(auto_peaks[:, 1], auto_peaks[:, 0], c='blue',
                    s=2, label='Auto Peaks')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Time (ms)')
    plt.xlim((t0_ind[0], t0_ind[-1]))
    plt.xlim((v_ind[0], v_ind[-1]))
    ax.invert_yaxis()
    plt.legend(loc=0)

    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close('all')


def plot_feature_map(feature_map_tensor, save_path='xxx'):
    Para = {
        'ws': [5, 5, 5, 10, 10, 10, 15, 15, 15],
        'st': [1, 2, 3, 1, 2, 3, 1, 2, 3],
        'eec': [1, 1.5, 2, 1, 1.5, 2, 1, 1.5, 2],
        'ln': [5, 8, 12, 5, 8, 12, 5, 8, 12]
    }

    def trim_axs(axs, N):
        axs = axs.flat
        for ax in axs[N:]:
            ax.remove()
        return axs[:N]

    cases = ['ws=%d st=%d eec=%d ln=%d' % 
            (Para['ws'][i], Para['st'][i], Para['eec'][i], Para['ln'][i]) for i in range(9)]
    axs = plt.figure(figsize=(15, 15), constrained_layout=True).subplots(3, 3)
    axs = trim_axs(axs, len(cases))
    count = 0
    spectrum_array = feature_map_tensor[1:]
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 23,
    }
    for ax, case in zip(axs, cases):
        spectrum = spectrum_array[count]
        origin_pwr = 255 - (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum)) * 255
        data_plot = origin_pwr.astype(np.uint8)
        data_plot_hot = cv2.applyColorMap(data_plot, cv2.COLORMAP_JET)
        ax.imshow(data_plot_hot, aspect='auto')
        ax.set_title(case, font1)
        count += 1

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close('all')


# resize the spectrum
def resize_spectrum(spectrum, resize):
    spectrum = transforms.ToPILImage()(spectrum)
    resize_spec = np.array(transforms.Resize(size=resize)(spectrum))
    return resize_spec


# Plot CMP gather
def plot_gather(gather_data, t_ind, group_ind, save_path=None):
    # plt.rcParams['font.sans-serif'] = 'Times New Roman'

    # init data
    group_ind = np.array(group_ind)
    group_ind = (group_ind - np.min(group_ind)).astype(int)
    center_single = np.linspace(np.min(group_ind), np.max(group_ind), gather_data.shape[1])
    o_d = center_single[1] - center_single[0]
    cmp_data_pos = gather_data.copy()
    cmp_data_neg = gather_data.copy()
    cmp_data_pos[cmp_data_pos < 0] = 0
    cmp_data_neg[cmp_data_neg > 0] = 0

    # scale the amplitude of cmp_data to range (-0.5*d, 0.5*d)
    pos_ptp, neg_ptp = np.ptp(cmp_data_pos, axis=0), np.ptp(cmp_data_neg, axis=0)
    cmp_data_pos = cmp_data_pos / pos_ptp * o_d
    cmp_data_neg = cmp_data_neg / neg_ptp * o_d

    # plot cmp trace
    fig, ax = plt.subplots(figsize=(20, 10), dpi=90)
    # negative part
    for i in range(gather_data.shape[1]):
        ax.plot(center_single[i] + cmp_data_neg[:, i], t_ind, linewidth=1, c='black')
    # positive part
    for i in range(gather_data.shape[1]):
        ax.fill_betweenx(t_ind, center_single[i], center_single[i] + cmp_data_pos[:, i], facecolor='black')
    ax.set_xlim(min(group_ind), max(group_ind))
    ax.set_ylim(min(t_ind), max(t_ind))
    ax.set_xlabel('index')
    ax.set_ylabel('time (ms)')

    ax.invert_yaxis()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')

    plt.close('all')


# plot stk velocity curve
def plot_stk_vel(vel_array, save_path):
    plt.figure(figsize=(2, 10), dpi=300)
    ax = plt.gca()
    for k in range(vel_array.shape[0]):
        VC = vel_array[k, vel_array[k, :, 0]> 0, :]
        plt.plot(VC[:, 0], VC[:, 1], label=str(k), linewidth=1)
        plt.scatter(VC[:, 0], VC[:, 1], c='blue', s=3)
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Time (ms)')
    ax.invert_yaxis()
    plt.legend(loc=1)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close('all')


# plot the data sample distributions
def PlotSampleDistributions(SampleDict, save_path='xxx'):
    plt.figure(figsize=(8, 4), dpi=300)
    CList = ['red', 'blue', 'm']
    SSize = [8, 8, 8]
    for i, (part, Sample) in enumerate(SampleDict.items()):
        SampleS = [elm.split('_') for elm in Sample]
        SampleA = np.array(SampleS, dtype=np.int).reshape((-1, 2))
        plt.scatter(SampleA[:, 1], SampleA[:, 0], s=SSize[i], c=CList[i], label=part)
    plt.ylabel('Line')
    plt.xlabel('CDP')
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')

# plot the SP distributions
def PlotSPDistributions(AllIndex, SPIndex, save_path='xxx'):
    USPIndex = AllIndex - SPIndex
    plt.figure(figsize=(8, 4), dpi=100)
    USPScatter = np.array([elm.split('_') for elm in list(USPIndex)], dtype=np.int32).reshape((-1, 2))
    SPScatter = np.array([elm.split('_') for elm in list(SPIndex)], dtype=np.int32).reshape((-1, 2))
    plt.scatter(USPScatter[:, 1], USPScatter[:, 0], s=1, c='black')
    plt.scatter(SPScatter[:, 1], SPScatter[:, 0], s=2, c='red', label='Seed Points')
    plt.ylabel('Line')
    plt.xlabel('CDP')
    plt.legend()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close('all')


# Plot Velocity Field
def PlotVelField(VelField, cdpList, tInd, vInd, LineName, save_path):
    plt.figure(figsize=(20, 10), dpi=300)
    cdpShow = [''] * len(cdpList)
    tshow = [''] * len(tInd)
    cdpInd = np.linspace(0, len(cdpList)-1, num=20).astype(np.int)
    tIndex = np.linspace(0, len(tInd)-1, num=20).astype(np.int)
    for i in tIndex:
        tshow[i] = tInd[i]
    for i in cdpInd:
        cdpShow[i] = cdpList[i]

    # heatmap
    h = sns.heatmap(data=VelField, cmap='jet', linewidths=0, annot=False, cbar=False,
                    vmax=vInd[-1], vmin=vInd[0], xticklabels=cdpShow, yticklabels=tshow,
                    cbar_kws={'label': 'Velocity (m/s)'})

    # color bar
    cb = h.figure.colorbar(h.collections[0])
    cb.ax.tick_params(labelsize=15)

    plt.title('Line %s Velocity Field' % LineName, fontsize=25)
    plt.xlabel('CDP', fontsize=20)
    plt.ylabel('t0', fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.close('all')


def PlotStkGather(gather_data, t_ind, group_ind, save_path='xxx'):
    # init data
    group_ind = np.array(group_ind)
    group_ind = (group_ind - np.min(group_ind)).astype(int)
    center_single = np.linspace(np.min(group_ind), np.max(group_ind), gather_data.shape[1])
    o_d = center_single[1] - center_single[0]
    cmp_data_pos = gather_data.copy()
    cmp_data_neg = gather_data.copy()
    cmp_data_pos[cmp_data_pos < 0] = 0
    cmp_data_neg[cmp_data_neg > 0] = 0

    # scale the amplitude of cmp_data to range (-0.5*d, 0.5*d)
    pos_ptp, neg_ptp = np.ptp(cmp_data_pos, axis=0), np.ptp(cmp_data_neg, axis=0)
    cmp_data_pos = cmp_data_pos / pos_ptp * o_d * 2
    cmp_data_neg = cmp_data_neg / neg_ptp * o_d * 2

    # plot cmp trace
    fig, ax = plt.subplots(figsize=(20, 10), dpi=90)
    # negative part
    for i in range(gather_data.shape[1]):
        ax.plot(center_single[i] + cmp_data_neg[:, i], t_ind, linewidth=0.03, c='black')
    # positive part
    for i in range(gather_data.shape[1]):
        ax.fill_betweenx(t_ind, center_single[i], center_single[i] + cmp_data_pos[:, i], facecolor='black')
    ax.set_xlim(min(group_ind), max(group_ind))
    ax.set_ylim(min(t_ind), max(t_ind))
    ax.set_xlabel('index')
    ax.set_ylabel('time (ms)')

    ax.invert_yaxis()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close('all')


# PastProcessing Visual Work
def PPVisual(SegMat, YEnergy, PeakInd):
    plt.figure(figsize=(5, 10), dpi=100)
    ax1 = plt.subplot2grid((10, 5), (0, 0), colspan=4, rowspan=10)
    ax2 = plt.subplot2grid((10, 5), (0, 4), colspan=1, rowspan=10)

    # plot SegMat
    origin_pwr = (SegMat - np.min(SegMat)) / (np.max(SegMat) - np.min(SegMat)) * 255
    data_plot = origin_pwr.astype(np.uint8).squeeze()
    data_plot_hot = cv2.applyColorMap(data_plot, cv2.COLORMAP_HOT)  # COLORMAP_JET COLORMAP_HOT
    ax1.imshow(data_plot_hot, aspect='auto')
    ax1.set_xlabel('Velocity')
    ax1.set_ylabel('Time')
    ax1.set_xticks([])
    ax1.set_yticks([])
    # plot Energy Stack 
    tInd = np.arange(SegMat.shape[0])
    ax2.plot(YEnergy, tInd, c='red', linewidth=0.5)
    ax2.scatter(YEnergy[PeakInd], PeakInd, c='blue', s=5)
    # ax2.set_xticks([])
    ax2.set_ylim(min(tInd), max(tInd))
    ax2.set_yticks([])
    ax2.invert_yaxis()

    # save the fig
    if not os.path.exists('result/process'):
        os.mkdir('result/process')
    plt.savefig('result/process/PPVisual.png', dpi=100, bbox_inches='tight')
    print('Save to result/process/PPVisual.png')
    plt.close('all')


# hist for energy amp
def EnergyHist(energy):
    plt.figure(figsize=(5, 5), dpi=100)
    plt.hist(energy)
    if not os.path.exists('result/process'):
        os.mkdir('result/process')
    plt.savefig('result/process/EnergyHist.png', dpi=100, bbox_inches='tight')
    plt.close('all')


# visual enhanced process
def EnhancedProcess(ProcessDict, SavePath):
    for name, spectrum in ProcessDict.items():
        plt.figure(figsize=(2, 10), dpi=300)
        origin_pwr = 255 - (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum)) * 255
        data_plot = origin_pwr.astype(np.uint8).squeeze()
        data_plot_hot = cv2.applyColorMap(data_plot, cv2.COLORMAP_JET)  # COLORMAP_JET COLORMAP_HOT
        plt.imshow(data_plot_hot, aspect='auto')
        plt.axis('off')
        plt.savefig(SavePath.replace('.png', '_S%d.png' % int(name)), dpi=300, bbox_inches='tight')
        plt.close('all')


# plot the feature map
def FeatureMap(array, SavePath=None, cmap='Hot'):
    ScaledArray = (array - np.min(array)) / (np.max(array) - np.min(array)) * 255
    ScaledArray = ScaledArray.astype(np.uint8).squeeze()
    plt.figure(figsize=(1, 2), dpi=300)
    if cmap == 'Hot':
        plt.imshow(ScaledArray, cmap='seismic', aspect='auto')
    else:
        print(ScaledArray.shape)
        plt.imshow(ScaledArray, cmap='gray', aspect='auto')
    plt.axis('off')
    if SavePath is None:
        plt.show()
    else:
        plt.savefig(SavePath, dpi=150, bbox_inches='tight')
    plt.close('all')

###################################
# Plot for test Part
###################################

# 1 hist for VMAE
def VMAEHist(VMAEList, SavePath=None):
    plt.figure(figsize=(5, 5), dpi=100)
    plt.hist(VMAEList)
    if SavePath is None:
        plt.show()
    else:
        plt.savefig(SavePath, dpi=100, bbox_inches='tight')
    plt.close('all')


# 2.1 Pwr and Seg Map
def PwrASeg(Pwr, Seg, SavePath=None):
    _, axs = plt.subplots(1, 2)
    # scale data mat
    ScaledPwr = (Pwr - np.min(Pwr)) / (np.max(Pwr) - np.min(Pwr)) * 255
    ScaledPwr = ScaledPwr.astype(np.uint8).squeeze()
    ScaledSeg = (Seg - np.min(Seg)) / (np.max(Seg) - np.min(Seg)) * 255
    ScaledSeg = ScaledSeg.astype(np.uint8).squeeze()
    axs[0].imshow(ScaledPwr, cmap='seismic', aspect='auto')
    axs[1].imshow(ScaledSeg, cmap='gray', aspect='auto')
    plt.axis('off')
    if SavePath is None:
        plt.show()
    else:
        plt.savefig(SavePath, dpi=100, bbox_inches='tight')
    plt.close('all')


# 2.2 Seg with AP and MP 
def SegPick(Seg, t0Vec, vVec, AP, MP, SavePath=None):
    Seg = cv2.resize(np.squeeze(Seg), (len(vVec), len(t0Vec)))
    ScaledSeg = (Seg - np.min(Seg)) / (np.max(Seg) - np.min(Seg)) * 255
    ScaledSeg = ScaledSeg.astype(np.uint8).squeeze()
    plt.figure(figsize=(2, 10), dpi=150)
    label = ['Auto Velocity', 'Manual Velocity']
    col = ['r', 'darkorange']
    for ind, VelC in enumerate([AP, MP]):
        VCCP = copy.deepcopy(VelC)
        VCCP[:, 0] = (VCCP[:, 0]-t0Vec[0]) / (t0Vec[1]-t0Vec[0])
        VCCP[:, 1] = (VCCP[:, 1]-vVec[0]) / (vVec[1]-vVec[0])
        plot_curve(VCCP, col[ind], label[ind])   
    plt.legend()
    plt.imshow(ScaledSeg, aspect='auto', cmap='gray')
    plt.xlabel('Velocity (m/s)')
    set_axis(vVec, t0Vec)
    plt.ylabel('Time (ms)')
    if SavePath is None:
        plt.show()
    else:
        plt.savefig(SavePath, dpi=100, bbox_inches='tight')
    plt.close('all')


# 2.3 CMP gather and NMO result
def CMPNMO(Gth, NMOGth, SavePath=None):
    _, axs = plt.subplots(1, 2)
    # scale data mat
    ScaledGth = (Gth - np.min(Gth)) / (np.max(Gth) - np.min(Gth)) * 255
    ScaledGth = ScaledGth.astype(np.uint8).squeeze()
    ScaledNMO = (NMOGth - np.min(NMOGth)) / (np.max(NMOGth) - np.min(NMOGth)) * 255
    ScaledNMO = ScaledNMO.astype(np.uint8).squeeze()
    axs[0].imshow(ScaledGth, cmap='seismic', aspect='auto')
    axs[1].imshow(ScaledNMO, cmap='seismic', aspect='auto')
    plt.axis('off')
    if SavePath is None:
        plt.show()
    else:
        plt.savefig(SavePath, dpi=100, bbox_inches='tight')
    plt.close('all')