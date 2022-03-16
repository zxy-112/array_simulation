import numpy as np
import matplotlib.pyplot as plt
import zxy_tools as zt
from array_simulation_initialization import *

array = at.Array(ele_num)  # 创建阵列
array.set_expect_azimuth(azimuth=theta_of_expect_signal)  # 设置期望角度
noise_power = array.get_sigma_power()
noise_amplitude = np.sqrt(noise_power)

# region 生成相干干扰信号
theta_of_inter = np.array(theta_of_coherent_inter)
INR_coherent = np.array(INR_coherent)
amplitude_coherent_inter = np.power(10, INR_coherent / 20) * noise_amplitude  # 相干干扰信号幅度
s_inter = []  # 存储相干干扰信号
for amplitude in amplitude_coherent_inter:
    temp_lfm = at.Lfm(bandwidth=bandwidth, pulsewidth=pulsewidth)
    temp_lfm.signal_points = signal_points  # 增加采样率
    temp_lfm.amplitude = amplitude  # 设置信号幅度
    temp_lfm.add_random_phase()  # 增加随机初始相位
    s_inter.append(temp_lfm)
# endregion

# region 产生非相干干扰信号
theta_of_normal_inter = np.array(theta_of_normal_inter)
INR_normal = np.array(INR_normal)
fre_offset = np.array(fre_offset) * bandwidth
amplitude_normal_inter = np.power(10, INR_normal/20) * noise_amplitude
s_normal_inter = []  # 存储非相干干扰信号
for amplitude, offset in zip(amplitude_normal_inter, fre_offset):
    temp_lfm = at.Lfm(bandwidth=bandwidth, pulsewidth=pulsewidth)
    temp_lfm.signal_points = signal_points  # 增加采样率
    temp_lfm.fre_offset = offset  # 增加频率偏移来实现非相干
    temp_lfm.amplitude = amplitude  # 设置信号幅度
    temp_lfm.add_random_phase()  # 增加随机相位
    s_normal_inter.append(temp_lfm)
# endregion

# region 产生与相干干扰相干的期望信号
signal_amplitude = np.power(10, SNR/20) * noise_amplitude  # 期望信号幅度
expect_signal = at.Lfm(bandwidth=bandwidth, pulsewidth=pulsewidth)
expect_signal.signal_points = signal_points
expect_signal.amplitude = signal_amplitude  # 设置信号幅度
expect_signal.add_random_phase()  # 增加随机相位
# endregion

###########################################################################################################
# 绘制信号

# region 绘制期望信号和相干干扰信号
fig_name = '期望信号和相干干扰信号示意图'
fig, ax = plt.subplots(2, 1, gridspec_kw=gs_kw, figsize=figsize, num=fig_name)
figname_list.append(fig_name)
line1, = ax[0].plot(expect_signal.t, np.real(expect_signal.signal), label='期望信号')
line2, = ax[1].plot(expect_signal.t, np.real(s_inter[0].signal), color='red', label='相干干扰信号')
if title_flag:
    fig.suptitle(fig_name, fontproperties=myfont, fontsize=fontsize)
if legend_flag:
    fig.legend(handles=[line1, line2], prop=myfont, loc='upper right')
for item in ax:
    item.set_axis_off()
if save_flag:
    fig.savefig(fname=r'C:\Users\ZXY\Desktop\阵列仿真' + '\\' + fig_name + suffix, transparent=transparent, dpi=dpi)
# endregion

# region 绘制期望信号和非相干干扰信号
fig_name = '期望信号和非相干干扰信号示意图'
figname_list.append(fig_name)
fig, ax = plt.subplots(2, 1, gridspec_kw=gs_kw, figsize=figsize, num=fig_name)
line1, = ax[0].plot(expect_signal.t, np.real(expect_signal.signal), label='期望信号')
line2, = ax[1].plot(expect_signal.t, np.real(s_normal_inter[0].signal), color='red', label='非相干干扰信号')
if legend_flag:
    fig.legend(handles=[line1, line2], prop=myfont, loc='upper right')
if title_flag:
    fig.suptitle(fig_name, fontproperties=myfont, fontsize=fontsize)
for item in ax:
    item.set_axis_off()
if save_flag:
    fig.savefig(fname=r'C:\Users\ZXY\Desktop\阵列仿真' + '\\' + fig_name + suffix, transparent=transparent, dpi=dpi)
# endregion

# region 绘制期望信号和非相干干扰
fig_name = '期望信号和非相干干扰信号示意图2'
figname_list.append(fig_name)
fig, ax = plt.subplots(2, 1, gridspec_kw=gs_kw, figsize=figsize, num=fig_name)
line1, = ax[0].plot(expect_signal.t, np.real(expect_signal.signal), label='期望信号')
default_delay = expect_signal.delay
expect_signal.delay = 2  # 将时延设置为2
line2, = ax[1].plot(expect_signal.t, np.real(expect_signal.signal), color='red', label='非相干干扰信号')
expect_signal.delay = default_delay  # 设置为初始时延
if legend_flag:
    fig.legend(handles=[line1, line2], prop=myfont, loc='upper right')
if title_flag:
    fig.suptitle(fig_name, fontproperties=myfont, fontsize=fontsize)
for item in ax:
    item.set_axis_off()
if save_flag:
    fig.savefig(fname=r'C:\Users\ZXY\Desktop\阵列仿真' + '\\' + fig_name + suffix, transparent=transparent, dpi=dpi)
# endregion

###########################################################################################################
# 波束形成

# region 常规波束形成
response_traditional, theta = array.pattern(expect_azimuth=theta_of_expect_signal)

fig_name = '常规波束形成'
figname_list.append(fig_name)
fig, ax = plt.subplots(figsize=figsize, num=fig_name)
ax.plot(theta, zt.zxy_decibel(response_traditional))
if title_flag:
    ax.set_title(fig_name, fontproperties=myfont, fontsize=fontsize)
if save_flag:
    fig.savefig(fname=r'C:\Users\ZXY\Desktop\阵列仿真' + '\\' + fig_name + suffix, transparent=transparent, dpi=dpi)
# endregion

# region 只有干扰信号MVDR
array.add_signal({'signal': s_inter[0].signal, 'theta': theta_of_inter[0]})
my_weight = array.get_mvdr_weight(expect_azimuth=theta_of_expect_signal)
response, theta = array.pattern(weights=my_weight)  # 方向图

fig_name = '只有一个相干干扰信号没有期望信号下MVDR波束形成'
figname_list.append(fig_name)
fig, ax = plt.subplots(figsize=figsize, num=fig_name)
ax.plot(theta, zt.zxy_decibel(response))
if title_flag:
    ax.set_title(fig_name, fontproperties=myfont, fontsize=fontsize)
if save_flag:  # 存储方向图
    fig.savefig(fname=r'C:\Users\ZXY\Desktop\阵列仿真' + '\\' + fig_name + suffix, transparent=transparent, dpi=dpi)
# endregion
array.clear_signal()

# region 只有期望信号MVDR
array.add_signal({'signal': expect_signal.signal, 'theta': theta_of_expect_signal})
my_weight = array.get_mvdr_weight(expect_azimuth=theta_of_expect_signal)
response, theta = array.pattern(weights=my_weight)

fig_name = '只有期望信号下MVDR波束形成结果'
figname_list.append(fig_name)
fig, ax = plt.subplots(figsize=figsize, num=fig_name)
ax.plot(theta, zt.zxy_decibel(response))
if title_flag:
    ax.set_title(fig_name, fontproperties=myfont, fontsize=fontsize)
if save_flag:
    fig.savefig(fname=r'C:\Users\ZXY\Desktop\阵列仿真' + '\\' + fig_name + suffix, transparent=transparent, dpi=dpi)

special_figname = '有无干扰下MVDR波束形成后的信号波形'
begin_num = 0
figname_list.append(special_figname + str(begin_num))
signal_after_beamform = np.conjugate(array.get_mvdr_weight().reshape((1, -1))) @ array.get_output()
signal_after_beamform = signal_after_beamform.flatten()
only_expect_mvdr_result = signal_after_beamform  # 用于后续绘图
special_fig, special_ax = plt.subplots(2, 1, gridspec_kw=gs_kw, figsize=figsize, num=figname_list[-1])
special_line1, = special_ax[0].plot(expect_signal.t, np.real(signal_after_beamform), label='只有期望信号')
if title_flag:
    special_fig.suptitle(figname_list[-1][:-1], fontproperties=myfont, fontsize=fontsize)

fig_name = '无干扰下MVDR波束形成后的信号波形'
figname_list.append(fig_name)
fig, ax = plt.subplots(figsize=figsize, num=fig_name)
ax.plot(expect_signal.t, np.real(signal_after_beamform))
if title_flag:
    ax.set_title(fig_name, fontproperties=myfont, fontsize=fontsize)
if save_flag:
    fig.savefig(fname=r'C:\Users\ZXY\Desktop\阵列仿真' + '\\' + fig_name + suffix, transparent=transparent, dpi=dpi)

begin_num = begin_num + 1
figname_list.append(special_figname + str(begin_num))
special_fig2, special_ax2 = plt.subplots(2, 1, gridspec_kw=gs_kw, figsize=figsize, num=figname_list[-1])
special_line3, = special_ax2[0].plot(expect_signal.t, np.real(signal_after_beamform), label='只有期望信号')
if title_flag:
    special_fig2.suptitle(figname_list[-1][:-1], fontproperties=myfont, fontsize=fontsize)

figname_list.append('有无干扰下协方差矩阵构建波束形成后的信号波形')
special_fig3, special_ax3 = plt.subplots(2, 1, gridspec_kw=gs_kw, figsize=figsize, num=figname_list[-1])
special_line5, = special_ax3[0].plot(expect_signal.t, np.real(signal_after_beamform), label='只有期望信号')
if title_flag:
    special_fig3.suptitle(figname_list[-1], fontproperties=myfont, fontsize=fontsize)
# endregion
array.clear_signal()

# region 只有两个相干干扰
for signal_item, theta_item in zip(s_inter, theta_of_inter):
    array.add_signal({'signal': signal_item.signal, 'theta': theta_item})
my_weight = array.get_mvdr_weight(expect_azimuth=theta_of_expect_signal)
response, theta = array.pattern(weights=my_weight)  # MVDR方向图

fig_name = '只有两个相干干扰MVDR波束形成'
figname_list.append(fig_name)
fig, ax = plt.subplots(figsize=figsize, num=fig_name)
ax.plot(theta, zt.zxy_decibel(response))
if title_flag:
    ax.set_title(fig_name, fontproperties=myfont, fontsize=fontsize)
if save_flag:
    fig.savefig(fname=r'C:\Users\ZXY\Desktop\阵列仿真' + '\\' + fig_name + suffix, transparent=transparent, dpi=dpi)
# endregion
array.clear_signal()

# region 期望信号和一个相干干扰
array.add_signal({'signal': s_inter[0].signal, 'theta': theta_of_inter[0]})
array.add_signal({'signal': expect_signal.signal, 'theta': theta_of_expect_signal})

# region MVDR波束形成
my_weight = array.get_mvdr_weight(expect_azimuth=theta_of_expect_signal)
response, theta = array.pattern(weights=my_weight)  # mvdr方向图

# region 绘图
fig_name = '期望信号和一个相干干扰MVDR波束形成'
figname_list.append(fig_name)
fig, ax = plt.subplots(figsize=figsize, num=fig_name)
ax.plot(theta, zt.zxy_decibel(response))
if title_flag:
    ax.set_title(fig_name, fontproperties=myfont, fontsize=fontsize)
if save_flag:  # 存储方向图
    fig.savefig(fname=r'C:\Users\ZXY\Desktop\阵列仿真' + '\\' + fig_name + suffix, transparent=transparent, dpi=dpi)
# endregion

signal_after_beamform = np.conjugate(array.get_mvdr_weight().reshape((1, -1))) @ array.get_output()
signal_after_beamform = signal_after_beamform.flatten()
special_line2, = special_ax[1].plot(expect_signal.t, np.real(signal_after_beamform), label='有相干干扰', color='red')
if legend_flag:
    special_fig.legend(handles=[special_line1, special_line2], prop=myfont, loc='upper right')
# endregion

# region 协方差矩阵构建波束形成
guide_vec = array.get_guide_vec(azimuth=theta_of_inter[0])
guide_vec = guide_vec.reshape((-1, 1))
power_of_inter = np.power(10, INR_coherent[0]/10)
cov_mat = power_of_inter * guide_vec @ np.conjugate(guide_vec.T) + np.eye(guide_vec.size, dtype=guide_vec.dtype)
weights = np.linalg.pinv(cov_mat) @ array.get_guide_vec(azimuth=theta_of_expect_signal).reshape((-1, 1))
response, theta = array.pattern(expect_azimuth=theta_of_expect_signal, weights=weights)  # 协方差矩阵构建波束形成方向图
response = zt.zxy_decibel(response)

fig_name = '协方差矩阵构建波束形成'
figname_list.append(fig_name)
fig, ax = plt.subplots(figsize=figsize, num=fig_name)
ax.plot(theta, response)
if title_flag:
    ax.set_title(fig_name, fontproperties=myfont, fontsize=fontsize)
if save_flag:
    fig.savefig(fname=r'C:\Users\ZXY\Desktop\阵列仿真' + '\\' + fig_name + suffix, transparent=transparent, dpi=dpi)

signal_after_beamform = np.conjugate(weights.T) @ array.get_output()
signal_after_beamform = signal_after_beamform.flatten()
special_line6, = special_ax3[1].plot(expect_signal.t, np.real(signal_after_beamform), label='有相干干扰', color='red')
if legend_flag:
    special_fig3.legend(handles=[special_line5, special_line6], prop=myfont, loc='upper right')
if save_flag:  # 存储构建协方差矩阵波束形成后的信号波形
    special_fig3.savefig(fname=r'C:\Users\ZXY\Desktop\阵列仿真\构建协方差矩阵波束形成后的信号波形.png',
                         transparent=transparent, dpi=dpi)
# endregion

# region mcmv波束形成
weights = array.get_mcmv_weight(expect_azimuth=theta_of_expect_signal,
                                coherent_inter_azimuth=(theta_of_coherent_inter[0],))
response, theta = array.pattern(weights=weights)

fig_name = 'MCMV算法波束形成'
figname_list.append(fig_name)
fig, ax = plt.subplots(figsize=figsize, num=fig_name)
ax.plot(theta, zt.zxy_decibel(response))
if title_flag:
    ax.set_title(fig_name, fontproperties=myfont, fontsize=fontsize)
if save_flag:
    fig.savefig(fname=r'C:\Users\ZXY\Desktop\阵列仿真' + '\\' + fig_name + suffix, transparent=transparent, dpi=dpi)

signal_after_beamform = np.conjugate(weights.T) @ array.get_output()
signal_after_beamform = signal_after_beamform.flatten()
fig_name = 'MCMV波束形成后的信号波形'
if show_only_signal:
    fig, ax = plt.subplots(2, 1, gridspec_kw=gs_kw, figsize=figsize, num=fig_name)
    line7, = ax[0].plot(expect_signal.t, np.real(only_expect_mvdr_result), label='无干扰')
    line8, = ax[1].plot(expect_signal.t, np.real(signal_after_beamform), label='有相干干扰', color='red')
    if title_flag:
        fig.suptitle(fig_name, fontproperties=myfont, fontsize=fontsize)
    if legend_flag:
        fig.legend(handles=[line7, line8], prop=myfont)
    if save_flag:
        fig.savefig(fname=r'C:\Users\ZXY\Desktop\阵列仿真' + '\\' + fig_name + suffix, transparent=transparent, dpi=dpi)
else:
    fig, ax = plt.subplots(figsize=figsize, num=fig_name)
    ax.plot(expect_signal.t, np.real(signal_after_beamform))
    if title_flag:
        ax.set_title(fig_name, fontproperties=myfont, fontsize=fontsize)
    if save_flag:
        fig.savefig(fname=r'C:\Users\ZXY\Desktop\阵列仿真' + '\\' + fig_name + suffix, transparent=transparent, dpi=dpi)
# endregion

# region ctmv波束形成
weights = array.get_ctmv_weight(expect_azimuth=theta_of_expect_signal,
                                coherent_inter_azimuth=(theta_of_coherent_inter[0], ))
response, theta = array.pattern(weights=weights)

fig_name = 'ctmv算法波束形成'
figname_list.append(fig_name)
fig, ax = plt.subplots(figsize=figsize, num=fig_name)
ax.plot(theta, at.zxy_decibel(response))
if title_flag:
    ax.set_title(fig_name, fontproperties=myfont, fontsize=fontsize)
if save_flag:
    fig.savefig(fname=r'C:\Users\ZXY\Desktop\阵列仿真' + '\\' + fig_name + suffix, transparent=transparent, dpi=dpi)

signal_after_beamform = np.conjugate(weights.T) @ array.get_output()
signal_after_beamform = signal_after_beamform.flatten()
fig_name = 'CTMV波束形成后的信号波形'
if show_only_signal:
    fig, ax = plt.subplots(2, 1, gridspec_kw=gs_kw, figsize=figsize, num=fig_name)
    line9, = ax[0].plot(expect_signal.t, np.real(only_expect_mvdr_result), label='无干扰')
    line10, = ax[1].plot(expect_signal.t, np.real(signal_after_beamform), label='有相干干扰', color='red')
    if title_flag:
        fig.suptitle(fig_name, fontproperties=myfont, fontsize=fontsize)
    if legend_flag:
        fig.legend(handles=[line9, line10], prop=myfont)
    if save_flag:
        fig.savefig(fname=r'C:\Users\ZXY\Desktop\阵列仿真' + '\\' + fig_name + suffix, transparent=transparent, dpi=dpi)
else:
    fig, ax = plt.subplots(figsize=figsize, num=fig_name)
    ax.plot(expect_signal.t, np.real(signal_after_beamform))
    if title_flag:
        ax.set_title(fig_name, fontproperties=myfont, fontsize=fontsize)
    if save_flag:
        fig.savefig(fname=r'C:\Users\ZXY\Desktop\阵列仿真' + '\\' + fig_name + suffix, transparent=transparent, dpi=dpi)
# endregion

# endregion
array.clear_signal()

# region 期望信号和非相干干扰信号
array.add_signal({'signal': expect_signal.signal, 'theta': theta_of_expect_signal})
array.add_signal({'signal': s_normal_inter[0].signal, 'theta': theta_of_normal_inter[0]})
my_weight = array.get_mvdr_weight(expect_azimuth=theta_of_expect_signal)
response, theta = array.pattern(weights=my_weight)  # 方向图

fig_name = '期望信号和非相干干扰信号下MVDR波束形成'
figname_list.append(fig_name)
fig, ax = plt.subplots(figsize=figsize, num=fig_name)
ax.plot(theta, zt.zxy_decibel(response))
if title_flag:
    ax.set_title(fig_name, fontproperties=myfont, fontsize=fontsize)
if save_flag:  # 存储方向图
    fig.savefig(fname=r'C:\Users\ZXY\Desktop\阵列仿真' + '\\' + fig_name + suffix, transparent=transparent, dpi=dpi)

signal_after_beamform = np.conjugate(array.get_mvdr_weight().reshape((1, -1))) @ array.get_output()  # MVDR后信号波形
signal_after_beamform = signal_after_beamform.flatten()
special_line4, = special_ax2[1].plot(expect_signal.t, np.real(signal_after_beamform), label='有非相干干扰', color='red')
if legend_flag:
    special_fig2.legend(handles=[special_line3, special_line4], prop=myfont, loc='upper right')

for item in np.hstack((special_ax, special_ax2)):
    item.set_axis_off()
if save_flag:  # 存储两个MVDR波束形成后的信号波形，分别对应相干干扰和非相干干扰
    special_fig.savefig(fname=r'C:\Users\ZXY\Desktop\阵列仿真\MVDR波束形成后的波形.png', transparent=transparent, dpi=dpi)
    special_fig2.savefig(fname=r'C:\Users\ZXY\Desktop\阵列仿真\MVDR波束形成后的波形2.png', transparent=transparent, dpi=dpi)
# endregion
array.clear_signal()

# region 期望信号和两个相干干扰

# endregion
###########################################################################################################

if save_flag:
    plt.close('all')


def my_one_plot():
    pass


def my_two_plot():
    pass
