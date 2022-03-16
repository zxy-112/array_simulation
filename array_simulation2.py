# -*- coding: utf-8 -*-
# @Time    : 2021/11/21 17:52
# @Author  : zxy
# @Email   : 
# @File    : array_simulation2.py
# @software: PyCharm

from array_simulation_initialization2 import *
import zxy_tools as zt


def my_one_plot_setter(fighandle, axhandle):
    if title_flag:
        axhandle.set_title(fig_name, fontproperties=myfont, fontsize=fontsize)
    if save_flag:
        fighandle.savefig(fname=r'C:\Users\ZXY\Desktop\阵列仿真' + '\\' + fig_name + suffix,
                          transparent=transparent, dpi=dpi)


def my_two_plot_setter(fighandle, linehandle):
    if legend_flag:
        fighandle.legend(handles=linehandle, prop=myfont, loc='upper right')
    if title_flag:
        fighandle.suptitle(figname_list[-1], fontproperties=myfont, fontsize=fontsize)
    if save_flag:
        fighandle.savefig(fname=r'C:\Users\ZXY\Desktop\阵列仿真' + '\\' + figname_list[-1] + suffix,
                          transparent=transparent, dpi=dpi)


def my_plot_two(signal1, signal2, label='有相干干扰'):
    if show_only_signal:
        fig_inter, ax_inter = plt.subplots(2, 1, gridspec_kw=gs_kw, figsize=figsize, num=figname_list[-1])
        line1_inter, = ax_inter[0].plot(expect_signal.t, np.real(signal1), label='无干扰')
        line2_inter, = ax_inter[1].plot(expect_signal.t, np.real(signal2), label=label, color='red')
        if not axis_flag2:
            for item_inter in ax_inter:
                item_inter.set_axis_off()
        my_two_plot_setter(fighandle=fig_inter, linehandle=[line1_inter, line2_inter])
    else:
        fig_inter, ax_inter = plt.subplots(figsize=figsize, num=figname_list[-1])
        ax_inter.plot(expect_signal.t, np.real(signal_after_beamform))
        my_one_plot_setter(fighandle=fig_inter, axhandle=ax_inter)


# region 绘制信号
# region 绘制期望信号和相干干扰信号
fig_name = '期望信号和相干干扰信号示意图'
figname_list.append(fig_name)
fig, ax = plt.subplots(2, 1, gridspec_kw=gs_kw, figsize=figsize, num=figname_list[-1])
line1, = ax[0].plot(expect_signal.t, np.real(expect_signal.signal), label='期望信号')
line2, = ax[1].plot(expect_signal.t, np.real(s_inter[0].signal), color='red', label='相干干扰信号')
if not axis_flag1:
    for item in ax:
        item.set_axis_off()
my_two_plot_setter(fighandle=fig, linehandle=[line1, line2])
# endregion


# region 绘制期望信号和非相干干扰信号
fig_name = '期望信号和非相干干扰信号示意图'
figname_list.append(fig_name)
fig, ax = plt.subplots(2, 1, gridspec_kw=gs_kw, figsize=figsize, num=figname_list[-1])
line1, = ax[0].plot(expect_signal.t, np.real(expect_signal.signal), label='期望信号')
line2, = ax[1].plot(expect_signal.t, np.real(s_normal_inter[0].signal), color='red', label='非相干干扰信号')
if not axis_flag1:
    for item in ax:
        item.set_axis_off()
my_two_plot_setter(fighandle=fig, linehandle=[line1, line2])
# endregion


# region 绘制期望信号和非相干干扰
fig_name = '期望信号和非相干干扰信号示意图2'
figname_list.append(fig_name)
fig, ax = plt.subplots(2, 1, gridspec_kw=gs_kw, figsize=figsize, num=figname_list[-1])
line1, = ax[0].plot(expect_signal.t, np.real(expect_signal.signal), label='期望信号')
default_delay = expect_signal.delay
expect_signal.delay = 2  # 将时延设置为2
line2, = ax[1].plot(expect_signal.t, np.real(expect_signal.signal), color='red', label='非相干干扰信号')
expect_signal.delay = default_delay  # 设置为初始时延
if not axis_flag1:
    for item in ax:
        item.set_axis_off()
my_two_plot_setter(fighandle=fig, linehandle=[line1, line2])
# endregion
# endregion


# region 常规波束形成
response_traditional, theta = array.pattern(expect_azimuth=theta_of_expect_signal)

fig_name = '常规波束形成'
figname_list.append(fig_name)
fig, ax = plt.subplots(figsize=figsize, num=figname_list[-1])
ax.plot(theta, zt.zxy_decibel(response_traditional))
my_one_plot_setter(fighandle=fig, axhandle=ax)
# endregion


# region 只有干扰信号MVDR
array.add_signal({'signal': s_inter[0].signal, 'theta': theta_of_coherent_inter[0]})
my_weight = array.get_mvdr_weight(expect_azimuth=theta_of_expect_signal)
response, theta = array.pattern(weights=my_weight)  # 方向图

fig_name = '只有一个相干干扰信号没有期望信号下MVDR波束形成'
figname_list.append(fig_name)
fig, ax = plt.subplots(figsize=figsize, num=figname_list[-1])
ax.plot(theta, zt.zxy_decibel(response))
my_one_plot_setter(fighandle=fig, axhandle=ax)

array.clear_signal()
# endregion


# region 只有期望信号
array.add_signal({'signal': expect_signal.signal, 'theta': theta_of_expect_signal})
my_weight = array.get_mvdr_weight(expect_azimuth=theta_of_expect_signal)
response, theta = array.pattern(weights=my_weight)

fig_name = '无干扰信号下MVDR波束形成结果'
figname_list.append(fig_name)
fig, ax = plt.subplots(figsize=figsize, num=figname_list[-1])
ax.plot(theta, zt.zxy_decibel(response))
my_one_plot_setter(fighandle=fig, axhandle=ax)

signal_after_beamform = np.conjugate(array.get_mvdr_weight().reshape((1, -1))) @ array.get_output()
signal_after_beamform = signal_after_beamform.flatten()
only_expect_mvdr_result = signal_after_beamform  # 用于后续绘图

fig_name = '无干扰下MVDR波束形成后的信号波形'
figname_list.append(fig_name)
fig, ax = plt.subplots(figsize=figsize, num=figname_list[-1])
ax.plot(expect_signal.t, np.real(signal_after_beamform))
my_one_plot_setter(fighandle=fig, axhandle=ax)

signal_after_beamform = (np.conjugate(array.get_guide_vec(azimuth=theta_of_expect_signal).reshape((1, -1)))
                         @ array.get_output()).flatten()
only_expect_signal = signal_after_beamform

fig_name = '无干扰下常规波束形成后的信号波形'
figname_list.append(fig_name)
fig, ax = plt.subplots(figsize=figsize, num=figname_list[-1])
ax.plot(expect_signal.t, np.real(only_expect_signal))
my_one_plot_setter(fighandle=fig, axhandle=ax)

array.clear_signal()
# endregion


# region 只有两个相干干扰
for signal_item, theta_item in zip(s_inter[0:2], theta_of_coherent_inter[0:2]):
    array.add_signal({'signal': signal_item.signal, 'theta': theta_item})
my_weight = array.get_mvdr_weight(expect_azimuth=theta_of_expect_signal)
response, theta = array.pattern(weights=my_weight)  # MVDR方向图

fig_name = '只有两个相干干扰MVDR波束形成'
figname_list.append(fig_name)
fig, ax = plt.subplots(figsize=figsize, num=figname_list[-1])
ax.plot(theta, zt.zxy_decibel(response))
my_one_plot_setter(fighandle=fig, axhandle=ax)

array.clear_signal()
# endregion


# region 期望信号和一个相干干扰
array.add_signal({'signal': s_inter[0].signal, 'theta': theta_of_coherent_inter[0]})
array.add_signal({'signal': expect_signal.signal, 'theta': theta_of_expect_signal})

# region MVDR波束形成
my_weight = array.get_mvdr_weight(expect_azimuth=theta_of_expect_signal)
response, theta = array.pattern(weights=my_weight)  # mvdr方向图

fig_name = '期望信号和一个相干干扰MVDR波束形成'
figname_list.append(fig_name)
fig, ax = plt.subplots(figsize=figsize, num=figname_list[-1])
ax.plot(theta, zt.zxy_decibel(response))
ax.plot([0, 0], [-60, 0], linestyle='--', color='red')
ax.plot([30, 30], [-60, 0], linestyle='--', color='red')
ax.grid(b=True)
ax.set_ylim(-60, 0)
ax.set_xlabel('角度（°）', fontproperties=myfont)
ax.set_ylabel('方向图增益(dB)', fontproperties=myfont)
my_one_plot_setter(fighandle=fig, axhandle=ax)

signal_after_beamform = np.conjugate(array.get_mvdr_weight().reshape((1, -1))) @ array.get_output()
signal_after_beamform = signal_after_beamform.flatten()

fig_name = '有无相干干扰MVDR波束形成后的信号波形'
figname_list.append(fig_name)
my_plot_two(signal1=only_expect_mvdr_result, signal2=signal_after_beamform)
# endregion

# region 协方差矩阵构建波束形成
guide_vec = array.get_guide_vec(azimuth=theta_of_coherent_inter[0])
guide_vec = guide_vec.reshape((-1, 1))
power_of_inter = np.power(10, INR_coherent[0]/10)
cov_mat = power_of_inter * guide_vec @ np.conjugate(guide_vec.T) + np.eye(guide_vec.size, dtype=guide_vec.dtype)
weights = np.linalg.pinv(cov_mat) @ array.get_guide_vec(azimuth=theta_of_expect_signal).reshape((-1, 1))
response, theta = array.pattern(expect_azimuth=theta_of_expect_signal, weights=weights)  # 协方差矩阵构建波束形成方向图
response = zt.zxy_decibel(response)

fig_name = '协方差矩阵构建波束形成'
figname_list.append(fig_name)
fig, ax = plt.subplots(figsize=figsize, num=figname_list[-1])
ax.plot(theta, response)
my_one_plot_setter(fighandle=fig, axhandle=ax)

signal_after_beamform = np.conjugate(weights.T) @ array.get_output()
signal_after_beamform = signal_after_beamform.flatten()

fig_name = '有无相干干扰下协方差矩阵构建波束形成后的信号波形'
figname_list.append(fig_name)
my_plot_two(signal1=only_expect_signal, signal2=signal_after_beamform)
# endregion

# region mcmv波束形成
weights = array.get_mcmv_weight(expect_azimuth=theta_of_expect_signal,
                                coherent_inter_azimuth=(theta_of_coherent_inter[0],))
response, theta = array.pattern(weights=weights)

fig_name = 'MCMV算法波束形成'
figname_list.append(fig_name)
fig, ax = plt.subplots(figsize=figsize, num=figname_list[-1])
ax.plot(theta, zt.zxy_decibel(response))
my_one_plot_setter(fighandle=fig, axhandle=ax)

signal_after_beamform = np.conjugate(weights.T) @ array.get_output()
signal_after_beamform = signal_after_beamform.flatten()

fig_name = '有无相干干扰MCMV波束形成后的信号波形'
figname_list.append(fig_name)
my_plot_two(signal1=only_expect_signal, signal2=signal_after_beamform)
# endregion

# region ctmv波束形成
weights = array.get_ctmv_weight(expect_azimuth=theta_of_expect_signal,
                                coherent_inter_azimuth=(theta_of_coherent_inter[0], ))
response, theta = array.pattern(weights=weights)

fig_name = 'ctmv算法波束形成'
figname_list.append(fig_name)
fig, ax = plt.subplots(figsize=figsize, num=figname_list[-1])
ax.plot(theta, at.zxy_decibel(response))
my_one_plot_setter(fighandle=fig, axhandle=ax)

signal_after_beamform = np.conjugate(weights.T) @ array.get_output()
signal_after_beamform = signal_after_beamform.flatten()
fig_name = '有无相干干扰CTMV波束形成后的信号波形'
figname_list.append(fig_name)
my_plot_two(signal1=only_expect_signal, signal2=signal_after_beamform)
# endregion

array.clear_signal()
# endregion


# region 期望信号和非相干干扰信号
array.add_signal({'signal': expect_signal.signal, 'theta': theta_of_expect_signal})
array.add_signal({'signal': s_normal_inter[0].signal, 'theta': theta_of_normal_inter[0]})

my_weight = array.get_mvdr_weight(expect_azimuth=theta_of_expect_signal)
response, theta = array.pattern(weights=my_weight)  # 方向图

fig_name = '期望信号和非相干干扰信号下MVDR波束形成'
figname_list.append(fig_name)
fig, ax = plt.subplots(figsize=figsize, num=figname_list[-1])
ax.plot(theta, zt.zxy_decibel(response))
my_one_plot_setter(fighandle=fig, axhandle=ax)

signal_after_beamform = np.conjugate(array.get_mvdr_weight().reshape((1, -1))) @ array.get_output()  # MVDR后信号波形
signal_after_beamform = signal_after_beamform.flatten()

fig_name = '有无非相干干扰MVDR波束形成后的信号波形'
figname_list.append(fig_name)
my_plot_two(signal1=only_expect_signal, signal2=signal_after_beamform, label='有非相干干扰')

array.clear_signal()
# endregion


# region 期望信号和两个相干干扰
pass
# endregion

plt.show()
if save_flag:
    plt.close('all')
