# -*- coding: utf-8 -*-
# @Time    : 2021/11/21 17:55
# @Author  : zxy
# @Email   : 
# @File    : array_simulation_initialization2.py
# @software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from functools import partial
import array_tools as at

# region 仿真参数设置
ele_num = 16  # 阵元个数
theta_of_coherent_inter = [30, 60]  # 相干干扰信号的DOA
INR_coherent = [10, 10]  # 干噪比
theta_of_expect_signal = 0  # 期望信号角度
SNR = 0  # 信噪比
figsize = (5, 4)  # 绘图窗口大小
ylim = (-60, 0)
bandwidth = 10  # 带宽
pulsewidth = 10  # 脉冲宽度
theta_of_normal_inter = [-25, 5]  # 非相干干扰信号的DOA
INR_normal = [10, 10]  # 非相干干扰信号的干噪比
fre_offset = [1, 0.2]  # 频偏，对应为带宽的多少倍
myfont = fm.FontProperties(fname=r'C:\Windows\Fonts\STFANGSO.TTF')  # 中文字体文件路径
gs_kw = {'hspace': 0}
save_flag = False  # 保存仿真结果的标志
title_flag = False  # 有无标题的标志
legend_flag = True  # 有无图例的标志
show_only_signal = True  # 是否绘制仅有期望信号波束形成后的信号波形
axis_flag1 = False  # 是否画示意图的坐标轴
axis_flag2 = True  # 是否画坐标轴
grid_flag = True  # 是否显示网格
figname_list = []  # fig名字的列表
suffix = '.png'  # 保存绘图的文件格式.
suffix2 = '.mp4'
transparent = True  # 保存绘图时背景是否透明
dpi = 600  # 保存绘图时的dpi
fontsize = 15  # 标题的文字大小
save_path = r'C:\Users\ZXY\Desktop\阵列仿真' + '\\'
s_inter_phase = []  # 相干干扰的初始相位
save_config = {'dpi': dpi, 'transparent': transparent}
# endregion

# region 参数检查
if not isinstance(ele_num, int) or ele_num <= 0:
    raise ValueError
if len(theta_of_coherent_inter) != len(INR_coherent):
    raise ValueError
if len(theta_of_normal_inter) != len(INR_normal):
    raise ValueError
if len(INR_normal) != len(fre_offset):
    raise ValueError
# endregion

array = at.Array(ele_num)  # 创建阵列
array.set_expect_azimuth(azimuth=theta_of_expect_signal)  # 设置期望角度

temp_var = at.Lfm(bandwidth=bandwidth, pulsewidth=pulsewidth)
signal_points = temp_var.minimum_points * 128  # 信号采样点数
del temp_var
noise_power = array.get_sigma_power()  # 噪声功率
noise_amplitude = np.sqrt(noise_power)  # 噪声幅度


# region 生成相干干扰信号
theta_of_coherent_inter = np.array(theta_of_coherent_inter)
INR_coherent = np.array(INR_coherent)
amplitude_coherent_inter = np.power(10, INR_coherent / 20) * noise_amplitude  # 相干干扰信号幅度
s_inter = []  # 存储相干干扰信号
for amplitude in amplitude_coherent_inter:
    temp_lfm = at.Lfm(bandwidth=bandwidth, pulsewidth=pulsewidth)
    temp_lfm.signal_points = signal_points  # 增加采样率
    temp_lfm.amplitude = amplitude  # 设置信号幅度
    random_phase = temp_lfm.add_random_phase()  # 增加随机初始相位
    s_inter_phase.append(random_phase)
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

t = expect_signal.t

array.add_signal({'signal': expect_signal.signal, 'theta': theta_of_expect_signal})
my_weight = array.get_guide_vec(azimuth=theta_of_expect_signal)
only_expect_signal_after_beamform = np.conjugate(my_weight.T) @ array.get_output()
only_expect_signal_after_beamform = only_expect_signal_after_beamform.flatten()  # 仅有期望信号常规波束形成
array.clear_signal()


def plot_1(x, y, name, set_ylim=True, **kwargs):
    fig, ax = plt.subplots(figsize=figsize, num=name)
    ax.plot(x, y)
    if title_flag:
        ax.set_title(name, fontproperties=myfont, fontsize=fontsize)
    if grid_flag:
        ax.grid(b=True)
    legend_list = []
    if 'theta_of_coherent_inter' in kwargs:
        flag = False
        if isinstance(kwargs['theta_of_coherent_inter'], (tuple, list, np.ndarray)):
            for item in kwargs['theta_of_coherent_inter']:
                line, = ax.plot([item, item], [ylim[0], ylim[1]], linestyle='--', color='red', label='相干干扰')
                if not flag:
                    legend_list.append(line)
                    flag = True
        elif isinstance(kwargs['theta_of_coherent_inter'], (int, float, np.number)):
            item = kwargs['theta_of_coherent_inter']
            line, = ax.plot([item, item], [ylim[0], ylim[1]], linestyle='--', color='red', label='相干干扰')
            legend_list.append(line)
    if 'theta_of_normal_inter' in kwargs:
        flag = False
        if isinstance(kwargs['theta_of_normal_inter'], (tuple, list, np.ndarray)):
            for item in kwargs['theta_of_normal_inter']:
                line, = ax.plot([item, item], [ylim[0], ylim[1]], linestyle='--', color='orange', label='非相干干扰')
                if not flag:
                    legend_list.append(line)
                    flag = True
        elif isinstance(kwargs['theta_of_normal_inter'], (int, float, np.number)):
            item = kwargs['theta_of_normal_inter']
            line, = ax.plot([item, item], [ylim[0], ylim[1]], linestyle='--', color='orange', label='非相干干扰')
            legend_list.append(line)
    if 'theta_of_expect_signal' in kwargs:
        flag = False
        if isinstance(kwargs['theta_of_expect_signal'], (tuple, list, np.ndarray)):
            for item in kwargs['theta_of_expect_signal']:
                line, = ax.plot([item, item], [ylim[0], ylim[1]], linestyle='--', color='purple', label='期望信号')
                if not flag:
                    legend_list.append(line)
                    flag = True
        elif isinstance(kwargs['theta_of_expect_signal'], (int, float, np.number)):
            item = kwargs['theta_of_expect_signal']
            line, = ax.plot([item, item], [ylim[0], ylim[1]], linestyle='--', color='purple', label='期望信号')
            legend_list.append(line)
    if set_ylim:
        ax.set_ylim(ylim)
    ax.set_xlabel('角度(°)', fontproperties=myfont)
    if legend_flag:
        if legend_list:
            ax.legend(handles=legend_list, prop=myfont)
    if save_flag:
        fig.savefig(fname=save_path + name + suffix, transparent=transparent, dpi=dpi)
    return fig, ax


pp1 = partial(plot_1, theta_of_expect_signal=theta_of_expect_signal,
              theta_of_normal_inter=theta_of_normal_inter[0])


def plot_1_annotate():
    pass


def plot_2(x1, y1, x2, y2, name, label=None):
    fig, ax = plt.subplots(2, 1, gridspec_kw=gs_kw, figsize=figsize, num=name)
    if label is None:
        line1, = ax[0].plot(x1, y1)
        line2, = ax[1].plot(x2, y2, color='red')
    else:
        line1, = ax[0].plot(x1, y1, label=label[0])
        line2, = ax[1].plot(x2, y2, label=label[1], color='red')
    if not axis_flag2:
        for item in ax:
            item.set_axis_off()
    else:
        for item in ax:
            item.set_yticks([])
        ax[-1].set_xlabel('us')
    if legend_flag and not(label is None):
        fig.legend(handles=[line1, line2], prop=myfont, loc='upper right')
    if title_flag:
        fig.suptitle(name, fontproperties=myfont, fontsize=fontsize)
    if save_flag:
        fig.savefig(fname=save_path + name + suffix, transparent=transparent, dpi=dpi)
    return fig, ax
