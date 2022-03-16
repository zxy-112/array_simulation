# -*- coding: utf-8 -*-
# @Time    : 2021/11/16 21:28
# @Author  : zxy
# @Email   : 
# @File    : array_simulation_initialization.py
# @software: PyCharm

import matplotlib.font_manager as fm
import array_tools as at

# region 仿真参数设置
ele_num = 16  # 阵元个数
theta_of_coherent_inter = [30, 60]  # 相干干扰信号的DOA
INR_coherent = [10, 10]  # 干噪比
theta_of_expect_signal = 0  # 期望信号角度
SNR = 0  # 信噪比
figsize = (6, 5)  # 绘图窗口大小
bandwidth = 10  # 带宽
pulsewidth = 10  # 脉冲宽度
theta_of_normal_inter = [30]  # 非相干干扰信号的DOA
INR_normal = [10]  # 非相干干扰信号的干噪比
fre_offset = [2]  # 频偏，对应为带宽的多少倍
myfont = fm.FontProperties(fname=r'C:\Windows\Fonts\STFANGSO.TTF')  # 中文字体文件路径
gs_kw = {'hspace': 0}
save_flag = False  # 保存仿真结果的标志
title_flag = False  # 有无标题的标志
legend_flag = True  # 有无图例的标志
show_only_signal = True  # 是否绘制仅有期望信号波束形成后的信号波形
figname_list = []  # fig名字的列表
suffix = '.png'  # 保存绘图的文件格式
transparent = True  # 保存绘图时背景是否透明
dpi = 600  # 保存绘图时的dpi
fontsize = 15  # 标题的文字大小
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
temp_var = at.Lfm(bandwidth=bandwidth, pulsewidth=pulsewidth)
signal_points = temp_var.minimum_points * 8  # 信号采样点数
del temp_var
# endregion
