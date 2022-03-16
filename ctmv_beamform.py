# -*- coding: utf-8 -*-
# @Time    : 2021/11/22 15:25
# @Author  : zxy
# @Email   : 
# @File    : ctmv_beamform.py
# @software: PyCharm

from array_simulation_initialization2 import *
import zxy_tools as zt


# region 一个相干干扰
label = ('无干扰', '有相干干扰')
array.add_signal({'signal': s_inter[0].signal, 'theta': theta_of_coherent_inter[0]})
array.add_signal({'signal': expect_signal.signal, 'theta': theta_of_expect_signal})
array.add_signal({'signal': s_normal_inter[0].signal, 'theta': theta_of_normal_inter[0]})

weights = array.get_ctmv_weight(expect_azimuth=theta_of_expect_signal,
                                coherent_inter_azimuth=(theta_of_coherent_inter[0],))
response, theta = array.pattern(weights=weights)

name = '一个相干干扰CTMV算法波束形成'
plot_1(theta, zt.zxy_decibel(response), name, theta_of_coherent_inter=theta_of_coherent_inter[0],
       theta_of_expect_signal=theta_of_expect_signal, theta_of_normal_inter=theta_of_normal_inter[0])

signal_after_beamform = np.conjugate(weights.T) @ array.get_output()
signal_after_beamform = signal_after_beamform.flatten()

name = '一个相干干扰CTMV波束形成后的信号波形'
plot_2(t, np.real(only_expect_signal_after_beamform), t, np.real(signal_after_beamform), name, label)

array.clear_signal()
# endregion

plt.show()
if save_flag:
    plt.close('all')
