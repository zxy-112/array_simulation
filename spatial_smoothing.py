# -*- coding: utf-8 -*-
# @Time    : 2021/12/9 14:30
# @Author  : zxy
# @Email   : 
# @File    : spatial_smoothing.py
# @software: PyCharm

from array_simulation_initialization2 import *

add_expect_signal = partial(array.add_signal, {'signal': expect_signal.signal, 'theta': theta_of_expect_signal})
add_coherent_inter = partial(array.add_signal, {'signal': s_inter[0].signal, 'theta': theta_of_coherent_inter[0]})
add_normal_inter = partial(array.add_signal, {'signal': s_normal_inter[0].signal, 'theta': theta_of_normal_inter[0]})

array.clear_signal()
add_expect_signal()
add_coherent_inter()
add_normal_inter()

smooth_result = array.spatial_smoothing(num=5, expect_azimuth=theta_of_expect_signal)
pp1(smooth_result['theta'], at.zxy_decibel(smooth_result['pattern']), name='空间平滑方向图5',
    theta_of_coherent_inter=theta_of_coherent_inter[0])

plot_2(t,
       np.real(only_expect_signal_after_beamform),
       t, np.real(smooth_result['output_signal']),
       name='空间平滑后信号波形5',
       label=('期望信号', '有干扰'))
