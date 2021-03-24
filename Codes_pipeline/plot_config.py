#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:25:22 2021

@author: jacoposala
"""



from TC_and_TS_define_param import grid_lower, grid_upper, grid_stride, folder2use, depth_layers, var2use


def plot_config():
    if var2use == 'Salinity':
        plot_config.clim = 0.4
    elif var2use == 'Temperature': 
        plot_config.clim = 2.0
    return plot_config.clim

