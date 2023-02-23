#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 12:47:47 2020

Analysis of DALES outputs

@author: alessandrosavazzi
"""

#%% DALES_MOMENTUM_BUDGET.py
# 

#%%                             Libraries
###############################################################################
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import DivergingNorm
import matplotlib.animation as animation
import os
from glob import glob
from datetime import datetime, timedelta
import sys
from sklearn.cluster import KMeans
import matplotlib.pylab as pylab
from pylab import *
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
params = {'legend.fontsize': 'large',
         'axes.labelsize': 20,
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large',
         'figure.figsize':[10,7],
         'figure.titlesize':20}
pylab.rcParams.update(params)

my_source_dir = os.path.abspath('{}/../../../My_source_codes')
sys.path.append(my_source_dir)
from My_thermo_fun import *

def logic(index,first_line=4):
    if ((index-3)%levels+3 == 0) or ((index-2)%levels+3 == 0) or (index<first_line):
       return True
    return False

def adjust_lightness(color, amount=0.7):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(max(0, min(1, amount * c[0])),\
                               max(0, min(1, amount * c[1])),\
                               max(0, min(1, amount * c[2])))
#%%                         Open Files
case       = '20200202_12_clim'
expnr      = '004'
snap_time ='2020-02-03T10:00'  # LT    

data_dir = '/Users/acmsavazzi/Documents/WORK/PhD_Year1/DATA/DALES/DALES_atECMWF/outputs/20200202_12_clim/'
save_dir   = data_dir

srt_time   = np.datetime64('2020-02-02T20')
#%%     OPTIONS FOR PLOTTING

# col=['b','r','g','orange','k']
col=['red','coral','maroon','blue','cornflowerblue','darkblue','green','lime','forestgreen','m']
height_lim = [0,3800]        # in m

#%%                             Import
###############################################################################
crossxz = xr.open_mfdataset(data_dir+'Exp_'+expnr+'/crossxz*.nc',combine='by_coords',decode_times=False)
crossyz = xr.open_mfdataset(data_dir+'Exp_'+expnr+'/crossyz*.nc',combine='by_coords',decode_times=False)


crossxz['time'] = srt_time + crossxz.time.astype("timedelta64[s]")
crossyz['time'] = srt_time + crossyz.time.astype("timedelta64[s]")
#%% 
var = 'u'
plt.figure()
crossxz[var].sel(time=snap_time).plot(x='xm')
plt.ylim(height_lim)
plt.title(snap_time)



#%%
print('end.')


