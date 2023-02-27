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
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import DivergingNorm
import matplotlib.animation as animation
import os
from glob import glob
from datetime import datetime, timedelta
import sys
import matplotlib.pylab as pylab
from pylab import *
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
height_lim = [0,3800*0.001]        # in km

#%%                             Import
###############################################################################
crossxz = xr.open_mfdataset(data_dir+'Exp_'+expnr+'/crossxz*.nc',combine='by_coords',decode_times=False)
crossyz = xr.open_mfdataset(data_dir+'Exp_'+expnr+'/crossyz*.nc',combine='by_coords',decode_times=False)
## convert time from seconds to date
crossxz['time'] = srt_time + crossxz.time.astype("timedelta64[s]")
crossyz['time'] = srt_time + crossyz.time.astype("timedelta64[s]")

## from m to km 
crossxz['xt']=crossxz['xt']*0.001
crossxz['yt']=crossxz['yt']*0.001
crossxz['zt']=crossxz['zt']*0.001
crossxz['zm']=crossxz['zm']*0.001
crossyz['yt']=crossyz['yt']*0.001
crossyz['xt']=crossyz['xt']*0.001
crossyz['zt']=crossyz['zt']*0.001
crossyz['zm']=crossyz['zm']*0.001

## interpolate coordinates to single grid (not zm and zt)
crossxz['xm'] = crossxz['xt']
crossyz['ym'] = crossyz['yt']
crossxz = crossxz.assign_coords({"xm": ("xm", crossxz.xm.values)})
crossyz = crossyz.assign_coords({"ym": ("ym", crossyz.ym.values)})



#%%
## find outline of clouds for 'snap_time' 

for section in ['xz','yz']:
    if section =='xz':
        mask = np.nan_to_num((crossxz['ql'].where(crossxz['ql']>0.0001)\
                          .sel(time=snap_time)).values)
    if section =='yz':
        mask = np.nan_to_num((crossyz['ql'].where(crossyz['ql']>0.0001)\
                          .sel(time=snap_time)).values)
    mask[mask > 0] = 3
    kernel = np.ones((4,4))
    C      = ndi.convolve(mask, kernel, mode='constant', cval=0)
    outer  = np.where( (C>=3) & (C<=12 ), 1, 0)
    # add variable cloud contour
    # works only for 1 time stamp 
    if section =='xz':
        crossxz['cloud'] = (('zt', 'xm'), outer)
    if section =='yz':
        crossyz['cloud'] = (('zt', 'ym'), outer)


crossxz['uw'] = crossxz['u']*crossxz['w']

#%% ##############     PLOTTING       ##############
####################################################

## cross-yz
var='u'
plt.figure()
crossxz[var].sel(time=snap_time).plot(x='xm',vmin=-8)
crossxz['cloud'].where(crossxz['cloud'] > 0).plot(cmap='binary',\
                                                  add_colorbar=False,vmin=0,vmax=0.5)
plt.axvline(crossyz.xt.values,c='k',ls='--',lw=1)
plt.axhline(0.2,c='k',ls='--',lw=1)
plt.ylim(height_lim)
plt.title(snap_time,fontsize=20)


# var='v'
# plt.figure()
# crossxz[var].sel(time=snap_time).plot(x='xt',vmin=-8)
# crossxz['cloud'].where(crossxz['cloud'] > 0).plot(cmap='binary',\
#                                                   add_colorbar=False,vmin=0,vmax=0.5)
# plt.axvline(crossyz.xt.values,c='k',ls='--',lw=1)
# plt.axhline(0.2,c='k',ls='--',lw=1)
# plt.ylim(height_lim)
# plt.title(snap_time,fontsize=20)


var='w'
plt.figure()
crossxz[var].sel(time=snap_time).plot(x='xt',vmin=-1)
crossxz['cloud'].where(crossxz['cloud'] > 0).plot(cmap='binary',\
                                                  add_colorbar=False,vmin=0,vmax=0.5)
plt.axvline(crossyz.xt.values,c='k',ls='--',lw=1)
plt.axhline(0.2,c='k',ls='--',lw=1)
plt.ylim(height_lim)
plt.title(snap_time,fontsize=20)


## cross-yz
var='v'
plt.figure()
crossyz[var].sel(time=snap_time).plot(x='ym',vmin=-8)
crossyz['cloud'].where(crossyz['cloud'] > 0).plot(cmap='binary',\
                                                  add_colorbar=False,vmin=0,vmax=0.5)
plt.axvline(crossxz.yt.values,c='k',ls='--',lw=1)
plt.axhline(0.2,c='k',ls='--',lw=1)
plt.ylim(height_lim)
plt.title(snap_time,fontsize=20)

var='w'
plt.figure()
crossyz[var].sel(time=snap_time).plot(x='yt',vmin=-1)
crossyz['cloud'].where(crossyz['cloud'] > 0).plot(cmap='binary',\
                                                  add_colorbar=False,vmin=0,vmax=0.5)
plt.axvline(crossxz.yt.values,c='k',ls='--',lw=1)
plt.axhline(0.2,c='k',ls='--',lw=1)
plt.ylim(height_lim)
plt.title(snap_time,fontsize=20)



#%%
print('end.')


