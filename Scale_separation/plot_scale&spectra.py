#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 16:57:47 2022

@author: acmsavazzi
"""

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import gc
import sys
sys.path.insert(1, '/Users/acmsavazzi/Documents/WORK/PhD_Year2/Coding/Scale_separation/')
from functions import *
from dataloader import DataLoaderDALES
import argparse
import xarray as xr

#%% Import variables 
lp = '/Users/acmsavazzi/Documents/WORK/PhD_Year1/DATA/DALES/DALES_atECMWF/outputs/20200202_12_clim/Exp_002'

## For spectral analysis
s_time      = np.load(lp+'/spec_time.npy')
s_plttime   = np.load(lp+'/spec_plttime.npy')
s_pltz      = np.load(lp+'/spec_pltz.npy')
s_ztlim     = np.load(lp+'/spec_zt.npy')
s_u         = np.load(lp+'/spec_u.npy')
s_v         = np.load(lp+'/spec_v.npy')
s_w         = np.load(lp+'/spec_w.npy')
s_uw        = np.load(lp+'/spec_uw.npy')
s_vw        = np.load(lp+'/spec_vw.npy')
s_k1d       = np.load(lp+'/spec_k1d.npy')

## For scale separation 
time    = np.load(lp+'/scale_time.npy')
plttime = np.load(lp+'/scale_plttime.npy')
ztlim   = np.load(lp+'/scale_zt.npy')
klps    = np.load(lp+'/scale_klps.npy')
    
u = np.load(lp+'/scale_u.npy')
v = np.load(lp+'/scale_v.npy')
w = np.load(lp+'/scale_w.npy')
    
u_pf       = np.load(lp+'/scale_u_pf.npy')
v_pf       = np.load(lp+'/scale_v_pf.npy')
w_pf       = np.load(lp+'/scale_w_pf.npy')
u_pfw_pf   = np.load(lp+'/scale_u_pfw_pf.npy')
u_psfw_psf = np.load(lp+'/scale_u_psfw_psf.npy')
v_pfw_pf   = np.load(lp+'/scale_v_pfw_pf.npy')
v_psfw_psf = np.load(lp+'/scale_v_psfw_psf.npy')


# Domain averaged profiles
profiles = xr.open_mfdataset(lp+'/profiles.002.nc')
profiles['time'] = np.datetime64('2020-02-02') + profiles.time.astype("timedelta64[s]")

#%% small adjustments

uw_p = u_pfw_pf + u_psfw_psf
vw_p = v_pfw_pf + v_psfw_psf

#%% PLOTTING 

## Spectral analysis 
for it in range(len(s_plttime)):
    plt.figure()
    for iz in range(len(s_pltz)):
        plt.plot(s_k1d,s_k1d*s_uw[it,iz,:],label=str(int(s_ztlim[s_pltz[iz]]))+' m')
    # plt.xscale('log')
    plt.xlabel('Wavelength  [km]')
    plt.ylabel('Energy []')
    plt.legend()
    plt.title('Spectra uw at '+np.datetime_as_string(s_time[it], unit='h'))

#%%
col=['r','b','c','g']

## Scale separation
for it in range(len(time)):
    plt.figure(figsize=(5,8))
    plt.plot(uw_p[0,it,:],ztlim,c='k',lw=3,label='Total')
    for k in range(len(klps)-1):
        plt.plot(u_pfw_pf[k,it,:],ztlim,c=col[k],ls='--')
        plt.plot(u_psfw_psf[k,it,:],ztlim,c=col[k],label='SFS '+str(150/(klps*2)[k])+' km')
    # profiles.uwr.sel(zm=ztlim,method='nearest').sel(time=time[it],method='nearest').plot(y='zm',ls=':',label='profiles')
    # plt.plot((u_pfw_pf + u_psfw_psf)[it,:],ztlim,c='k',ls= ':',alpha=0.5,label='sum')
    plt.legend()
    plt.axvline(0,c='grey',alpha=0.6,lw=0.5)
    plt.xlim([-0.001,0.012])
    plt.ylim([500,2300])
    plt.title('Mean UW at '+np.datetime_as_string(time[it], unit='h'))
    
    
    
    
    
    
    
    
    
    
