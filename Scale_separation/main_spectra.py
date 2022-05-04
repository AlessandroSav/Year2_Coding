#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 10:46:53 2022

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

#%%
##### NOTATIONS
# _av = domain average 
# _p  = domain perturbation (prime)
# sf  = sub filter scale 
# f   = filter scale
# t   = total grid laevel 
# m   = middle of the grid 

mod = 'dales'
# lp = 
lp = '/Users/acmsavazzi/Documents/WORK/PhD_Year1/DATA/DALES/DALES_atECMWF/outputs/20200202_12_clim/Exp_002'
itmin = 1
itmax = 24
di    = 8       # delta time for plots 
izmin = 0
izmax = 120
dz    = 40
casenr = '002'
store = True

#domain size from namotions
xsize      =  150000 # m
ysize      =  150000 # m
cu         = -6 # m/s
vu         = 0 # m/s

#%%
if mod == 'dales':
    dl = DataLoaderDALES(lp,casenr=casenr)
    
time = dl.time
zt = dl.zt
zm = dl.zm
xt = dl.xt
xm = dl.xm
yt = dl.yt
ym = dl.ym
time1d = dl.time1d

# FIXME temporary hardcoding of dx/dy for data that does not have xf/yf as variables
dx = np.diff(xt)[0]
dy = np.diff(yt)[0] # Assumes uniform horizontal spacing

ztlim = zt[izmin:izmax]
zmlim = zm[izmin:izmax]

# time and hight for plotting
plttime = np.arange(itmin, itmax, di)
pltheights    = [150,500,1500,3000]
pltz = []
for iz in pltheights:
    pltz = np.append(pltz,np.argmin(abs(ztlim.values-iz))).astype(int)

# pltz    = np.arange(0, izmax-izmin, dz)
##########
start_d = int(casenr)//2 +1
if (int(casenr) % 2) == 0:
    start_h = 12
else: start_h = 0
time = time + (np.datetime64('2020-02-'+str(start_d).zfill(2)+'T'+str(start_h).zfill(2)+':30') - time[0])

#%%
# Wave lenght related variables
N = xt.size; N2 = N//2

## initialise variables for spectral analysis 
spec_u  = np.zeros((len(plttime),len(pltz),N2))
spec_v  = np.zeros((len(plttime),len(pltz),N2))
spec_w  = np.zeros((len(plttime),len(pltz),N2))
spec_uw = np.zeros((len(plttime),len(pltz),N2))
spec_vw = np.zeros((len(plttime),len(pltz),N2))

#%% Loop in time
for i in range(len(plttime)):
    print('Processing time step', i+1, '/', len(plttime))
    
    # 3D fields
    # qt = dl.load_qt(plttime[i], izmin, izmax)
    wm1 = dl.load_wm(plttime[i], izmin, izmax)
    wm2 = dl.load_wm(plttime[i],izmin+1,izmax+1)
    # thlp = dl.load_thl(plttime[i], izmin, izmax)
    # qlp = dl.load_ql(plttime[i], izmin, izmax)
    u = dl.load_u(plttime[i], izmin, izmax) + cu
    v = dl.load_v(plttime[i], izmin, izmax) + vu
    w = (wm1 + wm2)*0.5 ### grid is stretched !!! # from w at midlevels caclculate w at full levels

    

    ### spectral analysis at specific levels
    for iz in range(len(pltz)):
        print('Computing spectra at time step', i+1, '/', len(plttime),
              ', height', iz+1,'/',len(pltz))
        k1d,spec_u[i,iz,:]  = compute_spectrum(u[pltz[iz],:,:], dx)
        k1d,spec_v[i,iz,:]  = compute_spectrum(v[pltz[iz],:,:], dx)
        k1d,spec_w[i,iz,:]  = compute_spectrum(w[pltz[iz],:,:], dx)
        k1d,spec_uw[i,iz,:] = compute_spectrum(u[pltz[iz],:,:], dx,\
                                               cloud_scalar_2=w[pltz[iz],:,:])
        k1d,spec_vw[i,iz,:] = compute_spectrum(v[pltz[iz],:,:], dx,\
                                               cloud_scalar_2=w[pltz[iz],:,:])
            
if store:        
    np.save(lp+'/spec_time.npy',time[plttime])
    np.save(lp+'/spec_plttime.npy',plttime)
    np.save(lp+'/spec_pltz.npy',pltz)
    np.save(lp+'/spec_zt.npy',ztlim)
    
    np.save(lp+'/spec_u.npy',spec_u)
    np.save(lp+'/spec_v.npy',spec_v)
    np.save(lp+'/spec_w.npy',spec_w)
    np.save(lp+'/spec_uw.npy',spec_uw)
    np.save(lp+'/spec_vw.npy',spec_vw)
    np.save(lp+'/spec_k1d.npy',k1d)
    
    
#%% PLOT
#%% 
sizes = xsize / np.arange(1, 1512// 2 + 1) 

# k1d = frquency
# lam = wavelenght 
# lam = xsize when k1d=pi

lam = (xsize*(k1d/np.pi))
nx  = np.pi/k1d

for it in range(len(plttime)):
    plt.figure()
    for iz in range(len(pltz)):
        plt.plot(lam/1000,k1d*spec_uw[it,iz,:],label=str(int(ztlim[pltz[iz]].values))+' m')
    plt.xscale('log')
    plt.xlabel('Wavelength  [km]')
    plt.ylabel('Energy []')
    plt.legend()
    plt.title('Spectra uw at '+time[plttime[it]].dt.strftime('%Y-%m-%dT%H:%M').values)

#%% cumulative
for it in range(len(plttime)):
    plt.figure()
    for iz in range(len(pltz)):
        plt.plot(lam/1000,np.cumsum(k1d*spec_uw[it,iz,:])\
                   /max(np.cumsum(k1d*spec_uw[it,iz,:]))\
                     ,label=str(int(ztlim[pltz[iz]].values))+' m')
    plt.xscale('log')
    plt.xlabel('Wavelength  [km]')
    plt.legend()
    plt.title('Cumulative spectra uw at '+time[plttime[it]].dt.strftime('%Y-%m-%dT%H:%M').values)





