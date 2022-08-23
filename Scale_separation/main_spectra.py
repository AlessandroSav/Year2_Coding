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

mod    = 'dales'
casenr = '004'
# lp = '/Users/acmsavazzi/Documents/WORK/PhD_Year1/DATA/DALES/DALES_atECMWF/outputs/20200202_12_clim/Exp_'+casenr
# lp =  '/Users/acmsavazzi/Documents/Mount1/DALES/Experiments/EUREC4A/Exp_ECMWF/20200202_12/Exp_'+casenr
lp =  '/Users/acmsavazzi/Documents/Mount/Raw_Data/Les/Eurec4a/20200202_12_clim/Exp_'+casenr
save_data_dir   = '/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/DALES'

itmin = 0
itmax = 24
di    = 2       # delta time for plots 
izmin = 2100
izmax = None
dz    = 40
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
# dx = np.diff(xt)[0]
# dy = np.diff(yt)[0] # Assumes uniform horizontal spacing

dx = xsize/xt.size  # in metres
dy = ysize/yt.size  # in metres


# time and hight for plotting
plttime = np.arange(itmin, itmax, di)
pltheights    = [10,650,2100,4000]
pltheights    = 400
if izmax!= None:
    ztlim = zt[izmin:izmax]
    zmlim = zm[izmin:izmax]
    pltz = []
    for iz in pltheights:
        pltz = np.append(pltz,np.argmin(abs(ztlim.values-iz))).astype(int)
else:
    pltz    = np.argmin(abs(zt.values-pltheights)).astype(int)
    izmin = pltz
##########
if (int(casenr) % 2) == 0:
    start_d = int(casenr)//2 +1
    start_h = 12
else:
    start_d = int(casenr)//2 +2
    start_h = 0
time = np.array(time,dtype='timedelta64[s]') + (np.datetime64('2020-02-'+str(start_d).zfill(2)+'T'+str(start_h).zfill(2)+':30'))
##################
############
###### SOMEHOW TIME IS WRONG and needs this 
time = time -np.timedelta64(12,'h') - np.timedelta64(30,'m')
############
##################
#%%
# Wave lenght related variables
N = xt.size; N2 = N//2
## initialise variables for spectral analysis 
if izmax!=None:
    spec_u  = np.zeros((len(plttime),len(pltz),N2))
    spec_v  = np.zeros((len(plttime),len(pltz),N2))
    spec_w  = np.zeros((len(plttime),len(pltz),N2))
    spec_uw = np.zeros((len(plttime),len(pltz),N2))
    spec_vw = np.zeros((len(plttime),len(pltz),N2))
else:
    spec_u  = np.zeros((len(plttime),N2))
    spec_v  = np.zeros((len(plttime),N2))
    spec_w  = np.zeros((len(plttime),N2))
    spec_uw = np.zeros((len(plttime),N2))
    spec_vw = np.zeros((len(plttime),N2))
    
    spec_uw_p = np.zeros((len(plttime),N2))
    

#%% Loop in time
for i in range(len(plttime)):
    print('Processing time step', i+1, '/', len(plttime))
    
    # 3D fields
    # qt = dl.load_qt(plttime[i], izmin, izmax)
    wm1 = dl.load_wm(plttime[i], izmin, izmax)
    if izmax!= None:
        wm2 = dl.load_wm(plttime[i],izmin+1,izmax+1)
    else:
        wm2 = dl.load_wm(plttime[i],izmin+1)
    # thlp = dl.load_thl(plttime[i], izmin, izmax)
    # qlp = dl.load_ql(plttime[i], izmin, izmax)
    u = dl.load_u(plttime[i], izmin, izmax) + cu
    v = dl.load_v(plttime[i], izmin, izmax) + vu
    w = (wm1 + wm2)*0.5 ### grid is stretched !!! # from w at midlevels caclculate w at full levels

    if izmax==None:
        # averages and perturbations 
        u_av  = np.mean(u,axis=(0,1))
        v_av  = np.mean(v,axis=(0,1))
        w_av  = 0
        u_p   = u - u_av[np.newaxis,np.newaxis]
        v_p   = v - v_av[np.newaxis,np.newaxis]
        w_p   = w - w_av    


    if izmax != None:
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
    else:
        print('Computing spectra...')
        ### spectral analysis at specific levels
        k1d,spec_u[i,:]  = compute_spectrum(u[:,:], dx)
        k1d,spec_v[i,:]  = compute_spectrum(v[:,:], dx)
        k1d,spec_w[i,:]  = compute_spectrum(w[:,:], dx)
        k1d,spec_uw[i,:] = compute_spectrum(u[:,:], dx,\
                                               cloud_scalar_2=w[:,:])
        k1d,spec_vw[i,:] = compute_spectrum(v[:,:], dx,\
                                               cloud_scalar_2=w[:,:])
            
        k1d,spec_uw_p[i,:] = compute_spectrum(u_p[:,:], dx,\
                                       cloud_scalar_2=w[:,:])
            
if store:        
    print('Saving data...')
    np.save(save_data_dir+'/spec_time.npy',time[plttime])
    np.save(save_data_dir+'/spec_plttime.npy',plttime)
    np.save(save_data_dir+'/spec_pltz.npy',pltz)
    np.save(save_data_dir+'/spec_zt.npy',zt)
    
    np.save(save_data_dir+'/spec_u_'+str(pltheights)+'.npy',spec_u)
    np.save(save_data_dir+'/spec_v_'+str(pltheights)+'.npy',spec_v)
    np.save(save_data_dir+'/spec_w_'+str(pltheights)+'.npy',spec_w)
    np.save(save_data_dir+'/spec_uw_'+str(pltheights)+'.npy',spec_uw)
    np.save(save_data_dir+'/spec_vw_'+str(pltheights)+'.npy',spec_vw)
    np.save(save_data_dir+'/spec_k1d_'+str(pltheights)+'.npy',k1d)
    
    
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
        plt.plot(nx/1000,k1d*spec_uw[it,iz,:],label=str(int(ztlim[pltz[iz]].values))+' m')
    plt.xscale('log')
    plt.xlabel('Wavelength  [km]')
    plt.ylabel('Energy []')
    plt.legend()
    plt.title('Spectra uw at '+time[plttime[it]].dt.strftime('%Y-%m-%dT%H:%M').values)

#%% cumulative
if izmax!=None:
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

else:
    wavelenght = np.pi / k1d
    
    # for it in range(len(plttime)):
    for it in [1]:
        plt.figure()        
        plt.plot(wavelenght/1000,np.cumsum(spec_uw[it,:])\
                # /max(np.cumsum(spec_uw[it,:]))\
                 ,label='Resolved ' +str(int(zt[pltz].values))+' m')
        # plt.plot(wavelenght/1000,(max(np.cumsum(spec_uw[it,:]))-np.cumsum(spec_uw[it,:]))/1000\
        #         # /max(np.cumsum(spec_uw[it,:]))\
        #          ,label='Sub-filter scale', ls='--')
        # plt.axvline(2*np.pi*60/150000,c='k',lw=0.5)
        plt.axvline(2.5,c='k',lw=0.5)
        plt.xscale('log')
        plt.xlabel('Gridsize  [km]')
        # plt.xlabel('N of boxes  ')
        plt.legend()
        plt.title('Cumulative spectra uw at '+np.datetime_as_string(time[plttime[it]], unit='h'))



