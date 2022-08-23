#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 13:13:10 2022

@author: acmsavazzi
"""

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import gc
import os
from glob import glob
import sys
import argparse
import xarray as xr
sys.path.insert(1, '/Users/acmsavazzi/Documents/WORK/PhD_Year2/Coding/Scale_separation/')
from functions import *

sys.path.insert(1, '/Users/acmsavazzi/Documents/WORK/My_source_codes')
from My_thermo_fun import *
#%%

mod = 'harmonie'
harmonie_time_to_keep = '202002010000-'
lp = '/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/HARMONIE/cy43_clim/'
# lp  = os.path.abspath('/Users/acmsavazzi/Documents/Mount1/Raw_Data/HARMONIE/BES_harm43h22tg3_fERA5_exp0/2020')+'/'
write_dir = os.path.abspath('{}/../../../DATA/HARMONIE/cy43_clim')+'/'
itmin = 1
itmax = 24
di    = 8       # delta time for plots 
zmin = 0
zmax = 5000
store = True
klps = [187.5,75,30,10]

load_data = False

dx = 2500               # model resoluton in m
dt = 75                 # model  timestep [seconds]
step = 3600             # output timestep [seconds]
domain_name = 'BES'
lat_select = 13.2806    # HALO center 
lon_select = -57.7559   # HALO center 
buffer = 30             # buffer of 150 km around (75 km on each side) the gridpoint 30 * 2 * 2.5 km

##############################
##### NOTATIONS #####
# _av = domain average 
# _p  = domain perturbation (prime)
# sf  = sub filter scale 
# f   = filter scale

# t   = total grid laevel 
# m   = middle of the grid 
##############################
#%%
if load_data:
    harm_avg = xr.open_mfdataset('/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/HARMONIE/cy43_clim/average_300km/my_harm_for_LES_forcing.nc',combine='by_coords')
    if mod == 'harmonie':
        print("Reading HARMONIE raw outputs.") 
        ### 3D fields
        nc_files = []
        variables=['ua_','va_','wa_','p_','ta_','hus_','uflx','vflx']
        EXT = "*_Slev_*.nc"
        for file in glob(os.path.join(lp, EXT)):
            if harmonie_time_to_keep in file:
                if any(var in file for var in variables):
                    nc_files.append(file)
        try:
            dl  = xr.open_mfdataset(nc_files, combine='by_coords')
        except TypeError:
            dl  = xr.open_mfdataset(nc_files)
        # select a smaller area for comparison with DALES
        j,i = np.unravel_index(np.sqrt((dl.lon-lon_select)**2 + (dl.lat-lat_select)**2).argmin(), dl.lon.shape)
        dl = dl.isel(x=slice(i-buffer,i+buffer),y=slice(j-buffer,j+buffer))   
    
        #make sure harm_avg covers the same time span as dl
        dl = dl.sel(time=harm_avg.time)
        # from an average define geopotential height at each time
        zz    = harm_avg['z']
        z_ref = zz.mean('time').values
        for a in range(len(dl.time)):
            x = dl.isel(time=a)
            x['lev'] = np.flip(zz.isel(time=a).values)
            x = x.interp(lev=z_ref)
            if 'dl_geo' in globals():
                dl_geo = xr.concat([dl_geo,x],dim='time')
            else:
                dl_geo = x
        del dl
        ## select only lower levels
        dl_geo = dl_geo.sel(lev=slice(0,6000)) # MAKE THIS SELECTION ONLY WHEN SAVING
        
    # dl_geo = dl_geo.rename({'ta':'T','hus':'qt','lev':'z','va':'v','ua':'u','wa':'w'})
    dl_geo = dl_geo.rename({'lev':'z','va':'v','ua':'u','wa':'w'})
    dl_geo = dl_geo.drop('Lambert_Conformal')
    
    dl_geo['up_wp'] = ((dl_geo['u']-dl_geo['u'].mean(dim=('x','y'))) * \
                      (dl_geo['w']-dl_geo['w'].mean(dim=('x','y'))))\
                        .mean(dim=('x','y'))  
    dl_geo['vp_wp'] = ((dl_geo['v']-dl_geo['v'].mean(dim=('x','y'))) * \
                      (dl_geo['w']-dl_geo['w'].mean(dim=('x','y'))))\
                        .mean(dim=('x','y'))
    
    dl_geo.to_netcdf(write_dir+'my_3d_harm_clim_lev_.nc')

else:
    dl_geo = xr.open_mfdataset('/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/HARMONIE/cy43_clim/my_3d_harm_clim_lev_.nc',combine='by_coords')
print("Data load completed.") 
#%% Time and hight for plotting
# 
time = dl_geo.time.values
z = dl_geo.z.values
plttime = dl_geo.time.values
pltheights    = [10,500,650,1500,4000]
pltz = []
for ii in pltheights:
    pltz = np.append(pltz,np.argmin(abs(z-ii))).astype(int)
   
    
### DEACCUMULATE
step = 3600 # output timestep [seconds]
for var in ['uflx_conv','uflx_turb']:
    dl_geo[var] = (dl_geo[var].diff('time')) * step**-1  # gives values per second

#%% Initialize np arrays
# Wave lenght related variables
N = dl_geo.x.size; N2 = N//2

## initialise variables for spectral analysis 
spec_u  = np.zeros((len(plttime),len(pltz),N2))
spec_v  = np.zeros((len(plttime),len(pltz),N2))
spec_w  = np.zeros((len(plttime),len(pltz),N2))
spec_uw = np.zeros((len(plttime),len(pltz),N2))
spec_vw = np.zeros((len(plttime),len(pltz),N2))

uw_p_av = np.zeros((len(plttime),len(z)))
vw_p_av = np.zeros((len(plttime),len(z)))
#%% Loop in time
for i in range(len(plttime)):
    print('Processing time step', i+1, '/', len(plttime))
    
    # 3D fields
    # qt = dl.load_qt(plttime[i], izmin, izmax)
    # thlp = dl.load_thl(plttime[i], izmin, izmax)
    # qlp = dl.load_ql(plttime[i], izmin, izmax)
    u = dl_geo['u'].sel(time=plttime[i]).values
    v = dl_geo['v'].sel(time=plttime[i]).values
    w = dl_geo['w'].sel(time=plttime[i]).values

    u_av  = np.mean(u,axis=(1,2))
    v_av  = np.mean(v,axis=(1,2))
    w_av  = np.mean(w,axis=(1,2))
    u_p   = u - u_av[:,np.newaxis,np.newaxis]
    v_p   = v - v_av[:,np.newaxis,np.newaxis]
    w_p   = w - w_av[:,np.newaxis,np.newaxis]
    
    uw_p  = u_p * w_p
    vw_p  = v_p * w_p
    
    uw_p_av[i,:] = np.mean(uw_p,axis=(1,2))
    vw_p_av[i,:] = np.mean(vw_p,axis=(1,2))

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
    np.save(lp+'/spec_time_HAR.npy',time)
    np.save(lp+'/spec_plttime_HAR.npy',plttime)
    np.save(lp+'/spec_pltz_HAR.npy',pltz)
    # np.save(lp+'/spec_zt_HAR.npy',ztlim)
    
    np.save(lp+'/spec_u_HAR.npy',spec_u)
    np.save(lp+'/spec_v_HAR.npy',spec_v)
    np.save(lp+'/spec_w_HAR.npy',spec_w)
    np.save(lp+'/spec_uw_HAR.npy',spec_uw)
    np.save(lp+'/spec_vw_HAR.npy',spec_vw)
    np.save(lp+'/spec_k1d_HAR.npy',k1d)
    
    np.save(lp+'/scale_up_wp_HAR.npy',uw_p_av)
    np.save(lp+'/scale_vp_wp_HAR.npy',vw_p_av)
    np.save(lp+'/scale_zt_HAR.npy',z)


#%% PLOTTING
xsize = N*dx
lam = (xsize*(k1d/np.pi))
nx  = np.pi/k1d
# k1d = frquency
# lam = wavelenght 
# lam = xsize when k1d=pi

#%%
wavelenght = np.pi/(k1d*1000)
## Spectral analysis 
# for it in range(len(s_plttime)):
for it in [3,14,38,45]:
    # plt.figure()
    # for iz in range(len(pltz)):
    #     plt.plot(wavelenght,k1d*spec_uw[it,iz,:],\
    #              ls='--',label='z: '+str(int(z[pltz[iz]]))+' m')
    # plt.axvline(2.5,c='k',lw=0.5)
    # plt.xscale('log')
    # plt.xlabel('Wavelength  [km]',fontsize=17)
    # plt.ylabel('Spectral density',fontsize=17)
    # plt.legend(fontsize=15)
    # # plt.ylim([None,0.11])
    # plt.title('Spectra uw at '+np.datetime_as_string(time[it], unit='m'),fontsize=18)
    # plt.title('Spectra uw.  Time: Feb-3 21:00UTC',fontsize=18)
    # plt.savefig(save_dir+'uw_spectra.pdf')
    
    # cumulative
    plt.figure()
    for iz in range(len(pltz)):
        plt.plot(wavelenght,1- np.cumsum(k1d*spec_uw[it,iz,:])\
                    /max(np.cumsum(k1d*spec_uw[it,iz,:]))\
                      ,label=str(int(z[pltz[iz]]))+' m')
        # plt.plot(wavelenght,np.cumsum(k1d*spec_uw[it,iz,:])\
        #             /max(np.cumsum(k1d*spec_uw[it,iz,:]))\
        #               ,label=None,ls='--')
    plt.axvline(2.5,c='k',lw=0.5)
    plt.xscale('log')
    plt.xlabel('Wavelength  [km]')
    plt.ylabel('%')
    plt.legend()
    plt.title('Cumulative spectra uw at '+np.datetime_as_string(time[it], unit='h'))
    
#%% 
for var in ['uflx_turb']:
    fig, axs = plt.subplots(figsize=(19,5))
    harm_clim_avg[var].sel(z=slice(0,4500)).plot(x='time',vmax=0.05)
    for ii in np.arange(srt_time, end_time):
        plt.axvline(x=ii,c='k')
    plt.suptitle('Param momentum flux form HARMONIE')




