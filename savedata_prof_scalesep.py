#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 13 Sept 13:13:10 2022

@author: acmsavazzi
"""
import numpy as np
import netCDF4 as nc
import gc
import os
from os.path import exists
import sys
sys.path.insert(1, os.path.abspath('.'))
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

for casenr in ['001','002','003','004','005','006','007','008','009','010',\
                '011','012','013','014','015','016','017']:
# for casenr in ['018','019']:
   
    print('################## \n ### Exp_'+casenr+'###',flush=True)
    ## running on staffumbrella
    lp = os.path.abspath('../../../../Raw_Data/Les/Eurec4a/20200202_12_clim/Exp_'+casenr)
    save_dir   = os.path.abspath('../../../DATA/Exp_'+casenr)
    ## running on Local
    # lp =  '/Users/acmsavazzi/Documents/Mount/Raw_Data/Les/Eurec4a/20200202_12_clim/Exp_'+casenr
    # save_dir   = '/Users/acmsavazzi/Documents/WORK/PhD_Year1/DATA/DALES/DALES_atECMWF/outputs/20200202_12_clim/Exp_'+casenr
    
    itmin = 1       # first timestep to consider
    itmax = 24      # last timestep to consider
    di    = 1       # delta time to consider (1-> 30 min)     
    izmin_parts =[0,23,43,63,83]
    izmax_parts =[22,42,62,82,103] # Level 103 is 3993m
    store = True
    #domain size from namotions
    xsize      =  150000 # m
    ysize      =  150000 # m
    cu         = -6 # m/s
    vu         = 0 # m/s
    
    klps = [30,]         ## halfh the number of grids after coarsening (klp=30 -> coarsen to 2.5km)
    #%%
    # print('Path: '+lp+' exists: '+str(exists(lp)))
    # print('File: merged_fielddump_wind Exists: '+str(exists(lp+'/merged_fielddump_wind.'+casenr+'.nc')))
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
    dx = xsize/xt.size  # in metres
    dy = ysize/yt.size  # in metres
    ##########  All these dz to be checked !!!
    # Vertical differences
    dzt = np.zeros(zm.shape)
    dzt[:-1] = np.diff(zm) # First value is difference top 1st cell and surface
    dzt[-1] = dzt[-2]
    dzm = np.zeros(zt.shape)
    dzm[1:] = np.diff(zt) # First value is difference mid 1st cell and mid 1st cell below ground
    dzm[0] = 2*zt[1]
    
    plttime = np.arange(itmin, itmax, di)    
    
    ##################
    ############
    if (int(casenr) % 2) == 0:
        start_d = int(casenr)//2 +1
    else:
        start_d = int(casenr)//2 +2
    start_h = 0
    ###### Exp_001 and Exp_002 have wrong times
    if casenr == '001' or casenr=='002':
        time = time + 34385400  +1800
    time = np.array(time,dtype='timedelta64[s]') + (np.datetime64('2020-02-'+str(start_d).zfill(2)+'T'+str(start_h).zfill(2)+':00'))
    ############
    ##################
        
    ztlim = zt[izmin_parts[0]:izmax_parts[-1]]
    zmlim = zm[izmin_parts[0]:izmax_parts[-1]]
    
    #### initialise variables for scale separation
    u_p_avtime          = np.zeros((plttime.size,izmax_parts[-1]-izmin_parts[0]))
    v_p_avtime          = np.zeros((plttime.size,izmax_parts[-1]-izmin_parts[0]))
    w_p_avtime          = np.zeros((plttime.size,izmax_parts[-1]-izmin_parts[0]))
    #
    u_pf_avtime         = np.zeros((len(klps),plttime.size,izmax_parts[-1]-izmin_parts[0]))
    v_pf_avtime         = np.zeros((len(klps),plttime.size,izmax_parts[-1]-izmin_parts[0]))
    w_pf_avtime         = np.zeros((len(klps),plttime.size,izmax_parts[-1]-izmin_parts[0]))
    #
    u_pfw_pf_avtime     = np.zeros((len(klps),plttime.size,izmax_parts[-1]-izmin_parts[0]))
    u_psfw_psf_avtime   = np.zeros((len(klps),plttime.size,izmax_parts[-1]-izmin_parts[0]))
    v_pfw_pf_avtime     = np.zeros((len(klps),plttime.size,izmax_parts[-1]-izmin_parts[0]))
    v_psfw_psf_avtime   = np.zeros((len(klps),plttime.size,izmax_parts[-1]-izmin_parts[0]))
    #
    uw_pf_avtime        = np.zeros((len(klps),plttime.size,izmax_parts[-1]-izmin_parts[0]))
    vw_pf_avtime        = np.zeros((len(klps),plttime.size,izmax_parts[-1]-izmin_parts[0]))
    uw_psf_avtime       = np.zeros((len(klps),plttime.size,izmax_parts[-1]-izmin_parts[0]))
    vw_psf_avtime       = np.zeros((len(klps),plttime.size,izmax_parts[-1]-izmin_parts[0]))
    #%% Loop in time
    ## make pltz time dependent so that ath each time in plttime you can select cloud top and cloud base
    
    for i in range(len(plttime)):
        print('Processing time step', i+1, '/', len(plttime),flush=True)
        
        for part in range(len(izmin_parts)):
            izmin=izmin_parts[part]
            izmax=izmax_parts[part]
        
            # 3D fields
            # qt = dl.load_qt(plttime[i], izmin, izmax)
            wm1 = dl.load_wm(plttime[i], izmin, izmax)
            wm2 = dl.load_wm(plttime[i],izmin+1, izmax+1)
            # thlp = dl.load_thl(plttime[i], izmin, izmax)
            # qlp = dl.load_ql(plttime[i], izmin, izmax)
            u = dl.load_u(plttime[i], izmin, izmax) + cu
            v = dl.load_v(plttime[i], izmin, izmax) + vu
            w = (wm1 + wm2)*0.5 ### grid is stretched !!! # from w at midlevels caclculate w at full levels
            print('Fields loaded')
            # averages and perturbations 
            u_av  = np.mean(u,axis=(1,2))
            v_av  = np.mean(v,axis=(1,2))
            w_av  = 0
            u_p   = u - u_av[:,np.newaxis,np.newaxis]
            v_p   = v - v_av[:,np.newaxis,np.newaxis]
            w_p   = w - w_av
            
            for k in range(len(klps)):
                print('Processing scale', k+1, '/', len(klps))
                klp=klps[k]
                #
                if klp > 0:
                    f_scale = xsize/(klp*2)  # m
                elif klp == 0:
                    f_scale = xsize
                else: print('Warning: Cutoff wavenumber for lw-pass filter smaller than 0.')
                
                # Mask for low-pass filtering
                circ_mask = np.zeros((xt.size,xt.size))
                rad = getRad(circ_mask)
                circ_mask[rad<=klp] = 1
            
                #filtered U
                u_pf  = lowPass(u_p, circ_mask)
                u_psf = u_p - u_pf
                #filtered V
                v_pf = lowPass(v_p, circ_mask)
                v_psf = v_p - v_pf
                #filtered W total level
                w_pf  = lowPass(w_p, circ_mask)
                w_psf = w_p - w_pf   
                 
                ## Fluxes
                # filtered and sub-filtered fluxes without the cross-terms
                u_pfw_pf   = u_pf  * w_pf 
                u_psfw_psf = u_psf * w_psf
                v_pfw_pf   = v_pf  * w_pf 
                v_psfw_psf = v_psf * w_psf
                # Fluxes with the cross-terms
                # uw_p = (u_pf + u_psf) * (w_pf + w_psf)  
                uw_p  = u_p * w_p
                vw_p  = v_p * w_p
                # # filtered fluxes
                uw_pf = lowPass(uw_p, circ_mask)
                vw_pf   = lowPass(vw_p, circ_mask)
                # # subgrid fluxes
                uw_psf = uw_p - uw_pf
                vw_psf    = vw_p - vw_pf    
                ## Put results into variables 
                print('Averaging fields...')
                u_p_avtime[i,izmin:izmax] = np.mean(u_p,axis=(1,2))
                v_p_avtime[i,izmin:izmax] = np.mean(v_p,axis=(1,2))
                w_p_avtime[i,izmin:izmax] = np.mean(w_p,axis=(1,2))
                #
                u_pf_avtime[k,i,izmin:izmax] = np.mean(u_pf,axis=(1,2))
                v_pf_avtime[k,i,izmin:izmax] = np.mean(v_pf,axis=(1,2))
                w_pf_avtime[k,i,izmin:izmax] = np.mean(w_pf,axis=(1,2))
                #
                u_pfw_pf_avtime[k,i,izmin:izmax]   = np.mean(u_pfw_pf,axis=(1,2))
                u_psfw_psf_avtime[k,i,izmin:izmax] = np.mean(u_psfw_psf,axis=(1,2))
                v_pfw_pf_avtime[k,i,izmin:izmax]   = np.mean(v_pfw_pf,axis=(1,2))
                v_psfw_psf_avtime[k,i,izmin:izmax] = np.mean(v_psfw_psf,axis=(1,2))
                #
                uw_pf_avtime[k,i,izmin:izmax]   = np.mean(uw_pf,axis=(1,2))
                vw_pf_avtime[k,i,izmin:izmax]   = np.mean(vw_pf,axis=(1,2))
                uw_psf_avtime[k,i,izmin:izmax]  = np.mean(uw_psf,axis=(1,2))
                vw_psf_avtime[k,i,izmin:izmax]  = np.mean(vw_psf,axis=(1,2))
                gc.collect()
                #### Momentum fluxes divergence 
                # to be added...
    
    if store:  
        print('Saving data...')     
        # df = xr.DataArray(u_pf_avtime, coords=[('klp',klps),('time', time), ('z', ztlim)])
        np.save(save_dir+'/scale_time_prof_'+casenr+'.npy',time[plttime])
        np.save(save_dir+'/scale_plttime_prof_'+casenr+'.npy',plttime)
        np.save(save_dir+'/scale_zt_prof_'+casenr+'.npy',ztlim.values)
        np.save(save_dir+'/scale_klps_prof_'+casenr+'.npy',klps)
        print('Sved general variables') 
        np.save(save_dir+'/scale_u_prof_'+casenr+'.npy',u_p_avtime)
        np.save(save_dir+'/scale_v_prof_'+casenr+'.npy',v_p_avtime)
        np.save(save_dir+'/scale_w_prof_'+casenr+'.npy',w_p_avtime)
        np.save(save_dir+'/scale_u_pf_prof_'+casenr+'.npy',u_pf_avtime)
        np.save(save_dir+'/scale_v_pf_prof_'+casenr+'.npy',v_pf_avtime)
        np.save(save_dir+'/scale_w_pf_prof_'+casenr+'.npy',w_pf_avtime)
        print('Sved u, v, w')
        np.save(save_dir+'/scale_u_pfw_pf_prof_'+casenr+'.npy',u_pfw_pf_avtime)
        np.save(save_dir+'/scale_u_psfw_psf_prof_'+casenr+'.npy',u_psfw_psf_avtime)
        np.save(save_dir+'/scale_v_pfw_pf_prof_'+casenr+'.npy',v_pfw_pf_avtime)
        np.save(save_dir+'/scale_v_psfw_psf_prof_'+casenr+'.npy',v_psfw_psf_avtime)
        np.save(save_dir+'/scale_uw_pf_prof_'+casenr+'.npy',uw_pf_avtime)
        np.save(save_dir+'/scale_uw_psf_prof_'+casenr+'.npy',uw_psf_avtime)
        np.save(save_dir+'/scale_vw_pf_prof_'+casenr+'.npy',vw_pf_avtime)
        np.save(save_dir+'/scale_vw_psf_prof_'+casenr+'.npy',vw_psf_avtime)
        print('Sved fluxes')
print('END')    
