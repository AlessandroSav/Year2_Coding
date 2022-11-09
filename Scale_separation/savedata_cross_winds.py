#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 13:13:10 2022

@author: acmsavazzi
"""
import numpy as np
import matplotlib.pyplot as plt
import gc
import os
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
# for casenr in ['011','012','013','014','015','016','017']:
# for casenr in ['018','019']:
   
    print('################## \n ### Exp_'+casenr+'###')
    pltheights = 1500  # in m  # height at which to compute the scale separation 
    klp = 30         ## halfh the number of grids after coarsening (klp=30 -> coarsen to 2.5km)
    
    ## running on staffumbrella
    # lp = os.path.abspath('../../../Raw_Data/Les/Eurec4a/20200202_12/Exp_'+casenr)
    # save_dir   = lp
    ## running on Local
    lp =  '/Users/acmsavazzi/Documents/Mount/Raw_Data/Les/Eurec4a/20200202_12_clim/Exp_'+casenr
    # save_dir   = '/Users/acmsavazzi/Documents/WORK/PhD_Year1/DATA/DALES/DALES_atECMWF/outputs/20200202_12_clim/Exp_'+casenr
    save_dir   = '/Users/acmsavazzi/Documents/Mount/PhD_Year2/DATA/Exp_'+casenr

    
    itmin = 1       # first timestep to consider
    itmax = 24      # last timestep to consider
    di    = 2       # delta time to consider (1-> 30 min) 
    store = True
    #domain size from namotions
    xsize      =  150000 # m
    ysize      =  150000 # m
    cu         = -6 # m/s
    vu         = 0 # m/s

    #%%
    dl = DataLoaderDALES(lp,casenr=casenr)
    time = dl.time
    zt = dl.zt
    zm = dl.zm
    xt = dl.xt
    xm = dl.xm
    yt = dl.yt
    ym = dl.ym
    
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
    # plttime = np.unique(np.sort(np.append(plttime,3)))
    pltz    = np.argmin(abs(zt.values-pltheights)).astype(int)
    
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


    #### initialise variables for scale separation
    u_time          = np.zeros((xt.size,yt.size,plttime.size))
    v_time          = np.zeros((xt.size,yt.size,plttime.size))
    #
    u_pf_time          = np.zeros((xt.size,yt.size,plttime.size))
    v_pf_time          = np.zeros((xt.size,yt.size,plttime.size))
    w_pf_time          = np.zeros((xt.size,yt.size,plttime.size))
    #
    u_psf_time          = np.zeros((xt.size,yt.size,plttime.size))
    v_psf_time          = np.zeros((xt.size,yt.size,plttime.size))
    w_psf_time          = np.zeros((xt.size,yt.size,plttime.size))
    #
    uw_p_time          = np.zeros((xt.size,yt.size,plttime.size))
    vw_p_time          = np.zeros((xt.size,yt.size,plttime.size))
    #
    u_pfw_pf_time          = np.zeros((xt.size,yt.size,plttime.size))
    v_pfw_pf_time          = np.zeros((xt.size,yt.size,plttime.size))   

    #%% Loop in time
    ## make pltz time dependent so that ath each time in plttime you can select cloud top and cloud base
    
    for i in range(len(plttime)):
        print('Processing time step', i+1, '/', len(plttime))
        
        # 3D fields
        # qt = dl.load_qt(plttime[i], izmin, izmax)
        wm1 = dl.load_wm(plttime[i], pltz)
        wm2 = dl.load_wm(plttime[i],pltz+1)
        # thlp = dl.load_thl(plttime[i], izmin, izmax)
        # qlp = dl.load_ql(plttime[i], izmin, izmax)
        u = dl.load_u(plttime[i], pltz) + cu
        v = dl.load_v(plttime[i], pltz) + vu
        w = (wm1 + wm2)*0.5 ### grid is stretched !!! # from w at midlevels caclculate w at full levels
        print('Fields loaded')
        # averages and perturbations 
        u_av  = np.mean(u,axis=(0,1))
        v_av  = np.mean(v,axis=(0,1))
        w_av  = 0
        u_p   = u - u_av[np.newaxis,np.newaxis]
        v_p   = v - v_av[np.newaxis,np.newaxis]
        w_p   = w - w_av
        
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

        gc.collect()
        #### Momentum fluxes divergence 
        # to be added...
        
        
        u_time[:,:,i]   = (u)
        v_time[:,:,i]   = (v)
        
        u_pf_time[:,:,i]   = (u_pf)
        v_pf_time[:,:,i]   = (v_pf)
        w_pf_time[:,:,i]   = (w_pf)
        
        u_psf_time[:,:,i]   = (u_psf)
        v_psf_time[:,:,i]   = (v_psf)
        w_psf_time[:,:,i]   = (w_psf)
        
        uw_p_time[:,:,i]   = (uw_p)
        vw_p_time[:,:,i]   = (vw_p)
        
        u_pfw_pf_time[:,:,i]   = (u_pfw_pf)
        v_pfw_pf_time[:,:,i]   = (v_pfw_pf)
         
         
    
    if store:  
        print('Saving data...')     
        
        da_scales = xr.Dataset(
            {'u_pf':(
                ('heights','klps','x','y','time'),
                u_pf_time[np.newaxis,np.newaxis],
                )},
                coords={'height':zt[pltz].values,'klp':klp,'x':xt.values,'y':yt.values,'time':time[plttime]},)
        ## up-filter prime fields
        da_scales['v_pf']       =(('heights','klps','x','y','time'),v_pf_time[np.newaxis,np.newaxis])
        da_scales['w_pf']       =(('heights','klps','x','y','time'),w_pf_time[np.newaxis,np.newaxis])
        ## sub-filter prime fields
        da_scales['u_psf']       =(('heights','klps','x','y','time'),u_psf_time[np.newaxis,np.newaxis])
        da_scales['v_psf']       =(('heights','klps','x','y','time'),v_psf_time[np.newaxis,np.newaxis])
        da_scales['w_psf']       =(('heights','klps','x','y','time'),w_psf_time[np.newaxis,np.newaxis])
        
        ## total fields
        da_scales['u']       =(('heights','x','y','time'),u_time[np.newaxis])
        da_scales['v']       =(('heights','x','y','time'),v_time[np.newaxis])
        
        ### fluxes ###
        da_scales['uw_p']       =(('heights','x','y','time'),uw_p_time[np.newaxis])
        da_scales['vw_p']       =(('heights','x','y','time'),vw_p_time[np.newaxis])
        ##
        da_scales['u_pfw_pf']       =(('heights','klps','x','y','time'),u_pfw_pf_time[np.newaxis,np.newaxis])
        da_scales['v_pfw_pf']       =(('heights','klps','x','y','time'),v_pfw_pf_time[np.newaxis,np.newaxis])
    
        da_scales.to_netcdf(save_dir+'/cross_field'+str(pltheights)+'m_filter_'+casenr+'.nc')
        # da_scales.to_netcdf(lp+'/cross_field'+str(pltheights)+'m_filter_'+casenr+'.nc')

print('END')    