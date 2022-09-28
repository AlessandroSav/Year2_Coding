#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 16:57:47 2022

@author: acmsavazzi
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from matplotlib.colors import DivergingNorm
import netCDF4 as nc
import gc
import sys
sys.path.insert(1, '/Users/acmsavazzi/Documents/WORK/PhD_Year2/Coding/Scale_separation/')
from functions import *
from dataloader import DataLoaderDALES
import argparse
import xarray as xr

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

col=['r','b','c','g','k']
style={'mean'            :'-' , 'q2'              :'--',\
        'q1'              :':' , 'q3'              :':'}

    
    
#%% Import variables 
## HARMONIE 
lp_H = '/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/HARMONIE/cy43_clim'
## DALES
casenr  = '004'
lp      = '/Users/acmsavazzi/Documents/WORK/PhD_Year1/DATA/DALES/DALES_atECMWF/outputs/20200202_12_clim/Exp_'+casenr
spec          = True
scale_sep     = True
load_profiles = True
Dfields       = False
save_dir   = '/Users/acmsavazzi/Documents/WORK/PhD_Year2/Figures/'
xsize = 150000 #m
#%% From HARMONIE
dl_geo = xr.open_mfdataset('/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/HARMONIE/cy43_clim/my_3d_harm_clim_lev_.nc',combine='by_coords')

cl_max_HAR = np.load('/Users/acmsavazzi/Documents/WORK/PhD_Year2/Figures/cl_max_HARM.npy', allow_pickle=True)

if spec:
    ## For spectral analysis
    s_time_H      = np.load(lp_H+'/spec_time_HAR.npy', allow_pickle=True) #there is a problem with time
    s_plttime_H   = np.load(lp_H+'/spec_plttime_HAR.npy', allow_pickle=True)
    s_pltz_H      = np.load(lp_H+'/spec_pltz_HAR.npy', allow_pickle=True)
    # s_ztlim_H     = np.load(lp_H+'/spec_zt_HAR.npy', allow_pickle=True)
    s_u_H         = np.load(lp_H+'/spec_u_HAR.npy', allow_pickle=True)
    s_v_H         = np.load(lp_H+'/spec_v_HAR.npy', allow_pickle=True)
    s_w_H         = np.load(lp_H+'/spec_w_HAR.npy', allow_pickle=True)
    s_uw_H        = np.load(lp_H+'/spec_uw_HAR.npy', allow_pickle=True)
    s_vw_H        = np.load(lp_H+'/spec_vw_HAR.npy', allow_pickle=True)
    s_k1d_H       = np.load(lp_H+'/spec_k1d_HAR.npy', allow_pickle=True)
    
if scale_sep:
    z_H          = np.load(lp_H+'/scale_z_HAR.npy', allow_pickle=True)
    up_wp_H      = np.load(lp_H+'/scale_up_wp_HAR.npy', allow_pickle=True)
    vp_wp_H      = np.load(lp_H+'/scale_vp_wp_HAR.npy', allow_pickle=True)

#%% From DALES
if spec:

    ## For spectral analysis
    s_time      = np.load(lp+'/spec_time.npy', allow_pickle=True) #there is a problem with time
    s_plttime   = np.load(lp+'/spec_plttime.npy', allow_pickle=True)
    s_pltz      = np.load(lp+'/spec_pltz.npy', allow_pickle=True)
    s_ztlim     = np.load(lp+'/spec_zt.npy', allow_pickle=True)
    s_u         = np.load(lp+'/spec_u.npy', allow_pickle=True)
    s_v         = np.load(lp+'/spec_v.npy', allow_pickle=True)
    s_w         = np.load(lp+'/spec_w.npy', allow_pickle=True)
    s_uw        = np.load(lp+'/spec_uw.npy', allow_pickle=True)
    s_vw        = np.load(lp+'/spec_vw.npy', allow_pickle=True)
    s_k1d       = np.load(lp+'/spec_k1d.npy', allow_pickle=True)
    
    dir_dales = '/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/DALES/'
    s_time      = np.load(dir_dales+'/spec_time.npy', allow_pickle=True) #there is a problem with time
    s_plttime   = np.load(dir_dales+'/spec_plttime.npy', allow_pickle=True)
    s_pltz      = np.load(dir_dales+'/spec_pltz.npy', allow_pickle=True)
    s_ztlim     = np.load(dir_dales+'/spec_zt.npy', allow_pickle=True)
    for ii in [400,]:
        ## For spectral analysis
        s_u         = np.load(dir_dales+'/spec_u_'+str(ii)+'.npy', allow_pickle=True)
        s_v         = np.load(dir_dales+'/spec_v_'+str(ii)+'.npy', allow_pickle=True)
        s_w         = np.load(dir_dales+'/spec_w_'+str(ii)+'.npy', allow_pickle=True)
        s_uw        = np.load(dir_dales+'/spec_uw_'+str(ii)+'.npy', allow_pickle=True)
        s_vw        = np.load(dir_dales+'/spec_vw_'+str(ii)+'.npy', allow_pickle=True)
        s_k1d       = np.load(dir_dales+'/spec_k1d_'+str(ii)+'.npy', allow_pickle=True)

if scale_sep:
    cases = ['004',]
    for casenr in cases:
        lp      = '/Users/acmsavazzi/Documents/WORK/PhD_Year1/DATA/DALES/DALES_atECMWF/outputs/20200202_12_clim/Exp_'+casenr

        ## For scale separation
        if 'time_' in locals():
            time_ = np.append(time_,np.load(lp+'/scale_time.npy', allow_pickle=True))
        else:
            try:
                time_    = np.load(lp+'/scale_time.npy', allow_pickle=True)
            except:
                None
            
        plttime = np.load(lp+'/scale_plttime.npy', allow_pickle=True)
        ztlim   = np.load(lp+'/scale_zt.npy', allow_pickle=True)
        klps    = np.load(lp+'/scale_klps.npy', allow_pickle=True)
         
        u = np.load(lp+'/scale_u.npy', allow_pickle=True)
        v = np.load(lp+'/scale_v.npy', allow_pickle=True)
        w = np.load(lp+'/scale_w.npy', allow_pickle=True)
            
        u_pf       = np.load(lp+'/scale_u_pf.npy', allow_pickle=True)
        v_pf       = np.load(lp+'/scale_v_pf.npy', allow_pickle=True)
        w_pf       = np.load(lp+'/scale_w_pf.npy', allow_pickle=True)
        
        if 'u_pfw_pf' in locals():
            u_pfw_pf   = np.append(u_pfw_pf, np.load(lp+'/scale_u_pfw_pf.npy', allow_pickle=True))
        else:
            u_pfw_pf   = np.load(lp+'/scale_u_pfw_pf.npy', allow_pickle=True)
        if 'u_psfw_psf' in locals():
            u_psfw_psf = np.append(u_psfw_psf,np.load(lp+'/scale_u_psfw_psf.npy', allow_pickle=True))
        else:        
            u_psfw_psf = np.load(lp+'/scale_u_psfw_psf.npy', allow_pickle=True)
        
        v_pfw_pf   = np.load(lp+'/scale_v_pfw_pf.npy', allow_pickle=True)
        v_psfw_psf = np.load(lp+'/scale_v_psfw_psf.npy', allow_pickle=True)

if load_profiles:
    # Domain averaged profiles
    profiles = xr.open_mfdataset(lp+'/profiles.'+casenr+'.nc')
    profiles['time'] = np.datetime64('2020-02-02') + profiles.time.astype("timedelta64[s]")

time = time_
#%% small adjustments
time = time -np.timedelta64(12,'h') - np.timedelta64(30,'m')
# s_time = s_time -np.timedelta64(12,'h') - np.timedelta64(30,'m')
if scale_sep:
    uw_p = u_pfw_pf + u_psfw_psf
    vw_p = v_pfw_pf + v_psfw_psf
    
### group by U variance 
# first quaritle
u2r_Q1 = np.quantile(profiles["u2r"].sel(zt=slice(0,200)).mean('zt'),0.25)
time_u2r_Q1 = profiles.where(profiles.sel(zt=slice(0,200)).mean('zt').u2r < u2r_Q1,drop=True).time
time_u2r_Q1 = np.unique(time_u2r_Q1.dt.round('H').values.astype('datetime64[s]'))
# third quartile
u2r_Q3 = np.quantile(profiles["u2r"].sel(zt=slice(0,200)).mean('zt'),0.75)
time_u2r_Q3 = profiles.where(profiles.sel(zt=slice(0,200)).mean('zt').u2r > u2r_Q3,drop=True).time
time_u2r_Q3 = np.unique(time_u2r_Q3.dt.round('H').values.astype('datetime64[s]'))


area_cel_H = 2500 * 2500 # m2
area_cel_D = (150000/1512)**2 # m2

#%% Calculate tendencies 
# dz = np.diff(ztlim)
# duwdz_f =  (u_pfw_pf[k,:,1:] - u_pfw_pf[k,:,:-1])/dz[np.newaxis,np.newaxis,:]
# duwdz_sf =  (u_psfw_psf[k,:,1:] - u_psfw_psf[k,:,:-1])/dz[np.newaxis,np.newaxis,:]

# plt.figure()
# plt.plot(3600*np.mean(duwdz_f[0,:,:],axis=0),ztlim[1:])
# plt.plot(3600*np.mean(duwdz_sf[0,:,:],axis=0),ztlim[1:])
# plt.axvline(0,c='k',lw=0.5)
# plt.xlim(-0.1,0.1)

dl_geo['uflx_par'] = dl_geo['uflx_conv']+dl_geo['uflx_turb']
step = 3600 # output timestep [seconds]
for var in ['uflx_conv','uflx_turb']:
    dl_geo[var] = (dl_geo[var].diff('time')) * step**-1  # gives values per second
mean_uflx_par = dl_geo['uflx_par'].mean(dim=['x','y'])

par_H  = np.zeros(len(s_pltz_H))
for ii in range(len(s_pltz_H)-1):
    par_H[ii]=mean_uflx_par.isel(z=s_pltz_H[ii]).sel(time='2020-02-03T14')

#%% PLOTTING 
##############################
##############################
###### Spectral plot variables saved for height 
# sizes = xsize / np.arange(1, 1512// 2 + 1) 

# # k1d = frquency
# # lam = wavelenght 
# # lam = xsize when k1d=pi

# lam = (xsize*(s_k1d/np.pi))
# nx  = np.pi/(s_k1d*1000)

# nx_H = np.pi/(s_k1d_H*1000)

# for it in [1]:
#     plt.figure()        
#     ### DALES 
#     plt.plot(nx,np.cumsum(s_k1d * s_uw[it,:]/1000)\
#             # /max(np.cumsum(spec_uw[it,:]))\
#              ,label='Resolved ' +str(int(s_ztlim[s_pltz]))+' m')
#     # plt.plot(wavelenght/1000,(max(np.cumsum(spec_uw[it,:]))-np.cumsum(spec_uw[it,:]))/1000\
#     #         # /max(np.cumsum(spec_uw[it,:]))\
#     #          ,label='Sub-filter scale', ls='--')
    
#     ### HARMONIE
#     # plt.plot(nx_H, par_H[2] + np.cumsum(s_k1d_H*s_uw_H[38,2,:])\
#     #                   ,c=col[2],ls='--',label=None)
    
    
    
#     # plt.axvline(2*np.pi*60/150000,c='k',lw=0.5)
#     plt.axvline(2.5,c='k',lw=0.5)
#     plt.axvline(150/1512,c='k',lw=0.5)
#     plt.axvline(150,c='k',lw=0.5)
#     plt.xscale('log')
#     plt.xlabel('Gridsize  [km]')
#     # plt.xlabel('N of boxes  ')
#     plt.legend()
#     plt.title('Cumulative spectra uw at '+np.datetime_as_string(s_time[s_plttime[it]], unit='h'))



#%%
##############################
############################## 
s_ktemp = s_k1d/(150000/1512)

wavelenght_H = np.pi/(s_k1d_H*1000)
it = 14
iz = 2
plt.plot(s_k1d_H,s_uw_H[it,iz,:],\
         c=col[iz],label='z: '+str(int(s_ztlim[s_pltz_H[iz]]))+' m')
# plt.yscale('log')
plt.legend()
plt.title('HARMONIE Spectra uw at '+np.datetime_as_string(s_time_H[it], unit='m'),fontsize=18)
#%%


if spec:
    # wavelenght = 0.19841* np.pi/s_k1d
    wavelenght = 2*np.pi/(s_k1d*1000) ## in DALES it's wrong because dx was wrong when doing spectral function
    wavelenght_H = np.pi/(s_k1d_H*1000)
    ## Spectral analysis 
    # for it in range(len(s_plttime)):
    for it in [3,17]:
        plt.figure()
        for iz in range(len(s_pltz[:-1])):
            plt.plot(wavelenght,s_k1d*s_uw[it,iz,:],\
                     c=col[iz],label='z: '+str(int(s_ztlim[s_pltz[iz]]))+' m')
            # plt.plot(wavelenght_H,s_k1d_H*s_uw_H[it+12,iz,:],\
            #           c=col[iz],ls='--',label=None)
        plt.axvline(2.5,c='k',lw=0.5)
        plt.xscale('log')
        # plt.yscale('log')
        plt.xlabel('Wavelength  [km]',fontsize=17)
        plt.ylabel('Spectral density',fontsize=17)
        plt.legend(fontsize=15)
        # plt.ylim([10**-6,10**-0.8])
        plt.title('Spectra uw at '+np.datetime_as_string(s_time[it], unit='m'),fontsize=18)
        # plt.title('Spectra uw.  Time: Feb-3 21:00UTC',fontsize=18)
        # plt.savefig(save_dir+'uw_spectra.pdf')
        
        
        
        # cumulative

                                
                                
        plt.figure()
        for iz in range(len(s_pltz)):
            b = np.abs(1000*(wavelenght[1:] - wavelenght[:-1]))
            h = ((s_ktemp*s_uw[it,iz,:])[1:] + (s_ktemp*s_uw[it,iz,:])[:-1]) / 2
            cumulat_flx = np.cumsum(b*h)/1512
            
            b_H = np.abs(1000*(wavelenght_H[1:] - wavelenght_H[:-1]))
            h_H = ((s_k1d_H*s_uw_H[it+37,iz,:])[1:] + (s_k1d_H*s_uw_H[it+37,iz,:])[:-1]) / 2
            cumulat_flx_H = np.cumsum(b_H * h_H)
                        
            ### DALES
            # plt.plot(np.flip(wavelenght[:-1]),cumulat_flx\
            #          ,c=col[iz],label='z: '+str(int(s_ztlim[s_pltz[iz]]))+' m')

            plt.plot(wavelenght,1- np.cumsum(s_uw[it,iz,:])\
                   /max(np.cumsum(s_uw[it,iz,:]))\
                     ,label=str(int(s_ztlim[s_pltz[iz]]))+' m',c=col[iz])

            # plt.plot((wavelenght), np.abs(np.cumsum(s_ktemp*s_uw[it,iz,:])\
            #             -max(np.cumsum(s_ktemp*s_uw[it,iz,:])))\
            #               ,label=str(int(s_ztlim[s_pltz[iz]]))+' m',c=col[iz])
            ### HARMONIE
            plt.plot(np.flip(wavelenght_H[:-1]), par_H[iz] + cumulat_flx_H\
                      ,c=col[iz],ls='--',label=None)
                
            plt.plot(wavelenght_H, np.abs(np.cumsum(s_k1d_H*s_uw_H[it+12,iz,:])\
                      -max(np.cumsum(s_k1d_H*s_uw_H[it+12,iz,:])))\
                      ,c=col[iz],ls='--',label=None)
            plt.plot(wavelenght,np.cumsum(s_k1d*s_uw[it,iz,:])\
                        /max(np.cumsum(s_k1d*s_uw[it,iz,:]))\
                          ,label=None,ls='--',c=col[iz])
        plt.axvline(2.5,c='k',lw=0.5)
        plt.xscale('log')
        # plt.yscale('log')
        plt.xlabel('Wavelength  [km]')
        plt.ylabel('%')
        plt.legend()
        plt.title('Cumulative spectra uw at '+np.datetime_as_string(s_time[it], unit='h'))
        # plt.savefig(save_dir+'spectra'+np.datetime_as_string(s_time[it], unit='m')+'.pdf', bbox_inches="tight")
#%% Cumulative spectral plot
    # cumulative
    ## First normalize the data 
    s_uw_norm = 1- np.cumsum(s_k1d*s_uw,2)\
                    /np.max(np.cumsum(s_k1d*s_uw,2),2)[:,:,None]
    # s_uw_norm = np.cumsum(s_k1d*s_uw,2)
                  
    ## Second compute the statisitcs
    s_uw_distr = {}
    s_uw_distr['mean']=np.mean(s_uw_norm,axis=0)
    s_uw_distr['q1']=np.quantile(s_uw_norm,0.25,axis=0)
    s_uw_distr['q2']=np.quantile(s_uw_norm,0.5,axis=0)
    s_uw_distr['q3']=np.quantile(s_uw_norm,0.75,axis=0)
    # Plot
    plt.figure()
    for iz in range(len(s_pltz)):
        # for ii in s_uw_distr.keys():
        for ii in ['mean','q2']:
            plt.plot(wavelenght,s_uw_distr[ii][iz,:]\
                      ,\
                    c=col[iz],ls=style[ii])  
        plt.fill_between(wavelenght,s_uw_distr['q1'][iz,:]\
                    ,s_uw_distr['q3'][iz,:]\
                    ,alpha=0.2,color=col[iz],\
                    label=str(int(s_ztlim[s_pltz[iz]]))+' m')
        # plt.hist(np.sum(s_k1d*s_uw,2))
    plt.axvline(2.5,c='k',lw=0.5)
    plt.xscale('log')
    plt.xlabel('Wavelength  [km]')
    plt.ylabel('%')
    plt.legend()
    plt.title('Mean cumulative spectra uw',fontsize = 20)
    # plt.savefig(save_dir+'spectra_distribution_uw.pdf', bbox_inches="tight")
    
    ## Only for times with high variance of momentum flux (3rd quartile)
    # compute the statisitcs
    s_uw_distr = {}
    s_uw_distr['mean']=np.mean(s_uw_norm[np.nonzero(np.in1d(time, time_u2r_Q3))[0],:,:],axis=0)
    s_uw_distr['q1']=np.quantile(s_uw_norm[np.nonzero(np.in1d(time, time_u2r_Q3))[0],:,:],0.25,axis=0)
    s_uw_distr['q2']=np.quantile(s_uw_norm[np.nonzero(np.in1d(time, time_u2r_Q3))[0],:,:],0.5,axis=0)
    s_uw_distr['q3']=np.quantile(s_uw_norm[np.nonzero(np.in1d(time, time_u2r_Q3))[0],:,:],0.75,axis=0)
    # Plot
    plt.figure()
    for iz in range(len(s_pltz)):
        # for ii in s_uw_distr.keys():
        for ii in ['mean','q2']:
            plt.plot(wavelenght,s_uw_distr[ii][iz,:]\
                      ,\
                    c=col[iz],ls=style[ii])  
        plt.fill_between(wavelenght,s_uw_distr['q1'][iz,:]\
                    ,s_uw_distr['q3'][iz,:]\
                    ,alpha=0.2,color=col[iz],\
                    label=str(int(s_ztlim[s_pltz[iz]]))+' m')
    plt.axvline(2.5,c='k',lw=0.5)
    plt.xscale('log')
    plt.xlabel('Wavelength  [km]')
    plt.ylabel('%')
    plt.legend()
    plt.title('Mean cumulative spectra uw',fontsize = 20)
    plt.suptitle('3st quaritle of uw')
    # plt.savefig(save_dir+'spectra_distribution_uw_Q3.pdf', bbox_inches="tight")
    
    ## Only for times with low variance of momentum flux (1rd quartile)
    # compute the statisitcs
    s_uw_distr = {}
    s_uw_distr['mean']=np.mean(s_uw_norm[np.nonzero(np.in1d(time, time_u2r_Q1))[0],:,:],axis=0)
    s_uw_distr['q1']=np.quantile(s_uw_norm[np.nonzero(np.in1d(time, time_u2r_Q1))[0],:,:],0.25,axis=0)
    s_uw_distr['q2']=np.quantile(s_uw_norm[np.nonzero(np.in1d(time, time_u2r_Q1))[0],:,:],0.5,axis=0)
    s_uw_distr['q3']=np.quantile(s_uw_norm[np.nonzero(np.in1d(time, time_u2r_Q1))[0],:,:],0.75,axis=0)
    # Plot
    plt.figure()
    for iz in range(len(s_pltz)):
        # for ii in s_uw_distr.keys():
        for ii in ['mean','q2']:
            plt.plot(wavelenght,s_uw_distr[ii][iz,:]\
                      ,\
                    c=col[iz],ls=style[ii])  
        plt.fill_between(wavelenght,s_uw_distr['q1'][iz,:]\
                    ,s_uw_distr['q3'][iz,:]\
                    ,alpha=0.2,color=col[iz],\
                    label=str(int(s_ztlim[s_pltz[iz]]))+' m')
    plt.axvline(2.5,c='k',lw=0.5)
    plt.xscale('log')
    plt.xlabel('Wavelength  [km]')
    plt.ylabel('%')
    plt.legend()
    plt.title('Mean cumulative spectra uw',fontsize = 20)
    plt.suptitle('1st quaritle of uw')
    # plt.savefig(save_dir+'spectra_distribution_uw_Q1.pdf', bbox_inches="tight")
    
#%% 
# ### Here statisitcs are compute before normalising the data
#     s_uw_distr = {}
#     s_uw_distr['mean']=np.mean(s_uw,axis=0)
#     s_uw_distr['q1']=np.quantile(s_uw,0.25,axis=0)
#     s_uw_distr['q2']=np.quantile(s_uw,0.5,axis=0)
#     s_uw_distr['q3']=np.quantile(s_uw,0.75,axis=0)
    
#     plt.figure()
#     for iz in range(len(s_pltz[0:2])):
#         # for ii in s_uw_distr.keys():
#         for ii in ['mean','q2']:
#             plt.plot(wavelenght,1- np.cumsum(s_k1d*s_uw_distr[ii][iz,:])\
#                     /max(np.cumsum(s_k1d*s_uw_distr[ii][iz,:]))\
#                       ,\
#                     c=col[iz],ls=style[ii])
                
#         plt.fill_between(wavelenght,1- np.cumsum(s_k1d*s_uw_distr['q1'][iz,:])\
#                 /max(np.cumsum(s_k1d*s_uw_distr['q1'][iz,:]))\
#                     ,1- np.cumsum(s_k1d*s_uw_distr['q3'][iz,:])\
#                 /max(np.cumsum(s_k1d*s_uw_distr['q3'][iz,:]))\
#                     ,alpha=0.2,color=col[iz],\
#                     label=str(int(s_ztlim[s_pltz[iz]]))+' m')
#         # plt.plot(wavelenght[:,None],(1- np.cumsum(s_k1d*s_uw[:,iz,:],axis=1)\
#                     # /np.max(np.cumsum(s_k1d*s_uw[:,iz,:],axis=1),axis=1)[:, np.newaxis]).T\
#                     #   ,label=str(int(s_ztlim[s_pltz[iz]]))+' m',c=col[iz])
#     plt.axvline(2.5,c='k',lw=0.5)
#     plt.xscale('log')
#     plt.xlabel('Wavelength  [km]')
#     plt.ylabel('%')
#     plt.legend()
#     plt.title('Mean cumulative spectra uw',fontsize = 20)
#     # # plt.savefig(save_dir+'spectra_distribution.pdf', bbox_inches="tight")


#%% Non normalized cumulative spectra

for it in range(len(plttime[0:2])):
    plt.figure()
    for iz in range(len(s_pltz)):
        plt.plot(wavelenght,np.flip(np.cumsum(s_k1d*s_uw[it,iz,:]))\
                     ,label=str(int(s_ztlim[s_pltz[iz]]))+' m')
    plt.xscale('log')
    plt.xlabel('Wavelength  [km]')
    plt.legend()
    plt.title('Cumulative spectra uw at '+np.datetime_as_string(s_time[s_plttime[it]], unit='h'))
#%%
if scale_sep:
    
    k=2
    uw_sf_tav = np.mean(u_psfw_psf,axis=1)[k,:]
    uw_f_tav = np.mean(u_pfw_pf,axis=1)[k,:]
    uw_tot_tav = uw_sf_tav + uw_f_tav + profiles['uws'].sel(time=time[:-1]).interp(zm=ztlim).mean('time')
    
    ## Scale separation
    # for it in range(len(time)):
    plt.figure(figsize=(4,9))
    
    
    # ((dl_geo['uflx_conv']+dl_geo['uflx_turb']).mean(dim=['x','y'])+\
    #   dl_geo['up_wp']).sel(time=time).mean('time').sel(z=slice(0,4500)).\
    #     plot(y='z',c='r',ls='--',lw=2,label='HARMONIE Tot')
        
    # ((dl_geo['uflx_conv']+dl_geo['uflx_turb']).mean(dim=['x','y'])\
    #   ).sel(time=time).mean('time').sel(z=slice(0,4500)).\
    #     plot(y='z',c='g',ls='--',label='HARMONIE Param')
        
    # # ((dl_geo['uflx_conv']).mean(dim=['x','y'])\
    # #   ).sel(time=time[it]).sel(z=slice(0,4500)).\
    # #     plot(y='z',c='b',label='HARMONIE conv',lw=1)
    # # ((dl_geo['uflx_turb']).mean(dim=['x','y'])\
    # #   ).sel(time=time[it]).sel(z=slice(0,4500)).\
    # #     plot(y='z',c='g',label='HARMONIE turb')
        
    # dl_geo['up_wp'].sel(time=time).mean('time').sel(z=slice(0,4500)).\
    #     plot(y='z',c='k',ls='--',label='HARMONIE Resol')
        


    
    # for k in range(len(klps)-1):
    ###
    plt.plot(uw_sf_tav + uw_f_tav,ztlim,lw=2,c='r',label='DALES Tot')
    # total from profiles
    # plt.plot(profiles['uwt'].sel(time=time[:-1]).interp(zm=ztlim).mean('time'),ztlim,c='k',lw=3,label='DALES Tot')
    # total flux 
    # plt.plot(uw_tot_tav,ztlim,c='k',lw=3,label='DALES Tot')
    ###
    #sub filter flux
    plt.plot(uw_sf_tav,ztlim,c='g',label='DALES SFS <'+str(150/(klps*2)[k])+' km')
    ###
    #filter flux
    plt.plot(uw_f_tav,ztlim,c='k',ls='-',label='DALES FS >'+str(150/(klps*2)[k])+' km')
    
    
    plt.plot(profiles['uws'].sel(time=time[:-1]).interp(zm=ztlim).mean('time'),ztlim,c='b',ls=':',label='DALES Sub-grid')
    # plt.plot((u_pfw_pf + u_psfw_psf)[it,:],ztlim,c='k',ls= ':',alpha=0.5,label='sum')
    plt.legend()
    plt.axvline(0,c='grey',alpha=0.6,lw=0.5)
    # plt.xlim([-0.013,0.05])
    plt.ylim([0,4000])
    plt.xlabel(r"$m^2 /s^2$")
    # plt.title('Mean UW at '+np.datetime_as_string(time[it], unit='h'))
    plt.title('Mean UW averaged')
    # plt.savefig(save_dir+'poster_uw_momflux_prof_av.pdf', bbox_inches="tight")
    
    
    
    #%%
    
    ## Scale separation
    # for it in range(len(time)):
    for it in [1,8]:
        plt.figure(figsize=(4,9))
        
        
        # ((dl_geo['uflx_conv']+dl_geo['uflx_turb']).mean(dim=['x','y'])+\
        #   dl_geo['up_wp']).sel(time=time[it]).sel(z=slice(0,4500)).\
        #     plot(y='z',c='r',ls='--',lw=3,label='HARMONIE Tot')
            
        # ((dl_geo['uflx_conv']+dl_geo['uflx_turb']).mean(dim=['x','y'])\
        #   ).sel(time=time[it]).sel(z=slice(0,4500)).\
        #     plot(y='z',c='g',lw=2,ls='--',label='HARMONIE Param')
            
        # # ((dl_geo['uflx_conv']).mean(dim=['x','y'])\
        # #   ).sel(time=time[it]).sel(z=slice(0,4500)).\
        # #     plot(y='z',c='b',label='HARMONIE conv',lw=1)
        # # ((dl_geo['uflx_turb']).mean(dim=['x','y'])\
        # #   ).sel(time=time[it]).sel(z=slice(0,4500)).\
        # #     plot(y='z',c='g',label='HARMONIE turb')
            
        # dl_geo['up_wp'].sel(time=time[it]).sel(z=slice(0,4500)).\
        #     plot(y='z',c='k',lw=2,ls='--',label='HARMONIE Resol')
            


        
        # for k in range(len(klps)-1):
        # DLES_tot =  u_psfw_psf[k,it,:] + u_pfw_pf[k,it,:] + profiles['uws'].sel(time=slice(time[it-1],time[it])).interp(zm=ztlim).mean('time')
        
        # plt.plot(DLES_tot,ztlim,c='k',lw=3,label='DALES Tot')
        plt.plot(uw_p[0,it,:],ztlim,c='r',lw=3,label='D. Tot')
        
       
        for k in [2]:
            plt.plot(u_psfw_psf[k,it,:],ztlim,c='g',lw=2,label='D. SFS <'+str(150/(klps*2)[k])+' km')
            plt.plot(u_pfw_pf[k,it,:],ztlim,c='k',lw=2,ls='-',label='D. FS >'+str(150/(klps*2)[k])+' km')
        profiles['uws'].sel(zm=ztlim,method='nearest').sel(time=time[it-1],method='nearest').\
            plot(y='zm',ls=':',c='b',label='D. Param')
        # plt.plot((u_pfw_pf + u_psfw_psf)[it,:],ztlim,c='k',ls= ':',alpha=0.5,label='sum')
        plt.legend()
        plt.xlabel(r'$m^2/s^2$')
        plt.axvline(0,c='grey',alpha=0.6,lw=0.5)
        # plt.xlim([-0.034,0.061])
        plt.ylim([0,4000])
        plt.title('Mean UW at '+np.datetime_as_string(time[it], unit='h'))
        # plt.savefig(save_dir+'poster_uw_momflux_prof_'+np.datetime_as_string(time[it])+'.pdf', bbox_inches="tight")
        # plt.savefig(save_dir+'DALES_uw_momflux_prof_'+np.datetime_as_string(time[it])+'.pdf', bbox_inches="tight")

# 
#%% Mean scale separation 

    fig, ax1 = plt.subplots(figsize=(5,9))
    
    ax1.plot(np.convolve(np.mean(100*u_pfw_pf/uw_p,axis=1)[1,:],np.ones(10) / 10,mode='same'),
                 ztlim,c=col[1],ls='-',label='F',lw=3)
    ax1.plot(np.convolve(np.mean(100*u_psfw_psf/uw_p,axis=1)[1,:],np.ones(10) / 10,mode='same'),
                 ztlim,c=col[2],ls='-',label='SF',lw=3)
    ax1.axvline(0,c='k',lw=0.5)
    ax1.set_xlabel('%')
    ax1.set_ylabel('z [m]')
    plt.legend()
    ax2 = ax1.twiny()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_xlabel('Zonal momentum flux', color=color)
    ax2.tick_params(axis='x', labelcolor=color)
    ax2.axvline(0,c='r',lw=0.5)
    ax2.plot(np.mean(uw_p,axis=1)[1,:],ztlim,c='r',lw=1,label='Total')
    # plt.ylim([500,3000])
    #%%
    it = 1
    k = 2
    fig, ax1 = plt.subplots(figsize=(5,9))
    ax1.plot(((100*abs(u_pfw_pf)/(abs(u_pfw_pf)+abs(u_psfw_psf)))[k,it,:]),
                  ztlim,c=col[1],ls='-',label='F',lw=3)
    ax1.plot((100*abs(u_psfw_psf)/(abs(u_pfw_pf)+abs(u_psfw_psf)))[k,it,:],
                  ztlim,c=col[2],ls='-',label='SF',lw=3)
    ax1.axvline(0,c='k',lw=0.5)
    ax1.set_xlabel('%')
    ax1.set_ylabel('z [m]')
    plt.legend()
    # ax1.set_xlim([-10,100])
    ax2 = ax1.twiny()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_xlabel('Zonal momentum flux', color=color)
    ax2.tick_params(axis='x', labelcolor=color)
    ax2.axvline(0,c='r',lw=0.5)
    ax2.plot((uw_p)[k,it,:],ztlim,c='r',lw=1,label='Total')
    # ax2.plot(((u_pfw_pf)[k,it,:]),
    #               ztlim,c=col[1],ls='-',label='F',lw=3)
    # ax2.plot(((u_psfw_psf)[k,it,:]),
    #               ztlim,c=col[2],ls='-',label='SF',lw=3)
    plt.ylim([0,3500])
    plt.legend()
    plt.title(s_time[plttime[it]])
    

#%% Time series of momentum flux profiles (and wind variance)

    # plttime = profiles.time[plttime*2]
    ## contour   
    k = 2
    x,y = np.meshgrid(time,ztlim/1000)
    
    level_bound  =  np.linspace(-0.009, 0.028, 20)
    level_bound =  np.linspace(-0.03, 0.06, 30)
    
    fig, axs = plt.subplots(3,figsize=(12,9))
    # fig.suptitle('')
    
    CS = axs[0].contourf(x,y,uw_p[k,:,:].T,cmap=cm.PiYG_r,\
                         levels=level_bound,extend='both',norm=DivergingNorm(0))
    cbar = fig.colorbar(CS, ax=axs[0],pad = 0.09)
    cbar.ax.locator_params(nbins=5)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    axs[0].set_ylabel('z [km]')
    axs[0].set_xticks([])
    axs[0].set_title('Total flux',fontsize=18)
    cbar.ax.set_ylabel(r'uw [$m^2/s^2$]',fontsize=17)
    
    # for ii in range(len(time_u2r_Q1)):
    #     axs[0].axvspan(time_u2r_Q1[ii]-np.timedelta64(30,'m'),\
    #                          time_u2r_Q1[ii]+np.timedelta64(30,'m'),\
    #                              alpha=0.2,color='b',lw=0)
    # for ii in range(len(time_u2r_Q3)):
    #     axs[0].axvspan(time_u2r_Q3[ii]-np.timedelta64(30,'m'),\
    #                          time_u2r_Q3[ii]+np.timedelta64(30,'m'),\
    #                              alpha=0.2,color='r',lw=0)
    
    # axs[0].axvline(time[1],c='k',lw=0.5)
    # axs[1].axvline(time[1],c='k',lw=0.5)
    # axs[2].axvline(time[1],c='k',lw=0.5)
    # axs[0].axvline(time[8],c='k',lw=0.5)
    # axs[1].axvline(time[8],c='k',lw=0.5)
    # axs[2].axvline(time[8],c='k',lw=0.5)
     
    #plot variance of u
    if load_profiles:
        ax2 = axs[0].twinx()
        profiles.u2r.sel(zt=slice(0,200)).mean('zt').plot(x='time',ax=ax2)
        # profiles.w2r.sel(zm=slice(1000,1800)).mean('zm').plot(x='time',ax=ax2)
        ax2.set_xticks([])
        ax2.set_ylabel('u variance (0-200m)',fontsize=15)
        
    # axs[0].set_yticks([])

    CS1 = axs[1].contourf(x,y,(u_pfw_pf[k,:,:]).T,\
                          levels=level_bound*0.5,extend='both',\
                          cmap=cm.PiYG_r,norm=DivergingNorm(0))
    cbar1 = fig.colorbar(CS1, ax=axs[1],pad = 0.09)
    cbar1.ax.locator_params(nbins=5)
    cbar1.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    axs[1].set_ylabel('z [km]')
    axs[1].set_xticks([])
    axs[1].set_title('Filter scale flux (>2.5 km)',fontsize=18)
    cbar1.ax.set_ylabel('uw_F',fontsize=17)
    
    CS2 = axs[2].contourf(x,y,(u_psfw_psf[k,:,:]).T,\
                          levels=level_bound*0.5,extend='both',\
                          cmap=cm.PiYG_r,norm=DivergingNorm(0))
    cbar2 = fig.colorbar(CS2, ax=axs[2],pad = 0.09)
    cbar2.ax.locator_params(nbins=5)
    cbar2.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    axs[2].set_ylabel('z [km]')
    axs[2].set_title('Sub-filter scale flux (<2.5 km)',fontsize=18)
    cbar2.ax.set_ylabel('uw_SF',fontsize=17)
    
    # axs[0].set_ylim([400,None])
    # axs[1].set_ylim([400,None])
    # axs[2].set_ylim([400,None])
    axs[2].set_xticklabels([x.hour for x in time.tolist()], rotation = 45)
    axs[2].set_xlabel('hours from Feb-3 00:00',fontsize=14)
    # plt.savefig(save_dir+'mom_flux_contour.pdf', bbox_inches="tight")
    
    
    
#%% time series flux HARMONIE 
# dl_geo = xr.open_mfdataset('/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/HARMONIE/cy43_clim/my_3d_harm_clim_lev_.nc',combine='by_coords')
tmser = xr.open_mfdataset(lp+'/tmser.'+casenr+'.nc')
tmser['time'] = np.datetime64('2020-02-02') + tmser.time.astype("timedelta64[s]") 

k = 2
x,y = np.meshgrid(time,ztlim/1000)

level_bound  =  np.linspace(-0.009, 0.028, 20)
level_bound =  np.linspace(-0.03, 0.06, 30)

fig, axs = plt.subplots(2,2,figsize=(23,15))
# fig.suptitle('')
#########
######### DALES filter scale
CS0 = axs[0,0].contourf(x,y,(u_pfw_pf[k,:,:]).T,\
                      levels=level_bound*0.5,extend='both',\
                      cmap=cm.PiYG_r,norm=DivergingNorm(0))
# cbar0 = fig.colorbar(CS0, ax=axs[0,0],pad = 0.09)
# cbar0.ax.locator_params(nbins=5)
# cbar1.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
axs[0,0].set_ylabel('z [km]')
# axs[0,0].set_xticks([])
axs[0,0].set_title('DALES Filter scale flux (>2.5 km)',fontsize=24)

# cbar0.ax.set_ylabel('uw_F',fontsize=17)
#########
######### DALES sub-filter scale
CS1 = axs[0,1].contourf(x,y,(u_psfw_psf[k,:,:]).T,\
                      levels=level_bound*0.5,extend='both',\
                      cmap=cm.PiYG_r,norm=DivergingNorm(0))
# cbar1 = fig.colorbar(CS1, ax=axs[0,1],pad = 0.09)
cbar1.ax.locator_params(nbins=5)
cbar1.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
# axs[0,1].set_ylabel('z [km]')
# axs[0,1].set_yticks([])
axs[0,1].set_title('DALES Sub-filter scale flux (<2.5 km)',fontsize=24)
# cbar1.ax.set_ylabel('uw_SF',fontsize=17)

# axs[0].set_ylim([400,None])
# axs[1].set_ylim([400,None])
# axs[2].set_ylim([400,None])
axs[1,0].set_xticklabels([x.hour for x in time.tolist()], rotation = 45)
axs[1,0].set_xlabel('hours from Feb-3 00:00',fontsize=20)
# plt.savefig(save_dir+'mom_flux_contour.pdf', bbox_inches="tight")
#########
######### HARMONIE reolved
x_H,y_H = np.meshgrid(dl_geo.time.sel(time=slice(time[0],time[-1])),dl_geo.z.sel(z=slice(0,4500)))
CS2 = axs[1,0].contourf(x_H,y_H/1000,\
                (dl_geo['up_wp']).sel(time=slice(time[0],time[-1])).sel(z=slice(0,4500)).T,\
                      levels=level_bound*0.5,extend='both',\
                      cmap=cm.PiYG_r,norm=DivergingNorm(0))
axs[1,0].set_title('HARMONIE resolved',fontsize=24)
    
CS3 = axs[1,1].contourf(x_H,y_H/1000,\
                (dl_geo['uflx_conv']+dl_geo['uflx_turb']).mean(dim=['x','y'])\
                 .sel(time=slice(time[0],time[-1])).sel(z=slice(0,4500)).T,\
                      levels=level_bound*0.5,extend='both',\
                      cmap=cm.PiYG_r,norm=DivergingNorm(0))
axs[1,1].set_title('HARMONIE parameterized',fontsize=24)
    
axs[0,0].set_xticklabels([x.hour for x in time.tolist()], rotation = 45)    
axs[0,1].set_xticklabels([x.hour for x in time.tolist()], rotation = 45)
axs[1,0].set_xticklabels([x.hour for x in time.tolist()], rotation = 45)
axs[1,1].set_xticklabels([x.hour for x in time.tolist()], rotation = 45)
axs[1,1].set_xlabel('hours from Feb-3 00:00',fontsize=20)

axs[1,0].set_ylabel('z [km]')
cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
fig.colorbar(CS1, cax=cbar_ax,format="%.2f",label='U momentum flux')
## CLOUD TOP
axs[0,0].plot(tmser.time,tmser["zc_max"].rolling(time=10, center=True).mean()/1000)
axs[0,1].plot(tmser.time,tmser["zc_max"].rolling(time=10, center=True).mean()/1000)

axs[1,0].plot(dl_geo.time,cl_max_HAR[:len(dl_geo.time)]/1000)
axs[1,0].set_xlim(time[0],time[-1])
axs[1,1].plot(dl_geo.time,cl_max_HAR[:len(dl_geo.time)]/1000)
axs[1,1].set_xlim(time[0],time[-1])
axs[1,0].set_ylim([0,4.5])
axs[1,1].set_ylim([0,4.5])



# plt.savefig(save_dir+'mom_flux_contour_HARM&DALES.pdf', bbox_inches="tight")







# step = 3600 # output timestep [seconds]
# for var in ['uflx_conv','uflx_turb']:
#     dl_geo[var] = (dl_geo[var].diff('time')) * step**-1  # gives values per second
# plt.figure(figsize=(19,5))
# # dl_geo.isel(time=slice(37,48)).sel(z=slice(0,4500))['up_wp'].plot(y='z',\
# #                           cmap=cm.PiYG_r,\
# #                          vmax=0.03,vmin=-0.015,norm=DivergingNorm(0))
# # (dl_geo['uflx_conv']+dl_geo['uflx_turb']).mean(dim=['x','y']).isel(time=slice(37,48)).sel(z=slice(0,4500)).plot(y='z',\
# #                           cmap=cm.PiYG_r,\
# #                          vmax=0.03,vmin=-0.015,norm=DivergingNorm(0))
# ((dl_geo['uflx_conv']+dl_geo['uflx_turb']).mean(dim=['x','y'])dl_geo['up_wp']).isel(time=slice(37,48)).sel(z=slice(0,4500)).plot(y='z',\
#                           cmap=cm.PiYG_r,\
#                          vmax=0.06,vmin=-0.02,norm=DivergingNorm(0))
# plt.suptitle('Resolved u momentum flux form HARMONIE')
#%%

#%%
level_bound =  np.linspace(-0.000008, 0.000005, 30)
level_bound =  np.linspace(-0.015, 0.03, 50)
plt.figure(figsize=(14,6))
plt.contourf(s_time_H[37:48],z_H/1000,up_wp_H[37:48,:-3].T,\
             levels=level_bound,extend='both',cmap=cm.PiYG_r,norm=DivergingNorm(0))
# plt.contourf(s_time_H,z_H/1000,up_wp_H.T/area_cel_H,\
#              levels=level_bound,extend='both',cmap=cm.PiYG_r)
plt.ylim([0,4.5])
plt.colorbar()
# for ii in np.arange(s_time_H[0].astype('datetime64[D]'), s_time_H[-1].astype('datetime64[D]')):
#     plt.axvline(x=ii,c='k')

x,y = np.meshgrid(time,ztlim/1000)

# ## Plot flux divided by area of a grid cell 
# plt.figure()
# plt.contourf(x,y,uw_p[2,:,:].T/area_cel_D,\
#              levels=level_bound,extend='both',cmap=cm.PiYG_r)
# plt.ylim([0,4.5])
# plt.colorbar()
#%% plot maps
if Dfields:
    z_plot = 4

    fig,axs = plt.subplots(nrows=1,sharex=True,figsize=(8,8))
    
    sc1 = axs.imshow(wspd[z_plot,:,:],
    # sc1 = axs.imshow(np.mean(wspd,axis=0),
                       aspect='auto',cmap='RdPu',
                       vmin=4,
                       vmax=17,
                       extent=[0,150,0,150])
    axs.set_xlabel(r"x[km]",fontsize=16)
    axs.set_ylabel(r'y [km]',fontsize=16)
    axs.invert_yaxis()
    # axs.title.set_text('Wind speed. Time:  '+'. Z:'+str(int(ztlim[z_plot].values))+' m' )
    axs.set_title('Time: Feb-3 14:00 UTC.  '+'      z: '+str(int(zt[z_plot].values))+' m',fontsize = 18)
    pos1 = axs.get_position()
    cbax1 = fig.add_axes([0.95, pos1.ymin, 0.01, pos1.height])
    cb1 = fig.colorbar(sc1, cax=cbax1)
    cb1.ax.set_ylabel(r'Wind speed [m/s]',fontsize=16)
    # fig.tight_layout()
    # plt.savefig(save_dir+'wspd_field_100m.pdf', bbox_inches="tight")
#%% plot profiles
if load_profiles: 
    print('Nothing to plot...')
    
    
#%% SAME NOW FOR HARMONIE


    
    
    
    
