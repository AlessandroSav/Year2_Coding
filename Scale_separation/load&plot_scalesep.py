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

heights=[1500,]

## running on staffumbrella
# lp = 
# save_dir   = ''
## running on Local
base_dir   = '/Users/acmsavazzi/Documents/WORK/PhD_Year1/DATA/DALES/DALES_atECMWF/outputs/20200202_12_clim'
# fig_dir = 
save_dir = '/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/DALES/'
############
## read klps once for all
casenr = '001'
ih = 0
klps    = np.load(base_dir+'/Exp_'+casenr+'/scale_klps_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
############

time=np.empty(0,dtype='datetime64')
u_av = np.empty([len(heights),0])
v_av = np.empty([len(heights),0])
w_av = np.empty([len(heights),0])
#
u_pf = np.empty([len(heights),len(klps),0])
v_pf = np.empty([len(heights),len(klps),0])
w_pf = np.empty([len(heights),len(klps),0])
u_pfw_pf    = np.empty([len(heights),len(klps),0])
u_psfw_psf  = np.empty([len(heights),len(klps),0])
v_pfw_pf    = np.empty([len(heights),len(klps),0])
v_psfw_psf  = np.empty([len(heights),len(klps),0])
uw_pf    = np.empty([len(heights),len(klps),0])
uw_psf  = np.empty([len(heights),len(klps),0])
vw_pf    = np.empty([len(heights),len(klps),0])
vw_psf  = np.empty([len(heights),len(klps),0])

for casenr in ['001','002','003','004','005','006','007','008','009','010']:
    ## running on staffumbrella
    # lp = 
    ## running on Local
    lp   = base_dir+'/Exp_'+casenr
    
    ih = 0
    ### These variables should be the same for all heights, so only open one
    time_temp    = np.load(lp+'/scale_time_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)     
    # klps    = np.load(lp+'/scale_klps_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
    ### initialize based on size
    heights_exact = np.zeros((len(heights)))
    
    u_av_temp = np.zeros((len(heights),len(time_temp)))
    v_av_temp = np.zeros((len(heights),len(time_temp)))
    w_av_temp = np.zeros((len(heights),len(time_temp))) 
    u_pf_temp        = np.zeros((len(heights),len(klps),len(time_temp)))
    v_pf_temp        = np.zeros((len(heights),len(klps),len(time_temp)))
    w_pf_temp        = np.zeros((len(heights),len(klps),len(time_temp)))
    u_pfw_pf_temp    = np.zeros((len(heights),len(klps),len(time_temp)))
    u_psfw_psf_temp  = np.zeros((len(heights),len(klps),len(time_temp)))
    v_pfw_pf_temp    = np.zeros((len(heights),len(klps),len(time_temp)))
    v_psfw_psf_temp  = np.zeros((len(heights),len(klps),len(time_temp)))
    #
    uw_pf_temp       = np.zeros((len(heights),len(klps),len(time_temp)))
    uw_psf_temp      = np.zeros((len(heights),len(klps),len(time_temp)))
    vw_pf_temp       = np.zeros((len(heights),len(klps),len(time_temp)))
    vw_psf_temp      = np.zeros((len(heights),len(klps),len(time_temp)))
    
    for ih in range(len(heights)):
    ## For scale separation 
        heights_exact[ih]  = np.load(lp+'/scale_zt_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
    
        u_av_temp[ih,:] = np.load(lp+'/scale_u_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
        v_av_temp[ih,:] = np.load(lp+'/scale_v_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
        w_av_temp[ih,:] = np.load(lp+'/scale_w_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
        #
        u_pf_temp[ih,:,:]       = np.load(lp+'/scale_u_pf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
        v_pf_temp[ih,:,:]       = np.load(lp+'/scale_v_pf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
        w_pf_temp[ih,:,:]       = np.load(lp+'/scale_w_pf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
        u_pfw_pf_temp[ih,:,:]   = np.load(lp+'/scale_u_pfw_pf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
        u_psfw_psf_temp[ih,:,:] = np.load(lp+'/scale_u_psfw_psf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
        v_pfw_pf_temp[ih,:,:]   = np.load(lp+'/scale_v_pfw_pf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
        v_psfw_psf_temp[ih,:,:] = np.load(lp+'/scale_v_psfw_psf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
        #
        uw_pf_temp[ih,:,:]   = np.load(lp+'/scale_uw_pf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
        uw_psf_temp[ih,:,:] = np.load(lp+'/scale_uw_psf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
        vw_pf_temp[ih,:,:]   = np.load(lp+'/scale_vw_pf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
        vw_psf_temp[ih,:,:] = np.load(lp+'/scale_vw_psf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
        
    ## Append all times together
    time = np.append(time,time_temp)
    u_av=np.append(u_av,u_av_temp,1)
    v_av=np.append(v_av,v_av_temp,1)
    w_av=np.append(w_av,w_av_temp,1)
    #
    u_pf=np.append(u_pf,u_pf_temp,2)
    v_pf=np.append(v_pf,v_pf_temp,2)
    w_pf=np.append(w_pf,w_pf_temp,2)
    u_pfw_pf    = np.append(u_pfw_pf,   u_pfw_pf_temp,  2)
    u_psfw_psf  = np.append(u_psfw_psf, u_psfw_psf_temp,2)
    v_pfw_pf    = np.append(v_pfw_pf,   v_pfw_pf_temp,  2)
    v_psfw_psf  = np.append(v_psfw_psf, v_psfw_psf_temp,2)
    uw_pf       = np.append(uw_pf,      uw_pf_temp,     2)
    uw_psf      = np.append(uw_psf,     uw_psf_temp,    2)
    vw_pf       = np.append(vw_pf,      vw_pf_temp,     2)
    vw_psf      = np.append(vw_psf,     vw_psf_temp,    2)
    
    
#%%   
da_scales = xr.Dataset(
    {'u_pf':(
        ('height','klp','time'),
        u_pf,
        )},
        coords={'height':heights,'klp':klps,'time':time},)

da_scales['v_pf']       =(('height','klp','time'),v_pf)
da_scales['w_pf']       =(('height','klp','time'),w_pf)
da_scales['u_pfw_pf']   =(('height','klp','time'),u_pfw_pf)
da_scales['u_psfw_psf'] =(('height','klp','time'),u_psfw_psf)
da_scales['v_pfw_pf']   =(('height','klp','time'),v_pfw_pf)
da_scales['v_psfw_psf'] =(('height','klp','time'),v_psfw_psf)
da_scales['uw_pf']      =(('height','klp','time'),uw_pf)
da_scales['uw_psf']     =(('height','klp','time'),uw_psf)
da_scales['vw_pf']      =(('height','klp','time'),vw_pf)
da_scales['vw_psf']     =(('height','klp','time'),vw_psf)

da_scales['u_av']       =(('height','time'),u_av)
da_scales['v_av']       =(('height','time'),v_av)
da_scales['w_av']       =(('height','time'),w_av)

da_scales.to_netcdf(save_dir+'scale_sep_allExp.nc')
#%% PLOT togeterh the heights
xsize = 150000

f_scales = np.zeros(len(klps))
for k in range(len(klps)):   
    if klps[k] > 0:
        f_scales[k] = xsize/(klps[k]*2)  # m
    elif klps[k] == 0:
        f_scales[k] = xsize
        
it = 37
plt.figure()
for ih in range(len(heights)):
    # plt.plot(f_scales/1000,da_scales.u_psfw_psf.isel(height=ih),\
    #           lw=0.5,alpha=0.5,c='grey')
    plt.plot(f_scales/1000,da_scales.u_psfw_psf.isel(height=ih,time=it),\
              lw=2,ls='--',label=str(np.round(heights_exact[ih]))+'m')
    plt.plot(f_scales/1000,da_scales.u_psfw_psf.isel(height=ih).mean('time'),\
             lw=3,label='mean '+str(np.round(heights_exact[ih]))+'m')
    for day in ['02','03','04','05']:
        plt.plot(f_scales/1000,da_scales.u_psfw_psf.isel(height=ih).sel(time='2020-02-'+day).mean('time'),\
              lw=2,label='Feb-'+day)
    # plt.plot(f_scales/1000,np.quantile(u_psfw_psf[ih,:,:],0.25,axis=1),\
    #          lw=1,c='b',label='Q1 '+str(np.round(heights_exact[ih]))+'m')      
    # plt.plot(f_scales/1000,np.quantile(u_psfw_psf[ih,:,:],0.75,axis=1),\
    #          lw=1,c='b',label='Q3 '+str(np.round(heights_exact[ih]))+'m')   

# plt.plot(f_scales/1000,u_pfw_pf_avtime[:,it],label='Up-filter',c='b',ls='--')
plt.xscale('log')
plt.axhline(0,c='k',lw=0.5)
plt.axvline(2.5,c='k',lw=0.5)
plt.axvline(150,c='k',lw=0.5)
# plt.axvline(150/1512,c='k',lw=0.5)
plt.legend()
plt.ylabel(r'Zonal momentum flux [$m^2/s^2$]')
plt.xlabel(r'Resolution [$km$]')
# plt.title('carried at the sub-filter scale',fontsize=24)
plt.suptitle('Cumulative flux at '+str(time[it])[5:16],fontsize=20)
# plt.savefig(save_fig_dir+'cumulat_spectra_'+str(time[it])+'.pdf', bbox_inches="tight")    
    
    
    
    