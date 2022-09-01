#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 13:13:10 2022

@author: acmsavazzi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
import os
from glob import glob
import gc
import sys
sys.path.insert(1, '/Users/acmsavazzi/Documents/WORK/PhD_Year2/Coding/Scale_separation/')
from functions import *
from dataloader import DataLoaderDALES
import argparse
import xarray as xr

def logic(index,first_line=4):
    if ((index-3)%levels+3 == 0) or ((index-2)%levels+3 == 0) or (index<first_line):
       return True
    return False
#%%
##### NOTATIONS
# _av = domain average 
# _p  = domain perturbation (prime)
# sf  = sub filter scale 
# f   = filter scale
# t   = total grid laevel 
# m   = middle of the grid 

heights=[1500,]
levels =151

## running on staffumbrella
# lp = 
# save_dir   = ''
## running on Local
base_dir   = '/Users/acmsavazzi/Documents/WORK/PhD_Year1/DATA/DALES/DALES_atECMWF/outputs/20200202_12_clim'
# fig_dir = 
save_dir = '/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/DALES/'

expnr = ['001','002','003','004','005','006','007','008','009','010',\
                '011','012','013','014','015','016','017']
srt_time   = np.datetime64('2020-02-02')
############
## read klps once for all
casenr = '001'
ih = 0
klps    = np.load(base_dir+'/Exp_'+casenr+'/scale_klps_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
############

prof_files      = []
tmser_files     = []
moments_files   = []
print("Finding output files.")  
for path,subdir,files in os.walk(base_dir):
    if path[-3:] in expnr: 
        for file in glob(os.path.join(path, 'profiles*.nc')):
            prof_files.append(file)
        for file in glob(os.path.join(path, 'tmser*.nc')):
            tmser_files.append(file)
        for file in glob(os.path.join(path, 'moments*')):
            moments_files.append(file)
            
####     profiles.nc    ####    
print("Reading DALES profiles.")      
profiles = xr.open_mfdataset(prof_files, combine='by_coords')
profiles['time'] = srt_time + profiles.time.astype("timedelta64[s]")
#remove last time step because it is a midnight of the day after
profiles = profiles.isel(time=slice(0,-1))
# interpolate half level to full level
profiles = profiles.interp(zm=profiles.zt)
profiles = profiles.rename({'zt':'z'})
####     tmser.nc   ####
print("Reading DALES time series.") 
tmser = xr.open_mfdataset(tmser_files, combine='by_coords')
tmser['time'] = srt_time + tmser.time.astype("timedelta64[s]")
####     moments.001    ####
print("Reading DALES moments.") 
colnames = ['lev','z','pres','thl2','thv2','th2','qt2','u2','v2','hght','w2','skew','sfs-tke']
moments  = []
for file in np.sort(moments_files):
    temp    = pd.read_csv(file,\
            skiprows=lambda x: logic(x),comment='#',\
            delimiter = " ",names=colnames,index_col=False,skipinitialspace=True)
    moments.append(temp)
moments = pd.concat(moments, axis=0, ignore_index=True)
moments['time'] = (moments.index.values//(levels-1))*(900)+900
moments.set_index(['time', 'z'], inplace=True)
moments = moments.to_xarray()
moments['time'] = srt_time + moments.time.astype("timedelta64[s]")



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

for casenr in expnr:
# for casenr in ['001','002','003','004','005','006','007','008']:
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
        
## normalize fluxes
# da_scales.max('klp') - da_scales.min('klp')
da_scales_norm = abs(da_scales)/abs(da_scales.sel(klp=min(klps)))

# da_scales_norm = abs(da_scales)/(da_scales.max('klp') - da_scales.min('klp'))


h   =  tmser.zi                 # boundary layer height 
hc  =  (tmser.zc_max-tmser.zb)  # cloud layer depth
f_scales_norm = f_scales[:,None] / (h+hc).sel(time=da_scales.time).values[None,:]



da_scales['f_scales_norm'] = (('klp','time'),f_scales_norm)
# da_scales_norm = (da_scales)/da_scales.max('klp') - da_scales.min('klp')

it = 24*3 + 14
plt.figure()
for ih in range(len(heights)):
    plt.plot(da_scales['f_scales_norm'].sel(time='2020-02-05'),da_scales_norm.u_psfw_psf.isel(height=ih).sel(time='2020-02-05'),\
              lw=0.5,alpha=0.5,c='grey')
    # plt.plot(f_scales_norm,(da_scales/da_scales.sel(klp=max(klps))).u_pfw_pf.isel(height=ih),\
    #           lw=0.5,alpha=0.5,c='green')
    # plt.plot(f_scales_norm[:,it],(da_scales/(da_scales.max('klp') - da_scales.min('klp'))).u_pfw_pf.isel(height=ih,time=it),\
    #           lw=1,alpha=0.5,ls='--',c='r')
    plt.plot(f_scales_norm[:,it],da_scales_norm.u_psfw_psf.isel(height=ih,time=it),\
              lw=1,c='b',ls='--',label=str(np.round(heights_exact[ih]))+'m')
    for day in ['02','03','04','05','06','07','08','09']:
        plt.plot(da_scales['f_scales_norm'].sel(time='2020-02-'+day).mean('time'),da_scales_norm.u_psfw_psf.isel(height=ih).sel(time='2020-02-'+day).mean('time'),\
              lw=2,label='Feb-'+day)
plt.ylim([0,4])
plt.xlim([0.01,40])
plt.xscale('log')
plt.axvline(0.5,c='k',lw=0.5)
plt.axhline(0,c='k',lw=0.5)

    
#%%
## normalize fluxes
# da_scales.max('klp') - da_scales.min('klp')
da_scales_norm = abs(da_scales)/abs(da_scales.sel(klp=min(klps)))

# da_scales_norm = abs(da_scales)/(abs(da_scales).max('klp') - abs(da_scales).min('klp'))

###
it = 24*3 + 14
plt.figure()
for ih in range(len(heights)):
    plt.plot(f_scales/1000,da_scales_norm.u_psfw_psf.isel(height=ih).sel(time='2020-02-05'),\
              lw=0.5,alpha=0.5,c='grey')
    plt.plot(f_scales/1000,da_scales_norm.u_psfw_psf.isel(height=ih,time=it),\
              lw=1,c='b',ls='--',label=str(np.round(heights_exact[ih]))+'m')
    # plt.plot(f_scales/1000,da_scales_norm.u_pfw_pf.isel(height=ih,time=it),\
    #           lw=1,c='r',ls='--',label=str(np.round(heights_exact[ih]))+'m')
    # plt.plot(f_scales/1000,da_scales.u_psfw_psf.isel(height=ih).mean('time')\
    #          /abs(da_scales.u_psfw_psf.isel(height=ih).mean('time')[-1]),\
    #           lw=3,ls='--',c='k',label='mean '+str(np.round(heights_exact[ih]))+'m')
    for day in ['02','03','04','05','06','07','08','09']:
# # for day in ['02','03','04','05']:
        plt.plot(f_scales/1000,da_scales_norm.u_psfw_psf.isel(height=ih).sel(time='2020-02-'+day).mean('time'),\
              lw=2,label='Feb-'+day)
            
        # plt.plot(f_scales/1000,da_scales.u_psfw_psf.isel(height=ih).sel(time='2020-02-'+day).mean('time')\
        #           /abs(da_scales.u_psfw_psf.isel(height=ih).sel(time='2020-02-'+day).mean('time')[-1]),\
        #       lw=2,label='Feb-'+day)
            
            
    # plt.plot(f_scales/1000,np.quantile(u_psfw_psf[ih,:,:],0.25,axis=1),\
    #          lw=1,c='b',label='Q1 '+str(np.round(heights_exact[ih]))+'m')      
    # plt.plot(f_scales/1000,np.quantile(u_psfw_psf[ih,:,:],0.75,axis=1),\
    #          lw=1,c='b',label='Q3 '+str(np.round(heights_exact[ih]))+'m')   

# plt.plot(f_scales/1000,u_pfw_pf_avtime[:,it],label='Up-filter',c='b',ls='--')
plt.xscale('log')
plt.ylim([0,+4])
plt.axhline(0,c='k',lw=0.5)
plt.axvline(2.5,c='k',lw=0.5)
plt.axvline(150,c='k',lw=0.5)
plt.axvline(150/1512,c='k',lw=0.5)
# plt.legend()
plt.ylabel(r'Zonal momentum flux [$m^2/s^2$]')
plt.xlabel(r'Resolution [$km$]')
# plt.title('carried at the sub-filter scale',fontsize=24)
plt.suptitle('Cumulative flux at '+str(time[it])[5:16],fontsize=20)
# plt.savefig(save_fig_dir+'cumulat_spectra_'+str(time[it])+'.pdf', bbox_inches="tight")    
    
    
#%% group on variance
plt.figure(figsize=(19,5))
moments.u2.sel(z=slice(0,200)).mean('z').plot(c='b')
(moments.u2.where(moments.u2.sel(z=slice(0,200)).mean('z')<0.8)).sel(z=slice(0,200)).mean('z').plot(c='orange')
(moments.u2.where(moments.u2.sel(z=slice(0,200)).mean('z')>4)).sel(z=slice(0,200)).mean('z').plot(c='green')


time_g1 = moments.where(moments.u2.sel(z=slice(0,200)).mean('z')<0.8,drop=True).time
time_g3 = moments.where(moments.u2.sel(z=slice(0,200)).mean('z')>4,drop=True).time
time_g2 = moments.where(np.logical_not(moments.time.isin(xr.concat((time_g1,time_g3),'time'))),drop=True).time



time_g1 = time_g1.where(time_g1.isin(da_scales.time),drop=True)
time_g2 = time_g2.where(time_g2.isin(da_scales.time),drop=True)
time_g3 = time_g3.where(time_g3.isin(da_scales.time),drop=True)


#%%
plt.figure()
for ih in range(len(heights)):
    # plt.plot(f_scales/1000,da_scales.u_psfw_psf.isel(height=ih).sel(time=time_g1),\
    #           lw=0.5,alpha=0.5,c='orange')
    # plt.plot(f_scales/1000,da_scales.u_psfw_psf.isel(height=ih).sel(time=time_g2),\
    #           lw=0.5,alpha=0.5,c='b')
    # plt.plot(f_scales/1000,da_scales.u_psfw_psf.isel(height=ih).sel(time=time_g3),\
    #           lw=0.5,alpha=0.5,c='green')
        
    plt.plot(f_scales/1000,da_scales.u_psfw_psf.isel(height=ih).sel(time=time_g1).mean('time'),\
              lw=2.5,c='orange')
    plt.plot(f_scales/1000,da_scales.u_psfw_psf.isel(height=ih).sel(time=time_g2).mean('time'),\
              lw=2.5,c='b')
    plt.plot(f_scales/1000,da_scales.u_psfw_psf.isel(height=ih).sel(time=time_g3).mean('time'),\
              lw=2.5,c='green')
# plt.ylim([0,+4])
plt.xscale('log')
plt.axhline(0,c='k',lw=0.5)
plt.axvline(2.5,c='k',lw=0.5)







