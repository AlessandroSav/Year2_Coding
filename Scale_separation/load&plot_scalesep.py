#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 13:13:10 2022

@author: acmsavazzi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import netCDF4 as nc
import os
from glob import glob
from sklearn.cluster import KMeans
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

heights=[200,650,1500,2600]
levels =151

## running on staffumbrella
# save_dir   = ''
## running on Local
base_dir   = '/Users/acmsavazzi/Documents/WORK/PhD_Year1/DATA/DALES/DALES_atECMWF/outputs/20200202_12_clim'
base_dir_scale   = '/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/DALES'

# fig_dir = 
save_dir = '/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/DALES/'

expnr = ['001','002','003','004','005','006','007','008','009','010',\
                '011','012','013','014','015','016','017']

srt_time   = np.datetime64('2020-02-02')
end_time   = np.datetime64('2020-02-11')
############
## read klps once for all
casenr = '001'
ih = 0
klps    = np.load(base_dir_scale+'/Exp_'+casenr+'/scale_klps_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
## read height once for all
zt    = np.load(base_dir_scale+'/Exp_'+casenr+'/scale_zt_prof_'+casenr+'.npy', allow_pickle=True) 
klp_prof    = np.load(base_dir_scale+'/Exp_'+casenr+'/scale_klps_prof_'+casenr+'.npy', allow_pickle=True)  

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
profiles['rain'] = profiles.rainrate/(24*28.94)
profiles['rain'].attrs["units"] = "mm/hour"
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
## organisation metric
df_org = pd.DataFrame()
## profiles
u_prof = np.empty([0,len(zt)])
v_prof = np.empty([0,len(zt)])
w_prof = np.empty([0,len(zt)])

u_pf_prof = np.empty([len(klp_prof),0,len(zt)])
v_pf_prof = np.empty([len(klp_prof),0,len(zt)])
w_pf_prof = np.empty([len(klp_prof),0,len(zt)])

u_pfw_pf_prof   = np.empty([len(klp_prof),0,len(zt)])
v_pfw_pf_prof   = np.empty([len(klp_prof),0,len(zt)])
u_psfw_psf_prof = np.empty([len(klp_prof),0,len(zt)])
v_psfw_psf_prof = np.empty([len(klp_prof),0,len(zt)])

uw_pf_prof  = np.empty([len(klp_prof),0,len(zt)])
vw_pf_prof  = np.empty([len(klp_prof),0,len(zt)])
uw_psf_prof = np.empty([len(klp_prof),0,len(zt)])
vw_psf_prof = np.empty([len(klp_prof),0,len(zt)])


## spectra
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
    lp   = base_dir_scale+'/Exp_'+casenr
    
    ## read organisation metrics
    df_org_temp = pd.read_hdf(lp+'/df_metrics.h5')
    ##################
    ############
    if (int(casenr) % 2) == 0:
        start_d = int(casenr)//2 +1
    else:
        start_d = int(casenr)//2 +2
    start_h = 0
    ###### Exp_001 and Exp_002 have wrong times
    if casenr == '001' or casenr=='002':
        df_org_temp.index = df_org_temp.index + 34385400  +1800
    df_org_temp.index = np.array(df_org_temp.index,dtype='timedelta64[s]') + \
        (np.datetime64('2020-02-'+str(start_d).zfill(2)+'T'+str(start_h).zfill(2)+':00'))
    ############
    ##################
    
    ih = 0
    ### These variables should be the same for all heights, so only open one
    time_temp    = np.load(lp+'/scale_time_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)     
    # klps    = np.load(lp+'/scale_klps_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
    ### initialize based on size
    heights_exact = np.zeros((len(heights)))
    
    ## load partitioned profiles
    u_prof_temp = np.load(lp+'/scale_u_prof_'+casenr+'.npy', allow_pickle=True)
    v_prof_temp = np.load(lp+'/scale_v_prof_'+casenr+'.npy', allow_pickle=True)
    w_prof_temp = np.load(lp+'/scale_w_prof_'+casenr+'.npy', allow_pickle=True)
    
    u_pf_prof_temp = np.load(lp+'/scale_u_pf_prof_'+casenr+'.npy', allow_pickle=True)
    v_pf_prof_temp = np.load(lp+'/scale_v_pf_prof_'+casenr+'.npy', allow_pickle=True)
    w_pf_prof_temp = np.load(lp+'/scale_w_pf_prof_'+casenr+'.npy', allow_pickle=True)
    
    u_pfw_pf_prof_temp = np.load(lp+'/scale_u_pfw_pf_prof_'+casenr+'.npy', allow_pickle=True)
    v_pfw_pf_prof_temp = np.load(lp+'/scale_v_pfw_pf_prof_'+casenr+'.npy', allow_pickle=True)
    u_psfw_psf_prof_temp = np.load(lp+'/scale_u_psfw_psf_prof_'+casenr+'.npy', allow_pickle=True)
    v_psfw_psf_prof_temp = np.load(lp+'/scale_v_psfw_psf_prof_'+casenr+'.npy', allow_pickle=True)

    uw_pf_prof_temp = np.load(lp+'/scale_uw_pf_prof_'+casenr+'.npy', allow_pickle=True)
    vw_pf_prof_temp = np.load(lp+'/scale_vw_pf_prof_'+casenr+'.npy', allow_pickle=True)
    uw_psf_prof_temp = np.load(lp+'/scale_uw_psf_prof_'+casenr+'.npy', allow_pickle=True)
    vw_psf_prof_temp = np.load(lp+'/scale_vw_psf_prof_'+casenr+'.npy', allow_pickle=True)
    
    
    ## load spectra at single hieght
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
        qtw_pf_temp[ih,:,:]   = np.load(lp+'/scale_qtw_pf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
        qtw_psf_temp[ih,:,:] = np.load(lp+'/scale_qtw_psf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)        
        
    ## Append all times together
    time = np.append(time,time_temp)
    ## org metrics
    df_org  = df_org.append(df_org_temp)
    ## profiles
    u_prof  =   np.append(u_prof,u_prof_temp,0)
    v_prof  =   np.append(v_prof,v_prof_temp,0)
    w_prof  =   np.append(w_prof,w_prof_temp,0)
    vqt_prof  =   np.append(qt_prof,qt_prof_temp,0)

    
    u_pf_prof   =   np.append(u_pf_prof,u_pf_prof_temp,1)
    v_pf_prof   =   np.append(v_pf_prof,v_pf_prof_temp,1)
    w_pf_prof   =   np.append(w_pf_prof,w_pf_prof_temp,1)
    qt_pf_prof   =   np.append(qt_pf_prof,qt_pf_prof_temp,1)

    
    u_pfw_pf_prof   = np.append(u_pfw_pf_prof,u_pfw_pf_prof_temp,1)
    v_pfw_pf_prof   = np.append(v_pfw_pf_prof,v_pfw_pf_prof_temp,1)
    qt_pfw_pf_prof   = np.append(qt_pfw_pf_prof,qt_pfw_pf_prof_temp,1)
    u_psfw_psf_prof = np.append(u_psfw_psf_prof,u_psfw_psf_prof_temp,1)
    v_psfw_psf_prof = np.append(v_psfw_psf_prof,v_psfw_psf_prof_temp,1)
    qt_psfw_psf_prof = np.append(qt_psfw_psf_prof,qt_psfw_psf_prof_temp,1)

    
    uw_pf_prof  = np.append(uw_pf_prof,uw_pf_prof_temp,1)
    vw_pf_prof  = np.append(vw_pf_prof,vw_pf_prof_temp,1)
    qtw_pf_prof  = np.append(qtw_pf_prof,qtw_pf_prof_temp,1)
    uw_psf_prof = np.append(uw_psf_prof,uw_psf_prof_temp,1)
    vw_psf_prof = np.append(vw_psf_prof,vw_psf_prof_temp,1)
    qtw_psf_prof = np.append(qtw_psf_prof,qtw_psf_prof_temp,1)


    ## spectra
    u_av=np.append(u_av,u_av_temp,1)
    v_av=np.append(v_av,v_av_temp,1)
    w_av=np.append(w_av,w_av_temp,1)
    qt_av=np.append(qt_av,qt_av_temp,1)
    #
    u_pf=np.append(u_pf,u_pf_temp,2)
    v_pf=np.append(v_pf,v_pf_temp,2)
    w_pf=np.append(w_pf,w_pf_temp,2)
    qt_pf=np.append(qt_pf,qt_pf_temp,2)
    u_pfw_pf    = np.append(u_pfw_pf,   u_pfw_pf_temp,  2)
    u_psfw_psf  = np.append(u_psfw_psf, u_psfw_psf_temp,2)
    v_pfw_pf    = np.append(v_pfw_pf,   v_pfw_pf_temp,  2)
    v_psfw_psf  = np.append(v_psfw_psf, v_psfw_psf_temp,2)
    qt_pfw_pf    = np.append(qt_pfw_pf,   qt_pfw_pf_temp,  2)
    qt_psfw_psf  = np.append(qt_psfw_psf, qt_psfw_psf_temp,2)
    uw_pf       = np.append(uw_pf,      uw_pf_temp,     2)
    uw_psf      = np.append(uw_psf,     uw_psf_temp,    2)
    vw_pf       = np.append(vw_pf,      vw_pf_temp,     2)
    vw_psf      = np.append(vw_psf,     vw_psf_temp,    2)
    qtw_pf       = np.append(qtw_pf,      qtw_pf_temp,     2)
    qtw_psf      = np.append(qtw_psf,     qtw_psf_temp,    2)
    
    
#%%   Convert to Xarray and Save 
## organisation metrics
df_org = df_org.apply(pd.to_numeric)
df_org = df_org.to_xarray().rename({'index':'time'})
# df_org.to_netcdf(save_dir+'df_org_allExp.nc')
## scales 
da_scales = xr.Dataset(
    {'u_pf':(
        ('height','klp','time'),
        u_pf,
        )},
        coords={'height':heights,'klp':klps,'time':time},)

da_scales['v_pf']       =(('height','klp','time'),v_pf)
da_scales['w_pf']       =(('height','klp','time'),w_pf)
da_scales['qt_pf']       =(('height','klp','time'),qt_pf)
da_scales['u_pfw_pf']   =(('height','klp','time'),u_pfw_pf)
da_scales['u_psfw_psf'] =(('height','klp','time'),u_psfw_psf)
da_scales['v_pfw_pf']   =(('height','klp','time'),v_pfw_pf)
da_scales['v_psfw_psf'] =(('height','klp','time'),v_psfw_psf)
da_scales['qt_pfw_pf']   =(('height','klp','time'),qt_pfw_pf)
da_scales['qt_psfw_psf'] =(('height','klp','time'),qt_psfw_psf)
da_scales['uw_pf']      =(('height','klp','time'),uw_pf)
da_scales['uw_psf']     =(('height','klp','time'),uw_psf)
da_scales['vw_pf']      =(('height','klp','time'),vw_pf)
da_scales['vw_psf']     =(('height','klp','time'),vw_psf)
da_scales['qtw_pf']      =(('height','klp','time'),qtw_pf)
da_scales['qtw_psf']     =(('height','klp','time'),qtw_psf)

da_scales['u_av']       =(('height','time'),u_av)
da_scales['v_av']       =(('height','time'),v_av)
da_scales['w_av']       =(('height','time'),w_av)
da_scales['qt_av']       =(('height','time'),qt_av)

da_scales.to_netcdf(save_dir+'scale_sep_allExp.nc')

## scale profiles
da_scales_prof = xr.Dataset(
    {'u_pf_prof':(
        ('klp','time','height'),
        u_pf_prof,
        )},
        coords={'klp':klp_prof,'time':time,'height':zt},)

da_scales_prof['v_pf']       =(('klp','time','height'),v_pf_prof)
da_scales_prof['w_pf']       =(('klp','time','height'),w_pf_prof)
da_scales_prof['u_pfw_pf']   =(('klp','time','height'),u_pfw_pf_prof)
da_scales_prof['u_psfw_psf'] =(('klp','time','height'),u_psfw_psf_prof)
da_scales_prof['v_pfw_pf']   =(('klp','time','height'),v_pfw_pf_prof)
da_scales_prof['v_psfw_psf'] =(('klp','time','height'),v_psfw_psf_prof)
da_scales_prof['uw_pf']      =(('klp','time','height'),uw_pf_prof)
da_scales_prof['uw_psf']     =(('klp','time','height'),uw_psf_prof)
da_scales_prof['vw_pf']      =(('klp','time','height'),vw_pf_prof)
da_scales_prof['vw_psf']     =(('klp','time','height'),vw_psf_prof)

da_scales_prof['u_av']       =(('time','height'),u_prof)
da_scales_prof['v_av']       =(('time','height'),v_prof)
da_scales_prof['w_av']       =(('time','height'),w_prof)

## fill the gap caused when creating files
da_scales_prof = da_scales_prof.where(da_scales_prof!=0)
da_scales_prof = da_scales_prof.bfill('height')


# da_scales_prof.to_netcdf(save_dir+'scale_sep_prof_allExp.nc')
#%% PLOT togeterh the heights


### From KLPS to resolution/size/scale of the filter
xsize = 150000
f_scales = np.zeros(len(klps))
for k in range(len(klps)):   
    if klps[k] > 0:
        f_scales[k] = xsize/(klps[k]*2)  # m
    elif klps[k] == 0:
        f_scales[k] = xsize

## HONNERT normalization of the filter scale
h   =  tmser.zi                 # boundary layer height 
hc  =  (tmser.zc_max-tmser.zb)  # cloud layer depth
hc  =  0
f_scales_norm = f_scales[:,None] / (h+hc).sel(time=da_scales.time).values[None,:]
da_scales['f_scales_norm'] = (('klp','time'),f_scales_norm)
     
#################
## normalize fluxes ##
da_scales_norm = (da_scales)/(da_scales.sel(klp=min(klps)))
#################

#%% FIGURE 1
cmap = matplotlib.cm.get_cmap('tab10')
######## All hours non normalized 
###
var = 'qt'
plt.figure(figsize=(10,7))
rgba = 1/8
iteration=0
for ih in range(len(heights)):
    for day in ['02','03','04','05','06','07','08','09']:
        iteration += 1
        plt.plot(f_scales/1000,\
                  da_scales[var+'_psfw_psf'].isel(height=ih).sel(time='2020-02-'+day),\
              lw=0.5,alpha=0.5,c=cmap(rgba*iteration))
        plt.plot(f_scales/1000,\
                  da_scales[var+'_psfw_psf'].isel(height=ih).sel(time='2020-02-'+day)\
                      .median('time'),\
              lw=3,label='Feb-'+day,c=cmap(rgba*iteration))
        
            
plt.xscale('log')
plt.axvline(2.5,c='k',lw=0.5)
# plt.axhline(0,c='k',lw=0.5)
# plt.ylim([-0.045,0.045])
plt.legend(frameon=False,fontsize=12)
plt.ylabel(r'Flux [$m^2/s^2$]',fontsize=18)
plt.xlabel(r'Filter size [km]',fontsize=18)
plt.suptitle('Sub-filter '+var+' flux \n Mean per day - at '+str(heights[ih])+' m',fontsize=20)

#%% FIGURE 2
plt.figure(figsize=(10,7))
rgba = 1/8
iteration=0
for ih in range(len(heights)):
    for day in ['02','03','04','05','06','07','08','09']:
        iteration += 1
        # plt.plot(da_scales['f_scales_norm'].sel(time='2020-02-'+day),\
        #           da_scales_norm.u_psfw_psf.isel(height=ih).sel(time='2020-02-'+day),\
        #       lw=0.5,alpha=0.3,c=cmap(rgba*iteration))
        plt.scatter(da_scales['f_scales_norm'].sel(time='2020-02-'+day),\
                  da_scales_norm.u_psfw_psf.isel(height=ih).sel(time='2020-02-'+day),\
              marker='.',alpha=0.3,c=cmap(rgba*iteration))
        plt.plot(da_scales['f_scales_norm'].sel(time='2020-02-'+day).mean('time'),\
                  da_scales_norm.u_psfw_psf.isel(height=ih).sel(time='2020-02-'+day).mean('time'),\
              lw=3,c=cmap(rgba*iteration))
            
plt.xscale('log')
# plt.axvline(2.5,c='k',lw=0.5)
plt.ylim([-0.5,1.5])
plt.axhline(0,c='k',lw=0.5)
plt.legend(frameon=False,fontsize=12)
plt.ylabel(r'Fractional flux',fontsize=18)
plt.xlabel(r'$\Delta$x / (h+hc)',fontsize=18)
plt.suptitle('Sub-filter zonal momentum flux \n Mean per day - at '+str(heights[ih])+' m',fontsize=20)

#%% CROSS TERMS CONTIBUTION 
plt.figure(figsize=(10,7))
rgba = 1/8
iteration=0
for ih in range(len(heights)):
    for day in ['02','03','04','05','06','07','08','09']:
        iteration += 1
        
        plt.plot(f_scales/1000,\
                 ((da_scales.u_pfw_pf + da_scales.u_psfw_psf) / da_scales.sel(klp=min(klps))['u_psfw_psf']).isel(height=ih).sel(time='2020-02-'+day),\
              lw=0.5,alpha=0.5,c=cmap(rgba*iteration))
            
plt.xscale('log')
plt.axvline(2.5,c='k',lw=0.5)

plt.legend(frameon=False,fontsize=12)
plt.ylabel(r'Flux [$m^2/s^2$]',fontsize=18)
plt.xlabel(r'Filter size [km]',fontsize=18)
plt.suptitle('Cross-filter scales zonal momentum flux \n at '+str(heights[ih])+' m',fontsize=20)

#%%
#################
## normalize fluxes ##
####
# da_scales.max('klp') - da_scales.min('klp')
da_scales_norm = (da_scales)/(da_scales.sel(klp=min(klps)))
# da_scales_norm = (da_scales)/(da_scales.sel(klp=30,method='nearest'))

# da_scales_norm = abs(da_scales)/(da_scales.max('klp') - da_scales.min('klp'))
# da_scales_norm = (da_scales)/da_scales.max('klp') - da_scales.min('klp')
####
################
it = 24*3 + 14  #time to plot as dashed 

######## plot non dimentional X axis (Honnert)
###
plt.figure(figsize=(10,7))
for ih in range(len(heights)):
    # plt.plot(da_scales['f_scales_norm'].sel(time='2020-02-05'),da_scales_norm.u_psfw_psf.isel(height=ih).sel(time='2020-02-05'),\
    #           lw=0.5,alpha=0.5,c='grey')
    # plt.plot(f_scales_norm,(da_scales/da_scales.sel(klp=max(klps))).u_pfw_pf.isel(height=ih),\
    #           lw=0.5,alpha=0.5,c='green')
    # plt.plot(f_scales_norm[:,it],(da_scales/(da_scales.max('klp') - da_scales.min('klp'))).u_pfw_pf.isel(height=ih,time=it),\
    #           lw=1,alpha=0.5,ls='--',c='r')
    # plt.plot(f_scales_norm[:,it],da_scales_norm.u_psfw_psf.isel(height=ih,time=it),\
    #           lw=1,c='b',ls='--',label=str(np.round(heights_exact[ih]))+'m')
    for day in ['02','03','04','05','06','07','08','09']:
        plt.scatter(da_scales['f_scales_norm'].sel(time='2020-02-'+day),\
                  da_scales_norm.u_psfw_psf.isel(height=ih).sel(time='2020-02-'+day)\
                    ,marker='.',alpha=0.3)
        # plt.plot(da_scales['f_scales_norm'].sel(time='2020-02-'+day).mean('time'),\
        #           da_scales_norm.u_psfw_psf.isel(height=ih).sel(time='2020-02-'+day)\
        #               .mean('time'),\
        #       lw=2,label='Feb-'+day)
            

            
plt.ylim([-3,3.5])
plt.xlim([0.02,90])
plt.xscale('log')
plt.axvline(0.5,c='k',lw=0.5)
plt.axhline(0,c='k',lw=0.5)
plt.legend(frameon=False)
# plt.ylabel(r'Zonal momentum flux [$m^2/s^2$]')
plt.ylabel(r'Fractional flux',fontsize=18)
plt.xlabel(r'$\Delta$x / (h+hc)',fontsize=18)
# plt.title('carried at the sub-filter scale',fontsize=24)
# plt.suptitle('Sub-filter flux at '+str(time[it])[5:16],fontsize=20)
plt.suptitle('Sub-filter zonal momentum flux \n Mean per day - at '+str(heights[ih])+' m',fontsize=20)

########
    
#%%
######## plot resolution [km] on the X axis
###
plt.figure(figsize=(10,7))
for ih in range(len(heights)):
    # plt.plot(f_scales/1000,da_scales_norm.u_psfw_psf.isel(height=ih).sel(time='2020-02-05'),\
    #           lw=0.5,alpha=0.5,c='grey')
    # plt.plot(f_scales/1000,da_scales_norm.u_psfw_psf.isel(height=ih,time=it),\
    #           lw=1,c='b',ls='--',label=str(np.round(heights_exact[ih]))+'m')
    # plt.plot(f_scales/1000,da_scales_norm.u_pfw_pf.isel(height=ih,time=it),\
    #           lw=1,c='r',ls='--',label=str(np.round(heights_exact[ih]))+'m')
    # plt.plot(f_scales/1000,da_scales.u_psfw_psf.isel(height=ih).mean('time')\
    #           /abs(da_scales.u_psfw_psf.isel(height=ih).mean('time')[-1]),\
    #           lw=3,ls='--',c='k',label='mean '+str(np.round(heights_exact[ih]))+'m')
    for day in ['02','03','04','05','06','07','08','09']:
        if day in ['02','05','07','09']:
            line=2
        else:line=1.5
    # for day in ['02','05','07','09']:
        plt.plot(f_scales/1000,da_scales_norm.u_psfw_psf.isel(height=ih).sel(time='2020-02-'+day).median('time'),\
              lw=line,label='Feb-'+day)
        plt.scatter(2.5,da_scales_norm.u_psfw_psf.isel(height=ih)\
                    .sel(klp=30,method='nearest').sel(time='2020-02-'+day).mean('time'),\
                        s=30*line)
            
        # plt.plot(f_scales/1000,da_scales.u_psfw_psf.isel(height=ih).sel(time='2020-02-'+day).mean('time')\
        #           /abs(da_scales.u_psfw_psf.isel(height=ih).sel(time='2020-02-'+day).mean('time')[-1]),\
        #       lw=2,label='Feb-'+day)
            
            
    # plt.plot(f_scales/1000,np.quantile(u_psfw_psf[ih,:,:],0.25,axis=1),\
    #          lw=1,c='b',label='Q1 '+str(np.round(heights_exact[ih]))+'m')      
    # plt.plot(f_scales/1000,np.quantile(u_psfw_psf[ih,:,:],0.75,axis=1),\
    #          lw=1,c='b',label='Q3 '+str(np.round(heights_exact[ih]))+'m')   

# plt.plot(f_scales/1000,u_pfw_pf_avtime[:,it],label='Up-filter',c='b',ls='--')
plt.xscale('log')
plt.ylim([-0.2,+1.2])
plt.axhline(0,c='k',lw=0.5)
plt.axvline(2.5,c='k',lw=0.5)
plt.axvline(150,c='k',lw=0.5)
plt.axvline(150/1512,c='k',lw=0.5)
plt.legend()
plt.ylabel(r'Zonal momentum flux [$m^2/s^2$]')
plt.ylabel(r'Fractional  flux')
plt.xlabel(r'Resolution [$km$]')
# plt.title('carried at the sub-filter scale',fontsize=24)
# plt.suptitle('Cumulative flux at '+str(time[it])[5:16],fontsize=20)
plt.suptitle('Sub-filter zonal momentum flux \n Median per day - at '+str(heights[ih])+' m',fontsize=20)
# plt.savefig(save_fig_dir+'cumulat_spectra_'+str(time[it])+'.pdf', bbox_inches="tight")    

# plt.savefig(save_fig_dir+'SF_Uflux_spectra_perDay_200.pdf', bbox_inches="tight")    
    

#%% Time series for one klp
klp = klps[(np.abs(klps - 30)).argmin()]


########################
###### This normalization is different !!!!
da_scales_norm = (da_scales)/(da_scales.u_psfw_psf + da_scales.u_pfw_pf)

# da_scales_norm = (da_scales)/(da_scales.sel(klp=min(klps)))
########################


plt.figure(figsize=(19,5))
for ih in range(len(heights)):
    plt.plot(da_scales_norm.time,da_scales_norm.u_psfw_psf.isel(height=ih).sel(klp=klp),\
              lw=2,c='green',label= 'SF <2.5km')
    plt.plot(da_scales_norm.time,da_scales_norm.u_pfw_pf.isel(height=ih).sel(klp=klp),\
              lw=2,c='k',label='UF >2.5km')
    plt.plot(da_scales_norm.time,(da_scales_norm.u_psfw_psf + da_scales_norm.u_pfw_pf).isel(height=ih).sel(klp=klp),\
              lw=1,c='r',label='total flux')
    # plt.plot(profiles.time,profiles.uwt.sel(z=heights[ih],method='nearest'),c='r',ls='--')
for ii in np.arange(srt_time, end_time):
    plt.axvline(x=ii,c='k')
plt.legend()
plt.ylim([-4,4])
plt.axhline(0,c='k',lw=0.5)
plt.title ('U momentum flux partitioning at '+str(heights[ih])+' m')
#%% group on variance
moments['ctop_var'] = moments.sel(z=tmser.sel(time=moments.time).zc_max - 400,method='nearest')['u2']

# plt.figure(figsize=(19,5))
# moments.u2.sel(z=slice(0,200)).mean('z').plot(c='b')
# (moments.u2.where(moments.u2.sel(z=slice(0,200)).mean('z')<0.8)).sel(z=slice(0,200)).mean('z').plot(c='orange')
# (moments.u2.where(moments.u2.sel(z=slice(0,200)).mean('z')>4)).sel(z=slice(0,200)).mean('z').plot(c='green')
# for ii in np.arange(srt_time, end_time):
#     plt.axvline(x=ii,c='k')

########
## grouping option 1: Based on surface variance
# time_g1 = moments.where(moments.u2.sel(z=slice(0,200)).mean('z')<0.8,drop=True).time
# time_g3 = moments.where(moments.u2.sel(z=slice(0,200)).mean('z')>4,drop=True).time
# time_g2 = moments.where(np.logical_not(moments.time.isin(xr.concat((time_g1,time_g3),'time'))),drop=True).time

########
## grouping option 2: Based on surface variance and cloud top variance
# time_g1 = moments.where(moments.u2.sel(z=slice(0,200)).mean('z')<0.8,drop=True).time
# time_g2 = moments.where((moments.u2.sel(z=slice(0,200)).mean('z')>=0.8) & (moments.ctop_var<0.6),drop=True).time
# time_g3 = moments.where(np.logical_not(moments.time.isin(xr.concat((time_g1,time_g2),'time'))),drop=True).time

########
## grouping option 3: Based on surface precipitation
time_g1 = profiles.where(profiles.rain.sel(z=slice(0,50)).mean('z').\
                         rolling(time=6, center=True).mean()<0.15,drop=True).time
time_g3 = profiles.where(profiles.rain.sel(z=slice(0,50)).mean('z').\
                         rolling(time=6, center=True).mean()>0.75,drop=True).time
time_g2 = profiles.where(np.logical_not(profiles.time.\
                        isin(xr.concat((time_g1,time_g3),'time'))),drop=True).time
########

##
time_g1 = time_g1.where(time_g1.isin(da_scales.time),drop=True)
time_g2 = time_g2.where(time_g2.isin(da_scales.time),drop=True)
time_g3 = time_g3.where(time_g3.isin(da_scales.time),drop=True)
##

# da_scales_norm = (da_scales)/(da_scales.sel(klp=min(klps)))
########
## grouping option 4: Based on K-mean clustering directly on the spectra
## CLUSTER on spectral shear 
# shear_cluster=KMeans(n_clusters=3,random_state=0,n_init=15,max_iter=10000,\
#                 tol=10**-7).fit(da_scales['u_psfw_psf'].sel(height=heights[0]).T)
# idx = np.argsort(shear_cluster.cluster_centers_.sum(axis=1))
# da_scales['group_kmean'] = (('time'), shear_cluster.labels_)

# time_g1 = moments.where(da_scales['group_kmean']==0,drop=True).time
# time_g2 = moments.where(da_scales['group_kmean']==1,drop=True).time
# time_g3 = moments.where(da_scales['group_kmean']==2,drop=True).time
# # time_g4 = moments.where(da_scales['group_kmean']==3,drop=True).time
# # time_g5 = moments.where(da_scales['group_kmean']==4,drop=True).time
#####



plt.figure(figsize=(19,5))
moments.u2.sel(z=slice(0,200)).mean('z').plot(c='k')
# moments.ctop_var.plot(c='r')
plt.scatter(moments.time.sel(time=time_g1),moments.u2.sel(z=slice(0,200),time=time_g1).mean('z'),c='orange',label='Group 1')
plt.scatter(moments.time.sel(time=time_g2),moments.u2.sel(z=slice(0,200),time=time_g2).mean('z'),c='b',label='Group 2')
plt.scatter(moments.time.sel(time=time_g3),moments.u2.sel(z=slice(0,200),time=time_g3).mean('z'),c='green',label='Group 3')

# plt.scatter(moments.time.sel(time=time_g4),moments.u2.sel(z=slice(0,200),time=time_g4).mean('z'))
# plt.scatter(moments.time.sel(time=time_g5),moments.u2.sel(z=slice(0,200),time=time_g5).mean('z'),c='k')
# moments.u2.sel(z=slice(0,200),time=time_g1).mean('z').plot(c='orange')
# moments.u2.sel(z=slice(0,200),time=time_g2).mean('z').plot(c='b')
# moments.u2.sel(z=slice(0,200),time=time_g3).mean('z').plot(c='green')
plt.legend()
for ii in np.arange(srt_time, end_time):
    plt.axvline(x=ii,c='k')
    
    
## Plot rain rate time series
plt.figure(figsize=(19,5))
profiles.rain.sel(z=slice(0,50)).mean('z').rolling(time=6, center=True).mean().plot(c='k')
# moments.ctop_var.plot(c='r')
plt.scatter(profiles.time.sel(time=time_g1),profiles.rain.sel(z=slice(0,50))\
            .mean('z').rolling(time=6, center=True).mean().sel(time=time_g1),c='orange',label='Group 1')
plt.scatter(profiles.time.sel(time=time_g2),profiles.rain.sel(z=slice(0,50))\
            .mean('z').rolling(time=6, center=True).mean().sel(time=time_g2),c='b',label='Group 2')
plt.scatter(profiles.time.sel(time=time_g3),profiles.rain.sel(z=slice(0,50))\
            .mean('z').rolling(time=6, center=True).mean().sel(time=time_g3),c='green',label='Group 3')

# plt.scatter(moments.time.sel(time=time_g4),moments.u2.sel(z=slice(0,200),time=time_g4).mean('z'))
# plt.scatter(moments.time.sel(time=time_g5),moments.u2.sel(z=slice(0,200),time=time_g5).mean('z'),c='k')
# moments.u2.sel(z=slice(0,200),time=time_g1).mean('z').plot(c='orange')
# moments.u2.sel(z=slice(0,200),time=time_g2).mean('z').plot(c='b')
# moments.u2.sel(z=slice(0,200),time=time_g3).mean('z').plot(c='green')
plt.legend()
plt.title('Surface rain rate')
for ii in np.arange(srt_time, end_time):
    plt.axvline(x=ii,c='k')
    


#%%
# plt.figure()
for ih in range(len(heights)):
    plt.figure()
    # plt.plot(f_scales/1000,da_scales.u_psfw_psf.isel(height=ih).sel(time=time_g1),\
    #           lw=0.5,alpha=0.5,c='orange')
    # plt.plot(f_scales/1000,da_scales.u_psfw_psf.isel(height=ih).sel(time=time_g2),\
    #           lw=0.5,alpha=0.5,c='b')
    # plt.plot(f_scales/1000,da_scales.u_psfw_psf.isel(height=ih).sel(time=time_g3),\
    #           lw=0.5,alpha=0.5,c='green')
        
    plt.plot(f_scales/1000,da_scales.u_psfw_psf.isel(height=ih).sel(time=time_g1).median('time'),\
              lw=2.5,c='orange',label='Group 1')
    plt.plot(f_scales/1000,da_scales.u_psfw_psf.isel(height=ih).sel(time=time_g2).median('time'),\
              lw=2.5,c='b',label='Group 2')
    plt.plot(f_scales/1000,da_scales.u_psfw_psf.isel(height=ih).sel(time=time_g3).median('time'),\
              lw=2.5,c='green',label='Group 3')
        
    # plt.plot(f_scales/1000,da_scales.u_psfw_psf.isel(height=ih).sel(time=time_g4).mean('time'),\
    #           lw=2.5)
    # plt.plot(f_scales/1000,da_scales.u_psfw_psf.isel(height=ih).sel(time=time_g5).mean('time'),\
    #           lw=2.5,c='k')
        
# plt.ylim([-0.02,+0.02])
    plt.xscale('log')
    plt.title(str(heights[ih]))
    plt.axhline(0,c='k',lw=0.5)
    plt.axvline(2.5,c='k',lw=0.5)
    plt.legend()

#%% PROFILES 
var='v'
plt.figure(figsize=(4,6))
## mean
da_scales_prof[var+"_pfw_pf"].mean('time').plot(y='height',ls='-',c='k',label='mean UF')
da_scales_prof[var+"_psfw_psf"].mean('time').plot(y='height',ls='-',c='green',label='mean SF')
da_scales_prof[var+"w_pf"].mean('time').plot(y='height',c='r',label='mean tot')


plt.fill_betweenx(da_scales_prof.height,\
                  da_scales_prof[var+"w_pf"].quantile(0.1,dim='time').sel(klp=30),\
                  da_scales_prof[var+"w_pf"].quantile(0.9,dim='time').sel(klp=30),\
                      color='r',alpha=0.2)
    
plt.fill_betweenx(da_scales_prof.height,\
                  da_scales_prof[var+"_pfw_pf"].quantile(0.1,dim='time').sel(klp=30),\
                  da_scales_prof[var+"_pfw_pf"].quantile(0.9,dim='time').sel(klp=30),\
                      color='k',alpha=0.2)
    
plt.fill_betweenx(da_scales_prof.height,\
                  da_scales_prof[var+"_psfw_psf"].quantile(0.1,dim='time').sel(klp=30),\
                  da_scales_prof[var+"_psfw_psf"].quantile(0.9,dim='time').sel(klp=30),\
                      color='g',alpha=0.2)
    
profiles[var+"wr"].mean('time').\
        plot(y='z',ls='--',c='r',label='Total from prof')   
    

plt.axvline(0,c='k',lw=0.5)
plt.axhline(200,c='k',ls='-',lw=0.5)
plt.axhline(650,c='k',ls='-',lw=0.5)
plt.axhline(1500,c='k',ls='-',lw=0.5)
plt.ylim([-10,4100])
plt.legend()
plt.xlabel(r''+var+' momentum flux [$m^2 / s^2$]')
plt.ylabel('Z [m]')
plt.title('Partitioning of momentum flux \n Filter scale = 2.5 km')

#%%
sel_time = ['2020-02-09T11','2020-02-09T17']
plt.figure(figsize=(4,6))
## mean
da_scales_prof["u_pfw_pf"].sel(time=slice(sel_time[0],sel_time[1])).mean('time').plot(y='height',ls='-',c='k',label='mean UF')
da_scales_prof["u_psfw_psf"].sel(time=slice(sel_time[0],sel_time[1])).mean('time').plot(y='height',ls='-',c='green',label='mean SF')
da_scales_prof["uw_pf"].sel(time=slice(sel_time[0],sel_time[1])).mean('time').plot(y='height',c='r',label='mean tot')


plt.fill_betweenx(da_scales_prof.height,\
                  da_scales_prof["uw_pf"].sel(time=slice(sel_time[0],sel_time[1])).quantile(0.1,dim='time').sel(klp=30),\
                  da_scales_prof["uw_pf"].sel(time=slice(sel_time[0],sel_time[1])).quantile(0.9,dim='time').sel(klp=30),\
                      color='r',alpha=0.2)
    
plt.fill_betweenx(da_scales_prof.height,\
                  da_scales_prof["u_pfw_pf"].sel(time=slice(sel_time[0],sel_time[1])).quantile(0.1,dim='time').sel(klp=30),\
                  da_scales_prof["u_pfw_pf"].sel(time=slice(sel_time[0],sel_time[1])).quantile(0.9,dim='time').sel(klp=30),\
                      color='k',alpha=0.2)
    
plt.fill_betweenx(da_scales_prof.height,\
                  da_scales_prof["u_psfw_psf"].sel(time=slice(sel_time[0],sel_time[1])).quantile(0.1,dim='time').sel(klp=30),\
                  da_scales_prof["u_psfw_psf"].sel(time=slice(sel_time[0],sel_time[1])).quantile(0.9,dim='time').sel(klp=30),\
                      color='g',alpha=0.2)

plt.axvline(0,c='k',lw=0.5)
plt.axhline(200,c='k',ls='-',lw=0.5)
plt.axhline(650,c='k',ls='-',lw=0.5)
plt.axhline(1500,c='k',ls='-',lw=0.5)
plt.legend()
plt.xlabel(r'Zonal momentum flux [$m^2 / s^2$]')
plt.ylabel('Z [m]')
plt.title('Partitioning of momentum flux \n Filter scale = 2.5 km')

#%%

groupnr = 0
for group in [time_g1,time_g2,time_g3]:
    groupnr +=1
    plt.figure(figsize=(4,6))
    da_scales_prof["u_pfw_pf"].sel(time=group).mean('time').plot(y='height',ls='--',c='k',label='up-filter')
    da_scales_prof["u_psfw_psf"].sel(time=group).mean('time').plot(y='height',ls=':',c='g',label='Sub-filter')
   
    # (da_scales_prof["u_psfw_psf"]+da_scales_prof["u_pfw_pf"]).sel(time=group).mean('time').plot(y='height',ls='-',c='r',label='sum')
    da_scales_prof["uw_pf"].sel(time=group).mean('time').plot(y='height',c='r',label='total flux')
    plt.axvline(0,c='k',lw=0.5)
    plt.xlim([-0.041,0.1])
    plt.axhline(200,c='k',ls='-',lw=0.5)
    plt.axhline(650,c='k',ls='-',lw=0.5)
    plt.axhline(1500,c='k',ls='-',lw=0.5)
    plt.legend()
    plt.title('Group '+str(groupnr)+'\n Filter scale = 2.5 km')
    






