#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 13:13:10 2022

@author: acmsavazzi
"""
import numpy as np
import pandas as pd
import netCDF4 as nc
import os
from glob import glob
import sys
sys.path.insert(1, '/Users/acmsavazzi/Documents/WORK/PhD_Year2/Coding/Scale_separation/')
from functions import *
from dataloader import DataLoaderDALES
import xarray as xr

#%%
##### NOTATIONS
# _av = domain average 
# _p  = domain perturbation (prime)
# sf  = sub filter scale 
# f   = filter scale
# t   = total grid laevel 
# m   = middle of the grid 

prof = True
spectra = False
org = False

#########
### if spectra == True 
# heights=[100,200,650,1500,2600]
heights=[100,'subCL']
save_ext = '100m_subCL'
#########

################################
## running on staffumbrella
# base_dir_scale
# save_dir   = ''
## running on Local
base_dir_scale  = '/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/DALES'
save_dir        = '/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/DALES/'
################################

expnr = ['001','002','003','004','005','006','007','008','009','010',\
                '011','012','013','014','015','016','017']
#%% Main
############
## read klps once for all
casenr = '001'
if spectra:
    ih = 0
    klps    = np.load(base_dir_scale+'/Exp_'+casenr+'/scale_klps_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
if prof:
    ## read height once for all
    zt    = np.load(base_dir_scale+'/Exp_'+casenr+'/scale_zt_prof_'+casenr+'.npy', allow_pickle=True) 
    klp_prof    = np.load(base_dir_scale+'/Exp_'+casenr+'/scale_klps_prof_'+casenr+'.npy', allow_pickle=True)  

time=np.empty(0,dtype='datetime64')
if org:
    ## organisation metric
    df_org = pd.DataFrame()

if prof:
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
    # qtw_psf_prof = np.empty([len(klp_prof),0,len(zt)])

if spectra:
    ## spectra
    u_av = np.empty([len(heights),0])
    v_av = np.empty([len(heights),0])
    w_av = np.empty([len(heights),0])
    qt_av = np.empty([len(heights),0])
    thl_av = np.empty([len(heights),0])

    #
    u_pf = np.empty([len(heights),len(klps),0])
    v_pf = np.empty([len(heights),len(klps),0])
    w_pf = np.empty([len(heights),len(klps),0])
    qt_pf = np.empty([len(heights),len(klps),0])
    thl_pf = np.empty([len(heights),len(klps),0])

    u_pfw_pf    = np.empty([len(heights),len(klps),0])
    u_psfw_psf  = np.empty([len(heights),len(klps),0])
    v_pfw_pf    = np.empty([len(heights),len(klps),0])
    v_psfw_psf  = np.empty([len(heights),len(klps),0])
    qt_pfw_pf    = np.empty([len(heights),len(klps),0])
    qt_psfw_psf  = np.empty([len(heights),len(klps),0])
    thl_pfw_pf    = np.empty([len(heights),len(klps),0])
    thl_psfw_psf  = np.empty([len(heights),len(klps),0])

    uw_pf    = np.empty([len(heights),len(klps),0])
    uw_psf  = np.empty([len(heights),len(klps),0])
    vw_pf    = np.empty([len(heights),len(klps),0])
    vw_psf  = np.empty([len(heights),len(klps),0])
    qtw_pf    = np.empty([len(heights),len(klps),0])
    qtw_psf  = np.empty([len(heights),len(klps),0])
    thlw_pf    = np.empty([len(heights),len(klps),0])
    thlw_psf  = np.empty([len(heights),len(klps),0])

for casenr in expnr:
    lp   = base_dir_scale+'/Exp_'+casenr
    
    if org:
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
    if prof:
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
        # qtw_psf_prof_temp = np.load(lp+'/scale_qtw_psf_prof_'+casenr+'.npy', allow_pickle=True)
    
    if spectra:
        ## load spectra at single hieght
        u_av_temp = np.zeros((len(heights),len(time_temp)))
        v_av_temp = np.zeros((len(heights),len(time_temp)))
        w_av_temp = np.zeros((len(heights),len(time_temp))) 
        qt_av_temp = np.zeros((len(heights),len(time_temp)))
        thl_av_temp = np.zeros((len(heights),len(time_temp)))

        u_pf_temp        = np.zeros((len(heights),len(klps),len(time_temp)))
        v_pf_temp        = np.zeros((len(heights),len(klps),len(time_temp)))
        w_pf_temp        = np.zeros((len(heights),len(klps),len(time_temp)))
        qt_pf_temp        = np.zeros((len(heights),len(klps),len(time_temp)))
        thl_pf_temp        = np.zeros((len(heights),len(klps),len(time_temp)))
        
        u_pfw_pf_temp    = np.zeros((len(heights),len(klps),len(time_temp)))
        u_psfw_psf_temp  = np.zeros((len(heights),len(klps),len(time_temp)))
        v_pfw_pf_temp    = np.zeros((len(heights),len(klps),len(time_temp)))
        v_psfw_psf_temp  = np.zeros((len(heights),len(klps),len(time_temp)))
        qt_pfw_pf_temp    = np.zeros((len(heights),len(klps),len(time_temp)))
        qt_psfw_psf_temp  = np.zeros((len(heights),len(klps),len(time_temp)))
        thl_pfw_pf_temp    = np.zeros((len(heights),len(klps),len(time_temp)))
        thl_psfw_psf_temp  = np.zeros((len(heights),len(klps),len(time_temp)))   
        #
        uw_pf_temp       = np.zeros((len(heights),len(klps),len(time_temp)))
        uw_psf_temp      = np.zeros((len(heights),len(klps),len(time_temp)))
        vw_pf_temp       = np.zeros((len(heights),len(klps),len(time_temp)))
        vw_psf_temp      = np.zeros((len(heights),len(klps),len(time_temp)))
        qtw_pf_temp       = np.zeros((len(heights),len(klps),len(time_temp)))
        qtw_psf_temp      = np.zeros((len(heights),len(klps),len(time_temp)))
        thlw_pf_temp       = np.zeros((len(heights),len(klps),len(time_temp)))
        thlw_psf_temp      = np.zeros((len(heights),len(klps),len(time_temp)))
    
        for ih in range(len(heights)):
        ## For scale separation 
            heights_exact[ih]  = np.load(lp+'/scale_zt_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
                
            u_av_temp[ih,:] = np.load(lp+'/scale_u_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
            v_av_temp[ih,:] = np.load(lp+'/scale_v_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
            w_av_temp[ih,:] = np.load(lp+'/scale_w_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
            qt_av_temp[ih,:] = np.load(lp+'/scale_qt_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
            thl_av_temp[ih,:] = np.load(lp+'/scale_thl_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
            #
            u_pf_temp[ih,:,:]       = np.load(lp+'/scale_u_pf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
            v_pf_temp[ih,:,:]       = np.load(lp+'/scale_v_pf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
            w_pf_temp[ih,:,:]       = np.load(lp+'/scale_w_pf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
            qt_pf_temp[ih,:,:]       = np.load(lp+'/scale_qt_pf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
            thl_pf_temp[ih,:,:]       = np.load(lp+'/scale_thl_pf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
            
            u_pfw_pf_temp[ih,:,:]   = np.load(lp+'/scale_u_pfw_pf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
            u_psfw_psf_temp[ih,:,:] = np.load(lp+'/scale_u_psfw_psf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
            v_pfw_pf_temp[ih,:,:]   = np.load(lp+'/scale_v_pfw_pf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
            v_psfw_psf_temp[ih,:,:] = np.load(lp+'/scale_v_psfw_psf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
            qt_pfw_pf_temp[ih,:,:]   = np.load(lp+'/scale_qt_pfw_pf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
            qt_psfw_psf_temp[ih,:,:] = np.load(lp+'/scale_qt_psfw_psf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
            thl_pfw_pf_temp[ih,:,:]   = np.load(lp+'/scale_thl_pfw_pf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
            thl_psfw_psf_temp[ih,:,:] = np.load(lp+'/scale_thl_psfw_psf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
            #
            uw_pf_temp[ih,:,:]   = np.load(lp+'/scale_uw_pf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
            uw_psf_temp[ih,:,:] = np.load(lp+'/scale_uw_psf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
            vw_pf_temp[ih,:,:]   = np.load(lp+'/scale_vw_pf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
            vw_psf_temp[ih,:,:] = np.load(lp+'/scale_vw_psf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
            qtw_pf_temp[ih,:,:]   = np.load(lp+'/scale_qtw_pf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
            qtw_psf_temp[ih,:,:] = np.load(lp+'/scale_qtw_psf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)  
            thlw_pf_temp[ih,:,:]   = np.load(lp+'/scale_thlw_pf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)
            thlw_psf_temp[ih,:,:] = np.load(lp+'/scale_thlw_psf_'+str(heights[ih])+'_'+casenr+'.npy', allow_pickle=True)  
        
    ## Append all times together
    time = np.append(time,time_temp)
    if org:
        ## org metrics
        df_org  = df_org.append(df_org_temp)
    if prof:
        ## profiles
        u_prof  =   np.append(u_prof,u_prof_temp,0)
        v_prof  =   np.append(v_prof,v_prof_temp,0)
        w_prof  =   np.append(w_prof,w_prof_temp,0)
        # vqt_prof  =   np.append(qt_prof,qt_prof_temp,0)
    
        u_pf_prof   =   np.append(u_pf_prof,u_pf_prof_temp,1)
        v_pf_prof   =   np.append(v_pf_prof,v_pf_prof_temp,1)
        w_pf_prof   =   np.append(w_pf_prof,w_pf_prof_temp,1)
        # qt_pf_prof   =   np.append(qt_pf_prof,qt_pf_prof_temp,1)
    
        u_pfw_pf_prof   = np.append(u_pfw_pf_prof,u_pfw_pf_prof_temp,1)
        v_pfw_pf_prof   = np.append(v_pfw_pf_prof,v_pfw_pf_prof_temp,1)
        # qt_pfw_pf_prof   = np.append(qt_pfw_pf_prof,qt_pfw_pf_prof_temp,1)
        u_psfw_psf_prof = np.append(u_psfw_psf_prof,u_psfw_psf_prof_temp,1)
        v_psfw_psf_prof = np.append(v_psfw_psf_prof,v_psfw_psf_prof_temp,1)
        # qt_psfw_psf_prof = np.append(qt_psfw_psf_prof,qt_psfw_psf_prof_temp,1)
        
        uw_pf_prof  = np.append(uw_pf_prof,uw_pf_prof_temp,1)
        vw_pf_prof  = np.append(vw_pf_prof,vw_pf_prof_temp,1)
        # qtw_pf_prof  = np.append(qtw_pf_prof,qtw_pf_prof_temp,1)
        uw_psf_prof = np.append(uw_psf_prof,uw_psf_prof_temp,1)
        vw_psf_prof = np.append(vw_psf_prof,vw_psf_prof_temp,1)
        # qtw_psf_prof = np.append(qtw_psf_prof,qtw_psf_prof_temp,1)
    if spectra:
        ## spectra
        u_av=np.append(u_av,u_av_temp,1)
        v_av=np.append(v_av,v_av_temp,1)
        w_av=np.append(w_av,w_av_temp,1)
        qt_av=np.append(qt_av,qt_av_temp,1)
        thl_av=np.append(thl_av,thl_av_temp,1)
        #
        u_pf=np.append(u_pf,u_pf_temp,2)
        v_pf=np.append(v_pf,v_pf_temp,2)
        w_pf=np.append(w_pf,w_pf_temp,2)
        qt_pf=np.append(qt_pf,qt_pf_temp,2)
        thl_pf=np.append(thl_pf,thl_pf_temp,2)
        
        u_pfw_pf    = np.append(u_pfw_pf,   u_pfw_pf_temp,  2)
        u_psfw_psf  = np.append(u_psfw_psf, u_psfw_psf_temp,2)
        v_pfw_pf    = np.append(v_pfw_pf,   v_pfw_pf_temp,  2)
        v_psfw_psf  = np.append(v_psfw_psf, v_psfw_psf_temp,2)
        qt_pfw_pf    = np.append(qt_pfw_pf,   qt_pfw_pf_temp,  2)
        qt_psfw_psf  = np.append(qt_psfw_psf, qt_psfw_psf_temp,2)
        thl_pfw_pf    = np.append(thl_pfw_pf,   thl_pfw_pf_temp,  2)
        thl_psfw_psf  = np.append(thl_psfw_psf, thl_psfw_psf_temp,2)
        
        uw_pf       = np.append(uw_pf,      uw_pf_temp,     2)
        uw_psf      = np.append(uw_psf,     uw_psf_temp,    2)
        vw_pf       = np.append(vw_pf,      vw_pf_temp,     2)
        vw_psf      = np.append(vw_psf,     vw_psf_temp,    2)
        qtw_pf       = np.append(qtw_pf,      qtw_pf_temp,     2)
        qtw_psf      = np.append(qtw_psf,     qtw_psf_temp,    2)
        thlw_pf       = np.append(thlw_pf,      thlw_pf_temp,     2)
        thlw_psf      = np.append(thlw_psf,     thlw_psf_temp,    2)
    
#%%   Convert to Xarray and Save 
if org:
    ## organisation metrics
    df_org = df_org.apply(pd.to_numeric)
    df_org = df_org.to_xarray().rename({'index':'time'})
    df_org.to_netcdf(save_dir+'df_org_allExp.nc')
if spectra:
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
    da_scales['thl_pf']       =(('height','klp','time'),thl_pf)

    da_scales['u_pfw_pf']   =(('height','klp','time'),u_pfw_pf)
    da_scales['u_psfw_psf'] =(('height','klp','time'),u_psfw_psf)
    da_scales['v_pfw_pf']   =(('height','klp','time'),v_pfw_pf)
    da_scales['v_psfw_psf'] =(('height','klp','time'),v_psfw_psf)
    da_scales['qt_pfw_pf']   =(('height','klp','time'),qt_pfw_pf)
    da_scales['qt_psfw_psf'] =(('height','klp','time'),qt_psfw_psf)
    da_scales['thl_pfw_pf']   =(('height','klp','time'),thl_pfw_pf)
    da_scales['thl_psfw_psf'] =(('height','klp','time'),thl_psfw_psf)

    da_scales['uw_pf']      =(('height','klp','time'),uw_pf)
    da_scales['uw_psf']     =(('height','klp','time'),uw_psf)
    da_scales['vw_pf']      =(('height','klp','time'),vw_pf)
    da_scales['vw_psf']     =(('height','klp','time'),vw_psf)
    da_scales['qtw_pf']      =(('height','klp','time'),qtw_pf)
    da_scales['qtw_psf']     =(('height','klp','time'),qtw_psf)
    da_scales['thlw_pf']      =(('height','klp','time'),thlw_pf)
    da_scales['thlw_psf']     =(('height','klp','time'),thlw_psf)

    da_scales['u_av']       =(('height','time'),u_av)
    da_scales['v_av']       =(('height','time'),v_av)
    da_scales['w_av']       =(('height','time'),w_av)
    da_scales['qt_av']       =(('height','time'),qt_av)
    da_scales['thl_av_av']       =(('height','time'),thl_av)

    da_scales.to_netcdf(save_dir+'scale_sep_allExp_'+save_ext+'.nc')

if prof:
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
    
    da_scales_prof.to_netcdf(save_dir+'scale_sep_prof_allExp.nc')

print('End.')

