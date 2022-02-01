#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 12:47:47 2020

Analysis of DALES outputs

@author: alessandrosavazzi
"""

#%% DALES_MOMENTUM_BUDGET.py
# 

#%%                             Libraries
###############################################################################
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
from glob import glob
import sys
from sklearn.cluster import KMeans
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'large',
         'axes.labelsize': 20,
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large',
         'figure.figsize':[10,7],
         'figure.titlesize':20}
pylab.rcParams.update(params)

my_source_dir = os.path.abspath('{}/../../../My_source_codes')
sys.path.append(my_source_dir)
from My_thermo_fun import *

def logic(index,first_line=4):
    if ((index-3)%levels+3 == 0) or ((index-2)%levels+3 == 0) or (index<first_line):
       return True
    return False

def calc_geo_height(ds_,fliplevels=False):
    if fliplevels==True:
        ds_['level']=np.flip(ds_.level)
    
    rho = 100.*ds_.p/(Rd*ds_.T*(1+0.61*(ds_.q)))
    k = np.arange(ds_.level[0]+(ds_.level[1]-ds_.level[0])/2,\
                  ds_.level[-1]+(ds_.level[1]-ds_.level[0])/2,\
                  ds_.level[1]-ds_.level[0])
    rho_interp = rho.interp(level=k)
    zz = np.zeros((len(ds_['time']),len(ds_['level'])))
    zz = ((100.*ds_.p.diff(dim='level').values)/(-1*rho_interp*g)).cumsum(dim='level')
    z = zz.interp(level=ds_.level,kwargs={"fill_value": "extrapolate"})
    ds_['z']=z
    return (ds_)

def adjust_lightness(color, amount=0.7):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(max(0, min(1, amount * c[0])),\
                               max(0, min(1, amount * c[1])),\
                               max(0, min(1, amount * c[2])))
#%%                         Open Files
###############################################################################
expnr      = ['001','002','003','004','005','006','007','008','009','010',\
              '011','012','013','014','015','016']
case       = '20200202_12'
casenr     = '001'

### Directories for runnin on VrLab
base_dir   = '/Users/acmsavazzi/Documents/Mount/DALES/'
Input_dir  = base_dir+'Cases/20200202_12_300km/'
Output_dir = base_dir+'Experiments/EUREC4A/Exp_ECMWF/'+case+'/'
save_dir   = '/Users/acmsavazzi/Documents/WORK/PhD_Year2/'
# fig_dir = os.path.abspath('{}/../../Figures/DALES')+'/'

### Directories for runnin on TU server
# base_dir   = 'staff-umbrella/cmtrace/Alessandro/'
# Input_dir  = base_dir+'Raw_Data/Les/Eurec4a/'+case+'/Exp_'+expnr[0]+'/'
# Output_dir = base_dir+'Raw_Data/Les/Eurec4a/'+case+'/'
# save_dir   = base_dir+'PhD_Year2/'

### Directories for local 
# Input_dir  = '/Users/acmsavazzi/Documents/WORK/PhD_Year1/DATA/DALES/DALES/Experiments/20200209_10/Exp_009/'
# Output_dir = Input_dir+'../'
# save_dir   = '/Users/acmsavazzi/Documents/WORK/PhD_Year2/'

###
srt_time   = np.datetime64('2020-02-02')
end_time   = np.datetime64('2020-02-09')

make_videos = False

hrs_inp = ((end_time - srt_time)/np.timedelta64(1, 'h')).astype(int)+1

#%%     OPTIONS FOR PLOTTING

# col=['b','r','g','orange','k']
col=['red','coral','maroon','blue','cornflowerblue','darkblue','green','lime','forestgreen']
height_lim = [0,5000]        # in m

#%%                             Import
###############################################################################

############                    NAMOPTIONS                         ############
# with open(Input_dir+'namoptions') as f:
with open('/Users/acmsavazzi/Documents/WORK/PhD_Year1/DATA/DALES/DALES/Experiments/20200209_10/Exp_009/namoptions') as f:
    for line in f:
        if "kmax" in line:
            levels = float(line.split('=')[1]) + 1  # number of vertical levels to split the input csv files
        if 'xlat' in line:
            domain_center={'lat':float(line.split('=')[1])}
        if 'xlon' in line:
            domain_center['lon']=float(line.split('=')[1])
            
############                    INPUT                              ############
####     backrad.inp    ####
backrad    = xr.open_dataset(Input_dir+'backrad.inp.'+casenr+'.nc')

####     prof.inp   ####
colnames = ['z (m)','thl (K)','qt (kg kg-1)','u (m s-1)','v (m s-1)','tke (m2 s-2)']
prof = pd.read_csv(Input_dir+'prof.inp.'+casenr,header = 2,names=colnames,\
                   index_col=False,delimiter = " ")
    
####    nudge.inp   ####
colnames = ['z','factor','u','v','w','thl','qt']    
nudge    = pd.read_csv(Input_dir+'nudge.inp.'+casenr,\
           skiprows=lambda x: logic(x),comment='#',\
           delimiter = " ",names=colnames,index_col=False)
nudge = nudge.apply(pd.to_numeric, errors='coerce')
nudge['time'] = nudge.index.values//levels
nudge.set_index(['time', 'z'], inplace=True)
nudge = nudge.to_xarray()
nudge = nudge.sel(time=range(hrs_inp))

####     ls_flux.inp    ####
# first read the surface values
colnames = ['time','wthl_s','wqt_s','th_s','qt_s','p_s']
ls_surf = pd.read_csv(Input_dir+'ls_flux.inp.'+casenr,header = 3,nrows=hrs_inp,\
                     names=colnames,index_col=False,delimiter = " ")
# second read the profiles
colnames = ['z','u_g','v_g','w_ls','dqtdx','dqtdy','dqtdt','dthldt','dudt','dvdt']
skip = 0
with open(Input_dir+'ls_flux.inp.'+casenr) as f:
    for line in f:
        if line and line != '\n':
            skip += 1
        else:
            break
ls_flux    = pd.read_csv(Input_dir+'ls_flux.inp.'+casenr,\
           skiprows=lambda x: logic(x,skip+1),comment='#',\
           delimiter = " ",names=colnames,index_col=False)
# somehow it still reads the header every hour, so...    
ls_flux = ls_flux.dropna()
ls_flux = ls_flux.apply(pd.to_numeric, errors='coerce')

ls_flux['time'] = (ls_flux.index.values//levels)-0.5
ls_flux['time'][ls_flux['time']<0] = 0

ls_flux.set_index(['time', 'z'], inplace=True)
ls_flux = ls_flux.to_xarray()
ls_surf['T_s'] = calc_T(ls_surf['th_s'],ls_surf['p_s'])
ls_surf['rho'] = calc_rho(ls_surf['p_s'],ls_surf['T_s'],ls_surf['qt_s'])

############                    OUTPUT                             ############
prof_files      = []
tmser_files     = []
samptend_files  = []
moments_files   = []
tmsurf_files    = [] 
cape_merg_files = []
for path,subdir,files in os.walk(Output_dir):
    if path[-3:] in expnr: 
        for file in glob(os.path.join(path, 'profiles*.nc')):
            prof_files.append(file)
        for file in glob(os.path.join(path, 'tmser*.nc')):
            tmser_files.append(file)
        for file in glob(os.path.join(path, 'samptend*.nc')):
            samptend_files.append(file)
        for file in glob(os.path.join(path, 'moments*')):
            moments_files.append(file)
        for file in glob(os.path.join(path, 'tmsurf*')):
            tmsurf_files.append(file)
        for file in glob(os.path.join(path, 'merged_cape*')):
            cape_merg_files.append(file)
            
####     profiles.nc    ####          
profiles = xr.open_mfdataset(prof_files, combine='by_coords')
profiles['time'] = srt_time + profiles.time.astype("timedelta64[s]")
#remove last time step because it is a midnight of the day after
profiles = profiles.isel(time=slice(0,-1))

####     tmser.nc   ####
tmser = xr.open_mfdataset(tmser_files, combine='by_coords')
# tmser['T_s'] = calc_T(tmser['thlskin'],ls_surf['p_s'])

####     samptend.nc    ####
samptend   = xr.open_mfdataset(samptend_files, combine='by_coords')
samptend['time'] = srt_time + samptend.time.astype("timedelta64[s]")

####     moments.001    ####
colnames = ['lev','z','pres','thl2','thv2','th2','qt2','u2','v2','hght','w2','skew','sfs-tke']
moments  = []
for file in moments_files:
    temp    = pd.read_csv(file,\
           skiprows=lambda x: logic(x),comment='#',\
           delimiter = " ",names=colnames,index_col=False,skipinitialspace=True)
    moments.append(temp)
moments = pd.concat(moments, axis=0, ignore_index=True)
moments['time'] = (moments.index.values//(levels-1))/4 + 0.25
moments.set_index(['time', 'z'], inplace=True)
moments = moments.to_xarray()

####     tmsurf.001     ####
colnames = ['time','ust','tst','qst','obukh','thls','z0','wthls','wthvs','wqls' ]
tmsurf   = []
for file in tmsurf_files:
    temp = pd.read_csv(file,header = 0,\
                     names=colnames,index_col=False,delimiter = " ",skipinitialspace=True)   
    tmsurf.append(temp)
tmsurf = pd.concat(tmsurf, axis=0, ignore_index=True)

if make_videos:
    ####     fielddump.nc    ####
    # fielddump  = xr.open_dataset(Output_dir+'fielddump.000.000.'+expnr+'.nc')
    
    ####     merged_cape.nc    ####
    cape   = xr.open_mfdataset(cape_merg_files[0], combine='by_coords')
    
    ####     merged_crossxy.nc    ####
    # crossxy_0001 = xr.open_dataset(Output_dir+'merged_crossxy_0001.'+expnr+'.nc')

#%% Import Harmonie large scale

#%% Import observations

#%% import IFS

#%%                         
###############################################################################
rho = ls_surf['rho'].mean()
# rho = 1.15


#%% convert xt, yt into lon, lat coordinates  


#%% SOME NEW VARIABLES

profiles['du_dz']=profiles.u.differentiate('zt')
profiles['dv_dz']=profiles.v.differentiate('zt')

profiles['duwt_dz']=profiles.uwt.differentiate('zm')
profiles['dvwt_dz']=profiles.vwt.differentiate('zm')

profiles['duwr_dz']=profiles.uwr.differentiate('zm')
profiles['dvwr_dz']=profiles.vwr.differentiate('zm')
profiles['duws_dz']=profiles.uws.differentiate('zm')
profiles['dvws_dz']=profiles.vws.differentiate('zm')



#%% Group by day
profiles_daily = profiles.resample(time='D').mean('time')

#%% Group by shape of some variables
####  K-mean clustering 
# CLUSTER on dayly shear profiles

### maybe cluster only based on the shape below 5 km?

shear_cluster=KMeans(n_clusters=min(profiles_daily.time.size,3),random_state=0,n_init=15,max_iter=10000,\
                tol=10**-7).fit(profiles_daily['du_dz'].sel(zt=slice(height_lim[0],height_lim[1])))
idx = np.argsort(shear_cluster.cluster_centers_.sum(axis=1))

profiles_daily['group'] = (('time'), shear_cluster.labels_)

profiles_daily_clusteres=profiles_daily.groupby('group').mean()

#%%                         PLOTTING
###############################################################################
# Variables to plot: 
# u, v, wsps, u'w', v'w', du_dz, dv_dz, du'w'_dz, dv'w'_dz, cloud_frac

# Separate contributors in the momentum budget.

# Subsets of data:
# 1) Domain and Temporl mean profiles
#
# 2) Domain mean profiles for daytime & nighttime separately
#
# 3) Domain mean profiles for subset of days 
#   a) arbitrarly choosen 
#   b) selcted with K-mean clustering on the shape of e.g. du_dz (shear)
#
# 4) Temporal mean profiles for cloudy regions, updrafts, downdrafts, environment separately


#%% 1) Domain and Temporl mean profiles

day_night=True
day_interval    = [10,16]
night_interval  = [22,4]

days = ['2020-02-09',]

def find_time_interval(time_1,time_2):
    if time_1 > time_2:
        temp= list(range(time_1,24)) + list(range(0,time_2+1))
    else: temp= list(range(time_1,time_2+1))
    return temp

for ii in ['all',]:
    if ii == 'all':
        hrs_to_plot = profiles
        title='Domain and Temporal mean'
    elif ii == 'days':
        hrs_to_plot = profiles.where(profiles.time.dt.strftime('%Y-%m-%d').isin(days),drop=True)
        title='Domain mean for '+ " ".join(days)
    if day_night:
        hrs_to_plot_day = hrs_to_plot.sel(time=hrs_to_plot['time.hour'].\
                           isin(find_time_interval(day_interval[0],day_interval[1])))
        hrs_to_plot_night = hrs_to_plot.sel(time=hrs_to_plot['time.hour'].\
                           isin(find_time_interval(night_interval[0],night_interval[1])))

    ## cloud fraction
    plt.figure(figsize=(6,9))
    plt.suptitle('Cloud fraction')
    hrs_to_plot['cfrac'].mean('time').plot(y='zt',c=col[0],lw=2,label='Cloud fraction')
    if day_night:      
        hrs_to_plot_day['cfrac'].mean('time').plot(y='zt',c=col[1],lw=1,label='Daytime')
        hrs_to_plot_night['cfrac'].mean('time').plot(y='zt',c=col[2],lw=1,label='Nighttime')
    plt.legend()
    plt.xlabel('%')
    plt.ylim(height_lim)
    plt.title(title)
    plt.savefig(save_dir+'Figures/mean_cfrac.pdf')
    ## winds
    plt.figure(figsize=(6,9))
    plt.suptitle('Winds')
    for idx,var in enumerate(['u','v']):
        hrs_to_plot[var].mean('time').plot(y='zt',c=col[idx*3],lw=2, label='DALES '+var)
        if day_night:
            hrs_to_plot_day[var].mean('time').plot(y='zt',c=col[idx*3+1],lw=1, label='Daytime '+var)
            hrs_to_plot_night[var].mean('time').plot(y='zt',c=col[idx*3+2],lw=1, label='Nighttime '+var)
    # if var in nudge:
    #     nudge[var].mean('time').plot(y='z',c=adjust_lightness(col[idx]),lw=0.8,label='HARMONIE '+var)
    plt.legend()
    plt.xlabel('m/s')
    plt.axvline(0,c='k',lw=0.5)
    plt.ylim(height_lim)
    plt.xlim([-12.5,0.5])
    plt.title(title)
    plt.savefig(save_dir+'Figures/mean_winds.pdf')
    
    ## momentum fluxes
    plt.figure(figsize=(6,9))
    plt.suptitle('Momentum fluxes')
    for idx,var in enumerate(['uw','vw']):
        for ii in ['t','r']:
            if ii == 't':
                hrs_to_plot[var+ii].mean('time').plot(y='zm',c=col[idx*3],ls='-',label=var+' tot')
                if day_night:
                    hrs_to_plot_day[var+ii].mean('time').plot(y='zm',c=col[idx*3+1],lw=1, label='Daytime tot '+var)
                    hrs_to_plot_night[var+ii].mean('time').plot(y='zm',c=col[idx*3+2],lw=1, label='Nighttime tot '+var)
            else:
                hrs_to_plot[var+ii].mean('time').plot(y='zm',c=col[idx*3],ls='--',label=var+' resolved')
                if day_night:
                    hrs_to_plot_day[var+ii].mean('time').plot(y='zm',c=col[idx*3+1],ls='--',lw=1, label='Daytime resolved '+var)
                    hrs_to_plot_night[var+ii].mean('time').plot(y='zm',c=col[idx*3+2],ls='--',lw=1, label='Nighttime resolved '+var)
    plt.title(title)
    plt.legend()
    plt.xlabel('Momentum flux (m2/s2)')
    plt.axvline(0,c='k',lw=0.5)
    plt.ylim(height_lim)
    plt.savefig(save_dir+'Figures/mean_momflux.pdf')
        
    ## counter gradient fluxes
    plt.figure(figsize=(6,9))
    plt.suptitle('Counter gradient transport')
    (hrs_to_plot['uwt'] * hrs_to_plot.du_dz.interp(zt=profiles.zm)).mean('time').\
        plot(y='zm',c=col[0],lw=2,label='uw du_dz')
    (hrs_to_plot['vwt'] * hrs_to_plot.dv_dz.interp(zt=profiles.zm)).mean('time').\
        plot(y='zm',c=col[3],lw=2,label='vw dv_dz')
    if day_night:
        (hrs_to_plot_day['uwt'] * hrs_to_plot_day.du_dz.interp(zt=profiles.zm)).mean('time').\
            plot(y='zm',c=col[1],lw=1,label='Daytime')
        (hrs_to_plot_night['uwt'] * hrs_to_plot_night.du_dz.interp(zt=profiles.zm)).mean('time').\
            plot(y='zm',c=col[2],lw=1,label='Nighttime')
    if day_night:
        (hrs_to_plot_day['vwt'] * hrs_to_plot_day.dv_dz.interp(zt=profiles.zm)).mean('time').\
            plot(y='zm',c=col[4],lw=1,label='Daytime')
        (hrs_to_plot_night['vwt'] * hrs_to_plot_night.dv_dz.interp(zt=profiles.zm)).mean('time').\
            plot(y='zm',c=col[5],lw=1,label='Nighttime')
    plt.title(title)
    plt.legend()
    plt.axvline(0,c='k',lw=0.5)
    plt.ylim(height_lim)
    plt.xlim([-0.00015,None])
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.savefig(save_dir+'Figures/mean_gradTransp.pdf')
    
    ## Momentum flux convergence
    plt.figure(figsize=(6,9))
    plt.suptitle('Momentum flux convergence')
    (-1*hrs_to_plot['duwt_dz']).mean('time').plot(y='zm',c='b',ls='-',label='- duw_dz')
    (-1*hrs_to_plot['dvwt_dz']).mean('time').plot(y='zm',c='r',ls='-',label='- dvw_dz')
    (-1*hrs_to_plot['duwr_dz']).mean('time').plot(y='zm',c='b',ls='--',label='- duwr_dz')
    (-1*hrs_to_plot['dvwr_dz']).mean('time').plot(y='zm',c='r',ls='--',label='- dvwr_dz')
    (-1*hrs_to_plot['duws_dz']).mean('time').plot(y='zm',c='b',ls=':',label='- duws_dz')
    (-1*hrs_to_plot['dvws_dz']).mean('time').plot(y='zm',c='r',ls=':',label='- dvws_dz')
    plt.title(title)
    plt.legend()
    plt.axvline(0,c='k',lw=0.5)
    plt.ylim(height_lim)
    plt.xlim([-0.0001,0.0002])
    plt.xlabel('m/s2')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.savefig(save_dir+'Figures/mean_momfluxconv.pdf')
    
#%%  TIMESERIES

## momentum fluxes
plt.figure(figsize=(19,5))
plt.suptitle('Momentum flux')
(profiles['uwt']).plot(y='zm',vmin=-0.25)
plt.ylim(height_lim)
plt.xlim([srt_time,end_time])
for ii in np.arange(srt_time, end_time):
    plt.axvline(x=ii,c='k')
# plt.savefig(save_dir+'Figures/tmser_momflux.pdf')

#%% DAILY MEANS
## momentum fluxes
plt.figure(figsize=(19,5))
profiles_daily['uwt'].plot(y='zm',vmin=-0.1)
plt.ylim(height_lim)
plt.savefig(save_dir+'Figures/daily_momflux.pdf')

#%% DAILY MEANS clustered

plt.figure(figsize=(5,9))
for key, group in profiles_daily.groupby('group'):
    group['du_dz'].mean('time').plot(y='zt',c=col[key*3],lw=3,label='Group '+str(key))
    group['du_dz'].plot.line(y='zt',c=col[key*3],lw=0.7,alpha=0.3,add_legend=False)
plt.legend()
plt.ylim(height_lim)
plt.axvline(0,c='k',lw=0.5)
plt.savefig(save_dir+'Figures/Kmean_Ushear.pdf')
## momentum fluxes
plt.figure(figsize=(5,9))
for key, group in profiles_daily.groupby('group'):
    group['uwt'].mean('time').plot(y='zm',c=col[key*3],lw=3,label='Group '+str(key))
    group['uwt'].plot.line(y='zm',c=col[key*3],lw=0.7,alpha=0.3,add_legend=False)
plt.legend()
plt.ylim(height_lim)
plt.axvline(0,c='k',lw=0.5)
plt.savefig(save_dir+'Figures/Kmean_Umomflux.pdf')


#%% TENDENCIES

acc_time = 3600*1
temp_hrs=[srt_time,end_time]

fig, axs = plt.subplots(1,2,figsize=(10,9))
axs = axs.ravel()
for ii in ['u','v']:
    if ii == 'u':
        temp=0
        axs[temp].plot(acc_time*samptend.utendlsall.\
                   sel(time=slice(temp_hrs[0],temp_hrs[1])).mean('time'),samptend.zt,c='k',label='Large scale')
        axs[temp].plot(acc_time*samptend.utendadvall.\
                   sel(time=slice(temp_hrs[0],temp_hrs[1])).mean('time'),samptend.zt,c='b',label='Advective')
        axs[temp].plot(acc_time*samptend.utenddifall.\
                   sel(time=slice(temp_hrs[0],temp_hrs[1])).mean('time'),samptend.zt,c='g',label='Diffusive')
        axs[temp].plot(acc_time*samptend.utendtotall.\
                   sel(time=slice(temp_hrs[0],temp_hrs[1])).mean('time'),samptend.zt,c='r',label='Net')
        # axs[temp].plot(acc_time*samptend.utendforall.\
        #            sel(time=slice(temp_hrs[0],temp_hrs[1])).mean('time'),samptend.zt,c='m',label='Other')       
        axs[temp].plot(acc_time*samptend.utendpoisall.\
                    sel(time=slice(temp_hrs[0],temp_hrs[1])).mean('time'),samptend.zt,c='m',label='Pres. grad.')        
        axs[temp].set_ylabel('Height (m)')
        axs[temp].legend()
    elif ii =='v':
        temp=1
        axs[temp].plot(acc_time*samptend.vtendlsall.\
                   sel(time=slice(temp_hrs[0],temp_hrs[1])).mean('time'),samptend.zt,c='k',label='Large scale')
        axs[temp].plot(acc_time*samptend.vtendadvall.\
                   sel(time=slice(temp_hrs[0],temp_hrs[1])).mean('time'),samptend.zt,c='b',label='Advective')
        axs[temp].plot(acc_time*samptend.vtenddifall.\
                   sel(time=slice(temp_hrs[0],temp_hrs[1])).mean('time'),samptend.zt,c='g',label='Diffusive')
        axs[temp].plot(acc_time*samptend.vtendtotall.\
                   sel(time=slice(temp_hrs[0],temp_hrs[1])).mean('time'),samptend.zt,c='r',label='Net')
        # axs[temp].plot(acc_time*samptend.vtendforall.\
        #            sel(time=slice(temp_hrs[0],temp_hrs[1])).mean('time'),samptend.zt,c='m',label='Other')   
        axs[temp].plot(acc_time*samptend.vtendpoisall.\
                    sel(time=slice(temp_hrs[0],temp_hrs[1])).mean('time'),samptend.zt,c='m',label='Pres. grad.')   

    axs[temp].set_title(ii+' tendency')
    axs[temp].axvline(0,c='k',lw=0.5)
    # axs[temp].set_ylim([0,ls_flux.z.max()])
    axs[temp].set_ylim(height_lim)
    axs[temp].set_xlabel('(m/s^2 / hour)')
#%% Make videos

#%%
print('end.')


