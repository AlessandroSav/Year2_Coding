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
import string
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import DivergingNorm
import matplotlib.animation as animation
import os
from glob import glob
from datetime import datetime, timedelta
import sys
from sklearn.cluster import KMeans
import matplotlib.pylab as pylab
from pylab import *
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
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
              '011','012','013','014','015','016','017']
case       = '20200202_12'
casenr     = '001'      # experiment number where to read input files 

### Directories for local
base_dir        = '/Users/acmsavazzi/Documents/WORK/PhD_Year1/DATA/DALES/'
Input_dir       = base_dir  + 'DALES/Cases/EUREC4A/20200202_12_300km_clim/'
# dales_exp_dir   = base_dir  + 'DALES_atECMWF/outputs/20200209_10/'
dales_exp_dir   = base_dir  + 'DALES_atECMWF/outputs/20200202_12_clim'
Output_dir      = base_dir  + 'DALES_atECMWF/outputs/20200202_12_clim/'

my_harm_dir     = '/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/HARMONIE/cy43_clim/average_300km/'
# IFS DATA
ifs_dir         = '/Users/acmsavazzi/Documents/WORK/Research/MyData/'
# OBS DATA
obs_dir         = '/Users/acmsavazzi/Documents/WORK/Research/MyData/'
#SAVE DIRECTORY 
save_dir        = '/Users/acmsavazzi/Documents/WORK/PhD_Year2/Manuscript/Figures/'

### times to read and to plot 
srt_time   = np.datetime64('2020-02-02')
end_time   = np.datetime64('2020-02-11')
temp_hrs   = [np.datetime64('2020-02-02'),np.datetime64('2020-02-11')]
hours = srt_time,srt_time + [np.timedelta64(2, 'h'),np.timedelta64(48, 'h'),\
                             np.timedelta64(108, 'h'),np.timedelta64(144, 'h')]
    
make_videos       = False
LES_forc_HARMONIE = True
harm_3d           = True
comp_observations = True
comp_experiments  = False

harmonie_dir   = base_dir+'Raw_Data/HARMONIE/BES_harm43h22tg3_fERA5_exp0/2020/'
harmonie_time_to_keep = '202002010000-'            

#%%
acc_time = 3600*1

#%%     OPTIONS FOR PLOTTING

# col=['b','r','g','orange','k']
col=['red','coral','maroon','blue','cornflowerblue','darkblue','green','lime','forestgreen','m']
height_lim = [0,4000]        # in m



proj=ccrs.PlateCarree()
coast = cartopy.feature.NaturalEarthFeature(\
        category='physical', scale='50m', name='coastline',
        facecolor='none', edgecolor='r')

#%%                             Import
###############################################################################

############                    NAMOPTIONS                         ############
print("Reading namoptions.")
# with open(Input_dir+'namoptions') as f:
with open(Output_dir+'Exp_'+expnr[0]+'/namoptions.'+expnr[0]) as f:
    for line in f:
        if "kmax" in line:
            levels = float(line.split('=')[1]) + 1  # number of vertical levels to split the input csv files
        if 'xlat' in line:
            domain_center={'lat':float(line.split('=')[1])}
        if 'xlon' in line:
            domain_center['lon']=float(line.split('=')[1])
            
############                    INPUT                              ############
####     backrad.inp    ####
print("Reading backrad.")
backrad    = xr.open_dataset(Input_dir+'backrad.inp.'+casenr+'.nc')

####     prof.inp   ####
print("Reading prof.inp.")
colnames = ['z (m)','thl (K)','qt (kg kg-1)','u (m s-1)','v (m s-1)','tke (m2 s-2)']
prof = pd.read_csv(Input_dir+'prof.inp.'+casenr,header = 2,names=colnames,\
                   index_col=False,delimiter = " ")
    
####    nudge.inp   ####
print("Reading nudge.inp.")
colnames = ['z','factor','u','v','w','thl','qt']    
nudge    = pd.read_csv(Input_dir+'nudge.inp.'+casenr,\
           skiprows=lambda x: logic(x),comment='#',\
           delimiter = " ",names=colnames,index_col=False)
nudge = nudge.apply(pd.to_numeric, errors='coerce')
nudge['time'] = nudge.index.values//levels
nudge.set_index(['time', 'z'], inplace=True)
nudge = nudge.to_xarray()
nudge['time'] = srt_time + nudge.time.astype("timedelta64[h]")
nudge = nudge.sel(time=slice(srt_time,end_time))

####     ls_flux.inp    ####
# hrs_inp = ((end_time - srt_time)/np.timedelta64(1, 'h')).astype(int)+1
with open(Input_dir+'ls_flux.inp.'+expnr[0]) as f:
    hrs_inp = 0
    for line in f:
        if '(s)' in line: hrs_inp = 0
        else: hrs_inp += 1  # number of timesteps in input files 
        if "z (m)" in line: break
    hrs_inp -= 3
# first read the surface values
print("Reading input surface fluxes.")
colnames = ['time','wthl_s','wqt_s','th_s','qt_s','p_s']
ls_surf = pd.read_csv(Input_dir+'ls_flux.inp.'+casenr,header = 3,nrows=hrs_inp,\
                     names=colnames,index_col=False,delimiter = " ")
ls_surf.set_index(['time'], inplace=True)
ls_surf = ls_surf.to_xarray()
ls_surf['time'] = srt_time + ls_surf.time.astype("timedelta64[s]")

# second read the profiles
print("Reading input forcing profiles.")
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
ls_flux['time'] = (ls_flux.index.values//levels)
ls_flux.set_index(['time', 'z'], inplace=True)
ls_flux = ls_flux.to_xarray()
ls_flux['time'] = srt_time + ls_flux.time.astype("timedelta64[h]") - np.timedelta64(30, 'm')
ls_flux['time'] = np.insert(ls_flux['time'][1:],0,srt_time)
ls_flux = ls_flux.sel(time=slice(srt_time,end_time))
ls_surf['T_s'] = calc_T(ls_surf['th_s'],ls_surf['p_s'])
ls_surf['rho'] = calc_rho(ls_surf['p_s'],ls_surf['T_s'],ls_surf['qt_s'])

############                    OUTPUT                             ############
prof_files      = []
tmser_files     = []
samptend_files  = []
moments_files   = []
tmsurf_files    = [] 
cape_merg_files = []
print("Finding output files.")  
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

####     samptend.nc    ####
print("Reading DALES tendencies.") 
samptend   = xr.open_mfdataset(np.sort(samptend_files), combine='by_coords')
samptend['time'] = srt_time + samptend.time.astype("timedelta64[s]")
# interpolate half level to full level
samptend = samptend.interp(zm=samptend.zt)
samptend = samptend.rename({'zt':'z'})

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


####     tmsurf.001     ####
print("Reading DALES surface values.") 
colnames = ['time','ust','tst','qst','obukh','thls','z0','wthls','wthvs','wqls' ]
tmsurf   = []
for file in np.sort(tmsurf_files):
    temp = pd.read_csv(file,header = 0,\
                     names=colnames,index_col=False,delimiter = " ",skipinitialspace=True)   
    tmsurf.append(temp)
tmsurf = pd.concat(tmsurf, axis=0, ignore_index=True)
tmsurf.set_index(['time'], inplace=True)
tmsurf = tmsurf.to_xarray()
tmsurf['time'] = srt_time + tmsurf.time.astype("timedelta64[s]")


if make_videos:
    ####     fielddump.nc    ####
    # fielddump  = xr.open_dataset(Output_dir+'fielddump.000.000.'+expnr+'.nc')
    
    ####     merged_cape.nc    ####
    cape_merg_files.sort()
    cape   = xr.open_mfdataset(cape_merg_files[1:], combine='by_coords',decode_times=False)
    cape['time'] = srt_time + cape.time.astype("timedelta64[s]")
    ####     merged_crossxy.nc    ####
    # crossxy_0001 = xr.open_dataset(Output_dir+'merged_crossxy_0001.'+expnr+'.nc')

#%% Import Harmonie
### Import raw Harmonie data
# This is too slow... need to find a better way. Maybe in a separate file open
# and save only the points and time neede for comparison.
if harm_3d:
    ### HARMONIE clim spatially averaged
    print("Reading HARMONIE clim spatial average.") 
    file = my_harm_dir+'LES_forcing_2020020200.nc'
    harm_clim_avg = xr.open_mfdataset(file)
    # harm_clim_avg = harm_clim_avg.mean(dim=['x', 'y'])
    
    harm_clim_avg = harm_clim_avg.sel(time=~harm_clim_avg.get_index("time").duplicated())
    harm_clim_avg = harm_clim_avg.interpolate_na('time')
    
    #
    z_ref = harm_clim_avg.z.mean('time')
    zz    = harm_clim_avg.z
    for var in list(harm_clim_avg.keys()):
        if 'level' in harm_clim_avg[var].dims:
            print("interpolating variable "+var)
            x = np.empty((len(harm_clim_avg['time']),len(harm_clim_avg['level'])))
            x[:] = np.NaN
            for a in range(len(harm_clim_avg.time)):
                x[a,:] = np.interp(z_ref,zz[a,:],harm_clim_avg[var].isel(time = a))            
            harm_clim_avg[var] = (("time","level"), x)    
    # convert model levels to height levels
    harm_clim_avg = harm_clim_avg.rename({'z':'geo_height'})
    harm_clim_avg = harm_clim_avg.rename({'level':'z','clw':'ql','cli':'qi'})
    harm_clim_avg["z"] = (z_ref-z_ref.min()).values
    harm_clim_avg['z'] = harm_clim_avg.z.assign_attrs(units='m',long_name='Height')
    
#%% Import observations
print("Reading observations.") 

## JOANNE tendencies
joanne = xr.open_dataset(obs_dir+'joanne_tend.nc') 
joanne = joanne.rename({'Fx':'F_u','Fy':'F_v'})
joanne = joanne.sel(time=slice(srt_time,end_time))

## radio and dropsondes 
ds_obs = {}
ds_obs['radio'] = xr.open_dataset(obs_dir+'nc_radio.nc').sel(launch_time=slice(srt_time,end_time)).rename({'q':'qt'})
ds_obs['drop'] = xr.open_dataset(obs_dir+'My_sondes.nc').sel(launch_time=slice(srt_time,end_time)).rename({'q':'qt'})


####
joanne['start_flight'] = ds_obs['drop']['launch_time'].resample(launch_time = "1D").first().dropna(dim='launch_time').rename({'launch_time':'time'})
joanne['end_flight']  = ds_obs['drop']['launch_time'].resample(launch_time = "1D").last().dropna(dim='launch_time').rename({'launch_time':'time'})
#everything in UTC
joanne['start_flight'] = joanne['start_flight'] + np.timedelta64(4,'h')
joanne['end_flight']   = joanne['end_flight']   + np.timedelta64(4,'h')
#%% import ERA5
print("Reading ERA5.") 
era5=xr.open_dataset(ifs_dir+'My_ds_ifs_ERA5.nc')

#%% Import scale separated fluxes 
da_scales      = xr.open_dataset('/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/DALES/scale_sep_allExp.nc')
da_scales_prof = xr.open_dataset('/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/DALES/scale_sep_prof_allExp.nc')

### From KLPS to resolution/size/scale of the filter
xsize = 150000
f_scales = np.zeros(len(da_scales.klp))
for k in range(len(da_scales.klp)):   
    if da_scales.klp[k] > 0:
        f_scales[k] = xsize/(da_scales.klp[k]*2).values  # m
    elif da_scales.klp[k] == 0:
        f_scales[k] = xsize

## HONNERT normalization of the filter scale
h   =  tmser.zi                 # boundary layer height 
hc  =  (tmser.zc_max-tmser.zb)  # cloud layer depth
hc  =  0
f_scales_norm = f_scales[:,None] / (h+hc).sel(time=da_scales.time).values[None,:]
da_scales['f_scales_norm'] = (('klp','time'),f_scales_norm)
     
#################
## normalize fluxes ##
da_scales_norm = (da_scales)/(da_scales.sel(klp=min(da_scales.klp).values))
#################
#%% Import organisatio metrics
da_org      = xr.open_dataset('/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/DALES/df_org_allExp.nc')
## exclude the first hour
da_org = da_org.isel(time=slice(11,-1))
da_org_norm = (da_org - da_org.min()) / (da_org.max() - da_org.min())
#%%                         
###############################################################################
rho = ls_surf['rho'].mean()
#%% convert xt, yt into lon, lat coordinates  

#%% SOME NEW VARIABLES
### for DALES
profiles = profiles.rename({'presh':'p'})
profiles['du_dz']   = profiles.u.differentiate('z')
profiles['dv_dz']   = profiles.v.differentiate('z')
profiles['duwt_dz'] = profiles.uwt.differentiate('z')
profiles['dvwt_dz'] = profiles.vwt.differentiate('z')
profiles['duwr_dz'] = profiles.uwr.differentiate('z')
profiles['dvwr_dz'] = profiles.vwr.differentiate('z')
profiles['duws_dz'] = profiles.uws.differentiate('z')
profiles['dvws_dz'] = profiles.vws.differentiate('z')
profiles['wspd']    = np.sqrt(profiles['u']**2 + profiles['v']**2)
profiles['th']      = profiles['thl'] + Lv / (cp * calc_exner(profiles['p'])) * profiles['ql']
profiles['T']       = calc_T(profiles['th'],profiles['p'])
profiles['K_dif_u'] = - profiles.uwt / profiles.du_dz
profiles['K_dif_v'] = - profiles.vwt / profiles.dv_dz
profiles['rain']    = profiles.rainrate/(24*28.94)
profiles['rain'].attrs["units"] = "mm/hour"
profiles['qt'] = profiles['qt'] *1000
profiles['qt'].attrs["units"] = "g/kg"

for var in ['u','v','thl','qt']:
    if var+'tendphyall' not in samptend:
        samptend[var+'tendphyall'] = samptend[var+'tendtotall'] - samptend[var+'tendlsall']


## for HARMONIE cy43
nudge['wspd']    = np.sqrt(nudge['u']**2    + nudge['v']**2)

            
### for HARMONIE cy43
if harm_3d:
    harm_clim_avg = harm_clim_avg.rename({'dtq_dyn':'dtqt_dyn','dtq_phy':'dtqt_phy'})
    harm_clim_avg['rho'] = calc_rho(harm_clim_avg['p'],harm_clim_avg['T'],harm_clim_avg['qt'])
    harm_clim_avg['wspd']= np.sqrt(harm_clim_avg['u']**2 + harm_clim_avg['v']**2)
    harm_clim_avg['th']  = calc_th(harm_clim_avg['T'],harm_clim_avg['p'])
    harm_clim_avg['thl'] = calc_thl(harm_clim_avg['th'],harm_clim_avg['ql'],harm_clim_avg['p'])
    for ii in ['phy','dyn']:
        harm_clim_avg['dtthl_'+ii]=calc_th(harm_clim_avg['dtT_'+ii],harm_clim_avg.p) - Lv / \
            (cp *calc_exner(harm_clim_avg.p)) * harm_clim_avg['dtqc_'+ii]
    
    ### DEACCUMULATE harmonie variables 
    ## most variables should already be deaccumulated
    # step = 3600 # output timestep [seconds]
    # for var in ['uflx_conv','uflx_turb','vflx_conv','vflx_turb']:
    #     harm_clim_avg[var] = (harm_clim_avg[var].diff('time')) * step**-1  # gives values per second
    # for var in list(harm_clim_avg.keys()):
    #     if 'dt' in var:
    #         harm_clim_avg[var] = (harm_clim_avg[var].diff('time')) * step**-1  # gives values per second

        
### for ERA5
if 'qt'not in (list(era5.keys())):
    era5 = era5.rename({'q':'qt'})
if 'T'not in (list(era5.keys())):
    era5 = era5.rename({'t':'T'})
if 'p'not in (list(era5.keys())):
    era5 = era5.rename({'P':'p'})
    era5['th']  = calc_th(era5['T'],era5['p'])
    # era5['thl'] = calc_thl(era5['th'],era5['ql'],era5['p'])
    
### for Observations
for ii in ds_obs.keys():
    if 'p'not in (list(ds_obs[ii].keys())):
        ds_obs[ii] = ds_obs[ii].rename({'pres':'p'})
    if 'qt'not in (list(ds_obs[ii].keys())):
        ds_obs[ii] = ds_obs[ii].rename({'q':'qt'})
    ds_obs[ii]['th']   = calc_th(ds_obs[ii]['T'],ds_obs[ii]['p'])
    # ds_obs[ii]['thl']  = calc_thl(ds_obs[ii]['th'],ds_obs[ii]['ql'],ds_obs[ii]['p'])
#convert to kelvin
ds_obs['drop']['T'] += 273.15

#%% Group by day
profiles_daily  = profiles.resample(time='D').mean('time')
tend_daily= (acc_time*samptend).resample(time='D').mean('time')
#%% Group by shape of some variables
####  K-mean clustering 

### cluster only based on the shape below certain km?
cluster_levels = [0, 3000]

# CLUSTER on dayly shear profiles
shear_cluster=KMeans(n_clusters=min(profiles_daily.time.size,3),random_state=0,n_init=15,max_iter=10000,\
                tol=10**-7).fit(profiles_daily['du_dz'].sel(z=slice(cluster_levels[0],cluster_levels[1])))
idx = np.argsort(shear_cluster.cluster_centers_.sum(axis=1))

# CLUSTER on dayly divergence
div_cluster=KMeans(n_clusters=min(profiles_daily.time.size,3),random_state=0,n_init=15,max_iter=10000,\
                tol=10**-7).fit(profiles_daily['duwt_dz'].sel(z=slice(cluster_levels[0],cluster_levels[1])))
idx = np.argsort(div_cluster.cluster_centers_.sum(axis=1))

# CLUSTER on large scale tendencies
LStend_cluster=KMeans(n_clusters=min(tend_daily.time.size,3),random_state=0,n_init=15,max_iter=10000,\
                tol=10**-7).fit(tend_daily['utendlsall'].sel(z=slice(cluster_levels[0],cluster_levels[1])))
idx = np.argsort(LStend_cluster.cluster_centers_.sum(axis=1))



profiles_daily['group_shear'] = (('time'), shear_cluster.labels_)
profiles_daily['group_div']   = (('time'), div_cluster.labels_)
tend_daily['group_LS']        = (('time'), LStend_cluster.labels_)
profiles_daily['group_LS']    = tend_daily.group_LS

profiles_daily_clusteres=profiles_daily.groupby('group_shear').mean()


#%%
### group by U variance 
# first quaritle
u2r_Q1 = np.quantile(profiles["u2r"].sel(z=slice(0,200)).mean('z'),0.25)
time_u2r_Q1 = profiles.where(profiles.sel(z=slice(0,200)).mean('z').u2r < u2r_Q1,drop=True).time
# time_u2r_Q1 = np.unique(time_u2r_Q1.dt.round('H').values.astype('datetime64[s]'))
# third quartile
u2r_Q3 = np.quantile(profiles["u2r"].sel(z=slice(0,200)).mean('z'),0.75)
time_u2r_Q3 = profiles.where(profiles.sel(z=slice(0,200)).mean('z').u2r > u2r_Q3,drop=True).time
# time_u2r_Q3 = np.unique(time_u2r_Q3.dt.round('H').values.astype('datetime64[s]'))
#%%                         PLOTTING
###############################################################################
print("Plotting.") 
#%% ## FIGURE 1 ##
fig, axs = plt.subplots(3,1,figsize=(19,19))
## panel a
profiles.cfrac.plot(y="z",cmap=plt.cm.Blues_r,vmax=0.1,vmin=0,ax=axs[0]\
                    ,cbar_kwargs=dict(orientation='horizontal',
                        pad=0.03, shrink=0.5,label='Fraction'))
ax2 = axs[0].twinx()
profiles.rain.sel(z=slice(0,50)).mean('z').rolling(time=6, center=True)\
    .mean().plot(x='time',ax=ax2,c='r',ls='-',label='Rain')
ax2.set_ylim([-0.01,4])
ax2.tick_params(axis='y', colors='red')
axs[0].set_title('Cloud fraction in DALES',fontsize = 22)
axs[0].set_ylabel(r'z ($m$)')
axs[0].set_ylim(height_lim)
ax2.set_ylabel(r'Rain rate [$W m^{-2}$]',color='r')
for tm in np.arange(srt_time, end_time):
    axs[0].axvline(x=tm,c='k')
    
## panel b
for level in [200]: # meters
    for var in ['wspd']:
        axs[1].plot(profiles.time,profiles[var].sel(z=level,method='nearest'),lw=3,c=col[3],label='DALES')
        if var in ds_obs['drop']:
            axs[1].scatter((ds_obs['drop'].launch_time  + np.timedelta64(4, 'h')).values,\
                    ds_obs['drop'].sel(Height=level,method='nearest').sel(launch_time=slice(srt_time,end_time))[var].values,c=col[2],alpha = 0.5,s=12,label='Dropsondes')
        if var in ds_obs['radio']:
            axs[1].scatter((ds_obs['radio'].launch_time + np.timedelta64(4, 'h')).values,\
                    ds_obs['radio'].sel(Height=level,method='nearest').sel(launch_time=slice(srt_time,end_time))[var].values,c=col[6],alpha = 0.5,s=12,label='Radiosondes')
        if harm_3d:
            axs[1].plot(harm_clim_avg.time,harm_clim_avg[var].sel(z=level,method='nearest'),lw=1.5,c=col[0],label='HARMONIE')
        if var in era5:
            axs[1].plot(era5.Date.sel(Date=slice(srt_time,end_time)),era5[var].sel(Height=level,method='nearest').\
                     sel(Date=slice(srt_time,end_time)).mean('Mypoint'),\
                     lw=1.5,ls='-',c=col[8], label='ERA5')
        # plt.xlabel('time')
        axs[1].set_ylabel(r'$m s^{-1}$')
        axs[1].set_title('Wind speed at '+str(level)+' m',size=22)
        axs[1].set_ylim([None,17])
        axs[1].legend(fontsize=15)


## panel c
layer = [0,750]
var = 'u'
rol = 10

dales_to_plot   = samptend.sel(z=slice(layer[0],layer[1])).mean('z')\
    .sel(time=slice(np.datetime64('2020-02-02'),np.datetime64('2020-02-11')))
h_clim_to_plot = harm_clim_avg.sel(z=slice(layer[0],layer[1])).mean('z')\
    .sel(time=slice(np.datetime64('2020-02-02'),np.datetime64('2020-02-11')))

## DALES 
(acc_time*dales_to_plot.rolling(time=rol*4).mean()[var+'tendtotall']).plot(c='r',label='DALES: Net',ax=axs[2])
(acc_time*dales_to_plot.rolling(time=rol*4).mean()[var+'tendlsall']).plot(c='k', label='DALES: LS',ax=axs[2])
(acc_time*dales_to_plot.rolling(time=rol*4).mean()[var+'tendphyall']).plot(c='g',label='DALES: Net - LS',ax=axs[2])

## HARMONIE cy43 clim
(acc_time*(h_clim_to_plot['dt'+var+'_dyn']+h_clim_to_plot['dt'+var+'_phy']).rolling(time=rol).mean()).plot(c='r',ls=':',label='HAR: Net',ax=axs[2])
(acc_time*h_clim_to_plot['dt'+var+'_dyn'].rolling(time=rol).mean()).plot(c='k',ls=':',label='HAR: Dyn',ax=axs[2])
(acc_time*h_clim_to_plot['dt'+var+'_phy'].rolling(time=rol).mean()).plot(c='g',ls=':',label='HAR: Phy',ax=axs[2])


axs[2].axhline(0,c='k',lw=0.5)
axs[2].set_title('Mean '+var+' tendency between '+str(layer[0])+' and '+str(layer[1])+' m',fontsize=22)
axs[2].legend(ncol=2,fontsize=15)
axs[2].set_ylabel(r'Tendency ($m s^{-1} hour^{-1}$)')
axs[2].set_ylim([-0.88,0.8])
axs[2].set_xlabel(None)

#####
for day in np.arange(srt_time,end_time):
    axs[0].axvline(x=day,c='k',lw=0.5)
    axs[1].axvline(x=day,c='k',lw=0.5)
    axs[2].axvline(x=day,c='k',lw=0.5)
axs[0].xaxis.set_visible(False) 
axs[1].xaxis.set_visible(False) 
axs[0].set_xlim([srt_time,end_time])
axs[1].set_xlim([srt_time,end_time])
axs[2].set_xlim([srt_time,end_time])
plt.tight_layout()
for n, ax in enumerate(axs):
    ax.text(0.95, 0.95, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=13)
plt.savefig(save_dir+'Figure1_tmser.pdf', bbox_inches="tight")
##################
#%% ## FIGURE 2 ##
cmap = matplotlib.cm.get_cmap('coolwarm')
rgba = 1/9
fig, axs = plt.subplots(1,4,figsize=(19,10))
for idx,var in enumerate(['u','v','thl','qt']):
    iteration = 0
    profiles[var].mean('time').plot(y='z',c='k',lw=4, label='Mean',ax=axs[idx])
    for day in np.unique(profiles.time.dt.day):
        iteration +=1
        profiles[var].sel(time='2020-02-'+str(day).zfill(2)).mean('time')\
            .plot(ax=axs[idx],y='z',c=cmap(rgba*iteration),lw=1.5,label='Feb-'+str(day).zfill(2))
   
    axs[idx].set_title(var,fontsize=22)        
    axs[idx].yaxis.set_visible(False) 
    axs[idx].set_ylim(height_lim)
    if var =='u':
        axs[idx].set_ylabel(r'z ($m$)')
        axs[idx].set_xlabel(r'$m s^{-1}$')
        axs[idx].set_xlim([-17,0])
        axs[idx].axvline(0,c='k',lw=0.5)
        axs[idx].yaxis.set_visible(True) 
    if var =='v':
        axs[idx].set_xlabel(r'$m s^{-1}$')
        axs[idx].set_xlim([-5.5,1.8])
        axs[idx].axvline(0,c='k',lw=0.5)
    if var =='thl':
        axs[idx].set_xlabel('$K$')
        axs[idx].set_xlim([295,321])
    if var =='qt':
        axs[idx].set_xlabel(r'$g kg^{-1}$')
        # axs[idx].set_xlim([300,330])
        axs[idx].legend(fontsize=15)
plt.tight_layout()
plt.savefig(save_dir+'Figure2_profiles.pdf', bbox_inches="tight")    
##################
#%% ## FIGURE 3 ##
bottom, top = 0.1, 0.9
left, right = 0.1, 0.8

fig, axs = plt.subplots(2,2,figsize=(22,12), gridspec_kw={'width_ratios': [1,6]})
fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, \
                    hspace=0.15, wspace=0.25)

for idx,var in enumerate(['uwt','vwt']):
    iteration = 0
    profiles[var].mean('time').plot(y='z',c='k',lw=4, label='Mean',ax=axs[idx,0])
    for day in np.unique(profiles.time.dt.day):
        iteration +=1
        profiles[var].sel(time='2020-02-'+str(day).zfill(2)).mean('time')\
            .plot(ax=axs[idx,0],y='z',c=cmap(rgba*iteration),lw=1.5,label='Feb-'+str(day).zfill(2))
   

    
    im = (profiles[var]).plot(y='z',vmax=0.1,vmin=-0.07,\
          cmap=cm.PiYG_r,norm=DivergingNorm(0),ax=axs[idx,1],\
              add_colorbar=True,cbar_kwargs={r'label':'$m^2 s^{-2}$'})

    axs[idx,0].yaxis.set_visible(True) 
    axs[idx,0].set_ylabel(r'z ($m$)')
    axs[idx,0].set_xlabel(r'$m^2 s^{-2}$')
    axs[idx,0].axvline(0,c='k',lw=0.5)
    axs[idx,1].yaxis.set_visible(False) 
    axs[idx,0].set_ylim(height_lim)
    axs[idx,1].set_ylim(height_lim)
    axs[idx,1].set_xlim([srt_time,end_time])
    for day in np.arange(srt_time,end_time):
        axs[idx,1].axvline(x=day,c='k',lw=0.5)

axs[0,1].set_title('Zonal momentum flux',fontsize=24)   
axs[1,1].set_title('Meridional momentum flux',fontsize=24)   
axs[0,1].xaxis.set_visible(False) 
axs[0,0].legend(fontsize=15)
axs[1,0].set_xlim([-0.045,0.045])
axs[1,1].set_xlabel(None)
for n, ax in enumerate(axs.flat):
    ax.text(0.9, 0.95, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=13)
# cbar_ax = fig.add_axes([0.9, 0.15, 0.01, 0.7])  # Left, bottom, width, height.
# cbar = fig.colorbar(im, cax=cbar_ax, extend='both', orientation='vertical')
# cbar.set_label(r'$m^2 s^{-1}$')
plt.tight_layout()
plt.savefig(save_dir+'Figure3_momFlux.pdf', bbox_inches="tight")  
##################
#%% ## FIGURE 4 ##
# snapshots 3x3 figure
### flower ### sugar ### gravel ###
#LWP  x    ###  x    ###   x    ###
#u    x    ###  x    ###   x    ###
#uw   x    ###  x    ###   x    ###

##################
#%% ## FIGURE 5 ##
for var in ['u_psfw_psf','v_psfw_psf']:
    fig, axs = plt.subplots(3,2,figsize=(12,12))
    for idcol in [0,1]:
        if idcol==0:
            # no normalisations
            da_toplot = da_scales_norm
        elif idcol==1:
            # normalise y axis
            da_toplot = da_scales_norm
        elif idcol==2:
            # normalise x and y axes
            da_toplot = da_scales_norm
            
        for idx,ih in enumerate(range(len(da_scales.height))):   
            iteration =0
            for day in ['02','03','04','05','06','07','08','09']:
                iteration +=1
                if idcol==1:
                    axs[idx,idcol].plot(da_scales['f_scales_norm'].\
                    resample(time='8h').mean('time').sel(time='2020-02-'+day),\
                      da_toplot.resample(time='8h').median('time')\
                          [var].isel(height=ih).sel(time='2020-02-'+day).T\
                        ,c=cmap(rgba*iteration),label=day)  
                else:
                    axs[idx,idcol].plot(f_scales/1000,\
                      da_toplot.resample(time='8h').median('time')\
                          [var].isel(height=ih).sel(time='2020-02-'+day).T\
                        ,c=cmap(rgba*iteration))   
            axs[idx,idcol].set_xscale('log')
            
            axs[idx,idcol].axhline(0,c='k',lw=0.5)
            axs[idx,0].axvline(2.5,c='k',lw=0.5)
            if idcol == 0:
                axs[idx,idcol].set_ylabel('At '+str(da_scales.height[ih].values)+' m')
            axs[idx,idcol].set_ylim([-0.1,1.1])
            axs[idx,1].yaxis.set_visible(False) 
    
    # axs[0,2].legend()
        
    axs[2,0].set_xlabel(r'Filter size ($km$)')
    axs[2,0].set_xlabel(r'Filter size $\Delta x$ ($km$)')
    axs[2,1].set_xlabel(r'Dimentionless $\frac{\Delta x}{h_b}$ ')
    axs[0,0].set_title('Standardized y axis',fontsize=21)  
    axs[0,1].set_title('Standardized y axis \n and dimentionless x axis',fontsize=21)  
    
    
    
    for n, ax in enumerate(axs.flat):
        ax.text(0.08, 0.9, string.ascii_uppercase[n], transform=ax.transAxes, 
                size=13)
    plt.tight_layout()
    plt.savefig(save_dir+'Figure5_'+var[0]+'_momFlux_spectra.pdf', bbox_inches="tight")  

##################
#%% ## FIGURE 7 ##
org_metric = 'iorg'
for var in ['u_psfw_psf','v_psfw_psf']:
    fig, axs = plt.subplots(3,2,figsize=(12,12))
    for idgroup in [0,1]:
        if idgroup==0:
        ### grouping by rain rate            
            time_g1 = profiles.where(profiles.rain.sel(z=slice(0,50)).mean('z').\
                              rolling(time=6, center=True).mean()<0.15,drop=True).time
            time_g3 = profiles.where(profiles.rain.sel(z=slice(0,50)).mean('z').\
                                     rolling(time=6, center=True).mean()>0.75,drop=True).time
            time_g2 = profiles.where(np.logical_not(profiles.time.\
                                    isin(xr.concat((time_g1,time_g3),'time'))),drop=True).time
            ########
        elif idgroup==1:
        ### grouping by organisation 
            time_g1 = profiles.where(da_org_norm[org_metric] < \
                                        da_org_norm[org_metric].quantile(0.25),drop=True).time
            time_g3 = profiles.where(da_org_norm[org_metric] > \
                                        da_org_norm[org_metric].quantile(0.75),drop=True).time
            time_g2 = profiles.where(np.logical_not(profiles.time.\
                                    isin(xr.concat((time_g1,time_g3),'time'))),drop=True).time
            
        ##
        time_g1 = time_g1.where(time_g1.isin(da_scales.time),drop=True)
        time_g2 = time_g2.where(time_g2.isin(da_scales.time),drop=True)
        time_g3 = time_g3.where(time_g3.isin(da_scales.time),drop=True)
            
        for idx,ih in enumerate(range(len(da_scales.height))):            
            axs[idx,idgroup].plot(f_scales/1000,da_scales_norm[var].isel(height=ih).sel(time=time_g1).median('time'),\
                      lw=2.5,c='orange',label='Group 1')
            axs[idx,idgroup].plot(f_scales/1000,da_scales_norm[var].isel(height=ih).sel(time=time_g2).median('time'),\
                      lw=2.5,c='b',label='Group 2')
            axs[idx,idgroup].plot(f_scales/1000,da_scales_norm[var].isel(height=ih).sel(time=time_g3).median('time'),\
                      lw=2.5,c='green',label='Group 3')
                
        # plt.ylim([-0.02,+0.02])
            
            axs[idx,idgroup].set_xscale('log')
            axs[idx,idgroup].axhline(0,c='k',lw=0.5)
            axs[idx,idgroup].axvline(2.5,c='k',lw=0.5)
            
            if idgroup == 0:
                axs[idx,idgroup].set_ylabel('At '+str(da_scales.height[ih].values)+' m')
        axs[0,idgroup].legend()
    axs[0,0].set_title('Grouping by rain rate',fontsize=24)  
    axs[0,1].set_title('Grouping by organisation',fontsize=24)  
    
    for n, ax in enumerate(axs.flat):
        ax.text(0.08, 0.9, string.ascii_uppercase[n], transform=ax.transAxes, 
                size=13)
    plt.tight_layout()
    plt.savefig(save_dir+'Figure7_'+var[0]+'_spectra_groups.pdf', bbox_inches="tight")  

##################
#%% ## FIGURE 6 ##
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


plt.legend()
plt.title('Surface rain rate',fontsize=22)
for ii in np.arange(srt_time, end_time):
    plt.axvline(x=ii,c='k')

plt.xlabel(None)
##################
#%% ## FIGURE 6 ##
# The box extends from the lower to upper quartile values of the data,
# with a line at the median. 
# The whiskers extend from the box to show the range of the data
ih = 650
fig, axs = plt.subplots(2,1,figsize=(12,7))
for idx, var in enumerate(['u_psfw_psf','v_psfw_psf']):
    iteration=-0.4
    for day in ['02','03','04','05','06','07','08','09']:
        iteration +=0.4
        for hour in ['00','12']:
            iteration +=0.3
            axs[idx].boxplot(da_scales_norm[var].sel(time=slice('2020-02-'+day+'T'+hour,\
                                    '2020-02-'+day+'T'+str(int(hour)+11)+':55'))\
                        .sel(height=ih,klp=30,method='nearest').values,\
                            positions=[round(iteration,1)],\
                    whis=2,showfliers=False,showmeans=True,meanline=False,widths=0.25)
    
    axs[idx].axhline(0,c='k',lw=0.5)
axs[0].set_ylim([-0,1.4])
axs[1].set_ylim([-0.45,2.1])
##################
#%% PLOT ORG METRICS
for var in da_org_norm:
    plt.figure(figsize=(9,5))
    da_org_norm[var].plot()
    
    da_org_norm.where(da_org_norm['iorg'] <= da_org_norm['iorg'].quantile(0.25),drop=True).time

#%%
for group_by in ['groups']:    
    ## momentum fluxes
    plt.figure(figsize=(6,9))
    plt.suptitle('Momentum fluxes')
    for idx,var in enumerate(['uw','vw']):
        for ii in ['t',]:
            if ii == 't':
                hrs_to_plot[var+ii].mean('time').plot(y='z',c=col[idx*3],ls='-',label=var+' tot')
                if group_by == 'flights':
                    profiles.where(profiles.time.dt.strftime('%Y-%m-%d')\
                                    .isin(joanne.time.dt.strftime('%Y-%m-%d')),drop=True)\
                        [var+ii].mean('time').plot(y='z',ls='--',c=col[idx*3],lw=2,label='Flights')
                if group_by =='groups':
                    for key in profiles_daily_clusteres['group_shear'].values:
                        profiles_daily_clusteres.sel(group_shear=key)[var+ii].plot(y='z',c=col[idx*3+key],lw=1,label='Group '+str(key))
                    title = 'Groups by wind shear'
                    plt.title(title)
                    
                if group_by == 'day_night':
                    hrs_to_plot_day[var+ii].mean('time').plot(y='z',c=col[idx*3+1],lw=1, label=str(day_interval)+' UTC tot')
                    hrs_to_plot_night[var+ii].mean('time').plot(y='z',c=col[idx*3+2],lw=1, label=str(night_interval)+' UTC tot')
            else:
                hrs_to_plot[var+ii].mean('time').plot(y='z',c=col[idx*3],ls='--',label=var+' resolved')
                if day_night:
                    hrs_to_plot_day[var+ii].mean('time').plot(y='z',c=col[idx*3+1],ls='--',lw=1, label=str(day_interval)+' UTC resolved')
                    hrs_to_plot_night[var+ii].mean('time').plot(y='z',c=col[idx*3+2],ls='--',lw=1, label=str(night_interval)+' UTC resolved')
    plt.legend()
    plt.xlabel('Momentum flux (m2/s2)')
    plt.axvline(0,c='k',lw=0.5)
    plt.ylim(height_lim)
    # plt.savefig(save_dir+'mean_momflux.pdf')
        
    ## counter gradient fluxes
    plt.figure(figsize=(6,9))
    plt.suptitle('Counter gradient transport')
    for idx,var in enumerate(['u','v']):
        (hrs_to_plot[var+'wt'] * hrs_to_plot['d'+var+'_dz']).mean('time').\
            plot(y='z',c=col[idx*3],lw=2,label=var+'w d'+var+'_dz')
        # if group_by == 'flights':
        #     (profiles[var+'wt'] * profiles['d'+var+'_dz']).where(profiles.time.dt.strftime('%Y-%m-%d')\
        #                 .isin(joanne.time.dt.strftime('%Y-%m-%d')),drop=True)\
        #         .mean('time').plot(y='z',ls='--',c=col[idx*3],lw=2,label='Flights')
        # if group_by =='groups':
        #     for key in profiles_daily_clusteres['group_shear'].values:
        #         (profiles_daily_clusteres[var+'wt'] * profiles_daily_clusteres['d'+var+'_dz'])\
        #             .sel(group_shear=key).plot(y='z',c=col[idx*3+key],lw=1,label='Group '+str(key))
        #         plt.title('Groups by wind shear')

        # if group_by == 'day_night':
        #     (hrs_to_plot_day[var+'wt'] * hrs_to_plot_day['d'+var+'_dz']).mean('time').\
        #         plot(y='z',c=col[idx*3+1],lw=1,label=str(day_interval)+' UTC')
        #     (hrs_to_plot_night[var+'wt'] * hrs_to_plot_night['d'+var+'_dz']).mean('time').\
        #         plot(y='z',c=col[idx*3+2],lw=1,label=str(night_interval)+' UTC')
    plt.legend()
    plt.axvline(0,c='k',lw=0.5)
    plt.ylim(height_lim)
    plt.xlim([-0.00012,None])
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    # plt.savefig(save_dir+'mean_gradTransp.pdf')
    
    ## Momentum flux convergence
    plt.figure(figsize=(6,9))
    plt.suptitle('Momentum flux convergence')
    (-1*hrs_to_plot['duwt_dz']).mean('time').plot(y='z',c='b',ls='-',label='- duw_dz')
    (-1*hrs_to_plot['dvwt_dz']).mean('time').plot(y='z',c='r',ls='-',label='- dvw_dz')
    # (-1*hrs_to_plot['duwr_dz']).mean('time').plot(y='z',c='b',ls='--',label='- duwr_dz')
    # (-1*hrs_to_plot['dvwr_dz']).mean('time').plot(y='z',c='r',ls='--',label='- dvwr_dz')
    # (-1*hrs_to_plot['duws_dz']).mean('time').plot(y='z',c='b',ls=':',label='- duws_dz')
    # (-1*hrs_to_plot['dvws_dz']).mean('time').plot(y='z',c='r',ls=':',label='- dvws_dz')
    plt.title(title)
    plt.legend()
    plt.axvline(0,c='k',lw=0.5)
    plt.ylim(height_lim)
    plt.xlim([-0.0001,0.0002])
    plt.xlabel('m/s2')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    # plt.savefig(save_dir+'mean_momfluxconv.pdf')
    
#%% momentum fluxes grouped by u variance 
days = ['2020-02-02','2020-02-03T23:50']
hrs_to_plot = profiles.where(profiles.time.dt.strftime('%Y-%m-%d').isin(days),drop=True)
fig, axs = plt.subplots(1,2,figsize=(12,12))
plt.suptitle('Momentum fluxes')
for idx,var in enumerate(['u','v']):
    axs[idx].plot(profiles[var+'wt'].sel(time=slice(days[0],days[1])).mean('time'),profiles['z'],\
                  c='r',ls='-',lw=3,label='Tot DALES')
    # axs[idx].plot(profiles[var+'wt'].sel(time=time_u2r_Q1.sel(time=slice(days[0],days[1]))).mean('time'),profiles['z'],\
    #               c='b',ls='-',label='Low u variance')
    # axs[idx].plot(profiles[var+'wt'].sel(time=time_u2r_Q3.sel(time=slice(days[0],days[1]))).mean('time'),profiles['z'],\
    #               c='r',ls='-',label='High u variance')
        
    ### from HARMONIE
    axs[idx].plot(((harm_clim_avg[var+'flx_conv']+harm_clim_avg[var+'flx_turb']) + \
          dl_geo.interp(z=harm_clim_avg['z'])[var+'p_wp'])\
              .sel(time=slice(days[0],days[1])).mean('time'),harm_clim_avg.z,\
              ls='--',c='r',lw=3,label='Tot HARMONIE')
        
    axs[idx].plot(harm_clim_avg[var+'flx_conv']\
              .sel(time=slice(days[0],days[1])).mean('time'),harm_clim_avg.z,\
              ls='--',c='b',lw=2,label='HARMONIE conv')
    axs[idx].plot(harm_clim_avg[var+'flx_turb']\
              .sel(time=slice(days[0],days[1])).mean('time'),harm_clim_avg.z,\
              ls='--',c='c',lw=2,label='HARMONIE turb')
    axs[idx].plot((dl_geo.interp(z=harm_clim_avg['z'])[var+'p_wp'])\
              .sel(time=slice(days[0],days[1])).mean('time'),harm_clim_avg.z,\
              ls='--',c='k',lw=2,label='HARMONIE resolved')
        
        
    axs[idx].axvline(0,c='k',lw=0.5)
    axs[0].legend(fontsize=15)
    axs[idx].set_xlabel('Momentum flux (m2/s2)')
    axs[idx].set_title(var +'flux',fontsize=24)

    axs[idx].set_ylim(height_lim)
# plt.savefig(save_dir+'mean_momflux_byu2r.pdf')

         # !!!!!


#%%  TIMESERIES

## momentum fluxes
for var in ['uwt']:
    fig, axs = plt.subplots(figsize=(19,5))
    # plt.suptitle('Meridional momentum flux and near-surface variance')
    # (profiles[var]*xr.ufuncs.sign(profiles[var[0]])).plot(y='z',vmax=0.1)
    (profiles[var]).plot(y='z',vmax=0.1,vmin=-0.07,cmap=cm.PiYG_r,norm=DivergingNorm(0))
    plt.ylim(height_lim)
    plt.xlim([srt_time,end_time])
    # plt.xlim(['2020-02-02T22','2020-02-05T02'])
    (tmser.zc_max.rolling(time=30, center=True).mean()-300).plot(c='b',ls='-',lw=1)
    for ii in np.arange(srt_time, end_time):
        plt.axvline(x=ii,c='k')

    
        
    # ax2 = axs.twinx()
    # profiles[var[0]+'2r'].sel(z=slice(0,200)).mean('z').plot(x='time',ax=ax2,c='k',label = r'$\sigma^2$'+var[0])
    # ax2.set_ylim([-0,13])
    # profiles.v2r.sel(z=slice(0,200)).mean('z').plot(x='time',ax=ax2,c='k')
    # profiles.rainrate.sel(z=slice(0,200)).mean('z').plot(x='time',ax=axs,c='b',label='Rain')
    # ax2.set_ylabel('$\sigma^2$'+var[0]+' [$m^2 / s^2$]')
    axs.set_ylabel('z [m]')
    axs.set_xlabel(None)
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
    # plt.savefig(save_dir+'poster_DALES_tmser_'+var+'_2days.pdf', bbox_inches="tight")
    
    

#%%
dl_geo = xr.open_mfdataset('/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/HARMONIE/cy43_clim/my_3d_harm_clim_lev_.nc',combine='by_coords')
step = 3600 # output timestep [seconds]
for var in ['uflx_conv','uflx_turb']:
    dl_geo[var] = (dl_geo[var].diff('time')) * step**-1  # gives values per second
## HARMONIE
for var in ['u']:
    # fig, axs = plt.subplots(figsize=(19,5))
    # (harm_clim_avg[var+'flx_conv']+harm_clim_avg[var+'flx_turb']).\
    #     isel(time=slice(37,48)).sel(z=slice(0,4500)).\
    #     plot(cmap=cm.PiYG_r,x='time',\
    #                   vmax=0.03,vmin=-0.015,norm=DivergingNorm(0))
    # # for ii in np.arange(srt_time, end_time):
    # #     plt.axvline(x=ii,c='k')
    # plt.suptitle('Param '+var+' momentum flux form HARMONIE')
   
    # plt.figure(figsize=(19,5))
    # dl_geo.sel(z=slice(0,4500))[var+'p_wp'].plot(y='z',\
    #                           cmap=cm.PiYG_r,\
    #                           vmax=None,vmin=None,norm=DivergingNorm(0))
    # plt.suptitle('Resolved '+var+' momentum flux form HARMONIE')
    # for ii in np.arange(srt_time, end_time):
    #     plt.axvline(x=ii,c='k')
    # plt.xlim(['2020-02-02T22','2020-02-05T02'])
    
    plt.figure(figsize=(19,5))
    ((harm_clim_avg[var+'flx_conv']+harm_clim_avg[var+'flx_turb']) + \
         dl_geo.interp(z=harm_clim_avg['z'])[var+'p_wp']).sel(z=slice(0,4500))\
   .plot(y='z',\
                              cmap=cm.PiYG_r,\
                             vmax=0.1,vmin=-0.07,norm=DivergingNorm(0),cbar_kwargs={'label': '$m^2/s^2$'})
    # plt.suptitle('Total '+var+' momentum flux from HARMONIE')
    plt.plot(dl_geo.time,cl_max[:len(dl_geo.time)],c='b',ls='-',lw=1)
    plt.ylim(height_lim)

    for ii in np.arange(srt_time, end_time):
        plt.axvline(x=ii,c='k')
    # plt.xlim(['2020-02-02T22','2020-02-05T02'])
    plt.savefig(save_dir+'poster_HARM_tmser_'+var+'w_2days.pdf', bbox_inches="tight")

#%% CLOUD LAYER timeseries
exp_prof  = profiles
exp_tmser = tmser

plt.figure(figsize=(19,5))
exp_prof.cfrac.plot(y="z",cmap=plt.cm.Blues_r,vmax=0.005,vmin=0)
profiles.rainrate.sel(z=slice(0,300)).mean('z').plot(x='time',ax=axs,alpha=0.7,ls='--',label='Rain')
# exp_tmser.zc_max.rolling(time=30, center=True).mean().plot(c='r',ls='-')
# exp_tmser.zc_av.plot(c='r',ls='--')
# exp_tmser.zb.plot(c='k')
# exp_tmser.zi.plot(c='k',ls='--')
plt.ylim([0,8000])
plt.title('DALES',fontsize = 21)
plt.ylabel('z [m]')
plt.xlabel('')
# plt.xlim(['2020-02-03','2020-02-04'])
# plt.axvline(x=np.datetime64('2020-02-03T12'),c='k')
for tm in np.arange(srt_time, end_time):
    plt.axvline(x=tm,c='k')
# plt.savefig(save_dir+'tmser_clfrac_DALES.pdf', bbox_inches="tight")
    
# plt.figure(figsize=(19,5))
# harm_clim_avg.cl.sel(time='2020-02-03').plot(x='time',cmap=plt.cm.Blues_r,vmax=0.05,vmin=0)
# exp_tmser.zc_max.rolling(time=30, center=True).mean().plot(c='r',ls='-')
# exp_tmser.zc_av.plot(c='r',ls='--')
# plt.ylim([0,8000])
# plt.xlim(['2020-02-03','2020-02-04'])
# plt.title('HARMONIE',fontsize = 21)
# plt.ylabel('z [m]')
# plt.xlabel('')
# plt.axvline(x=np.datetime64('2020-02-03T12'),c='k')
# for tm in np.arange(srt_time, end_time):
#     plt.axvline(x=tm,c='k')
# plt.savefig(save_dir+'tmser_clfrac_HARMONIE.pdf', bbox_inches="tight")
#%%
## Difference HARMONIE hind - DALES
exp_prof = profiles
exp_tend = samptend

####################
############
## check LS tendencies 
# exp_tend['qttendlsall'].sel(time=harm_clim_avg.sel(time=slice('2020-02-02T01','2020-02-02T11')).time).mean('time').plot(y='z')
# harm_clim_avg.sel(time=slice('2020-02-02T01','2020-02-02T11'))['dtqt_dyn'].mean('time').plot(y='z')

# var = 'u'
# (harm_clim_avg.sel(time=slice('2020-02-02T01','2020-02-02T11')).interp(z=exp_tend.z)['dt'+var+'_dyn'] - \
# exp_tend[var+'tendlsall'].sel(time=harm_clim_avg.sel(time=slice('2020-02-02T01','2020-02-02T11')).time)).plot(y='z')
############
####################
    
for ii in ['thl','qt','ql','wspd']:
# for ii in ['qttendlsall','utendlsall']:
    if ii == 'qt': 
        vmax = 0.005
        unit = 'kg/kg'
    elif ii == 'T' or ii == 'thl' or ii=='th':
        vmax = 4
        unit = 'K'
    elif ii=='u' or ii=='v': 
        vmax = 1
        unit ='m/s'
    else:
        vmax=None
        unit = '?'
    for mod in ['clim',]:
        plt.figure(figsize=(19,5))
        if mod == 'hind':
            if 'tend' in ii:
                (acc_time*((harm_hind_avg['300']['dt'+ii[0:3]+'_dyn'])\
                 .interp(z=exp_tend.z)-(exp_tend['thltendlsall']))).plot(x="time",cmap='seismic') 
            else:
                (harm_hind_avg['300'][ii].interp(z=exp_prof.z)-exp_prof[ii]).plot(x="time",cmap='seismic',vmax=vmax)
        
        elif mod == 'clim':
            if 'tend' in ii:
                ((harm_clim_avg['dt'+ii[0:-9]+'_dyn'])\
                  .interp(z=exp_tend.z)-exp_tend[ii]).plot(x="time",cmap='seismic') 
                    
                # ((harm_clim_avg['dt'+ii[0:-9]+'_dyn'])\
                #  .interp(z=ls_flux.z,time=ls_flux.time)-ls_flux['d'+ii[0:-9]+'dt']).plot(x="time",cmap='seismic') 
            else:
                (harm_clim_avg[ii].interp(z=exp_prof.z)-exp_prof[ii]).plot(x="time",cmap='seismic',vmax=vmax)
        plt.title(' HAMRONIE '+mod+' - DALES ('+unit+')',size=20)
        plt.ylim([0,8000])
        for tm in np.arange(srt_time, end_time):
            plt.axvline(x=tm,c='k')
    #%%    
## Difference HARMONIE hind - HARMONIE clim
for ii in ['u','v','T','qt','dtT_dyn','dtqt_dyn','dtu_dyn','dtv_dyn']:
    if ii == 'qt': 
        vmax = 0.005
        unit = 'kg/kg'
        acc_time = 1
    elif ii == 'dtqt_dyn':
        vmax = 0.001
        unit = 'kg/kg /hour'
        acc_time = 3600*1
    elif ii == 'T':
        vmax = 4
        unit = 'K'
        acc_time = 1
    elif ii == 'dtT_dyn':
        vmax = 1
        unit = 'K /hour'
        acc_time = 3600*1
    elif ii == 'u' or ii =='v': 
        vmax = 9
        unit ='m/s'
        acc_time = 1
    elif ii == 'dtu_dyn' or ii =='dtv_dyn':
        vmax = 1.5
        unit = 'm/s /hour'
        acc_time = 3600*1
    plt.figure(figsize=(19,5))
    (acc_time*(harm_hind_avg['300'][ii]-harm_clim_avg[ii].interp(z=harm_hind_avg['300'].z))).plot(x='time',cmap='seismic',vmax=vmax)
    # (nudge_clim[ii]-harm_clim_avg[ii].interp(z=nudge.z)).plot(x="time",cmap='seismic')
    plt.title('HAMRONIE hind - HARMONIE clim ('+unit+')',size=20)
    plt.ylim([0,8000])
    for tm in np.arange(srt_time, end_time):
        plt.axvline(x=tm,c='k')        
        
#%% CONTOUR for VARIABLES (any model)
# profiles['uwr_sign']=profiles.uwr*(-1* np.sign(profiles.u))
# profiles['vwr_sign']=profiles.vwr*(-1* np.sign(profiles.v))

for var in ['u','qt','uwr']:
    unit = 1
    vmin = None
    vmax = None
    if var == 'qt': 
        unit = 1000
        vmin = 0.5
        vmax = 15
    elif var == 'T':
        vmin = 282
        vmax = 300
    elif var == 'v':
        vmin = -6
        vmax = 3
    elif var == 'u':
        vmin = -14
        vmax = 1
    elif 'uw' in var:
        vmin = -0.06
        vmax = 0.08

    plt.figure(figsize=(19,5))
    (unit*profiles[var]).plot(x="time",vmin=vmin,vmax=vmax,cmap='bwr')
    # (unit*harm_hind_avg['300'][var]).plot(x="time",vmin=vmin,vmax=vmax)
    # (unit*harm_clim_avg[var]).plot(x="time",vmin=vmin,vmax=vmax)
    plt.ylim([0,5000])
    plt.title('Resolved zonal momentum flux '+var, size=20)
    for ii in np.arange(srt_time, end_time):
            plt.axvline(x=ii,c='k',lw=0.5)

        
       #%% 
### TENDENCIES 
for ii in ['qt', 'thl','u','v']:
    if ii == 'qt': 
        unit = 1000 
        vmax = 1
    elif ii=='thl': 
        unit = 1
        vmax = 1
    else: 
        unit = 1
        vmax = 1.5
    plt.figure(figsize=(19,5))
    (unit*3600*ls_flux['d'+ii+'dt']).plot(x='time',vmax=vmax)
    # (unit*3600*harm_hind_avg['300']['dt'+ii+'_dyn']).plot(x='time',vmax=vmax)
    # plt.gca().set_prop_cycle(None)
    # (unit*3600*24*samptend.qttendlsall).plot(x='time')
    plt.ylim([0,6000])
    plt.axvline(x=0,c='k',lw=0.5)
    plt.ylabel('m')
    if   ii == 'qt': plt.title('dqt_dt ($g/kg /hour$)',size=20)
    elif ii == 'thl': plt.title('dthl_dt ($K /hour$)',size=20)
    else : plt.title('d'+ii+'_dt ($m/s /hour$)',size=20)
    
    for ii in np.arange(srt_time, end_time):
            plt.axvline(x=ii,c='k')

#%%
### OBS AND MODELS TOGETHER
if comp_observations:
    for level in [200]: # meters
        for var in ['wspd']:
            ## Temperature    
            plt.figure(figsize=(15,6))
            plt.plot(profiles.time,profiles[var].sel(z=level,method='nearest'),lw=3,c=col[3],label='DALES')
            if comp_experiments:
                plt.plot(prof_isurf5.time,prof_isurf5[var].sel(z=level,method='nearest'),c=col[5],label='isurf5')
                # plt.plot(profiles_clim.time,profiles_clim[var].sel(z=level,method='nearest'),c=col[5],label='DALES clim')
            # plt.plot(harm_hind_avg['300'].time,harm_hind_avg['300'][var].sel(z=level,method='nearest'),c=col[0],label='HARMONIE cy40')
            if var in ds_obs['drop']:
                plt.scatter((ds_obs['drop'].launch_time  + np.timedelta64(4, 'h')).values,\
                        ds_obs['drop'].sel(Height=level,method='nearest').sel(launch_time=slice(srt_time,end_time))[var].values,c=col[2],alpha = 0.5,s=12,label='Dropsondes')
            if var in ds_obs['radio']:
                plt.scatter((ds_obs['radio'].launch_time + np.timedelta64(4, 'h')).values,\
                        ds_obs['radio'].sel(Height=level,method='nearest').sel(launch_time=slice(srt_time,end_time))[var].values,c=col[6],alpha = 0.5,s=12,label='Radiosondes')
            if harm_3d:
                plt.plot(harm_clim_avg.time,harm_clim_avg[var].sel(z=level,method='nearest'),lw=1.5,c=col[0],label='HARMONIE')
            if var in era5:
                plt.plot(era5.Date.sel(Date=slice(srt_time,end_time)),era5[var].sel(Height=level,method='nearest').\
                         sel(Date=slice(srt_time,end_time)).mean('Mypoint'),\
                         lw=1.5,ls='-',c=col[8], label='ERA5')
            # plt.xlabel('time')
            plt.ylabel('z [m/s]')
            plt.title('Wind speed at '+str(level)+' m',size=20)
            plt.xlim(temp_hrs)
            plt.ylim([None,17])
            plt.axvspan(srt_time,srt_time + np.timedelta64(2, 'h'), alpha=0.2, color='grey')
            plt.legend(fontsize=15)
            for day in np.arange(srt_time,end_time):
                plt.axvline(x=day,c='k',lw=0.5)
        plt.savefig(save_dir+'tmser_'+var+'_'+str(level)+'m.pdf')
          
#%%
    ### SURFACE LATENT HEAT FLUX 
    plt.figure(figsize=(15,6))
    plt.plot(tmser.time, ls_surf['rho'].mean() * tmser.wq * Lv,lw=2.5,c=col[3],label='DALES')
    if comp_experiments:
        plt.plot(tmser_isurf5.time, ls_surf['rho'].mean() * tmser_isurf5.wq * Lv,c=col[5],lw=0.7,label='DALES exp')
    # harm_hind_avg['300'].LE.plot()
    harm_clim_avg.hfls.mean(dim=['x','y']).plot(lw=2.5,c=col[0],label='HARMONIE cy43')
    xr.plot.scatter(Meteor,'time','LHF_bulk_mast',alpha = 0.6,s=12,c=col[2],label='Meteor')
    xr.plot.scatter(Meteor,'time','LHF_EC_mast',alpha = 0.4,s=12,label='EC')
    plt.xlabel('time')
    plt.ylabel('LH (W/m2)')
    plt.title('Surface latent heat flux',size=20)
    plt.xlim(temp_hrs)
    plt.axvspan(srt_time,srt_time + np.timedelta64(2, 'h'), alpha=0.2, color='grey')
    plt.legend(fontsize=15)
    for day in np.arange(srt_time,end_time):
        plt.axvline(x=day,c='k',lw=0.5)
    plt.savefig(save_dir+'LatentHeat_srf.pdf', bbox_inches="tight")
        
    ### SURFACE SENSIBLE HEAT FLUX
    plt.figure(figsize=(15,6))
    plt.plot(tmser.time, rho * tmser.wtheta * cp,lw=2.5,c=col[3],label='DALES')
    if comp_experiments:
        plt.plot(tmser_isurf5.time, rho * tmser_isurf5.wtheta * cp,c=col[5],lw=0.9,label='DALES exp')
    # harm_hind_avg['300'].H.plot(c=col[0],lw=2,label='HARMONIE_cy40 hind')
    harm_clim_avg.hfss.mean(dim=['x','y']).plot(lw=2.5,c=col[0],label='HARMONIE cy43')
    xr.plot.scatter(Meteor,'time','SHF_bulk_mast',alpha = 0.6,s=12,c=col[2],label='Meteor')
    xr.plot.scatter(Meteor,'time','SHF_EC_mast',alpha = 0.4,s=12,label='EC')
    plt.xlabel('time')
    plt.ylabel('SH ($W/m^2$)')
    plt.title('Surface sensible heat flux',size=20)
    plt.xlim(temp_hrs)
    plt.axvspan(srt_time,srt_time + np.timedelta64(2, 'h'), alpha=0.2, color='grey')
    plt.legend(fontsize=15)
    for day in np.arange(srt_time,end_time):
        plt.axvline(x=day,c='k',lw=0.5)
    plt.savefig(save_dir+'SensHeat_srf.pdf', bbox_inches="tight")
        
    # flux profiles
    # plt.figure(figsize=(10,9))
    # (profiles.rhof * profiles.wqtt * Lv).plot(y='z',vmin=0,vmax=+500, cmap='coolwarm')
    # plt.ylim([0,4000])
    # plt.title('Latent heat flux',size=20)
    # plt.figure(figsize=(10,9))
    # # sensible heat flux uses theta not thetaV
    # (profiles.rhof * profiles.wthvt * cp).plot(y='z',vmin=0,vmax=+40, cmap='coolwarm')
    # plt.ylim([0,4000])
    # plt.title('Sensible heat flux',size=20)
    
        
#%% DAILY MEANS
## momentum fluxes
# plt.figure(figsize=(19,5))
# profiles_daily['uwt'].plot(y='z',vmin=-0.1)
# plt.ylim(height_lim)
# plt.savefig(save_dir+'daily_momflux.pdf')

#%% DAILY MEANS clustered
group_by = 'group_LS'

for ii in ['u','v','du_dz','vwt','duwt_dz','K_dif_v']:
    plt.figure(figsize=(5,9))
    for key, group in profiles_daily.groupby(group_by):
        if 'K_dif' in ii:
            group[ii].rolling(z=5).mean().mean('time').plot(y='z',c=col[key*3],lw=3,label='Group '+str(key))
            group[ii].rolling(z=5).mean().plot.line(y='z',c=col[key*3],lw=0.7,alpha=0.3,add_legend=False)
        else:
            group[ii].mean('time').plot(y='z',c=col[key*3],lw=3,label='Group '+str(key))
            group[ii].plot.line(y='z',c=col[key*3],lw=0.7,alpha=0.3,add_legend=False)
    profiles[ii].rolling(z=5).mean().mean('time').plot(y='z',c='k',lw=3,label='Mean')
    plt.legend()
    plt.title(group_by,size=20)
    plt.ylim(height_lim)
    plt.axvline(0,c='k',lw=0.5)
    plt.xlabel(ii)
    if 'K_dif' in ii :
        plt.xlim([-100,100])
        plt.ylim([0,3000])
        if 'u' in ii:
            plt.xlabel('$K_u$ [$m^2 s^{-1}$]')
        elif 'v' in ii:
            plt.xlabel('$K_v$ [$m^2 s^{-1}$]')
            plt.legend().remove()
    elif ii=='u':
        plt.xlim([-16,4])
    elif ii=='v':
        plt.xlim([-10,4])
        plt.legend().remove()
    # plt.savefig(save_dir+'Kmean_'+ii+'.pdf')

#%%
for ii in ['utendtotall','vtendlsall','utendrestall']:
    plt.figure(figsize=(5,9))
    for key, group in tend_daily.groupby('group_LS'):
        if ii == 'utendrestall':
            (group['utendtotall']-group['utendlsall']).mean('time').plot(y='z',c=col[key*3],lw=3,label='Group '+str(key))
            (group['utendtotall']-group['utendlsall']).plot.line(y='z',c=col[key*3],lw=0.7,alpha=0.3,add_legend=False)
        else: 
            group[ii].mean('time').plot(y='z',c=col[key*3],lw=3,label='Group '+str(key))
            group[ii].plot.line(y='z',c=col[key*3],lw=0.7,alpha=0.5,add_legend=False)
    plt.legend()
    plt.title('group_LS',size=20)
    plt.ylim(height_lim)
    plt.axvline(0,c='k',lw=0.5)
    # plt.savefig(save_dir+'Kmean_'+ii+'.pdf')

#%% TENDENCIES
#%% CORRELATION PLOTS of tendencies 
layer = [0,750]
var = 'u'
color_hours = False
model = 'dales'

if model == 'harm':
    y  = 3600*(harm_clim_avg['dt'+var+'_dyn']+harm_clim_avg['dt'+var+'_phy']).sel(z=slice(layer[0],layer[1])).mean('z')
    x  = 3600*(harm_clim_avg).sel(z=slice(layer[0],layer[1])).mean('z')['dt'+var+'_dyn']
    y1 = 3600*(harm_clim_avg['dt'+var+'_dyn']+harm_clim_avg['dt'+var+'_phy']).resample(time='D').mean('time').sel(z=slice(layer[0],layer[1])).mean('z')
    x1 = 3600*(harm_clim_avg.resample(time='D').mean('time')).sel(z=slice(layer[0],layer[1])).mean('z')['dt'+var+'_dyn']
elif model =='dales':        
    y  = (3600*samptend).sel(z=slice(layer[0],layer[1])).mean('z')[var+'tendtotall']
    x  = (3600*samptend).sel(z=slice(layer[0],layer[1])).mean('z')[var+'tendlsall']
    y1 = tend_daily.sel(z=slice(layer[0],layer[1])).mean('z')[var+'tendtotall']
    x1 = tend_daily.sel(z=slice(layer[0],layer[1])).mean('z')[var+'tendlsall']

# y  = profiles.sel(z=slice(layer[0],layer[1])).mean('z')['duwt_dz']
# y1 = profiles_daily.sel(z=slice(layer[0],layer[1])).mean('z')['duwt_dz']

#### FIGURE 1
##################
plt.figure()
if color_hours:
    colors = cm.hsv(np.linspace(0, 1, 24))
    for hour in range(0,24):
        plt.scatter((3600*samptend[var+'tendlsall']).sel(z=slice(layer[0],layer[1])).mean('z').where(samptend.time.dt.hour==hour,drop=True),\
                  (3600*samptend[var+'tendtotall']).sel(z=slice(layer[0],layer[1])).mean('z').where(samptend.time.dt.hour==hour,drop=True),\
                  c=colors[hour],alpha=0.5,s=7)
        plt.scatter((3600*samptend[var+'tendlsall']).sel(z=slice(layer[0],layer[1])).mean('z').where(samptend.time.dt.hour==hour,drop=True).mean(),\
                  (3600*samptend[var+'tendtotall']).sel(z=slice(layer[0],layer[1])).mean('z').where(samptend.time.dt.hour==hour,drop=True).mean(),\
                    label=hour,c=colors[hour],marker='s',s=70)
        # plt.text((3600*samptend[var+'tendlsall']).sel(z=slice(layer[0],layer[1])).mean('z').where(samptend.time.dt.hour==hour,drop=True).mean(),\
        #          (3600*samptend[var+'tendtotall']).sel(z=slice(layer[0],layer[1])).mean('z').where(samptend.time.dt.hour==hour,drop=True).mean(),\
        #            hour,c=colors[hour],fontsize=12)
            
else:
    # plt.scatter(x,y,alpha=0.7,s=10)
    for idx,ii in enumerate(x1.time):
        plt.scatter(x1.sel(time=ii),y1.sel(time=ii),marker='s',c=col[idx-1],s=70,label=str(ii.values)[5:10])
        plt.scatter(x.sel(time=ii.dt.strftime('%Y-%m-%d').values),\
                    y.sel(time=ii.dt.strftime('%Y-%m-%d').values),c=col[idx-1],alpha=0.5,s=5)

################## 
plt.legend(ncol=2)
plt.plot([-1.1, 0.8], [-1.1, 0.8],c='grey',lw=1)
plt.plot([-1.1, 0.8], [0,0],c='grey',lw=0.5)
plt.plot([0,0], [-1.1, 0.8],c='grey',lw=0.5)

if var == 'thl':
    plt.xlim([-0.28, 0.25])
    plt.ylim([-0.4, 0.25]) 
elif var == 'qt':
    plt.xlim([-0.0005, 0.0003])
    plt.ylim([-0.00003, 0.00003]) 
else:
    plt.xlim([-1.1, 0.75])
    plt.ylim([-1.1, 0.75])
plt.ylabel('Tot tendency (m/s /hour)')
# plt.ylabel('Divergence duwt_dz ()')
plt.xlabel('LS tendency (m/s /hour)')
plt.title(model+': '+var+' in layer '+str(layer),size=20)
#%% time series of tendencies
layer = [0,750]
var = 'u'
rol = 10
composite = False

dales_to_plot   = samptend.sel(z=slice(layer[0],layer[1])).mean('z')\
    .sel(time=slice(np.datetime64('2020-02-02'),np.datetime64('2020-02-11')))
h_clim_to_plot = harm_clim_avg.sel(z=slice(layer[0],layer[1])).mean('z')\
    .sel(time=slice(np.datetime64('2020-02-02'),np.datetime64('2020-02-11')))

plt.figure(figsize=(15,6))
if composite:
    ## DALES
    3600*acc_time*dales_to_plot.groupby(dales_to_plot.time.dt.hour).mean()[var+'tendtotall'].plot(c='r',label='DALES: Tot')
    acc_time*dales_to_plot.groupby(dales_to_plot.time.dt.hour).mean()[var+'tendlsall'].plot(c='k', label='DALES: LS')
    acc_time*dales_to_plot.groupby(dales_to_plot.time.dt.hour).mean()[var+'tendphyall'].plot(c='c',label='DALES: Tot - LS')
    
    ## HARMONIE cy43 clim
    acc_time*(h_clim_to_plot['dt'+var+'_dyn']+h_clim_to_plot['dt'+var+'_phy'])\
        .groupby(h_clim_to_plot.time.dt.hour).mean().plot(c='r',ls=':',label='H.clim cy43: Tot')
    acc_time*h_clim_to_plot.groupby(h_clim_to_plot.time.dt.hour).mean()\
        ['dt'+var+'_dyn'].plot(c='k',ls=':',label='H.clim cy43: Dyn')
    acc_time*h_clim_to_plot.groupby(h_clim_to_plot.time.dt.hour).mean()\
        ['dt'+var+'_phy'].plot(c='c',ls=':',label='H.clim cy43: Phy') 
    
else:
    ## DALES 
    (acc_time*dales_to_plot.rolling(time=rol*4).mean()[var+'tendtotall']).plot(c='r',label='DALES: Net')
    (acc_time*dales_to_plot.rolling(time=rol*4).mean()[var+'tendlsall']).plot(c='k', label='DALES: LS')
    (acc_time*dales_to_plot.rolling(time=rol*4).mean()[var+'tendphyall']).plot(c='g',label='DALES: Net - LS')
    
    ## HARMONIE cy43 clim
    (acc_time*(h_clim_to_plot['dt'+var+'_dyn']+h_clim_to_plot['dt'+var+'_phy']).rolling(time=rol).mean()).plot(c='r',ls=':',label='HAR: Net')
    (acc_time*h_clim_to_plot['dt'+var+'_dyn'].rolling(time=rol).mean()).plot(c='k',ls=':',label='HAR: Dyn')
    (acc_time*h_clim_to_plot['dt'+var+'_phy'].rolling(time=rol).mean()).plot(c='g',ls=':',label='HAR: Phy')

    for day in np.arange(srt_time,end_time):
        plt.axvline(x=day,c='k',lw=0.5)
plt.axhline(0,c='k',lw=0.5)
plt.title('Mean '+var+' tendency between '+str(layer)+' m',fontsize=20)
plt.legend(ncol=2)
plt.ylabel('Tendency (m/s /hour)')

if var == 'thl':
    plt.ylim([-0.00005, 0.00005]) 
    plt.ylabel('Tendency (K /hour)')
elif var == 'qt':
    plt.ylim([-0.001, 0.001]) 
    plt.ylabel('Tendency (g/kg /hour)')
else:
    plt.ylim([-0.85,0.8])
    plt.ylabel('Tendency (m/s /hour)')
plt.xlim(temp_hrs)
plt.savefig(save_dir+'tmser_'+var+'tend_rol'+str(rol)+'.pdf', bbox_inches="tight")

#%% TAKE INTO ACCOUNT SIGN OF THE WIND COMPONENTS !!

# tend_sign = samptend
# for ii in list(samptend.keys()):
#     if 'vtend' in ii:
#         tend_sign[ii] = samptend[ii] * -1* np.sign(profiles.v)
#     elif 'utend' in ii:
#         tend_sign[ii] = samptend[ii] * -1* np.sign(profiles.u)
#     else: 
#         tend_sign[ii] = samptend[ii]

#%%
exp_tend = samptend

#%%    
days = ['2020-02-03T13','2020-02-03T14','2020-02-03T15']
# days = ['2020-02-03']
# days = ['2020-02-03T12','2020-02-03T13','2020-02-03T14',
#     '2020-02-03T15','2020-02-03T16','2020-02-03T17',\
#         '2020-02-03T18','2020-02-03T19','2020-02-03T20','2020-02-03T21','2020-02-03T22','2020-02-03T23']
# days = ['2020-02-02','2020-02-03','2020-02-04','2020-02-05','2020-02-06','2020-02-07',\
#         '2020-02-08','2020-02-09','2020-02-10','2020-02-11']
h_hind_to_plot = {}
cond_sampl = ['all']
for ii in ['days']:
    plt_obs = False
    if ii == 'all':
        tend_to_plot   = exp_tend.sel(time=slice(temp_hrs[0],temp_hrs[1]))
        h_clim_to_plot = harm_clim_avg.sel(time=slice(temp_hrs[0]+np.timedelta64(2,'h'),temp_hrs[1]-np.timedelta64(26,'h')))
        for avg in harm_avg_domains:
            h_hind_to_plot[avg] = harm_hind_avg[avg].sel(time=slice(temp_hrs[0]+np.timedelta64(2,'h'),temp_hrs[1]-np.timedelta64(26,'h')))
        title='Domain and Temporal mean'
    elif ii == 'days':
        tend_to_plot   = exp_tend.where(exp_tend.time.dt.strftime('%Y-%m-%dT%H').isin(days),drop=True)
        # h_clim_to_plot = harm_clim_avg.where(harm_clim_avg.time.dt.strftime('%Y-%m-%d').isin(days),drop=True)
        h_clim_to_plot = harm_clim_avg.where(harm_clim_avg.time.isin(tend_to_plot.time,),drop=True)
        tend_to_plot = tend_to_plot.sel(time=h_clim_to_plot.time)
        
        ls_flux_toplot = ls_flux.where(ls_flux.time.dt.strftime('%Y-%m-%d').isin(days),drop=True)
        for avg in harm_avg_domains:
            h_hind_to_plot[avg] = harm_hind_avg[avg].where(harm_hind_avg[avg].time.dt.strftime('%Y-%m-%d').isin(days),drop=True)
        title='Domain mean for '+ " ".join(days)
    elif ii == 'flights':
        if 'tend_to_plot' in locals(): del tend_to_plot
        for ii in joanne.time.dt.strftime('%Y-%m-%d').values:
            temp = exp_tend.sel(time=ii).where((exp_tend.time - np.timedelta64(10,'m')<=\
                                      joanne.end_flight.sel(time=ii)) & \
                                      (exp_tend.time + np.timedelta64(10,'m')>=\
                                      joanne.start_flight.sel(time=ii))\
                                      ,drop = True)
            if 'tend_to_plot' in locals():
                tend_to_plot = xr.concat([tend_to_plot,temp],dim='time')
            else: 
                tend_to_plot = temp
        
        
        h_clim_to_plot = harm_clim_avg.where(harm_clim_avg.time.dt.strftime('%Y-%m-%dT%H').isin(np.unique(tend_to_plot.time.dt.strftime('%Y-%m-%dT%H'))),drop=True)
        # for avg in harm_avg_domains:
            # h_hind_to_plot[avg] = harm_hind_avg[avg].where(harm_hind_avg[avg].time.dt.strftime('%Y-%m-%d').isin(joanne.time.dt.strftime('%Y-%m-%d')),drop=True)
        title='Domain mean for flight days'
        plt_obs = True

# h_clim_to_plot=h_clim_to_plot.where(h_clim_to_plot.time.dt.strftime('%Y-%m-%d').isin(days),drop=True)
# tend_to_plot=tend_to_plot.where(tend_to_plot.time.dt.strftime('%Y-%m-%d').isin(days),drop=True)

days_all = np.arange(temp_hrs[0], temp_hrs[1])
acc_time = 3600*1
# var=['thl','qt']
var=['u','v']
# for group in [0,1,2,3]:
#     if group <= profiles_daily.group.max()
#         days = profiles_daily.where(profiles_daily.group==group,drop=True).time
#     else: days = days_all

# samptend.where(samptend['time'].dt.strftime('%Y-%m-%d').isin(days_all),drop=True)


fig, axs = plt.subplots(1,len(var),figsize=(10,9))
axs = axs.ravel()
temp = 0
for ii in var:
    for samp in cond_sampl:
        if ii == 'qt': unit = 1000
        else: unit = 1
        axs[temp].plot(unit*acc_time*tend_to_plot[ii+'tendls'+samp].\
                    mean('time'),tend_to_plot.z,c='k',label='D. LS dyn.',lw=2)
        # axs[temp].plot(unit*acc_time*tend_to_plot[ii+'tendadv'+samp].\
        #             mean('time'),tend_to_plot.z,c='b',label='Advective')
        # axs[temp].plot(unit*acc_time*tend_to_plot[ii+'tenddif'+samp].\
        #             mean('time'),tend_to_plot.z,c='g',label='Diffusive') 
        # axs[temp].plot(unit*acc_time*tend_to_plot[ii+'tendfor'+samp].\
        #             mean('time'),tend_to_plot.z,c='orange',label='Other') 
        # if ii =='u' or ii=='v':
        #     axs[temp].plot(-unit*acc_time*profiles['d'+ii+'wt_dz'].\
        #                 where(profiles.time.dt.strftime('%Y-%m-%d').isin(days),drop=True).\
        #                 mean('time'),profiles.z,c='y',label='Vert adv tot') 
        #     axs[temp].plot(-unit*acc_time*profiles['d'+ii+'ws_dz'].\
        #                 where(profiles.time.dt.strftime('%Y-%m-%d').isin(days),drop=True).\
        #                 mean('time'),profiles.z,c='y',ls='--',label='Vert adv res') 
                
            # axs[temp].plot(unit*acc_time*tend_to_plot[ii+'tendpois'+samp].\
            #             mean('time'),tend_to_plot.z,c='y',label='Press') 
        # axs[temp].plot(unit*acc_time*tend_to_plot[ii+'tendaddon'+samp].\
        #             mean('time'),tend_to_plot.z,c='m',label='Addon') 
        
            
        if ii == 'qt':
            axs[temp].plot(unit*acc_time*(tend_to_plot[ii+'tendadv'+samp] + tend_to_plot[ii+'tenddif'+samp]+\
              tend_to_plot[ii+'tendmicro'+samp] + tend_to_plot[ii+'tendrad'+samp]+tend_to_plot[ii+'tendls'+samp]).\
               mean('time'),tend_to_plot.z,c='r',label='Net (sum)')
            axs[temp].plot(unit*acc_time*(tend_to_plot[ii+'tendadv'+samp] + tend_to_plot[ii+'tenddif'+samp]+\
                                          tend_to_plot[ii+'tendmicro'+samp] + tend_to_plot[ii+'tendrad'+samp]).\
                        mean('time'),tend_to_plot.z,c='c',label='Net - LS (sum)')
        else:
            axs[temp].plot(unit*acc_time*tend_to_plot[ii+'tendtot'+samp].\
               mean('time'),tend_to_plot.z,c='r',label='D. Net',lw=2)
            # axs[temp].plot(unit*acc_time*(tend_to_plot[ii+'tendtot'+samp] - tend_to_plot[ii+'tendls'+samp]).\
            #             mean('time'),tend_to_plot.z,c='c',label='Net - LS')
            axs[temp].plot(unit*acc_time*(tend_to_plot[ii+'tendadv'+samp] + tend_to_plot[ii+'tenddif'+samp]).\
                        mean('time'),tend_to_plot.z,c='g',label='D. Resolved\n+ sub-grid')
        # if ii =='thl' or ii=='qt':
        #     axs[temp].plot(unit*acc_time*tend_to_plot[ii+'tendmicro'+samp].\
        #             mean('time'),tend_to_plot.z,c='orange',label='Microphysics') 
        # if ii =='thl':
        #     axs[temp].plot(unit*acc_time*tend_to_plot[ii+'tendrad'+samp].\
        #                 mean('time'),tend_to_plot.z,c='g',label='Radiation')    
                
        # axs[temp].plot(unit*acc_time*ls_flux_toplot['d'+ii+'dt'].\
        #             mean('time'),ls_flux.z,c='k',ls='-.',lw=2,label='forcing')
    
    ## HARMONIE cy40 hind
    c=1
    # for avg in ['300']:
    #     c+=1
    #     axs[temp].plot(unit*acc_time*h_hind_to_plot[avg]['dt'+ii+'_dyn'].\
    #                 mean('time'),h_hind_to_plot[avg]['z'],c='k',ls='--',lw=2,label='H.hind dyn '+avg) 
    #     axs[temp].plot(unit*acc_time*h_hind_to_plot[avg]['dt'+ii+'_phy'].\
    #                 mean('time'),h_hind_to_plot[avg]['z'],c='c',ls='--',lw=2,label='H.hind phy '+avg) 
    #     axs[temp].plot(unit*acc_time*(h_hind_to_plot[avg]['dt'+ii+'_phy']+h_hind_to_plot[avg]['dt'+ii+'_dyn']).\
    #                 mean('time'),h_hind_to_plot[avg]['z'],c='r',ls='--',lw=2,label='H.hind net '+avg)

    ## HARMONIE cy43 clim
    # axs[temp].plot(unit*acc_time*h_clim_to_plot['dt'+ii+'_dyn'].\
    #             mean('time'),h_clim_to_plot['z'],c='k',ls='--',lw=2,label='H. dyn') 
    # axs[temp].plot(unit*acc_time*(h_clim_to_plot['dt'+ii+'_phy']+h_clim_to_plot['dt'+ii+'_dyn']).\
    #             mean('time'),h_clim_to_plot['z'],c='r',ls='--',lw=2,label='H. Net')
    # axs[temp].plot(unit*acc_time*h_clim_to_plot['dt'+ii+'_phy'].\
    #             mean('time'),h_clim_to_plot['z'],c='g',ls='--',lw=2,label='H. phy')

    # if ii == 'u' or ii == 'v':
    #     axs[temp].plot(unit*acc_time*h_clim_to_plot['dt'+ii+'_turb'].\
    #         mean('time'),h_clim_to_plot['z'],c='c',ls='--',lw=1,label='H. turb')
    #     axs[temp].plot(unit*acc_time*h_clim_to_plot['dt'+ii+'_conv'].\
    #         mean('time'),h_clim_to_plot['z'],c='b',ls='--',lw=1,label='H. conv')
    
    
    ## OBS
    # if plt_obs:
    #     if ii =='u' or ii=='v':
    #         axs[temp].plot(unit*acc_time*joanne['F_'+ii].\
    #                         mean('time'),joanne['Height'],c='green',ls='-.',label='Joanne F')
    #         axs[temp].plot(unit*acc_time*joanne['dyn_'+ii+'_tend'].\
    #                         mean('time'),joanne['Height'],c='k',ls='-.',label='Joanne dyn')
    #         axs[temp].plot(unit*acc_time*joanne['d'+ii+'dt_fl'].\
    #                         mean('time'),joanne['Height'],c='r',ls='-.',label='Joanne net')
        
    axs[temp].set_title(ii+' tendency',size=20)
    # axs[temp].set_title('Zonal momentum budget',size=18)
    axs[temp].axvline(0,c='k',lw=0.5)
    
    axs[temp].set_ylim(height_lim)
    # axs[temp].set_ylim([0,3100])
    if ii == 'thl':
        axs[temp].set_xlabel('K/s /hour')
        if samp == 'all':
            axs[temp].set_xlim([-0.31,+0.35])
        axs[temp].legend(frameon=False)
    elif ii == 'qt':
        axs[temp].set_xlabel('g/kg /hour')
        if samp == 'all':
            axs[temp].set_xlim([-0.25,+0.35])

        # axs[temp].legend(frameon=False)
    else:
        axs[temp].set_xlabel(r'($m/s^2$ / hour)',fontsize=18)
        if samp == 'all':
            # axs[temp].set_xlim([-0.8,+0.9])
            axs[temp].legend(frameon=False)
        # else: 
        #     axs[temp].set_xlim([-9,+11])
        # if ii=='u':
        #     axs[temp].legend(frameon=False)
    temp+=1
    
plt.suptitle(days[1])
    
# plt.savefig(save_dir+'/IFS_tend/budget_Har_DALES_'+days[0][5:]+'.pdf', bbox_inches="tight")
plt.savefig(save_dir+'/IFS_tend/budget_DALES_'+days[1][5:]+'.pdf', bbox_inches="tight")
####################################################################### 
#%% Make videos
if make_videos:

#%% FROM DALES
    print('Creating images for video')
    var = 'lwp'
    
    plt.figure()
    ax =cape.sel(time=ii)[var].plot(vmin=0,vmax=1,\
                            cmap=plt.cm.Blues_r,x='lon',y='lat',\
                            subplot_kws=dict(projection=proj))
    ax = plt.axes(projection=proj)
    ax.add_feature(coast, lw=2, zorder=7)
    plt.xlim([-60,-56.5])
    plt.ylim([12,14.5])
    gl = ax.gridlines(crs=proj, draw_labels=True)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabels_top = False
    gl.ylabels_right = False
    # plt.savefig(save_dir+'for_video_'+str(ii)+'.png')


#%% FROM HARMONIE
wa = xr.open_mfdataset('/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/HARMONIE/cy43_clim/average_300km/wa_Slev_fp_EUREC4Acircle_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002102300.nc')
h_dyn_v = xr.open_mfdataset('/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/HARMONIE/cy43_clim/average_300km/dtv_dyn_Slev_his_EUREC4Acircle_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002110000.nc')
h_dyn_u = xr.open_mfdataset('/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/HARMONIE/cy43_clim/average_300km/dtu_dyn_Slev_his_EUREC4Acircle_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002110000.nc')
h_dyn_u['dtu_dyn_deac'] = (h_dyn_u['dtu_dyn'].diff('time')) * 3600**-1  # gives values per second 
h_dyn_v['dtv_dyn_deac'] = (h_dyn_v['dtv_dyn'].diff('time')) * 3600**-1  # gives values per second 

#%%
lev = 7

# for ii in np.arange(wa.time[0].values, wa.time[-1].values,timedelta(hours = 20)):
    ### vertical velocity 
    # plt.figure()
    # ax =wa.sel(time=ii).isel(lev=lev)['wa'].plot(vmin=-0.1,vmax=0.1,\
    #                         cmap=plt.cm.seismic,x='lon',y='lat',\
    #                         subplot_kws=dict(projection=proj))
    # ax = plt.axes(projection=proj)
    # ax.add_feature(coast, lw=2, zorder=7)
    # plt.xlim([-59.7,-56.5])
    # plt.ylim([12,14.5])
    # gl = ax.gridlines(crs=proj, draw_labels=True)
    # gl.xformatter = LONGITUDE_FORMATTER
    # gl.yformatter = LATITUDE_FORMATTER
    # gl.xlabels_top = False
    # gl.ylabels_right = False
    
# for ii in np.arange(h_dyn_v.time[0].values, h_dyn_u.time[-1].values,timedelta(hours = 12)):
#     ### tendency
#     plt.figure()
#     ax =(3600*h_dyn_v.sel(time=ii).isel(lev=lev)['dtv_dyn_deac']).plot(vmin=-1,vmax=1,\
#                             cmap=plt.cm.seismic,x='lon',y='lat',\
#                             subplot_kws=dict(projection=proj))
#     ax = plt.axes(projection=proj)
#     ax.add_feature(coast, lw=2, zorder=7)
#     # plt.xlim([-59.7,-56.5])
#     # plt.ylim([12,14.5])
#     gl = ax.gridlines(crs=proj, draw_labels=True)
#     gl.xformatter = LONGITUDE_FORMATTER
#     gl.yformatter = LATITUDE_FORMATTER
#     gl.xlabels_top = False
#     gl.ylabels_right = False
#     plt.show()
    


# sizes= np.array([1,2,5,7,8,10,12,15,18,20,22,25,30,35,40,45,50])
# y1_std=[]
# y2_std=[]
# y1_mean=[]
# y2_mean=[]
# # wa_mean={}
# # wa_std={}

# # wa_mean[ii]=[]
# # wa_std[ii]=[]
# for x in sizes:
#     print('Calculating size '+str(x)+'...')
#     y1_mean=np.append(y1_mean,h_dyn_u.isel(lev=lev,x=range(70-x,70+x),y=range(60-x,60+x)).mean(dim=['x','y','time']).dtu_dyn_deac.values)
#     y2_mean=np.append(y2_mean,h_dyn_v.isel(lev=lev,x=range(70-x,70+x),y=range(60-x,60+x)).mean(dim=['x','y','time']).dtv_dyn_deac.values)

#     y1_std=np.append(y1_std,h_dyn_u.isel(lev=lev,x=range(70-x,70+x),y=range(60-x,60+x)).std(dim=['x','y']).mean('time').dtu_dyn_deac.values)
#     y2_std=np.append(y2_std,h_dyn_v.isel(lev=lev,x=range(70-x,70+x),y=range(60-x,60+x)).std(dim=['x','y']).mean('time').dtv_dyn_deac.values)
    
#     # wa_mean[ii]=np.append(wa_mean[ii],wa.isel(lev=lev,x=range(70-x,70+x),y=range(60-x,60+x)).mean(dim=['x','y','time']).wa.values)
#     # wa_std[ii]=np.append(wa_std[ii],wa.isel(lev=lev,x=range(70-x,70+x),y=range(60-x,60+x)).std(dim=['x','y']).mean('time').wa.values)

# fig,ax = plt.subplots()
# ax.plot(sizes*2.5*2,y1_mean*3600,c='r',label='U direction')
# # ax.fill_between(sizes*2.5*2, (3600*y1_mean-y1_std), (3600*y1_mean+y1_std),facecolor='r',alpha=0.5)
# ax2=ax.twinx()
# ax2.plot(sizes*2.5*2,y2_mean*3600,c='b',label='V direction')
# # ax2.fill_between(sizes*2.5*2, (3600*y2_mean-y2_std), (3600*y2_mean+y2_std),facecolor='b',alpha=0.5)
# ax.set_xlabel('Domain [$km^2$]')
# ax.set_ylabel("Mean dyn tend U [m/s /hour]",color="r")
# ax2.set_ylabel("Mean dyn tend V [m/s /hour]",color="b")
# # plt.ylabel('STD dyn tend [m/s /hour]')
# fig.legend(bbox_to_anchor=(1,1),bbox_transform=ax.transAxes)


# wa_mean.to_netcdf()
# wa_mean.to_netcdf(my_harm_dir+'wa_mean.nc')


# plt.figure(figsize=(19,5))
# for lev in [7]:
#     wa.isel(lev=lev,x=70,y=60).wa.rolling(time=2,center=True).mean().plot(x='time',lw=1.5,label='2.5x2.5 km')
#     wa.isel(lev=lev,x=slice(50,91),y=slice(40,81)).mean(dim=['x', 'y']).wa.rolling(time=2,center=True).mean().plot(x='time',lw=1.5,label='100x100 km')
#     wa.isel(lev=lev,x=slice(30,111),y=slice(20,101)).mean(dim=['x', 'y']).wa.rolling(time=2,center=True).mean().plot(x='time',lw=1.5,label='200x200 km')
#     # wa.isel(lev=lev,x=70,y=60).wa.plot(x='time',lw=0.5)
# plt.legend()
# plt.axhline(0,c='k',lw=0.5)
# plt.ylim([-0.1,0.1])
# pl.title('200 m')

# plt.figure(figsize=(19,5))
# for lev in [4,7,10,20]:
#     wa.isel(lev=lev,x=70,y=60).wa.rolling(time=3,center=True).mean().plot(x='time',lw=1.5)
#     # wa.isel(lev=lev,x=70,y=60).wa.plot(x='time',lw=0.5)
# plt.legend(['100 m', '200 m', '300 m', '1 km'])
# plt.axhline(0,c='k',lw=0.5)
# plt.ylim([-0.1,0.1])

# plt.figure(figsize=(19,5))
# for lev in [4,7,10,20]:
#     wa.isel(lev=lev,x=slice(50,90),y=slice(40,80)).mean(dim=['x', 'y']).wa.rolling(time=3,center=True).mean().plot(x='time',lw=1.5)
#     # wa.isel(lev=lev,x=70,y=60).wa.plot(x='time',lw=0.5)
# plt.legend(['100 m', '200 m', '300 m', '1 km'])
# plt.axhline(0,c='k',lw=0.5)
# plt.ylim([-0.1,0.1])




#%%
if make_videos:
    for ii in np.arange(nc_data_cl.time[0].values, nc_data_cl.time[-1].values,timedelta(hours = 10)):

        ########  cloud variables ########
        print('Creating images for video')
        var = 'clwvi'
        for ii in np.arange(srt_time, end_time,timedelta(hours = 2)):
            plt.figure()    
            ax =nc_data_cl.sel(time=ii)[var].plot(vmin=0,vmax=1,\
                                    cmap=plt.cm.Blues_r,x='lon',y='lat',\
                                    subplot_kws=dict(projection=proj))
            ax = plt.axes(projection=proj)
            ax.add_feature(coast, lw=2, zorder=7)
            plt.xlim([-60,-56.5])
            plt.ylim([12,14.5])
            gl = ax.gridlines(crs=proj, draw_labels=True)
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabels_top = False
            gl.ylabels_right = False
            plt.savefig(save_dir+'for_video_'+str(ii)+'.png')
        
        #get list of images to use for video
        images = [os.path.join(save_dir,img)
                       for img in os.listdir(save_dir)
                       if img.endswith(".png")]
        images.sort()
        
        #create video
        print('Creating video')
        fps = 4
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(images, fps=fps)
        clip.write_videofile(save_dir+'my_video.mp4')
        
        #remove images
        print('Removing images used for video')
        for image in images:
            os.remove(image)

#%% MOMENTS
layer = [0,300]
composite = False
# # groupby(dales_to_plot.time.dt.hour).mean()
# plt.figure(figsize=(19,5))
# # for var in ['w2']:
# for var in ['u2','v2','w2','thl2','qt2']:
#     if composite:
#         moments[var].groupby(moments.time.dt.hour).mean().sel(z=slice(layer[0],layer[1])).mean('z').plot(label=var)
#     else:     
#         moments[var].sel(z=slice(layer[0],layer[1])).mean('z').plot(x='time',label=var)
#         if comp_experiments:
#             mom_isurf5[var].sel(z=slice(layer[0],layer[1])).mean('z').plot(x='time',label=var)
#         plt.xlim(temp_hrs)
#         for ii in np.arange(srt_time, end_time):
#             plt.axvline(x=ii,c='k',lw=0.5,alpha=0.5)
# plt.legend()
# plt.title('Variance in layer '+str(layer),fontsize=20)

##variance compare layer 
moments['ctop_var'] = moments.sel(z=tmser.sel(time=moments.time).zc_max - 400,method='nearest')['u2']
fig, axs = plt.subplots(figsize=(19,5))
ax2 = axs.twinx()
for var in ['u2']:
    if composite:
        moments[var].groupby(moments.time.dt.hour).mean().sel(z=slice(layer[0],layer[1])).mean('z').plot(label=var)
    else:     
        moments[var].sel(z=slice(layer[0],layer[1])).mean('z').plot(x='time',label=var,lw=2,c='k')
        moments['ctop_var'].plot(x='time',label='u2 at cloud top',lw=2,c='r')
        profiles.rainrate.sel(z=slice(0,300)).mean('z').plot(x='time',ax=axs,alpha=0.7,ls='--',label='Rain')
        # tmser.zc_max.plot(x='time', label = 'cloud top')
        if comp_experiments:
            mom_isurf5[var].sel(z=slice(layer[0],layer[1])).mean('z').plot(x='time',label=var)
        plt.xlim(temp_hrs)
        for ii in np.arange(srt_time, end_time):
            plt.axvline(x=ii,c='k',lw=0.5,alpha=0.5)
plt.legend()
plt.title('Variance in layer '+str(layer),fontsize=20)

#%%

#%%
print('end.')


