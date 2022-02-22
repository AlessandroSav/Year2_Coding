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
              '011','012','013','014','015','016']
expnr      = ['014','015','016'] 
case       = '20200202_12'
casenr     = '002'      # experiment number where to read input files 

# expnr      = ['001']
# case       = '20200209_10'
# casenr     = '001'      # experiment number where to read input files 

### Directories for runnin on VrLab
# base_dir   = '/Users/acmsavazzi/Documents/Mount/DALES/'
# Input_dir  = base_dir+'Cases/20200202_12_300km/'
# Output_dir = base_dir+'Experiments/EUREC4A/Exp_ECMWF/'+case+'/'
save_dir   = '/Users/acmsavazzi/Documents/WORK/PhD_Year2/Figures/'
# fig_dir = os.path.abspath('{}/../../Figures/DALES')+'/'

### Directories for runnin on TU server
# base_dir   = 'staff-umbrella/cmtrace/Alessandro/'
base_dir   = '/Users/acmsavazzi/Documents/Mount/'
Input_dir  = base_dir+'Raw_Data/Les/Eurec4a/'+case+'/Exp_'+casenr+'/'
Output_dir = base_dir+'Raw_Data/Les/Eurec4a/'+case+'/'
# save_dir   = base_dir+'PhD_Year2/'

### Directories for local 
# Input_dir  = '/Users/acmsavazzi/Documents/WORK/PhD_Year1/DATA/DALES/DALES/Experiments/20200209_10/Exp_009/'
# Output_dir = Input_dir+'../'
# save_dir   = '/Users/acmsavazzi/Documents/WORK/PhD_Year2/'

### times to read and to plot 
srt_time   = np.datetime64('2020-02-02')
end_time   = np.datetime64('2020-02-12')
temp_hrs=[np.datetime64('2020-02-09'),np.datetime64('2020-02-10')]
hours = srt_time,srt_time + [np.timedelta64(2, 'h'),np.timedelta64(48, 'h'),\
                             np.timedelta64(108, 'h'),np.timedelta64(144, 'h')]

lat_select = 13.2806    # HALO center 
lon_select = -57.7559   # HALO center 
buffer = 30             # buffer of 150 km around (75 km on each side) the gridpoint 30 * 2 * 2.5 km

make_videos       = False
LES_forc_HARMONIE = True
harm_3d           = True
comp_observations = True

dales_exp_dir = base_dir+'Raw_Data/Les/Eurec4a/20200209_10/'

LES_forc_dir  = '/Users/acmsavazzi/Documents/WORK/PhD_Year1/DATA/HARMONIE/LES_forcing_300km/'

my_harm_dir   = '/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/'

harmonie_dir   = base_dir+'Raw_Data/HARMONIE/BES_harm43h22tg3_fERA5_exp0/2020/'
harmonie_time_to_keep = '202002010000-'

ifs_dir        = '/Users/acmsavazzi/Documents/WORK/Research/MyData/'
obs_dir        = '/Users/acmsavazzi/Documents/WORK/Research/MyData/'
Aers_Dship_dir = '/Users/acmsavazzi/Documents/WORK/Data/Aers-Dship/'

hrs_inp = ((end_time - srt_time)/np.timedelta64(1, 'h')).astype(int)+1

#%%     OPTIONS FOR PLOTTING

# col=['b','r','g','orange','k']
col=['red','coral','maroon','blue','cornflowerblue','darkblue','green','lime','forestgreen']
height_lim = [0,5000]        # in m

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
# first read the surface values
print("Reading input surface fluxes.")
colnames = ['time','wthl_s','wqt_s','th_s','qt_s','p_s']
ls_surf = pd.read_csv(Input_dir+'ls_flux.inp.'+casenr,header = 3,nrows=hrs_inp,\
                     names=colnames,index_col=False,delimiter = " ")
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
# tmser['T_s'] = calc_T(tmser['thlskin'],ls_surf['p_s'])

####     samptend.nc    ####
print("Reading DALES tendencies.") 
samptend   = xr.open_mfdataset(samptend_files, combine='by_coords')
samptend['time'] = srt_time + samptend.time.astype("timedelta64[s]")
# interpolate half level to full level
samptend = samptend.interp(zm=samptend.zt)
samptend = samptend.rename({'zt':'z'})

####     moments.001    ####
print("Reading DALES moments.") 
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
print("Reading DALES surface values.") 
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
    cape_merg_files.sort()
    cape   = xr.open_mfdataset(cape_merg_files[1:], combine='by_coords',decode_times=False)
    cape['time'] = srt_time + cape.time.astype("timedelta64[s]")
    ####     merged_crossxy.nc    ####
    # crossxy_0001 = xr.open_dataset(Output_dir+'merged_crossxy_0001.'+expnr+'.nc')

#%% Import DALES sensitivity experimetns
prof_files = [] 
samptend_files = []
for path,subdir,files in os.walk(dales_exp_dir):
    if path[-3:] in ['001','002']: 
        for file in glob(os.path.join(path, 'profiles*.nc')):
            prof_files.append(file)
        for file in glob(os.path.join(path, 'samptend*.nc')):
            samptend_files.append(file)

####     profiles.nc    ####    
print("Reading DALES exp profiles.")      
prof_isurf5 = xr.open_mfdataset(prof_files, combine='by_coords')
prof_isurf5['time'] = np.datetime64('2020-02-09') + prof_isurf5.time.astype("timedelta64[s]")
# interpolate half level to full level
prof_isurf5 = prof_isurf5.interp(zm=prof_isurf5.zt)
prof_isurf5 = prof_isurf5.rename({'zt':'z'})

####     samptend.nc    ####
print("Reading DALES exp tendencies.") 
tend_isurf5   = xr.open_mfdataset(samptend_files, combine='by_coords')
tend_isurf5['time'] = np.datetime64('2020-02-09') + tend_isurf5.time.astype("timedelta64[s]")
# interpolate half level to full level
tend_isurf5 = tend_isurf5.interp(zm=tend_isurf5.zt)
tend_isurf5 = tend_isurf5.rename({'zt':'z'})


#%% Import Harmonie
### Import large scale spatial means (used for LES forcing)
if LES_forc_HARMONIE:
    print("Reading HARMONIE spatial mean (used for LES forcing).") 
    harm_hind_avg = xr.open_dataset(LES_forc_dir+'LES_forcing_202002'+\
                                  srt_time.astype(str)[8:10]+'00.nc')
    for ii in np.arange(srt_time, end_time)[1:]:    
        harm_hind_avg = xr.concat((harm_hind_avg,xr.open_dataset(LES_forc_dir+\
                     'LES_forcing_202002'+ii.astype(str)[8:10]+'00.nc')),dim='time')
    
    harm_hind_avg = calc_geo_height(harm_hind_avg,fliplevels=True)
    z_ref = harm_hind_avg.z.mean('time')
    zz    = harm_hind_avg.z
    
    for var in list(harm_hind_avg.keys()):
        if 'level' in harm_hind_avg[var].dims:
            print("interpolating variable "+var)
            x = np.empty((len(harm_hind_avg['time']),len(harm_hind_avg['level'])))
            x[:] = np.NaN
            for a in range(len(harm_hind_avg.time)):
                x[a,:] = np.interp(z_ref,zz[a,:],harm_hind_avg[var].isel(time = a))            
            harm_hind_avg[var] = (("time","level"), x)    
    # convert model levels to height levels
    harm_hind_avg = harm_hind_avg.rename({'z':'geo_height'})
    harm_hind_avg = harm_hind_avg.rename({'level':'z'})
    harm_hind_avg = harm_hind_avg.rename({'q':'qt','dtq_phy':'dtqt_phy','dtq_dyn':'dtqt_dyn'})
    harm_hind_avg["z"] = (z_ref-z_ref.min()).values
    harm_hind_avg['z'] = harm_hind_avg.z.assign_attrs(units='m',long_name='Height')

### Import raw Harmonie data
# This is too slow... need to find a better way. Maybe in a separate file open
# and save only the points and time neede for comparison.
if harm_3d:
    ### HARMONIE clim spatially averaged
    print("Reading HARMONIE clim spatial average.") 
    file = my_harm_dir+'my_harm_clim_avg.nc'
    harm_clim_avg = xr.open_mfdataset(file)
    
### 2D fields    
###      # Read cloud fraction
print("Reading 2D HARMONIE data.") 
nc_files = []
for EXT in ["clt_his*.nc","cll_his*.nc","clm_his*.nc","clh_his*.nc","clwvi_his*.nc","clivi_his*.nc"]:
    for file in glob(os.path.join(harmonie_dir, EXT)):
        if harmonie_time_to_keep in file:
            nc_files.append(file) 
try:
    nc_data_cl  = xr.open_mfdataset(nc_files, combine='by_coords')
except TypeError:
    nc_data_cl  = xr.open_mfdataset(nc_files)
    


#%% Import observations
print("Reading observations.") 
colnames = ['day','hour','SST (K)','LH (W/m2)']
obs = pd.read_csv(Aers_Dship_dir+'bugts_all_CommaSeparated.csv'\
                   ,header = 1,names=colnames,\
                   index_col=False,delimiter = ";",decimal=",")
    
obs['month'] = (obs.day % 1 *100).round(0).astype(int)
obs['day']   = (obs.day // 1).astype(int)  
obs = obs.loc[obs.month==srt_time.astype(object).month].loc[obs.day>=srt_time.astype(object).day].loc[obs.day<=end_time.astype(object).day]
joanne = xr.open_dataset(obs_dir+'joanne_tend.nc') 
joanne = joanne.rename({'Fx':'F_u','Fy':'F_v'})

ds_obs = {}
ds_obs['radio'] = xr.open_dataset(obs_dir+'nc_radio.nc').sel(launch_time=slice(srt_time,end_time)).rename({'q':'qt'})
ds_obs['drop'] = xr.open_dataset(obs_dir+'My_sondes.nc').sel(launch_time=slice(srt_time,end_time)).rename({'q':'qt'})

#%% import ERA5
print("Reading ERA5.") 
era5=xr.open_dataset(ifs_dir+'My_ds_ifs_ERA5.nc')
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
profiles['th']      = profiles['thl'] + Lv / cp * calc_exner(profiles['p']) * profiles['ql']
profiles['T']       = calc_T(profiles['th'],profiles['p'])

## DALES isurf 5
prof_isurf5 = prof_isurf5.rename({'presh':'p'})
prof_isurf5['wspd']    = np.sqrt(prof_isurf5['u']**2 + prof_isurf5['v']**2)
prof_isurf5['th']   = prof_isurf5['thl'] + Lv / cp * calc_exner(prof_isurf5['p']) * prof_isurf5['ql']
prof_isurf5['T']    = calc_T(prof_isurf5['th'],prof_isurf5['p'])


## for HARMONIE cy40
nudge['wspd']    = np.sqrt(nudge['u']**2    + nudge['v']**2)

### for HARMONIE cy43
# convert temperature to potential temperature
harm_hind_avg['th']  = calc_th(harm_hind_avg.T,harm_hind_avg.p)
harm_hind_avg['thl'] = calc_thl(harm_hind_avg['th'],harm_hind_avg['ql'],harm_hind_avg['p'])
harm_hind_avg['wspd']= np.sqrt(harm_hind_avg['u']**2 + harm_hind_avg['v']**2)
for ii in ['phy','dyn']:
    harm_hind_avg['dtthl_'+ii]=calc_th(harm_hind_avg['dtT_'+ii],harm_hind_avg.p) - Lv / \
        (cp *calc_exner(harm_hind_avg.p)) * harm_hind_avg['dtqc_'+ii]
# denisy 
if harm_3d:
    if 'qt'not in (list(harm_clim_avg.keys())):
        harm_clim_avg = harm_clim_avg.rename({'q':'qt'})
    harm_clim_avg['rho'] = calc_rho(harm_clim_avg['p'],harm_clim_avg['T'],harm_clim_avg['qt'])
    harm_clim_avg['wspd']= np.sqrt(harm_clim_avg['u']**2 + harm_clim_avg['v']**2)
    harm_clim_avg['th']  = calc_th(harm_clim_avg['T'],harm_clim_avg['p'])
    # harm_clim_avg['thl'] = calc_thl(harm_clim_avg['th'],harm_clim_avg['ql'],harm_clim_avg['p'])
    
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
profiles_daily = profiles.resample(time='D').mean('time')
#%% Group by shape of some variables
####  K-mean clustering 
# CLUSTER on dayly shear profiles
### cluster only based on the shape below certain km?
cluster_levels = [0, 3000]

shear_cluster=KMeans(n_clusters=min(profiles_daily.time.size,3),random_state=0,n_init=15,max_iter=10000,\
                tol=10**-7).fit(profiles_daily['du_dz'].sel(z=slice(cluster_levels[0],cluster_levels[1])))
idx = np.argsort(shear_cluster.cluster_centers_.sum(axis=1))

profiles_daily['group'] = (('time'), shear_cluster.labels_)
profiles_daily_clusteres=profiles_daily.groupby('group').mean()
#%%                         PLOTTING
###############################################################################
print("Plotting.") 
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

for ii in ['days',]:
    if ii == 'all':
        hrs_to_plot = profiles.sel(time=slice(temp_hrs[0],temp_hrs[1]))
        title='Domain and Temporal mean'
    elif ii == 'days':
        hrs_to_plot = profiles.where(profiles.time.dt.strftime('%Y-%m-%d').isin(days),drop=True)
        title='Domain mean for '+ " ".join(days)
    elif ii == 'flights':
        hrs_to_plot = profiles.where(profiles.time.isin(joanne.time),drop=True)
        title='Domain mean for flight days'
    if day_night:
        hrs_to_plot_day = hrs_to_plot.sel(time=hrs_to_plot['time.hour'].\
                           isin(find_time_interval(day_interval[0],day_interval[1])))
        hrs_to_plot_night = hrs_to_plot.sel(time=hrs_to_plot['time.hour'].\
                           isin(find_time_interval(night_interval[0],night_interval[1])))

    ## cloud fraction
    plt.figure(figsize=(6,9))
    plt.suptitle('Cloud fraction')
    hrs_to_plot['cfrac'].mean('time').plot(y='z',c=col[0],lw=2,label='Cloud fraction')
    if day_night:      
        hrs_to_plot_day['cfrac'].mean('time').plot(y='z',c=col[1],lw=1,label=str(day_interval)+' UTC')
        hrs_to_plot_night['cfrac'].mean('time').plot(y='z',c=col[2],lw=1,label=str(night_interval)+' UTC')
    harm_clim_avg.sel(time=hrs_to_plot.time,method='nearest').cl.mean('time').plot(y='z',label='H. cy43 clim')
    plt.legend()
    plt.xlabel('%')
    plt.ylim(height_lim)
    plt.title(title)
    plt.savefig(save_dir+'mean_cfrac.pdf')
    ## winds
    plt.figure(figsize=(6,9))
    plt.suptitle('Winds')
    for idx,var in enumerate(['u','v']):
        hrs_to_plot[var].mean('time').plot(y='z',c=col[idx*3],lw=2, label='DALES '+var)
        if day_night:
            hrs_to_plot_day[var].mean('time').plot(y='z',c=col[idx*3+1],lw=1, label=str(day_interval)+' UTC')
            hrs_to_plot_night[var].mean('time').plot(y='z',c=col[idx*3+2],lw=1, label=str(night_interval)+' UTC')
    # if var in nudge:
    #     nudge[var].mean('time').plot(y='z',c=adjust_lightness(col[idx]),lw=0.8,label='HARMONIE '+var)
    plt.legend()
    plt.xlabel('m/s')
    plt.axvline(0,c='k',lw=0.5)
    plt.ylim(height_lim)
    plt.xlim([-12.5,0.5])
    plt.title(title)
    plt.savefig(save_dir+'mean_winds.pdf')
    
    ## momentum fluxes
    plt.figure(figsize=(6,9))
    plt.suptitle('Momentum fluxes')
    for idx,var in enumerate(['uw','vw']):
        for ii in ['t',]:
            if ii == 't':
                hrs_to_plot[var+ii].mean('time').plot(y='z',c=col[idx*3],ls='-',label=var+' tot')
                if day_night:
                    hrs_to_plot_day[var+ii].mean('time').plot(y='z',c=col[idx*3+1],lw=1, label=str(day_interval)+' UTC tot')
                    hrs_to_plot_night[var+ii].mean('time').plot(y='z',c=col[idx*3+2],lw=1, label=str(night_interval)+' UTC tot')
            else:
                hrs_to_plot[var+ii].mean('time').plot(y='z',c=col[idx*3],ls='--',label=var+' resolved')
                if day_night:
                    hrs_to_plot_day[var+ii].mean('time').plot(y='z',c=col[idx*3+1],ls='--',lw=1, label=str(day_interval)+' UTC resolved')
                    hrs_to_plot_night[var+ii].mean('time').plot(y='z',c=col[idx*3+2],ls='--',lw=1, label=str(night_interval)+' UTC resolved')
    plt.title(title)
    plt.legend()
    plt.xlabel('Momentum flux (m2/s2)')
    plt.axvline(0,c='k',lw=0.5)
    plt.ylim(height_lim)
    plt.savefig(save_dir+'mean_momflux.pdf')
        
    ## counter gradient fluxes
    plt.figure(figsize=(6,9))
    plt.suptitle('Counter gradient transport')
    (hrs_to_plot['uwt'] * hrs_to_plot.du_dz).mean('time').\
        plot(y='z',c=col[0],lw=2,label='uw du_dz')
    (hrs_to_plot['vwt'] * hrs_to_plot.dv_dz).mean('time').\
        plot(y='z',c=col[3],lw=2,label='vw dv_dz')
    if day_night:
        (hrs_to_plot_day['uwt'] * hrs_to_plot_day.du_dz).mean('time').\
            plot(y='z',c=col[1],lw=1,label=str(day_interval)+' UTC')
        (hrs_to_plot_night['uwt'] * hrs_to_plot_night.du_dz).mean('time').\
            plot(y='z',c=col[2],lw=1,label=str(night_interval)+' UTC')
        (hrs_to_plot_day['vwt'] * hrs_to_plot_day.dv_dz).mean('time').\
            plot(y='z',c=col[4],lw=1,label=str(day_interval)+' UTC')
        (hrs_to_plot_night['vwt'] * hrs_to_plot_night.dv_dz).mean('time').\
            plot(y='z',c=col[5],lw=1,label=str(night_interval)+' UTC')
    plt.title(title)
    plt.legend()
    plt.axvline(0,c='k',lw=0.5)
    plt.ylim(height_lim)
    plt.xlim([-0.00012,None])
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.savefig(save_dir+'mean_gradTransp.pdf')
    
    ## Momentum flux convergence
    plt.figure(figsize=(6,9))
    plt.suptitle('Momentum flux convergence')
    (-1*hrs_to_plot['duwt_dz']).mean('time').plot(y='z',c='b',ls='-',label='- duw_dz')
    (-1*hrs_to_plot['dvwt_dz']).mean('time').plot(y='z',c='r',ls='-',label='- dvw_dz')
    (-1*hrs_to_plot['duwr_dz']).mean('time').plot(y='z',c='b',ls='--',label='- duwr_dz')
    (-1*hrs_to_plot['dvwr_dz']).mean('time').plot(y='z',c='r',ls='--',label='- dvwr_dz')
    (-1*hrs_to_plot['duws_dz']).mean('time').plot(y='z',c='b',ls=':',label='- duws_dz')
    (-1*hrs_to_plot['dvws_dz']).mean('time').plot(y='z',c='r',ls=':',label='- dvws_dz')
    plt.title(title)
    plt.legend()
    plt.axvline(0,c='k',lw=0.5)
    plt.ylim(height_lim)
    plt.xlim([-0.0001,0.0002])
    plt.xlabel('m/s2')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.savefig(save_dir+'mean_momfluxconv.pdf')
    
#%%  TIMESERIES
## momentum fluxes
plt.figure(figsize=(19,5))
plt.suptitle('Momentum flux')
(profiles['uwt']).plot(y='z',vmin=-0.15)
plt.ylim(height_lim)
plt.xlim([srt_time,end_time])
for ii in np.arange(srt_time, end_time):
    plt.axvline(x=ii,c='k')
# plt.savefig(save_dir+'Figures/tmser_momflux.pdf')


## Difference HARMONIE - DALES
for ii in ['u','v','thl','qt']:
    plt.figure()
    (nudge[ii]-profiles[ii].interp(z=nudge.z)).plot(x="time",cmap='seismic')
    plt.title(ii+' HAMRONIE - DALES')
    plt.ylim([0,8000])
    for ii in np.arange(srt_time, end_time):
        plt.axvline(x=ii,c='k')
        
## Difference HARMONIE - HARMONIE clim
plt.figure()
(nudge['u']-harm_clim_avg['u'].interp(z=nudge.z)).plot(x="time",cmap='seismic')
plt.title('q HAMRONIE - HARMONIE clim')
plt.ylim([0,8000])
for ii in np.arange(srt_time, end_time):
    plt.axvline(x=ii,c='k')
    
## Difference HARMONIE clim - DALES
plt.figure()
(harm_clim_avg['u']-profiles['u'].interp(z=harm_clim_avg.z)).plot(x="time",cmap='seismic')
plt.title('q HAMRONIE clim - DALES')
plt.ylim([0,8000])
for ii in np.arange(srt_time, end_time):
    plt.axvline(x=ii,c='k')
        
        
# Zonal wind
plt.figure()
profiles.u.plot(x="time",cmap='coolwarm',vmin=-15,vmax=1)
plt.title('DALES')
for ii in np.arange(srt_time, end_time):
        plt.axvline(x=ii,c='k')
# Meridional wind
plt.figure()
profiles.v.plot(x="time",cmap='coolwarm',vmin=-7,vmax=5)
plt.title('DALES')
for ii in np.arange(srt_time, end_time):
        plt.axvline(x=ii,c='k')

#%%
### OBS AND MODELS TOGETHER
if comp_observations:
    for level in [40,2000,3000,5000]: # meters
        for var in ['T','qt','wspd']:
            ## Temperature    
            plt.figure(figsize=(15,6))
            plt.plot(profiles.time,profiles[var].sel(z=level,method='nearest'),c=col[3],label='DALES')
            plt.plot(prof_isurf5.time,prof_isurf5[var].sel(z=level,method='nearest'),c=col[1],label='isurf5')
            plt.plot(harm_hind_avg.time,harm_hind_avg[var].sel(z=level,method='nearest'),c=col[0],label='HARMONIE cy40')
            plt.scatter((ds_obs['drop'].launch_time  + np.timedelta64(4, 'h')).values,\
                        ds_obs['drop'].sel(Height=level,method='nearest').sel(launch_time=slice(srt_time,end_time))[var].values,c=col[2],alpha = 0.6,s=20,label='Dropsondes')
            plt.scatter((ds_obs['radio'].launch_time + np.timedelta64(4, 'h')).values,\
                        ds_obs['radio'].sel(Height=level,method='nearest').sel(launch_time=slice(srt_time,end_time))[var].values,c=col[6],alpha = 0.6,s=20,label='Radiosondes')
            plt.plot(era5.Date.sel(Date=slice(srt_time,end_time)),era5[var].sel(Height=level,method='nearest').sel(Date=slice(srt_time,end_time)).mean('Mypoint'), label='ERA5')
            if harm_3d:
                plt.plot(harm_clim_avg.time,harm_clim_avg[var].sel(z=level,method='nearest'),c=col[8],label='HARMONIE cy43 clim')
            plt.xlabel('time (hour)')
            plt.ylabel(var)
            plt.title( var+' at '+str(level)+' m',size=20)
            plt.xlim(temp_hrs)
            plt.axvspan(srt_time,srt_time + np.timedelta64(2, 'h'), alpha=0.2, color='grey')
            plt.legend()
            for day in np.arange(srt_time,end_time):
                plt.axvline(x=day,c='k',lw=0.5)
        
#%% DAILY MEANS
## momentum fluxes
plt.figure(figsize=(19,5))
profiles_daily['uwt'].plot(y='z',vmin=-0.1)
plt.ylim(height_lim)
plt.savefig(save_dir+'daily_momflux.pdf')

#%% DAILY MEANS clustered
for ii in ['du_dz','uwt','duwt_dz']:
    plt.figure(figsize=(5,9))
    for key, group in profiles_daily.groupby('group'):
        group[ii].mean('time').plot(y='z',c=col[key*3],lw=3,label='Group '+str(key))
        group[ii].plot.line(y='z',c=col[key*3],lw=0.7,alpha=0.3,add_legend=False)
    plt.legend()
    plt.ylim(height_lim)
    plt.axvline(0,c='k',lw=0.5)
    plt.savefig(save_dir+'Kmean_'+ii+'.pdf')

#%% TENDENCIES

for ii in ['all',]:
    plt_obs = False
    if ii == 'all':
        tend_to_plot   = samptend.sel(time=slice(temp_hrs[0],temp_hrs[1]))
        h_clim_to_plot = harm_clim_avg.sel(time=slice(temp_hrs[0],temp_hrs[1]))
        h_hind_to_plot = harm_hind_avg.sel(time=slice(temp_hrs[0],temp_hrs[1]))
        title='Domain and Temporal mean'
    elif ii == 'days':
        tend_to_plot   = samptend.where(samptend.time.dt.strftime('%Y-%m-%d').isin(days),drop=True)
        h_clim_to_plot = harm_clim_avg.where(harm_clim_avg.time.dt.strftime('%Y-%m-%d').isin(days),drop=True)
        h_hind_to_plot = harm_hind_avg.where(harm_hind_avg.time.dt.strftime('%Y-%m-%d').isin(days),drop=True)
        title='Domain mean for '+ " ".join(days)
    elif ii == 'flights':
        tend_to_plot   = samptend.where(samptend.time.dt.strftime('%Y-%m-%d').isin(joanne.time.dt.strftime('%Y-%m-%d')),drop=True)
        h_clim_to_plot = harm_clim_avg.where(harm_clim_avg.time.dt.strftime('%Y-%m-%d').isin(joanne.time.dt.strftime('%Y-%m-%d')),drop=True)
        h_hind_to_plot = harm_hind_avg.where(harm_hind_avg.time.dt.strftime('%Y-%m-%d').isin(joanne.time.dt.strftime('%Y-%m-%d')),drop=True)
        title='Domain mean for flight days'
        plt_obs = True

days_all = np.arange(temp_hrs[0], temp_hrs[1])
acc_time = 3600*1
var=['u','v']
# var=['qt','thl']
# for group in [0,1,2,3]:
#     if group <= profiles_daily.group.max()
#         days = profiles_daily.where(profiles_daily.group==group,drop=True).time
#     else: days = days_all

# samptend.where(samptend['time'].dt.strftime('%Y-%m-%d').isin(days_all),drop=True)


fig, axs = plt.subplots(1,len(var),figsize=(10,9))
axs = axs.ravel()
temp = 0
for ii in var:
    if ii == 'qt': unit = 1000
    else: unit = 1
    axs[temp].plot(unit*acc_time*tend_to_plot[ii+'tendlsall'].\
                mean('time'),tend_to_plot.z,c='k',label='Large scale')
    # axs[temp].plot(unit*acc_time*tend_to_plot[ii+'tendadvall'].\
    #             mean('time'),tend_to_plot.z,c='b',label='Advective')
    # axs[temp].plot(unit*acc_time*tend_to_plot[ii+'tenddifall'].\
    #             mean('time'),tend_to_plot.z,c='g',label='Diffusive')     
    axs[temp].plot(unit*acc_time*tend_to_plot[ii+'tendtotall'].\
                mean('time'),tend_to_plot.z,c='r',label='Net')
    axs[temp].plot(unit*acc_time*(tend_to_plot[ii+'tendtotall'] - tend_to_plot[ii+'tendlsall']).\
                mean('time'),tend_to_plot.z,c='c',label='Net - LS')
    if ii =='thl' or ii=='qt':
        axs[temp].plot(unit*acc_time*tend_to_plot[ii+'tendmicroall'].\
                mean('time'),tend_to_plot.z,c='orange',label='Microphysics') 
    if ii =='thl':
        axs[temp].plot(unit*acc_time*tend_to_plot[ii+'tendradall'].\
                    mean('time'),tend_to_plot.z,c='m',label='Radiation')    
            
    # axs[temp].plot(unit*acc_time*ls_flux['d'+ii+'dt'].\
    #             sel(time=slice(temp_hrs[0],temp_hrs[1])).mean('time'),ls_flux.z,c='k',ls='-.',lw=2,label='forcing')
    
    ## HARMONIE cy40 hind
    axs[temp].plot(unit*acc_time*h_hind_to_plot['dt'+ii+'_dyn'].\
                mean('time'),h_hind_to_plot['z'],c='k',ls='--',lw=2,label='H.hind dyn') 
    axs[temp].plot(unit*acc_time*h_hind_to_plot['dt'+ii+'_phy'].\
                mean('time'),h_hind_to_plot['z'],c='c',ls='--',lw=2,label='H.hind phy') 
    # axs[temp].plot(unit*acc_time*(h_hind_to_plot['dt'+ii+'_phy']+h_hind_to_plot['dt'+ii+'_dyn']).\
                # mean('time'),h_hind_to_plot['z'],c='r',ls='--',lw=2,label='H.hind net')

    ## HARMONIE cy43 clim
    axs[temp].plot(unit*60/acc_time*h_clim_to_plot['dt'+ii+'_dyn'].\
                mean('time'),h_clim_to_plot['z'],c='k',ls=':',lw=2,label='H.clim dyn') 
    axs[temp].plot(unit*60/acc_time*h_clim_to_plot['dt'+ii+'_phy'].\
                mean('time'),harm_clim_avg['z'],c='c',ls=':',lw=2,label='H.clim phy')
    axs[temp].plot(unit*60/acc_time*(h_clim_to_plot['dt'+ii+'_phy']+h_clim_to_plot['dt'+ii+'_dyn']).\
                mean('time'),harm_clim_avg['z'],c='r',ls=':',lw=2,label='H.clim net')
    
    ## OBS
    if plt_obs:
        if ii =='u' or ii=='v':
            axs[temp].plot(unit*acc_time*joanne['F_'+ii].\
                            mean('time'),joanne['Height'],c='c',ls='-.',label='Joanne F')
            axs[temp].plot(unit*acc_time*joanne['dyn_'+ii+'_tend'].\
                           mean('time'),joanne['Height'],c='k',ls='-.',label='Joanne dyn')
            axs[temp].plot(unit*acc_time*joanne['d'+ii+'dt_fl'].\
                           mean('time'),joanne['Height'],c='r',ls='-.',label='Joanne net')
        
    axs[temp].set_title(ii+' tendency')
    axs[temp].axvline(0,c='k',lw=0.5)
    axs[temp].set_xlim([-0.81,+0.81])
    axs[temp].set_ylim(height_lim)
    if ii == 'thl':
        axs[temp].set_xlabel('K/s /hour')
    elif ii == 'qt':
        axs[temp].set_xlabel('g/kg /hour')
        axs[temp].legend(frameon=False)
    else:
        axs[temp].set_xlabel('(m/s^2 / hour)')
        if ii=='v':
            axs[temp].legend(frameon=False)
    temp+=1
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
    plt.savefig(save_dir+'for_video_'+str(ii)+'.png')


#%% FROM HARMONIE
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

#%%

#%%

#%%
print('end.')


