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
              '011','012','013','014','015','016','017','018']
# expnr      = ['014','015','016'] 
case       = '20200202_12'
casenr     = '001'      # experiment number where to read input files 

# expnr      = ['001']
# case       = '20200209_10'
# casenr     = '001'      # experiment number where to read input files 

### Directories for runnin on VrLab
# base_dir   = '/Users/acmsavazzi/Documents/Mount/DALES/'
# Input_dir  = base_dir+'Cases/20200202_12_300km/'
# Output_dir = base_dir+'Experiments/EUREC4A/Exp_ECMWF/'+case+'/'
# save_dir   = '/Users/acmsavazzi/Documents/WORK/PhD_Year2/Figures/'
# fig_dir = os.path.abspath('{}/../../Figures/DALES')+'/'

### Directories for runnin on TU server
# base_dir   = 'staff-umbrella/cmtrace/Alessandro/'
# base_dir        = '/Users/acmsavazzi/Documents/Mount/'
# Input_dir       = base_dir+'Raw_Data/Les/Eurec4a/'+case+'/Exp_'+casenr+'/'
# dales_exp_dir   = base_dir+'Raw_Data/Les/Eurec4a/20200209_10/'
# Output_dir      = base_dir+'Raw_Data/Les/Eurec4a/'+case+'/'
# save_dir      = base_dir+'PhD_Year2/'
# LES_forc_dir    = '/Users/acmsavazzi/Documents/WORK/PhD_Year1/DATA/HARMONIE/LES_forcing_300km/'

### Directories for local
base_dir        = '/Users/acmsavazzi/Documents/WORK/PhD_Year1/DATA/DALES/'
Input_dir       = base_dir  + 'DALES/Cases/EUREC4A/20200202_12_300km/'
# dales_exp_dir   = base_dir  + 'DALES_atECMWF/outputs/20200209_10/'
dales_exp_dir   = base_dir  + 'DALES_atECMWF/outputs/20200202_12_clim'
Output_dir      = base_dir  + 'DALES_atECMWF/outputs/20200202_12_clim/'

# HARMONIE DATA 
LES_forc_dir    = base_dir  + '../HARMONIE/LES_forcing_300km/'

my_harm_dir     = '/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/HARMONIE/cy43_clim/average_300km/'
# IFS DATA
ifs_dir         = '/Users/acmsavazzi/Documents/WORK/Research/MyData/'
# OBS DATA
obs_dir         = '/Users/acmsavazzi/Documents/WORK/Research/MyData/'
Aers_Dship_dir  = '/Users/acmsavazzi/Documents/WORK/Data/Aers-Dship/'
#SAVE DIRECTORY 
save_dir        = '/Users/acmsavazzi/Documents/WORK/PhD_Year2/'

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

#%%     OPTIONS FOR PLOTTING

# col=['b','r','g','orange','k']
col=['red','coral','maroon','blue','cornflowerblue','darkblue','green','lime','forestgreen','m']
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
samptend   = xr.open_mfdataset(samptend_files, combine='by_coords')
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

#%% Import DALES sensitivity experimetns
if comp_experiments:
    prof_files      = [] 
    tmser_files     = []
    samptend_files  = []
    moments_files   = []
    for path,subdir,files in os.walk(dales_exp_dir):
        if path[-3:] in ['001','002','003','004','005','006','007','008','009','010']: 
            for file in glob(os.path.join(path, 'profiles*.nc')):
                prof_files.append(file)
            for file in glob(os.path.join(path, 'tmser*.nc')):
                tmser_files.append(file)
            for file in glob(os.path.join(path, 'samptend*.nc')):
                samptend_files.append(file)
            for file in glob(os.path.join(path, 'moments*')):
                moments_files.append(file)
    
    ####     profiles.nc    ####    
    print("Reading DALES exp profiles.")      
    prof_isurf5 = xr.open_mfdataset(prof_files, combine='by_coords')
    prof_isurf5['time'] = np.datetime64('2020-02-02') + prof_isurf5.time.astype("timedelta64[s]")
    # interpolate half level to full level
    prof_isurf5 = prof_isurf5.interp(zm=prof_isurf5.zt)
    prof_isurf5 = prof_isurf5.rename({'zt':'z'})
    
    ####     tmser.nc   ####
    print("Reading DALES exp time series.") 
    tmser_isurf5 = xr.open_mfdataset(tmser_files, combine='by_coords')
    tmser_isurf5['time'] = np.datetime64('2020-02-02') + tmser_isurf5.time.astype("timedelta64[s]")
    
    ####     samptend.nc    ####
    print("Reading DALES exp tendencies.") 
    tend_isurf5   = xr.open_mfdataset(samptend_files, combine='by_coords')
    tend_isurf5['time'] = np.datetime64('2020-02-02') + tend_isurf5.time.astype("timedelta64[s]")
    # interpolate half level to full level
    tend_isurf5 = tend_isurf5.interp(zm=tend_isurf5.zt)
    tend_isurf5 = tend_isurf5.rename({'zt':'z'})

    ####     moments.001    ####
    print("Reading DALES moments.") 
    colnames = ['lev','z','pres','thl2','thv2','th2','qt2','u2','v2','hght','w2','skew','sfs-tke']
    mom_isurf5  = []
    for file in np.sort(moments_files):
        temp    = pd.read_csv(file,\
                skiprows=lambda x: logic(x),comment='#',\
                delimiter = " ",names=colnames,index_col=False,skipinitialspace=True)
        mom_isurf5.append(temp)
    mom_isurf5 = pd.concat(mom_isurf5, axis=0, ignore_index=True)
    mom_isurf5['time'] = (mom_isurf5.index.values//(levels-1))*(900)+900
    mom_isurf5.set_index(['time', 'z'], inplace=True)
    mom_isurf5 = mom_isurf5.to_xarray()
    mom_isurf5['time'] = srt_time + mom_isurf5.time.astype("timedelta64[s]")

#%% Import Harmonie
### Import large scale spatial means (used for LES forcing)
print("Reading HARMONIE spatial mean (used for LES forcing).") 
harm_hind_avg = {}
if LES_forc_HARMONIE:
    harm_avg_domains = ['150','300','500']
else: harm_avg_domains = ['300']
for avg in harm_avg_domains:

    harm_hind_avg[avg] = xr.open_dataset(LES_forc_dir+'../LES_forcing_'+avg+'km/LES_forcing_202002'+\
                          srt_time.astype(str)[8:10]+'00.nc')
    for ii in np.arange(srt_time, end_time)[1:]:    
        harm_hind_avg[avg] = xr.concat((harm_hind_avg[avg],xr.open_dataset(LES_forc_dir+\
                 '../LES_forcing_'+avg+'km/LES_forcing_202002'+ii.astype(str)[8:10]+'00.nc')),dim='time')
    #
    harm_hind_avg[avg] = calc_geo_height(harm_hind_avg[avg],fliplevels=True)
    z_ref = harm_hind_avg[avg].z.mean('time')
    zz    = harm_hind_avg[avg].z
    
    for var in list(harm_hind_avg[avg].keys()):
        if 'level' in harm_hind_avg[avg][var].dims:
            print("interpolating variable "+var)
            x = np.empty((len(harm_hind_avg[avg]['time']),len(harm_hind_avg[avg]['level'])))
            x[:] = np.NaN
            for a in range(len(harm_hind_avg[avg].time)):
                x[a,:] = np.interp(z_ref,zz[a,:],harm_hind_avg[avg][var].isel(time = a))            
            harm_hind_avg[avg][var] = (("time","level"), x)    
    # convert model levels to height levels
    harm_hind_avg[avg] = harm_hind_avg[avg].rename({'z':'geo_height'})
    harm_hind_avg[avg] = harm_hind_avg[avg].rename({'level':'z'})
    harm_hind_avg[avg] = harm_hind_avg[avg].rename({'q':'qt','dtq_phy':'dtqt_phy','dtq_dyn':'dtqt_dyn'})
    harm_hind_avg[avg]["z"] = (z_ref-z_ref.min()).values
    harm_hind_avg[avg]['z'] = harm_hind_avg[avg].z.assign_attrs(units='m',long_name='Height')

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
    
### 2D fields    
###      # Read cloud fraction
### !!!  THIS SHOULD BE MOVED TO HARMONIE_BASIC.PY !!!
# print("Reading 2D HARMONIE data.") 
# nc_files = []
# for EXT in ["clt_his*.nc","cll_his*.nc","clm_his*.nc","clh_his*.nc","clwvi_his*.nc","clivi_his*.nc"]:
#     for file in glob(os.path.join(harmonie_dir, EXT)):
#         if harmonie_time_to_keep in file:
#             nc_files.append(file) 
# try:
#     nc_data_cl  = xr.open_mfdataset(nc_files, combine='by_coords')
# except TypeError:
#     nc_data_cl  = xr.open_mfdataset(nc_files)
    


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

## meteor fluxes
Meteor = xr.open_dataset(Aers_Dship_dir+'../EUREC4A_Meteor_surface_heat_fluxes_20200115_v1.0.nc')\
    .sel(time=slice(srt_time,end_time))


####
joanne['start_flight'] = ds_obs['drop']['launch_time'].resample(launch_time = "1D").first().dropna(dim='launch_time').rename({'launch_time':'time'})
joanne['end_flight']  = ds_obs['drop']['launch_time'].resample(launch_time = "1D").last().dropna(dim='launch_time').rename({'launch_time':'time'})
#everything in UTC
joanne['start_flight'] = joanne['start_flight'] + np.timedelta64(4,'h')
joanne['end_flight']   = joanne['end_flight']   + np.timedelta64(4,'h')
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
profiles['th']      = profiles['thl'] + Lv / (cp * calc_exner(profiles['p'])) * profiles['ql']
profiles['T']       = calc_T(profiles['th'],profiles['p'])
profiles['K_dif_u'] = - profiles.uwt / profiles.du_dz
profiles['K_dif_v'] = - profiles.vwt / profiles.dv_dz

for var in ['u','v','thl','qt']:
    if var+'tendphyall' not in samptend:
        samptend[var+'tendphyall'] = samptend[var+'tendtotall'] - samptend[var+'tendlsall']

## DALES isurf 5
if comp_experiments:
    prof_isurf5 = prof_isurf5.rename({'presh':'p'})
    prof_isurf5['wspd']    = np.sqrt(prof_isurf5['u']**2 + prof_isurf5['v']**2)
    prof_isurf5['th']   = prof_isurf5['thl'] + Lv / (cp * calc_exner(prof_isurf5['p'])) * prof_isurf5['ql']
    prof_isurf5['T']    = calc_T(prof_isurf5['th'],prof_isurf5['p'])


## for HARMONIE cy40
nudge['wspd']    = np.sqrt(nudge['u']**2    + nudge['v']**2)

### for HARMONIE cy40
# convert temperature to potential temperature
for avg in harm_avg_domains:
    harm_hind_avg[avg]['th']    = calc_th(harm_hind_avg[avg].T,harm_hind_avg[avg].p)
    harm_hind_avg[avg]['thl']   = calc_thl(harm_hind_avg[avg]['th'],harm_hind_avg[avg]['ql'],harm_hind_avg[avg]['p'])
    harm_hind_avg[avg]['wspd']  = np.sqrt(harm_hind_avg[avg]['u']**2 + harm_hind_avg[avg]['v']**2)
    harm_hind_avg[avg]['du_dz'] = harm_hind_avg[avg]['u'].differentiate('z')
    harm_hind_avg[avg]['dv_dz'] = harm_hind_avg[avg]['v'].differentiate('z')
    for ii in ['phy','dyn']:
        harm_hind_avg[avg]['dtthl_'+ii]=calc_th(harm_hind_avg[avg]['dtT_'+ii],harm_hind_avg[avg].p) - Lv / \
            (cp *calc_exner(harm_hind_avg[avg].p)) * harm_hind_avg[avg]['dtqc_'+ii]
            
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
harm_hind_daily = harm_hind_avg['300'].resample(time='D').mean('time')

acc_time = 3600*1
tend_daily= (acc_time*samptend).resample(time='D').mean('time')
comp_tend = (acc_time*samptend).groupby(samptend.time.dt.hour)
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

### grouped by large scale forcings
# for ii in ['u','v','vwt','qt','uwt']:
#     plt.figure(figsize=(6,9))
#     for key, group in profiles_daily.groupby(tend_daily['group_LS']):
#         group[ii].mean('time').plot(y='z',c=col[key*3],lw=3,label='Group '+str(key))
#         group[ii].plot.line(y='z',c=col[key*3],lw=0.7,alpha=0.5,add_legend=False)
#     plt.legend()
#     plt.title('group_LS',size=20)
#     plt.ylim(height_lim)
#     # plt.xlim([-17,3])
#     plt.axvline(0,c='k',lw=0.5)


#%%
### grouped by others
day_interval    = [10,16]
night_interval  = [22,4]

days = ['2020-02-04','2020-02-06']
def find_time_interval(time_1,time_2):
    if time_1 > time_2:
        temp= list(range(time_1,24)) + list(range(0,time_2+1))
    else: temp= list(range(time_1,time_2+1))
    return temp

ii = 'all'
if ii == 'all':
    hrs_to_plot = profiles.sel(time=slice(temp_hrs[0],temp_hrs[1]))
    label = 'All days'
    title='Domain and Temporal mean'
elif ii == 'days':
    hrs_to_plot = profiles.where(profiles.time.dt.strftime('%Y-%m-%d').isin(days),drop=True)
    label = str(days)
    title='Domain mean for '+ " ".join(days)


for group_by in ['groups']:        
    if group_by == 'day_night':
        hrs_to_plot_day = hrs_to_plot.sel(time=hrs_to_plot['time.hour'].\
                            isin(find_time_interval(day_interval[0],day_interval[1])))
        hrs_to_plot_night = hrs_to_plot.sel(time=hrs_to_plot['time.hour'].\
                            isin(find_time_interval(night_interval[0],night_interval[1])))

    ## cloud fraction
    plt.figure(figsize=(6,9))
    plt.suptitle('Cloud fraction')
    hrs_to_plot['cfrac'].mean('time').plot(y='z',c=col[0],lw=2,label='Cloud fraction')
    for t in profiles_daily.time:
        profiles_daily.sel(time=t)['cfrac'].plot(y='z',c='grey',lw=0.5,alpha=0.4)
    if group_by == 'flights':
        profiles.where(profiles.time.dt.strftime('%Y-%m-%d').isin(joanne.time.dt.strftime('%Y-%m-%d')),drop=True)['cfrac'].mean('time').plot(y='z',c=col[3],lw=2,label='Flights')
    if group_by =='groups':       
        for key in profiles_daily_clusteres['group_shear'].values:
            profiles_daily_clusteres.sel(group_shear=key)['cfrac'].plot(y='z',c=col[key*3],lw=1,label='Group '+str(key))
        title = 'Groups by wind shear'
        plt.title(title)
    if group_by == 'day_night':
        hrs_to_plot_day['cfrac'].mean('time').plot(y='z',c=col[1],lw=1,label=str(day_interval)+' UTC')
        hrs_to_plot_night['cfrac'].mean('time').plot(y='z',c=col[2],lw=1,label=str(night_interval)+' UTC')
    # harm_clim_avg.sel(time=hrs_to_plot.time,method='nearest').cl.mean('time').plot(y='z',label='H. cy43 clim')
    plt.legend()
    plt.xlabel('%')
    plt.ylim(height_lim)
    # plt.savefig(save_dir+'mean_cfrac.pdf')
    
    ## winds
    plt.figure(figsize=(6,9))
    plt.suptitle('Winds')
    for idx,var in enumerate(['u','v']):
        hrs_to_plot[var].mean('time').plot(y='z',c=col[idx*3],lw=2, label='DALES '+var)
        if group_by == 'flights':
            profiles.where(profiles.time.dt.strftime('%Y-%m-%d').isin(joanne.time.dt.strftime('%Y-%m-%d')),drop=True)[var].mean('time').plot(y='z',ls='--',c=col[idx*3],lw=2,label='Flights')
        if group_by =='groups':
            for key in profiles_daily_clusteres['group_shear'].values:
                profiles_daily_clusteres.sel(group_shear=key)[var].plot(y='z',c=col[idx*3+key],lw=1,label='Group '+str(key))
            title = 'Groups by wind shear'
            plt.title(title)
        if group_by == 'day_night':
            hrs_to_plot_day[var].mean('time').plot(y='z',c=col[idx*3+1],lw=1, label=str(day_interval)+' UTC')
            hrs_to_plot_night[var].mean('time').plot(y='z',c=col[idx*3+2],lw=1, label=str(night_interval)+' UTC')
    # if var in nudge:
    #     nudge[var].mean('time').plot(y='z',c=adjust_lightness(col[idx]),lw=0.8,label='HARMONIE '+var)
    plt.legend()
    plt.xlabel('m/s')
    plt.axvline(0,c='k',lw=0.5)
    plt.ylim(height_lim)
    plt.xlim([-12.5,0.5])
    plt.savefig(save_dir+'mean_winds.pdf')
    
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
    plt.savefig(save_dir+'mean_momflux.pdf')
        
    ## counter gradient fluxes
    plt.figure(figsize=(6,9))
    plt.suptitle('Counter gradient transport')
    for idx,var in enumerate(['u','v']):
        (hrs_to_plot[var+'wt'] * hrs_to_plot['d'+var+'_dz']).mean('time').\
            plot(y='z',c=col[idx*3],lw=2,label=var+'w d'+var+'_dz')
        if group_by == 'flights':
            (profiles[var+'wt'] * profiles['d'+var+'_dz']).where(profiles.time.dt.strftime('%Y-%m-%d')\
                        .isin(joanne.time.dt.strftime('%Y-%m-%d')),drop=True)\
                .mean('time').plot(y='z',ls='--',c=col[idx*3],lw=2,label='Flights')
        if group_by =='groups':
            for key in profiles_daily_clusteres['group_shear'].values:
                (profiles_daily_clusteres[var+'wt'] * profiles_daily_clusteres['d'+var+'_dz'])\
                    .sel(group_shear=key).plot(y='z',c=col[idx*3+key],lw=1,label='Group '+str(key))
                plt.title('Groups by wind shear')

        if group_by == 'day_night':
            (hrs_to_plot_day[var+'wt'] * hrs_to_plot_day['d'+var+'_dz']).mean('time').\
                plot(y='z',c=col[idx*3+1],lw=1,label=str(day_interval)+' UTC')
            (hrs_to_plot_night[var+'wt'] * hrs_to_plot_night['d'+var+'_dz']).mean('time').\
                plot(y='z',c=col[idx*3+2],lw=1,label=str(night_interval)+' UTC')
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
    plt.savefig(save_dir+'mean_momfluxconv.pdf')
    
#%%  TIMESERIES
## momentum fluxes
for var in ['uwt','vwt']:
    plt.figure(figsize=(19,5))
    plt.suptitle('Momentum flux')
    (profiles[var]).plot(y='z',vmin=-0.13)
    plt.ylim(height_lim)
    plt.xlim([srt_time,end_time])
    for ii in np.arange(srt_time, end_time):
        plt.axvline(x=ii,c='k')
    # plt.savefig(save_dir+'Figures/tmser_'+var+'.pdf')

#%% CLOUD LAYER timeseries
exp_prof  = profiles
exp_tmser = tmser

plt.figure(figsize=(19,5))
exp_prof.cfrac.plot(y="z",cmap=plt.cm.Blues_r,vmax=0.005,vmin=0)
exp_tmser.zc_max.rolling(time=30, center=True).mean().plot(c='r',ls='-')
exp_tmser.zc_av.plot(c='r',ls='--')
# exp_tmser.zb.plot(c='k')
plt.ylim([0,8000])
for tm in np.arange(srt_time, end_time):
    plt.axvline(x=tm,c='k')
    
plt.figure(figsize=(19,5))
harm_clim_avg.cl.plot(x='time',cmap=plt.cm.Blues_r,vmax=0.005,vmin=0)
exp_tmser.zc_max.rolling(time=30, center=True).mean().plot(c='r',ls='-')
exp_tmser.zc_av.plot(c='r',ls='--')
plt.ylim([0,7000])
plt.xlim([None,'2020-02-11'])
for tm in np.arange(srt_time, end_time):
    plt.axvline(x=tm,c='k')
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
    
acc_time = 3600*1
# for ii in ['thl','qt','ql','wspd']:
for ii in ['qttendlsall','utendlsall']:
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
for var in ['u','qt','T']:
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
    plt.figure(figsize=(19,5))
    # (unit*profiles[var]).plot(x="time",vmin=vmin,vmax=vmax)
    # (unit*harm_hind_avg['300'][var]).plot(x="time",vmin=vmin,vmax=vmax)
    (unit*harm_clim_avg[var]).plot(x="time",vmin=vmin,vmax=vmax)
    plt.ylim([0,5000])
    plt.title('HARMONIE clim '+var, size=20)
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
    # (unit*3600*ls_flux['d'+ii+'dt']).plot(x='time',vmax=vmax)
    (unit*3600*harm_hind_avg['300']['dt'+ii+'_dyn']).plot(x='time',vmax=vmax)
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
    for level in [1000]: # meters
        for var in ['thl','qt','wspd']:
            ## Temperature    
            plt.figure(figsize=(15,6))
            plt.plot(profiles.time,profiles[var].sel(z=level,method='nearest'),c=col[3],label='DALES')
            if comp_experiments:
                plt.plot(prof_isurf5.time,prof_isurf5[var].sel(z=level,method='nearest'),c=col[5],label='isurf5')
                # plt.plot(profiles_clim.time,profiles_clim[var].sel(z=level,method='nearest'),c=col[5],label='DALES clim')
            plt.plot(harm_hind_avg['300'].time,harm_hind_avg['300'][var].sel(z=level,method='nearest'),c=col[0],label='HARMONIE cy40')
            if var in ds_obs['drop']:
                plt.scatter((ds_obs['drop'].launch_time  + np.timedelta64(4, 'h')).values,\
                        ds_obs['drop'].sel(Height=level,method='nearest').sel(launch_time=slice(srt_time,end_time))[var].values,c=col[2],alpha = 0.5,s=12,label='Dropsondes')
            if var in ds_obs['radio']:
                plt.scatter((ds_obs['radio'].launch_time + np.timedelta64(4, 'h')).values,\
                        ds_obs['radio'].sel(Height=level,method='nearest').sel(launch_time=slice(srt_time,end_time))[var].values,c=col[6],alpha = 0.5,s=12,label='Radiosondes')
            if var in era5:
                plt.plot(era5.Date.sel(Date=slice(srt_time,end_time)),era5[var].sel(Height=level,method='nearest').sel(Date=slice(srt_time,end_time)).mean('Mypoint'), label='ERA5')
            if harm_3d:
                plt.plot(harm_clim_avg.time,harm_clim_avg[var].sel(z=level,method='nearest'),c=col[8],label='HARMONIE cy43 clim')
            plt.xlabel('time')
            plt.ylabel(var)
            plt.title( var+' at '+str(level)+' m',size=20)
            plt.xlim(temp_hrs)
            plt.axvspan(srt_time,srt_time + np.timedelta64(2, 'h'), alpha=0.2, color='grey')
            plt.legend()
            for day in np.arange(srt_time,end_time):
                plt.axvline(x=day,c='k',lw=0.5)
        # plt.savefig(save_dir+'wspd.pdf')
          
#%%
    ### SURFACE LATENT HEAT FLUX 
    plt.figure(figsize=(15,6))
    plt.plot(tmser.time, ls_surf['rho'].mean() * tmser.wq * Lv,c=col[1],lw=0.9,label='DALES')
    if comp_experiments:
        plt.plot(tmser_isurf5.time, ls_surf['rho'].mean() * tmser_isurf5.wq * Lv,c=col[5],lw=0.7,label='DALES exp')
    # harm_hind_avg['300'].LE.plot()
    harm_clim_avg.hfls.mean(dim=['x','y']).plot(c=col[6],lw=2,label='HARMONIE_cy43 clim')
    xr.plot.scatter(Meteor,'time','LHF_bulk_mast',alpha = 0.6,s=10,c=col[2],label='Meteor')
    xr.plot.scatter(Meteor,'time','LHF_EC_mast',alpha = 0.4,s=10,label='EC')
    plt.xlabel('time')
    plt.ylabel('LH (W/m2)')
    plt.title('Surface latent heat flux',size=20)
    plt.xlim(temp_hrs)
    plt.axvspan(srt_time,srt_time + np.timedelta64(2, 'h'), alpha=0.2, color='grey')
    plt.legend()
    for day in np.arange(srt_time,end_time):
        plt.axvline(x=day,c='k',lw=0.5)
        
    ### SURFACE SENSIBLE HEAT FLUX
    plt.figure(figsize=(15,6))
    plt.plot(tmser.time, rho * tmser.wtheta * cp,c=col[1],lw=0.7,label='DALES')
    if comp_experiments:
        plt.plot(tmser_isurf5.time, rho * tmser_isurf5.wtheta * cp,c=col[5],lw=0.9,label='DALES exp')
    harm_hind_avg['300'].H.plot(c=col[0],lw=2,label='HARMONIE_cy40 hind')
    harm_clim_avg.hfss.mean(dim=['x','y']).plot(c=col[6],lw=2,label='HARMONIE_cy43 clim')
    xr.plot.scatter(Meteor,'time','SHF_bulk_mast',alpha = 0.6,s=10,c=col[2],label='Meteor')
    xr.plot.scatter(Meteor,'time','SHF_EC_mast',alpha = 0.4,s=10,label='EC')
    plt.xlabel('time')
    plt.ylabel('SH ($W/m^2$)')
    plt.title('Surface sensible heat flux',size=20)
    plt.xlim(temp_hrs)
    plt.axvspan(srt_time,srt_time + np.timedelta64(2, 'h'), alpha=0.2, color='grey')
    plt.legend()
    for day in np.arange(srt_time,end_time):
        plt.axvline(x=day,c='k',lw=0.5)
        
    # flux profiles
    plt.figure(figsize=(10,9))
    (profiles.rhof * profiles.wqtt * Lv).plot(y='z',vmin=0,vmax=+500, cmap='coolwarm')
    plt.ylim([0,4000])
    plt.title('Latent heat flux',size=20)
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
var = 'v'
rol = 4
composite = True
acc_time = 3600*1

dales_to_plot   = samptend.sel(z=slice(layer[0],layer[1])).mean('z')\
    .sel(time=slice(np.datetime64('2020-02-02'),np.datetime64('2020-02-09')))
h_clim_to_plot = harm_clim_avg.sel(z=slice(layer[0],layer[1])).mean('z')\
    .sel(time=slice(np.datetime64('2020-02-02'),np.datetime64('2020-02-09')))

plt.figure(figsize=(15,6))
if composite:
    ## DALES
    acc_time*dales_to_plot.groupby(dales_to_plot.time.dt.hour).mean()[var+'tendtotall'].plot(c='r',label='DALES: Tot')
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
    acc_time*dales_to_plot.rolling(time=rol).mean()[var+'tendtotall'].plot(c='r',label='DALES: Tot')
    acc_time*dales_to_plot.rolling(time=rol).mean()[var+'tendlsall'].plot(c='k', label='DALES: LS')
    acc_time*dales_to_plot.rolling(time=rol).mean()[var+'tendphyall'].plot(c='c',label='DALES: Tot - LS')
    
    ## HARMONIE cy43 clim
    acc_time*(h_clim_to_plot['dt'+var+'_dyn']+h_clim_to_plot['dt'+var+'_phy']).plot(c='r',ls=':',label='H.clim cy43: Tot')
    acc_time*h_clim_to_plot['dt'+var+'_dyn'].plot(c='k',ls=':',label='H.clim cy43: Dyn')
    acc_time*h_clim_to_plot['dt'+var+'_phy'].plot(c='c',ls=':',label='H.clim cy43: Phy')

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
    plt.ylim([-0.0000001, 0.0000001]) 
    plt.ylabel('Tendency (g/kg /hour)')
else:
    plt.ylim([-0.0002,0.0002])
    plt.ylabel('Tendency (m/s /hour)')
    

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
days = ['2020-02-02','2020-02-05','2020-02-06','2020-02-07']
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
        tend_to_plot   = exp_tend.where(exp_tend.time.dt.strftime('%Y-%m-%d').isin(days),drop=True)
        h_clim_to_plot = harm_clim_avg.where(harm_clim_avg.time.dt.strftime('%Y-%m-%d').isin(days),drop=True)
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
        
        
        # h_clim_to_plot = harm_clim_avg.where(harm_clim_avg.time.dt.strftime('%Y-%m-%d').isin(joanne.time.dt.strftime('%Y-%m-%d')),drop=True)
        # for avg in harm_avg_domains:
            # h_hind_to_plot[avg] = harm_hind_avg[avg].where(harm_hind_avg[avg].time.dt.strftime('%Y-%m-%d').isin(joanne.time.dt.strftime('%Y-%m-%d')),drop=True)
        title='Domain mean for flight days'
        plt_obs = True

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
                    mean('time'),tend_to_plot.z,c='k',label='Large scale')
        axs[temp].plot(unit*acc_time*tend_to_plot[ii+'tendadv'+samp].\
                    mean('time'),tend_to_plot.z,c='b',label='Advective')
        axs[temp].plot(unit*acc_time*tend_to_plot[ii+'tenddif'+samp].\
                    mean('time'),tend_to_plot.z,c='g',label='Diffusive') 
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
               mean('time'),tend_to_plot.z,c='r',label='Net')
            # axs[temp].plot(unit*acc_time*(tend_to_plot[ii+'tendtot'+samp] - tend_to_plot[ii+'tendls'+samp]).\
            #             mean('time'),tend_to_plot.z,c='c',label='Net - LS')
            axs[temp].plot(unit*acc_time*(tend_to_plot[ii+'tendadv'+samp] + tend_to_plot[ii+'tenddif'+samp]).\
                        mean('time'),tend_to_plot.z,c='c',label='Adv + Dif')
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

    # ## HARMONIE cy43 clim
    # axs[temp].plot(unit*acc_time*h_clim_to_plot['dt'+ii+'_dyn'].\
    #             mean('time'),h_clim_to_plot['z'],c='k',ls=':',lw=2,label='H.clim dyn') 
    axs[temp].plot(unit*acc_time*h_clim_to_plot['dt'+ii+'_phy'].\
                mean('time'),h_clim_to_plot['z'],c='c',ls=':',lw=2,label='H.clim phy')
    axs[temp].plot(unit*acc_time*(h_clim_to_plot['dt'+ii+'_phy']+h_clim_to_plot['dt'+ii+'_dyn']).\
                mean('time'),h_clim_to_plot['z'],c='r',ls=':',lw=2,label='H.clim net')
    if ii == 'u' or ii == 'v':
        axs[temp].plot(unit*acc_time*h_clim_to_plot['dt'+ii+'_turb'].\
            mean('time'),h_clim_to_plot['z'],c='g',ls=':',lw=2,label='H.clim turb')
        axs[temp].plot(unit*acc_time*h_clim_to_plot['dt'+ii+'_conv'].\
            mean('time'),h_clim_to_plot['z'],c='b',ls=':',lw=2,label='H.clim conv')
    
    
    ## OBS
    if plt_obs:
        if ii =='u' or ii=='v':
            axs[temp].plot(unit*acc_time*joanne['F_'+ii].\
                            mean('time'),joanne['Height'],c='c',ls='-.',label='Joanne F')
            axs[temp].plot(unit*acc_time*joanne['dyn_'+ii+'_tend'].\
                            mean('time'),joanne['Height'],c='k',ls='-.',label='Joanne dyn')
            axs[temp].plot(unit*acc_time*joanne['d'+ii+'dt_fl'].\
                            mean('time'),joanne['Height'],c='r',ls='-.',label='Joanne net')
        
    axs[temp].set_title(ii+' tendency',size=20)
    # axs[temp].set_title('Zonal momentum budget',size=15)
    axs[temp].axvline(0,c='k',lw=0.5)
    
    axs[temp].set_ylim(height_lim)
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
        axs[temp].set_xlabel('(m/s^2 / hour)')
        if samp == 'all':
            axs[temp].set_xlim([-0.5,+0.5])
        else: 
            axs[temp].set_xlim([-9,+11])
        if ii=='v':
            axs[temp].legend(frameon=False)
    temp+=1
    
# plt.savefig(save_dir+'tendency_05_Feb.pdf')
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
layer = [800,1500]
composite = False
# groupby(dales_to_plot.time.dt.hour).mean()
plt.figure(figsize=(19,5))
# for var in ['w2']:
for var in ['u2','v2','w2','thl2','qt2']:
    if composite:
        moments[var].groupby(moments.time.dt.hour).mean().sel(z=slice(layer[0],layer[1])).mean('z').plot(label=var)
    else:     
        moments[var].sel(z=slice(layer[0],layer[1])).mean('z').plot(x='time',label=var)
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


