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
import matplotlib.colors as mcolors
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
         'axes.labelsize': 24,
         'axes.titlesize':'large',
         'xtick.labelsize':20,
         'ytick.labelsize':20,
         'figure.figsize':[10,7],
         'figure.titlesize':24}
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
# case       = '20200202_12'
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
srt_time   = np.datetime64('2020-02-01T20')
end_time   = np.datetime64('2020-02-10T08')
srt_plot   = np.datetime64('2020-02-02')
end_plot   = np.datetime64('2020-02-11')
temp_hrs   = [np.datetime64('2020-02-02'),np.datetime64('2020-02-11')]
# hours = srt_time,srt_time + [np.timedelta64(2, 'h'),np.timedelta64(48, 'h'),\
#                              np.timedelta64(108, 'h'),np.timedelta64(144, 'h')]
    
make_videos       = False
LES_forc_HARMONIE = True
harm_3d           = False
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
heights_to_plot=[100,200,1500]
heights_to_plot=['200','CLbase','midCL']


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
# print("Reading backrad.")
# backrad    = xr.open_dataset(Input_dir+'backrad.inp.'+casenr+'.nc')

####     prof.inp   ####
# print("Reading prof.inp.")
# colnames = ['z (m)','thl (K)','qt (kg kg-1)','u (m s-1)','v (m s-1)','tke (m2 s-2)']
# prof = pd.read_csv(Input_dir+'prof.inp.'+casenr,header = 2,names=colnames,\
#                    index_col=False,delimiter = " ")
    
####    nudge.inp   ####
# print("Reading nudge.inp.")
# colnames = ['z','factor','u','v','w','thl','qt']    
# nudge    = pd.read_csv(Input_dir+'nudge.inp.'+casenr,\
#            skiprows=lambda x: logic(x),comment='#',\
#            delimiter = " ",names=colnames,index_col=False)
# nudge = nudge.apply(pd.to_numeric, errors='coerce')
# nudge['time'] = nudge.index.values//levels
# nudge.set_index(['time', 'z'], inplace=True)
# nudge = nudge.to_xarray()
# nudge['time'] = srt_time + nudge.time.astype("timedelta64[h]")
# nudge = nudge.sel(time=slice(srt_time,end_time))
# nudge['wspd']    = np.sqrt(nudge['u']**2    + nudge['v']**2)

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
ls_surf.time.attrs["units"] = "Local Time"
ls_surf=ls_surf.sel(time=slice(srt_plot,end_time))

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
ls_flux.time.attrs["units"] = "Local Time"
ls_flux = ls_flux.sel(time=slice(srt_plot,end_time))
ls_surf['T_s'] = calc_T(ls_surf['th_s'],ls_surf['p_s'])
ls_surf['rho'] = calc_rho(ls_surf['p_s'],ls_surf['T_s'],ls_surf['qt_s'])
rho = ls_surf['rho'].mean()
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
profiles = profiles.sel(time=slice(srt_plot,end_time))
# interpolate half level to full level
profiles = profiles.interp(zm=profiles.zt)
profiles = profiles.rename({'zt':'z'})
profiles.time.attrs["units"] = "Local Time"

####     tmser.nc   ####
print("Reading DALES time series.") 
tmser = xr.open_mfdataset(tmser_files, combine='by_coords')
tmser['time'] = srt_time + tmser.time.astype("timedelta64[s]")
tmser.time.attrs["units"] = "Local Time"
tmser = tmser.sel(time=slice(srt_plot,end_time))


####     samptend.nc    ####
print("Reading DALES tendencies.") 
samptend   = xr.open_mfdataset(np.sort(samptend_files), combine='by_coords')
samptend['time'] = srt_time + samptend.time.astype("timedelta64[s]")
# interpolate half level to full level
samptend = samptend.interp(zm=samptend.zt)
samptend = samptend.rename({'zt':'z'})
samptend.time.attrs["units"] = "Local Time"
# samptend = samptend.sel(time=slice(srt_plot,end_time))


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
moments.time.attrs["units"] = "Local Time"
moments = moments.sel(time=slice(srt_plot,end_time))



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
tmsurf.time.attrs["units"] = "Local Time"
tmsurf = tmsurf.sel(time=slice(srt_plot,end_time))



if make_videos:
    ####     fielddump.nc    ####
    # fielddump  = xr.open_dataset(Output_dir+'fielddump.000.000.'+expnr+'.nc')
    
    ####     merged_cape.nc    ####
    cape_merg_files.sort()
    cape   = xr.open_mfdataset(cape_merg_files[1:], combine='by_coords',decode_times=False)
    cape['time'] = srt_time + cape.time.astype("timedelta64[s]")
    cape.time.attrs["units"] = "Local Time"
    cape = cape.sel(time=slice(srt_plot,end_time))

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
    harm_clim_avg['time'] = harm_clim_avg.time - np.timedelta64(4, 'h')
    harm_clim_avg.time.attrs["units"] = "Local Time"
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
ds_obs['radio'].launch_time.attrs["units"] = "Local Time"
ds_obs['drop'].launch_time.attrs["units"] = "Local Time"
####
joanne['start_flight'] = ds_obs['drop']['launch_time'].resample(launch_time = "1D").first().dropna(dim='launch_time').rename({'launch_time':'time'})
joanne['end_flight']  = ds_obs['drop']['launch_time'].resample(launch_time = "1D").last().dropna(dim='launch_time').rename({'launch_time':'time'})
joanne.time.attrs["units"] = "Local Time"

#%% import ERA5
print("Reading ERA5.") 
era5=xr.open_dataset(ifs_dir+'My_ds_ifs_ERA5.nc')
era5['Date'] = era5.Date - np.timedelta64(4, 'h')
era5.Date.attrs["units"] = "Local Time"

#%% Import scale separated fluxes 
print("Reading ScaleSep files.") 
da_scales      = xr.open_dataset('/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/DALES/scale_sep_allExp.nc')
da_scales_prof = xr.open_dataset('/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/DALES/scale_sep_prof_allExp.nc')

da_scales_prof_scalar = xr.open_dataset('/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/DALES/scale_sep_prof_allExp_scalar.nc')

############################
########################################################
############################
da_scales_100      = xr.open_dataset('/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/DALES/scale_sep_allExp_100m_200m_CL.nc')
da_scales_old = da_scales
da_scales   = xr.merge([da_scales_old,da_scales_100])
# da_scales   = da_scales.fillna(0)

da_scales = da_scales_100

############################
########################################################
############################

da_scales['time'] = da_scales.time  - np.timedelta64(4, 'h')
da_scales.time.attrs["units"] = "Local Time"
da_scales = da_scales.sel(time=slice(srt_plot,end_time))
da_scales_prof['time'] = da_scales_prof.time  - np.timedelta64(4, 'h')
da_scales_prof.time.attrs["units"] = "Local Time"
da_scales_prof = da_scales_prof.sel(time=slice(srt_plot,end_time))


### From KLPS to resolution/size/scale of the filter
xsize = 150000
f_scales = np.zeros(len(da_scales.klp))
for k in range(len(da_scales.klp)):   
    if da_scales.klp[k] > 0:
        f_scales[k] = xsize/(da_scales.klp[k]*2).values  # m
    elif da_scales.klp[k] == 0:
        f_scales[k] = xsize
#%% Import organisation metrics
print("Reading org metrics.") 
da_org      = xr.open_dataset('/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/DALES/df_org_allExp.nc')
da_org['time'] = da_org.time  - np.timedelta64(4, 'h')
da_org.time.attrs["units"] = "Local Time"
da_org = da_org.sel(time=slice(srt_plot,end_time))
da_org_norm = (da_org - da_org.min()) / (da_org.max() - da_org.min())
#%% convert xt, yt into lon, lat coordinates  

#%% Boundary layer height
print("Defining Boundary Layer.") 
# Above cloudheight with ql
#maxql = profiles['ql'].max('z')    # max of humiidity 
#imax = profiles['ql'].argmax('z')  # index of max humidity 
zmax = profiles['ql'].idxmax('z')  # height of max humidity
temp = profiles['ql'].where(profiles['z']>=zmax)
hc_ql = temp.where(lambda x: x<0.0000001).idxmax(dim='z')  #height of zero humidity after maximum

# Now calculate cloud base as the height where ql becomes >0 
temp = profiles['ql'].where(profiles['z']<=zmax)
cl_base = temp.where(lambda x: x<0.0000001).idxmax(dim='z')  

# Minimum theta_v flux
# hc_thlvw = profiles['wthlt'].idxmin('z')  # height of min flux
#  
# # Combine 
# if profiles['ql'].max('z') == 0:
#     HC = hc_thlvw
# else:
#     HC = hc_ql

######
## HONNERT normalization of the filter scale
f_scales_norm_ql = f_scales[:,None] / (hc_ql).sel(time=da_scales.time).values[None,:]
da_scales['f_scales_norm_ql'] = (('klp','time'),f_scales_norm_ql)
# h   =  tmser.zi                 # boundary layer height 
# hc  =  (tmser.zc_max-tmser.zb)  # cloud layer depth
# hc  =  0
# f_scales_norm = f_scales[:,None] / (h+hc).sel(time=da_scales.time).values[None,:]
# da_scales['f_scales_norm'] = (('klp','time'),f_scales_norm)

## Normalization with horizontal lenght 
f_scales_norm_horiz = f_scales[:,None] / (da_org['spectral_length_moment']).\
                                    sel(time=da_scales.time).values[None,:]
da_scales['f_scales_norm_ho'] = (('klp','time'),f_scales_norm_horiz)

######

##############################
########## ADD UNRESOLVED FLUX,
# define sub cloud layer and mid cloud layer 
subCL = profiles.sel(z=(cl_base/2),method='nearest')
subCL = subCL.assign_coords({"height": 'subCL'}).expand_dims(dim='height')
CLbase = profiles.sel(z=(cl_base),method='nearest')
CLbase = CLbase.assign_coords({"height": 'CLbase'}).expand_dims(dim='height')
midCL = profiles.sel(z=((hc_ql-cl_base)/2 + cl_base),method='nearest')
midCL = midCL.assign_coords({"height": 'midCL'}).expand_dims(dim='height')
layer_100 = profiles.sel(z=100,method='nearest')
layer_100 = layer_100.assign_coords({"height": '100'}).expand_dims(dim='height')
layer_200 = profiles.sel(z=200,method='nearest')
layer_200 = layer_200.assign_coords({"height": '200'}).expand_dims(dim='height')

temp = xr.merge([subCL.drop(['z','zm']),CLbase.drop(['z','zm']),midCL.drop(['z','zm']),\
                 layer_100.drop(['z','zm']),layer_200.drop(['z','zm'])])

####### THIS NEXT PART DOES NOT WORK!!!!!!! !!! !!!!!!!
#initialise new variabels
da_scales['v_psfw_psf_unres'] = da_scales['v_psfw_psf']     + temp['vws']
da_scales['u_psfw_psf_unres'] = da_scales['u_psfw_psf']     + temp['uws']
da_scales['qt_psfw_psf_unres'] = da_scales['qt_psfw_psf']   + temp['wqts']
da_scales['thl_psfw_psf_unres'] = da_scales['thl_psfw_psf'] + temp['wthls']


########################################
##############################

#################
## normalize fluxes ## !!!!
## here you should normalise for klp=0.5, but not all height have that, so take 0.75
da_scales_norm = (da_scales)/(da_scales.sel(klp=np.sort(da_scales.klp)[0]))

#################

#%% SOME NEW VARIABLES
print("Computing new variables.") 
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
col_groups = ['orange','blue','green']
#%% ## FIGURE 1 ##
fig, axs = plt.subplots(3,1,figsize=(19,19))
## panel a
profiles.cfrac.plot(y="z",cmap=plt.cm.Blues_r,vmax=0.1,vmin=0,ax=axs[0]\
                    ,cbar_kwargs=dict(orientation='horizontal',
                        pad=0.03, shrink=0.5,label='Fraction'))
    
hc_ql.plot(x='time',ax=axs[0],c='k',ls='-',label='Cloud top')
# h.plot(ax=axs[0])
ax2 = axs[0].twinx()
profiles.rain.sel(z=slice(0,50)).mean('z').rolling(time=6, center=True)\
    .mean().plot(x='time',ax=ax2,c='r',ls='-',label='Rain')
ax2.set_ylim([-0.01,4])
ax2.tick_params(axis='y', colors='red')
axs[0].set_title('Cloud fraction in DALES',fontsize = 35)
axs[0].set_ylabel(r'z ($m$)',fontsize=26)
axs[0].set_ylim(height_lim)
ax2.set_ylabel(r'Rain rate ($W m^{-2}$)',color='r')
axs[0].legend(fontsize=22)
axs[0].tick_params(axis='both', which='major', labelsize=30)
    
## panel b
for level in [200]: # meters
    for idx,var in enumerate(['u','v']):
        axs[idx+1].plot(profiles.time,profiles[var].sel(z=level,method='nearest'),lw=3,c=col[3],label='DALES')
        if var in ds_obs['drop']:
            axs[idx+1].scatter((ds_obs['drop'].launch_time).values,\
                    ds_obs['drop'].sel(Height=level,method='nearest').sel(launch_time=slice(srt_time,end_time))[var].values,c=col[2],alpha = 0.5,s=12,label='Dropsondes')
        if var in ds_obs['radio']:
            axs[idx+1].scatter((ds_obs['radio'].launch_time).values,\
                    ds_obs['radio'].sel(Height=level,method='nearest').sel(launch_time=slice(srt_time,end_time))[var].values,c=col[6],alpha = 0.5,s=12,label='Radiosondes')
        if harm_3d and var in harm_clim_avg:
            axs[idx+1].plot(harm_clim_avg.time,harm_clim_avg[var].sel(z=level,method='nearest'),lw=1.5,c=col[0],label='HARMONIE')
        if var in era5:
            axs[idx+1].plot(era5.Date.sel(Date=slice(srt_time,end_time)),era5[var].sel(Height=level,method='nearest').\
                     sel(Date=slice(srt_time,end_time)).mean('Mypoint'),\
                     lw=1.5,ls='-',c=col[8], label='ERA5')
        # plt.xlabel('time')
        axs[idx+1].set_ylabel(r'$m s^{-1}$',fontsize=26)
        
        axs[idx+1].tick_params(axis='y', which='major', labelsize=30)
axs[2].axhline(0,c='k',lw=0.5)    
        
axs[1].set_title('Zonal wind at '+str(level)+' m',size=35)
axs[2].set_title('Meridional wind at '+str(level)+' m',size=35)
# axs[1].set_ylim([-17.5,-1.5])
axs[1].set_ylim([-1.5,-17.5])
axs[2].set_ylim([4.5,-7.8])
axs[1].legend(fontsize=22)

axs[2].tick_params(axis='x', which='major', labelsize=30, rotation =25)



## panel c
# layer = [0,750]
# var = 'u'
# rol = 10

# dales_to_plot   = samptend.sel(z=slice(layer[0],layer[1])).mean('z')\
#     # .sel(time=slice(np.datetime64('2020-02-02'),np.datetime64('2020-02-11')))
# h_clim_to_plot = harm_clim_avg.sel(z=slice(layer[0],layer[1])).mean('z')\
#     # .sel(time=slice(np.datetime64('2020-02-02'),np.datetime64('2020-02-11')))

# ## DALES 
# (acc_time*dales_to_plot.rolling(time=rol*4).mean()[var+'tendtotall']).plot(c='r',label='DALES: Tot',ax=axs[2])
# (acc_time*dales_to_plot.rolling(time=rol*4).mean()[var+'tendlsall']).plot(c='k', label='DALES: LS',ax=axs[2])
# (acc_time*dales_to_plot.rolling(time=rol*4).mean()[var+'tendphyall']).plot(c='g',label='DALES: Tot - LS',ax=axs[2])

# ## HARMONIE cy43 clim
# (acc_time*(h_clim_to_plot['dt'+var+'_dyn']+h_clim_to_plot['dt'+var+'_phy']).rolling(time=rol).mean()).plot(c='r',ls=':',label='HAR: Tot',ax=axs[2])
# (acc_time*h_clim_to_plot['dt'+var+'_dyn'].rolling(time=rol).mean()).plot(c='k',ls=':',label='HAR: Dyn',ax=axs[2])
# (acc_time*h_clim_to_plot['dt'+var+'_phy'].rolling(time=rol).mean()).plot(c='g',ls=':',label='HAR: Phy',ax=axs[2])


# axs[2].axhline(0,c='k',lw=0.5)
# axs[2].set_title('Mean '+var+' tendency between '+str(layer[0])+' and '+str(layer[1])+' m',fontsize=22)
# axs[2].legend(ncol=2,fontsize=15)
# axs[2].set_ylabel(r'Tendency ($m s^{-1} hour^{-1}$)')
# axs[2].set_ylim([-0.88,0.8])
axs[2].set_xlabel(None)

#####
for day in np.arange(srt_plot,end_plot):
    axs[0].axvline(x=day,c='k',lw=0.5)
    axs[1].axvline(x=day,c='k',lw=0.5)
    axs[2].axvline(x=day,c='k',lw=0.5)
axs[0].xaxis.set_visible(False) 
axs[1].xaxis.set_visible(False) 
axs[0].set_xlim([srt_plot,end_time])
axs[1].set_xlim([srt_plot,end_time])
axs[2].set_xlim([srt_plot,end_time])
plt.tight_layout()
for n, ax in enumerate(axs):
    ax.text(0.95, 1.05, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=20)
# plt.savefig(save_dir+'Figure1_tmser.pdf', bbox_inches="tight")
##################
#%% ## FIGURE 2 ##
cmap = matplotlib.cm.get_cmap('coolwarm')
rgba = 1/8
fig, axs = plt.subplots(1,4,figsize=(19,10))
for idx,var in enumerate(['u','v','thl','qt']):
    iteration = 0
    axs[idx].tick_params(axis='both', which='major', labelsize=28)
    profiles[var].mean('time').plot(y='z',c='k',lw=4, label='Mean',ax=axs[idx])
    for day in np.unique(profiles.time.dt.day)[:-1]:
        iteration +=1
        profiles[var].sel(time='2020-02-'+str(day).zfill(2)).mean('time')\
            .plot(ax=axs[idx],y='z',c=cmap(rgba*iteration),lw=1.5,label='Feb-'+str(day).zfill(2))
   
    axs[idx].set_title(var,fontsize=40)        
    axs[idx].yaxis.set_visible(False) 
    axs[idx].set_ylim(height_lim)
    if var =='u':
        axs[idx].set_ylabel(r'z ($m$)',fontsize=30)
        axs[idx].set_xlabel(r'$m s^{-1}$',fontsize=30)
        axs[idx].set_xlim([-17,0])
        axs[idx].axvline(0,c='k',lw=0.5)
        axs[idx].yaxis.set_visible(True) 
    if var =='v':
        axs[idx].set_xlabel(r'$m s^{-1}$',fontsize=30)
        axs[idx].set_xlim([-5.5,1.8])
        axs[idx].axvline(0,c='k',lw=0.5)
    if var =='thl':
        axs[idx].set_xlabel('$K$',fontsize=30)
        axs[idx].set_xlim([295,321])
        axs[idx].legend(fontsize=22)
    if var =='qt':
        axs[idx].set_xlabel(r'$g kg^{-1}$',fontsize=30)
        # axs[idx].set_xlim([300,330])
       
plt.tight_layout()
for n, ax in enumerate(axs):
    ax.text(0.9, 0.92, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=30)
plt.savefig(save_dir+'Figure2_profiles.pdf', bbox_inches="tight")    
##################
#%% ## FIGURE 3 ## mom flux contours
bottom, top = 0.1, 0.9
left, right = 0.01, 0.9

fig, axs = plt.subplots(2,2,figsize=(29,15), gridspec_kw={'width_ratios': [1,6]})
fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, \
                    hspace=0.15, wspace=0.1)
    
    
for idx,var in enumerate(['uwt','vwt']):
    iteration = 0
    profiles[var].mean('time').plot(y='z',c='k',lw=4, label='Mean',ax=axs[idx,0])
    for day in np.unique(profiles.time.dt.day)[:-1]:
        iteration +=1
        profiles[var].sel(time='2020-02-'+str(day).zfill(2)).mean('time')\
            .plot(ax=axs[idx,0],y='z',c=cmap(rgba*iteration),lw=1.5,label='Feb-'+str(day).zfill(2))
   

    
    im = (profiles[var]).plot(y='z',vmax=0.1,vmin=-0.07,\
          cmap=cm.PiYG_r,norm=mcolors.TwoSlopeNorm(0),ax=axs[idx,1],\
              add_colorbar=True,cbar_kwargs={r'label':'$m^2 s^{-2}$'})
        
    # tmser.zi.plot(x='time',ax=axs[idx,1],c='b',ls='-',label='Boundary layer')
    hc_ql.plot(x='time',ax=axs[idx,1],c='k',ls='-',label='Cloud top')
    # hc_thlvw.plot(x='time',ax=axs[idx,1],c='k',ls='-',label='Boundary layer thlw')
    

    axs[idx,0].yaxis.set_visible(True) 
    axs[idx,0].set_ylabel(r'z ($m$)',fontsize=33)
    axs[idx,0].set_xlabel(r'$m^2 s^{-2}$',fontsize=33)
    axs[idx,0].axvline(0,c='k',lw=0.5)
    axs[idx,1].yaxis.set_visible(False) 
    axs[idx,0].set_ylim(height_lim)
    axs[idx,1].set_ylim(height_lim)
    axs[idx,1].set_xlim([srt_plot,end_time])
    axs[idx,0].tick_params(axis='both', which='major', labelsize=31)
    axs[idx,1].tick_params(axis='both', which='major', labelsize=31)
    for day in np.arange(srt_plot,end_plot):
        axs[idx,1].axvline(x=day,c='k',lw=0.5)

ax2 = axs[0,1].twinx()
(profiles['u']).sel(z=200,method='nearest')\
    .plot(x='time',ax=ax2,c='b',ls='-',label='u ($m s^{-1}$)')
ax2.set_ylim([-5,-28])
ax2.tick_params(axis='y', colors='b')
ax2.set_title('')
ax2.set_ylabel('u ($m s^{-1}$)',c='b',fontsize=28)
# ax2.legend(fontsize=26)

ax3 = axs[1,1].twinx()
(profiles['v']).sel(z=200,method='nearest')\
    .plot(x='time',ax=ax3,c='b',ls='-',label='v ($m s^{-1})$')
ax3.set_ylim([3,-20])
ax3.tick_params(axis='y', colors='b')
ax3.set_title('')
ax3.set_ylabel('v ($m s^{-1}$)',c='b',fontsize=28)
# ax3.legend(fontsize=26)

axs[0,1].set_title('Zonal momentum flux',fontsize=40)   
axs[1,1].set_title('Meridional momentum flux',fontsize=40)   
axs[0,1].xaxis.set_visible(False) 
axs[0,0].legend(fontsize=21)
axs[0,1].legend(fontsize=30,loc='upper left')
axs[1,0].set_xlim([-0.04,0.04])
axs[1,0].set_xticks([-0.03,0.03])
axs[1,1].set_xticks(np.arange(srt_plot,end_plot)[1:])

axs[1,1].set_xlabel(None)
for n, ax in enumerate(axs.flat):
    ax.text(0.9, 1.05, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=30)
    
    

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
#%% ## FIGURE 5 new ##
# f_scales_norm_horiz = f_scales[:,None] / (1/(da_org['open_sky'])).\
#                                     sel(time=da_scales.time).values[None,:]
f_scales_norm_horiz = f_scales[:,None] / ((np.power(da_org['spectral_length_moment'],1))*100).\
                                    sel(time=da_scales.time).values[None,:]
# f_scales_norm_horiz = f_scales[:,None] / ((np.power(da_org['iorg'],3))*1000).\
#                                     sel(time=da_scales.time).values[None,:]
da_scales['f_scales_norm_ho'] = (('klp','time'),f_scales_norm_horiz)

fil_size = 2  # km

ih = 'midCL'

dimensionelss = 'f_scales_norm_ql'
fig, axs = plt.subplots(2,2,figsize=(15,12))
for idx, var in enumerate(['u_psfw_psf_unres','v_psfw_psf_unres']):
    # normalised y axis
    da_toplot = da_scales_norm
    for idcol,scale in enumerate([1,2]): 
        if scale == 1:
            honnert = True
            axs[1,idcol].set_xlabel(r'Dimensionless $\frac{\Delta x}{h_b}$ ',fontsize=26)

        else:
            honnert =False
            axs[1,idcol].set_xlabel(r'Filter scale $\Delta x$ [km] ',fontsize=26)

        iteration =0
        for day in ['02','03','04','05','06','07','08','09']:
            iteration +=1
            if honnert==True:
                axs[idx,idcol].plot(da_scales[dimensionelss].\
                resample(time='8h').mean('time').sel(time='2020-02-'+day),\
                  da_toplot.resample(time='8h').median('time',skipna=True)\
                      [var].sel(height=ih).sel(time='2020-02-'+day).T\
                    ,c=cmap(rgba*iteration),label=day)  
                axs[idx,idcol].axvline(1,c='k',lw=0.5)
            else:
                axs[idx,idcol].plot(f_scales/1000,\
                  da_toplot.resample(time='8h').median('time')\
                      [var].sel(height=ih).sel(time='2020-02-'+day).T\
                    ,c=cmap(rgba*iteration))   
                axs[idx,idcol].axvline(fil_size,c='k',lw=0.5)
        axs[idx,idcol].set_xscale('log')
        
        axs[idx,idcol].axhline(0,c='k',lw=0.5)
        
        
        if idcol == 0:
            if var[0]=='u':
                axs[idx,idcol].set_ylabel('Zonal fux \n partition',fontsize=26)
            if var[0]=='v':
                axs[idx,idcol].set_ylabel('Meridional flux \npartition',fontsize=26)


        axs[idx,idcol].set_ylim([-0.15,1.15])
        axs[idx,1].yaxis.set_visible(False) 
    

        # axs[0,0].legend(ncol=3)
plt.suptitle('Middle of the cloud layer',fontsize=35)
# plt.suptitle('Cloud base',fontsize=35)
# axs[0,0].set_title('Zonal momentum flux',fontsize=21)  
# axs[0,1].set_title('Meridional momentum flux',fontsize=21)  
    
for n, ax in enumerate(axs.flat):
    ax.text(0.08, 0.9, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=20)
plt.tight_layout()
# plt.savefig(save_dir+'Figure5_norm&not_'+ih+'.pdf', bbox_inches="tight")  



#%% ## FIGURE 5 ##
# f_scales_norm_horiz = f_scales[:,None] / (1/(da_org['open_sky'])).\
#                                     sel(time=da_scales.time).values[None,:]
f_scales_norm_horiz = f_scales[:,None] / ((np.power(da_org['spectral_length_moment'],1))*100).\
                                    sel(time=da_scales.time).values[None,:]
# f_scales_norm_horiz = f_scales[:,None] / ((np.power(da_org['iorg'],3))*1000).\
#                                     sel(time=da_scales.time).values[None,:]
da_scales['f_scales_norm_ho'] = (('klp','time'),f_scales_norm_horiz)

honnert = False
dimensionelss = 'f_scales_norm_ql'
fig, axs = plt.subplots(3,2,figsize=(12,12))
for idcol, var in enumerate(['u_psfw_psf_unres','v_psfw_psf_unres']):
    # normalised y axis
    da_toplot = da_scales_norm
    for idx,ih in enumerate(heights_to_plot): 
        iteration =0
        for day in ['02','03','04','05','06','07','08','09']:
            iteration +=1
            if honnert==True:
                axs[idx,idcol].plot(da_scales[dimensionelss].\
                resample(time='8h').mean('time').sel(time='2020-02-'+day),\
                  da_toplot.resample(time='8h').median('time',skipna=True)\
                      [var].sel(height=ih).sel(time='2020-02-'+day).T\
                    ,c=cmap(rgba*iteration),label=day)  
                axs[idx,idcol].axvline(1,c='k',lw=0.5)
            else:
                axs[idx,idcol].plot(f_scales/1000,\
                  da_toplot.resample(time='8h').median('time')\
                      [var].sel(height=ih).sel(time='2020-02-'+day).T\
                    ,c=cmap(rgba*iteration))   
                axs[idx,idcol].axvline(2.5,c='k',lw=0.5)
        axs[idx,idcol].set_xscale('log')
        
        axs[idx,idcol].axhline(0,c='k',lw=0.5)
        
        
        if idcol == 0:
            axs[idx,idcol].set_ylabel('Flux partition \n at '+str(ih)+' m')
        axs[idx,idcol].set_ylim([-0.15,1.15])
        axs[idx,1].yaxis.set_visible(False) 
    
        # axs[0,1].legend()
    if honnert==True:
        if dimensionelss == 'f_scales_norm_ql':
            axs[2,idcol].set_xlabel(r'Dimensionless $\frac{\Delta x}{h_b}$ ')
        else:
            axs[2,idcol].set_xlabel(r'Dimensionless $\frac{\Delta x}{h_{horiz}}$ ')
    else:
        axs[2,idcol].set_xlabel(r'Filter size $\Delta x$ ($km$)')

axs[0,0].set_title('Zonal momentum flux',fontsize=21)  
axs[0,1].set_title('Meridional momentum flux',fontsize=21)  
    
for n, ax in enumerate(axs.flat):
    ax.text(0.08, 0.9, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=13)
plt.tight_layout()
# plt.savefig(save_dir+'Figure5_momFlux_spectra.pdf', bbox_inches="tight")  
##################
#%% ## FIGURE appendix ##
honnert = True
fig, axs = plt.subplots(3,2,figsize=(12,12))
for idcol, var in enumerate(['qt_psfw_psf_unres','thl_psfw_psf_unres']):
    # normalised y axis
    da_toplot = da_scales_norm
    for idx,ih in enumerate(heights_to_plot):   
        iteration =0
        for day in ['02','03','04','05','06','07','08','09']:
            iteration +=1
            if honnert==True:
                axs[idx,idcol].plot(da_scales['f_scales_norm_ql'].\
                resample(time='8h').mean('time').sel(time='2020-02-'+day),\
                  da_toplot.resample(time='8h').median('time')\
                      [var].sel(height=ih).sel(time='2020-02-'+day).T\
                    ,c=cmap(rgba*iteration),label=day)  
                axs[idx,idcol].axvline(1,c='k',lw=0.5)
            else:
                axs[idx,idcol].plot(f_scales/1000,\
                  da_toplot.resample(time='8h').median('time')\
                      [var].sel(height=ih).sel(time='2020-02-'+day).T\
                    ,c=cmap(rgba*iteration))   
                axs[idx,idcol].axvline(2.5,c='k',lw=0.5)
        axs[idx,idcol].set_xscale('log')
        
        axs[idx,idcol].axhline(0,c='k',lw=0.5)
        
        
        if idcol == 0:
            axs[idx,idcol].set_ylabel('Flux partition \n at '+str(ih)+' m')
        axs[idx,idcol].set_ylim([-0.15,1.15])
        axs[idx,1].yaxis.set_visible(False) 
    
        # axs[0,1].legend()
    if honnert==True:
        axs[2,idcol].set_xlabel(r'Dimensionless $\frac{\Delta x}{h_b}$ ')
    else:
        axs[2,idcol].set_xlabel(r'Filter size $\Delta x$ ($km$)')

axs[0,0].set_title('Qt flux',fontsize=21)  
axs[0,1].set_title('Thl flux',fontsize=21)  
    
for n, ax in enumerate(axs.flat):
    ax.text(0.08, 0.9, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=13)
plt.tight_layout()
# plt.savefig(save_dir+'Figure5_momFlux_spectra_scalars.pdf', bbox_inches="tight")  
##################

#%% ## FIGURE 6 ## Boxplot
# The box extends from the lower to upper quartile values of the data,
# with a line at the median. 
# The whiskers extend from the box to show the range of the data
honnert = False
if honnert:
    da_to_plot = da_scales_norm.where(abs(da_scales.f_scales_norm_ql-1)<=\
                                      abs(da_scales.f_scales_norm_ql-1).min('klp').max(),\
                                          drop=True) 
else:
    filter_size = 2  # km
    da_to_plot = da_scales_norm.sel(klp=150/(filter_size*2),method='nearest')
    
ih = 'midCL'
fig, axs = plt.subplots(2,1,figsize=(13,7))
for idx, var in enumerate(['u_psfw_psf_unres','v_psfw_psf_unres']):
    if 'lab' in locals(): del lab
    if 'x_ax' in locals(): del x_ax
    iteration=-0.4                            
    for day in ['02','03','04','05','06','07','08','09','10']:
        iteration +=0.4
        if day =='10': list_hours=['00',]
        else: list_hours=['00','08','16']
        for hour in list_hours:
            iteration +=0.3
            if honnert: 
                axs[idx].boxplot(da_to_plot[var].sel(time=slice('2020-02-'+day+'T'+hour,\
                                    '2020-02-'+day+'T'+str(int(hour)+7)+':55'))\
                        .sel(height=ih).mean('klp').values,\
                            positions=[round(iteration,1)],\
                    whis=1.8,showfliers=False,showmeans=True,meanline=False,widths=0.25,\
                        medianprops=dict(color="r", lw=2))   
            else:
                axs[idx].boxplot(da_to_plot[var].sel(time=slice('2020-02-'+day+'T'+hour,\
                                '2020-02-'+day+'T'+str(int(hour)+7)+':55'))\
                    .sel(height=ih).values,\
                        positions=[round(iteration,1)],\
                whis=1.8,showfliers=False,showmeans=True,meanline=False,widths=0.25,\
                    medianprops=dict(color="r", lw=2))   
        
            if 'lab' in locals():
                lab=np.append(lab,day+'-'+str(int(hour)+4))
                x_ax=np.append(x_ax,iteration)
            else:
                lab = day+'-'+str(int(hour)+4)
                x_ax=iteration
    # 1        
    # axs[idx].plot(x_ax,da_to_plot[var].isel(time=slice(0,-1)).sel(height=ih,method='nearest')\
    #                 .mean('klp').resample(time='8h').median('time'),c='r',lw=1)
    
    axs[idx].axhline(0,c='k',lw=0.5)
    axs[idx].set_xticklabels(lab, rotation=45 )
    axs[idx].set_ylabel('Flux partition \n at '+str(ih))
    axs[idx].set_ylim([-0.2,1.4])
    axs[idx].tick_params(axis='x', which='major', labelsize=16)

if honnert:
    axs[0].set_title('Zonal momentum flux - ' +r'Mesh $\frac{\Delta x}{h_b} = 1$',fontsize=21) 
    axs[1].set_title('Meridional momentum flux - ' +r'Mesh $\frac{\Delta x}{h_b} = 1$',fontsize=21) 
else:
    axs[0].set_title('Zonal momentum flux - ' +r'Mesh $\Delta x = $'+str(filter_size)+' km',fontsize=21) 
    axs[1].set_title('Meridional momentum flux - ' +r'Mesh $\Delta x = $'+str(filter_size)+' km',fontsize=21) 
    


for n, ax in enumerate(axs.flat):
    ax.text(0.97, 0.9, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=13)
plt.tight_layout()
# plt.savefig(save_dir+'Figure6_boxplot_'+str(ih)+'.pdf', bbox_inches="tight")  
##################
#%%
time_g1={}
time_g2={}
time_g3={}

for group in ['rain','iorg','du_dz','zi']:
    if group == 'rain':
    ### grouping by rain rate            
        time_g1[group] = profiles.where(profiles[group].sel(z=slice(0,50)).mean('z').\
                          rolling(time=6, center=True).mean() <= np.quantile(profiles[group].sel(z=slice(0,50)).mean('z').\
                                        values,0.25),drop=True).time
        time_g3[group] = profiles.where(profiles[group].sel(z=slice(0,50)).mean('z').\
                                  rolling(time=6, center=True).mean()>= np.quantile(profiles[group].sel(z=slice(0,50)).mean('z').\
                                        values,0.75),drop=True).time
    if group == 'du_dz':
    ###             
        time_g1[group] = profiles.where(profiles[group].sel(z=slice(0,50)).mean('z')\
                          <= np.quantile(profiles[group].sel(z=slice(0,50)).mean('z').\
                                        values,0.25),drop=True).time
        time_g3[group] = profiles.where(profiles[group].sel(z=slice(0,50)).mean('z')\
                          >= np.quantile(profiles[group].sel(z=slice(0,50)).mean('z').\
                                        values,0.75),drop=True).time
            
    elif group=='zi':
    ### grouping by boundary layer height            
        time_g1[group] = profiles.where(tmser[group] <= \
                                    np.quantile(tmser[group].values,0.25),drop=True).time
        time_g3[group] = profiles.where(tmser[group] >= \
                                    np.quantile(tmser[group].values,0.75),drop=True).time
        ########
    elif group in da_org_norm:
    ### grouping by organisation 
        time_g1[group] = profiles.where(da_org[group] <= \
                                    da_org[group].quantile(0.25),drop=True).time
        time_g3[group] = profiles.where(da_org[group] >= \
                                    da_org[group].quantile(0.75),drop=True).time
    
    time_g2[group] = profiles.where(np.logical_not(profiles.time.\
                            isin(xr.concat((time_g1[group],time_g3[group]),'time'))),drop=True).time
        
    ##
    time_g1[group] = time_g1[group].where(time_g1[group].isin(da_scales.time),drop=True)
    time_g2[group] = time_g2[group].where(time_g2[group].isin(da_scales.time),drop=True)
    time_g3[group] = time_g3[group].where(time_g3[group].isin(da_scales.time),drop=True)

#%% ## FIGURE 7 NEW ## timeseries rain and Iorg
## Plot rain rate time series
fig, axs = plt.subplots(2,1,figsize=(12,7))
for idgroup, group in enumerate(['iorg','rain']): 
    if group in profiles:
        da_org_norm['iorg'].plot(c='k',ax=axs[idgroup],lw=2,label='$I_{org}$')
        # da_org_norm['spectral_length_moment'].plot(c='g',ax=axs[idgroup],ls='--')


        # ((hc_ql - hc_ql.min()) / (hc_ql.max() - hc_ql.min())).plot(c='b',ax=axs[idgroup])

        temp = profiles[group].sel(z=slice(0,50)).mean('z')
        temp = ((temp - temp.min()) / (temp.max() - temp.min()))
        temp.rolling(time=6, center=True).mean().plot(c='r',ax=axs[idgroup],label='Rain rate')

            # hc_ql.plot(c='b',ax=axs[idgroup])
    else:
        da_org[group].plot(c='k',ax=axs[idgroup])
        # np.power(da_org_norm['spectral_length_moment'],1).plot(c='r',ax=axs[idgroup])
        # tmser['cfrac'].plot(c='r',ax=axs[idgroup])
        da_to_plot = da_org[group]     
        
        # moments.ctop_var.plot(c='r')
        axs[idgroup].scatter(profiles.time.sel(time=time_g1[group]),\
                             da_to_plot.sel(time=time_g1[group]),c='orange',label='Group 1')
        axs[idgroup].scatter(profiles.time.sel(time=time_g2[group]),\
                             da_to_plot.sel(time=time_g2[group]),c='b',label='Group 2')
        axs[idgroup].scatter(profiles.time.sel(time=time_g3[group]),\
                             da_to_plot.sel(time=time_g3[group]),c='green',label='Group 3')


    axs[idgroup].set_xlim([srt_plot,end_time])
axs[0].xaxis.set_visible(False) 
axs[0].legend(fontsize=17,bbox_to_anchor=(0.2, 0.96),ncol=3)
axs[1].legend(fontsize=17,bbox_to_anchor=(0.3, 0.96),ncol=2)
# axs[0].set_title('Surface rain-rate',fontsize=22)
axs[1].set_ylabel(r'Rescaled')
axs[0].set_ylabel(r'$I_{org}$')
# axs[1].set_title('Organisation',fontsize=22)
for ii in np.arange(srt_plot, end_plot):
    axs[0].axvline(x=ii,c='k')
    axs[1].axvline(x=ii,c='k')

plt.xlabel(None)
for n, ax in enumerate(axs.flat):
    ax.text(0.05, 0.9, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=18)
plt.tight_layout()
# plt.savefig(save_dir+'Figure7_tmser_groups_Iorg.pdf', bbox_inches="tight") 



#%% ## FIGURE 7 ## timeseries rain and Iorg
## Plot rain rate time series
fig, axs = plt.subplots(2,1,figsize=(12,7))
for idgroup, group in enumerate(['rain','iorg']): 
    if group in profiles:
        profiles[group].sel(z=slice(0,50)).mean('z').rolling(time=6, center=True).mean().plot(c='k',ax=axs[idgroup])
        da_to_plot = profiles[group].sel(z=slice(0,50))\
                .mean('z').rolling(time=6, center=True).mean()
    elif group in da_org_norm:
        if group == 'spectral_length_moment':
            (100*(np.power(da_org[group],1))).plot(c='r',ax=axs[idgroup])
            # hc_ql.plot(c='b',ax=axs[idgroup])
        else:
            da_org[group].plot(c='k',ax=axs[idgroup])
            # np.power(da_org_norm['spectral_length_moment'],1).plot(c='r',ax=axs[idgroup])
        # tmser['cfrac'].plot(c='r',ax=axs[idgroup])
        da_to_plot = da_org[group]     
    elif group in tmser:
        tmser[group].plot(c='k',ax=axs[idgroup])
        da_to_plot = tmser[group]
        
    if group in time_g1:
        # moments.ctop_var.plot(c='r')
        axs[idgroup].scatter(profiles.time.sel(time=time_g1[group]),\
                             da_to_plot.sel(time=time_g1[group]),c='orange',label='Group 1')
        axs[idgroup].scatter(profiles.time.sel(time=time_g2[group]),\
                             da_to_plot.sel(time=time_g2[group]),c='b',label='Group 2')
        axs[idgroup].scatter(profiles.time.sel(time=time_g3[group]),\
                             da_to_plot.sel(time=time_g3[group]),c='green',label='Group 3')


    axs[idgroup].set_xlim([srt_plot,end_time])
axs[0].xaxis.set_visible(False) 
axs[0].legend()
axs[0].set_title('Surface rain-rate',fontsize=22)
axs[0].set_ylabel(r'$mm \; h^{-1}$')
axs[1].set_ylabel(r'$I_{org}$')
axs[1].set_title('Organisation',fontsize=22)
for ii in np.arange(srt_plot, end_plot):
    axs[0].axvline(x=ii,c='k')
    axs[1].axvline(x=ii,c='k')

plt.xlabel(None)
for n, ax in enumerate(axs.flat):
    ax.text(0.97, 0.9, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=13)
plt.tight_layout()
# plt.savefig(save_dir+'Figure7_tmser_groups.pdf', bbox_inches="tight")  

##################
#%% For Pier
fil_size= 2.5
var ='u_psfw_psf_unres'

klp = 150/(2*fil_size)
fig, axs = plt.subplots(3,3,figsize=(15,10))
for idgroup, group in enumerate(['iorg','spectral_length_moment','hc_ql']): 
    for idx,ih in enumerate(heights_to_plot): 
        if group in da_org:
            axs[idx,idgroup].scatter(da_org[group].sel(time=da_scales.time),\
                                      (1-da_scales_norm[var]).sel(klp=klp,method='nearest').sel(height=ih),\
                                          alpha = 0.5,s=12)
            axs[idx,idgroup].plot([da_org[group].sel(time=da_scales.time).min(),\
                                   da_org[group].sel(time=da_scales.time).max()],\
                                  [-0.1, 1.1],c='grey',lw=1)
        else:
            axs[idx,idgroup].scatter(hc_ql.sel(time=da_scales.time),\
                                     (1-da_scales_norm[var]).sel(klp=klp,method='nearest').sel(height=ih),\
                                         alpha = 0.5,s=12)
            axs[idx,idgroup].plot([hc_ql.sel(time=da_scales.time).min(),\
                                   hc_ql.sel(time=da_scales.time).max()],\
                                  [-0.1, 1.1],c='grey',lw=1)
                    
        axs[idx,0].set_ylabel(str(1)+' - '+var)
        axs[idx,idgroup].set_ylim([-0.1,1.1])
        
        # axs[idx,idgroup].set_xscale('log')
    axs[2,idgroup].set_xlabel(group)

#%% ## FIGURE 8 NEW ## spectral plot by groups 
group = 'iorg'
klp      = 38
fil_size = round(150/(2*klp),1) 

fig, axs = plt.subplots(3,2,figsize=(12,12))
for idvar, var in enumerate(['u_psfw_psf_unres','v_psfw_psf_unres']):            
    for idx,ih in enumerate(heights_to_plot):  
        ##############
        ## Non Dimensional x axis
        ##############
        # ## median
        # axs[idx,idvar].plot(da_scales['f_scales_norm_ql'].sel(time=time_g1[group]).mean('time'),da_scales_norm[var].sel(height=ih).sel(time=time_g1[group]).median('time'),\
        #           lw=2.5,c='orange',label='Group 1')
        # axs[idx,idvar].plot(da_scales['f_scales_norm_ql'].sel(time=time_g2[group]).mean('time'),da_scales_norm[var].sel(height=ih).sel(time=time_g2[group]).median('time'),\
        #           lw=2.5,c='b',label='Group 2')
        # axs[idx,idvar].plot(da_scales['f_scales_norm_ql'].sel(time=time_g3[group]).mean('time'),da_scales_norm[var].sel(height=ih).sel(time=time_g3[group]).median('time'),\
        #           lw=2.5,c='green',label='Group 3')
        # ## quartiles        
        # axs[idx,idvar].fill_between(da_scales['f_scales_norm_ql'].sel(time=time_g1[group]).mean('time'),\
        #               da_scales_norm[var].sel(height=ih).sel(time=time_g1[group]).chunk(dict(time=-1)).quantile(0.25,dim='time'),\
        #               da_scales_norm[var].sel(height=ih).sel(time=time_g1[group]).chunk(dict(time=-1)).quantile(0.75,dim='time'),\
        #                   color='orange',alpha=0.1)
        # axs[idx,idvar].fill_between(da_scales['f_scales_norm_ql'].sel(time=time_g2[group]).mean('time'),\
        #               da_scales_norm[var].sel(height=ih).sel(time=time_g2[group]).chunk(dict(time=-1)).quantile(0.25,dim='time'),\
        #               da_scales_norm[var].sel(height=ih).sel(time=time_g2[group]).chunk(dict(time=-1)).quantile(0.75,dim='time'),\
        #                   color='b',alpha=0.1)
        # axs[idx,idvar].fill_between(da_scales['f_scales_norm_ql'].sel(time=time_g3[group]).mean('time'),\
        #               da_scales_norm[var].sel(height=ih).sel(time=time_g3[group]).chunk(dict(time=-1)).quantile(0.25,dim='time'),\
        #               da_scales_norm[var].sel(height=ih).sel(time=time_g3[group]).chunk(dict(time=-1)).quantile(0.75,dim='time'),\
        #                   color='green',alpha=0.1)
        # axs[idx,idvar].axvline(1,c='k',lw=0.5)
        ##############
        ## Dimensional x axis
        ##############
        ## median
        axs[idx,idvar].plot(f_scales/1000,da_scales_norm[var].sel(height=ih).sel(time=time_g1[group]).median('time'),\
                  lw=2.5,c='orange',label='Unorganised')
        axs[idx,idvar].plot(f_scales/1000,da_scales_norm[var].sel(height=ih).sel(time=time_g2[group]).median('time'),\
                  lw=2.5,c='b',label='Group 2')
        axs[idx,idvar].plot(f_scales/1000,da_scales_norm[var].sel(height=ih).sel(time=time_g3[group]).median('time'),\
                  lw=2.5,c='green',label='Organised')
        ## quartiles        
        axs[idx,idvar].fill_between(f_scales/1000,\
                      da_scales_norm[var].sel(height=ih).sel(time=time_g1[group]).chunk(dict(time=-1)).quantile(0.25,dim='time'),\
                      da_scales_norm[var].sel(height=ih).sel(time=time_g1[group]).chunk(dict(time=-1)).quantile(0.75,dim='time'),\
                          color='orange',alpha=0.1)
        axs[idx,idvar].fill_between(f_scales/1000,\
                      da_scales_norm[var].sel(height=ih).sel(time=time_g2[group]).chunk(dict(time=-1)).quantile(0.25,dim='time'),\
                      da_scales_norm[var].sel(height=ih).sel(time=time_g2[group]).chunk(dict(time=-1)).quantile(0.75,dim='time'),\
                          color='b',alpha=0.1)
        axs[idx,idvar].fill_between(f_scales/1000,\
                      da_scales_norm[var].sel(height=ih).sel(time=time_g3[group]).chunk(dict(time=-1)).quantile(0.25,dim='time'),\
                      da_scales_norm[var].sel(height=ih).sel(time=time_g3[group]).chunk(dict(time=-1)).quantile(0.75,dim='time'),\
                          color='green',alpha=0.1)
        axs[idx,idvar].axvline(fil_size,c='k',lw=0.5)
            
        axs[idx,idvar].set_ylim([-0.1,+1.1])
        axs[idx,idvar].set_xscale('log')
        axs[idx,idvar].axhline(0,c='k',lw=0.5)
        axs[idx,idvar].set_yticks([0,0.5,1])
        
        
        if idvar == 0:
            if 'CL' in ih:
                axs[idx,idvar].set_ylabel('Flux partition \n at '+str(ih))
            else:
                axs[idx,idvar].set_ylabel('Flux partition \n at '+str(ih)+' m')
    axs[0,0].legend(loc='lower right',fontsize=20)
    ##############
    # axs[2,idvar].set_xlabel(r'Dimensionless $\frac{\Delta x}{h_b}$ ')  
    axs[2,idvar].set_xlabel(r'Filter scale $\Delta x$ [km]')  
    ##############
    if var[0]=='u':
        axs[0,idvar].set_title('Zonal component',fontsize=24)
    elif var[0]=='v':
        axs[0,idvar].set_title('Meridional component',fontsize=24)
plt.suptitle(r'Grouping by $I_{org}$',fontsize=26)


for n, ax in enumerate(axs.flat):
    ax.text(0.08, 0.9, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=13)
plt.tight_layout()
plt.savefig(save_dir+'Figure8_spectra_groups.pdf', bbox_inches="tight")  


#%% ## FIGURE 8 ## spectral plot by groups 
for var in ['u_psfw_psf_unres','v_psfw_psf_unres']:
    fig, axs = plt.subplots(3,2,figsize=(12,12))
    for idgroup, group in enumerate(['rain','iorg']):            
        for idx,ih in enumerate(heights_to_plot):  
            
            ## Non Dimensional x axis
            axs[idx,idgroup].plot(da_scales['f_scales_norm_ql'].sel(time=time_g1[group]).mean('time'),da_scales_norm[var].sel(height=ih).sel(time=time_g1[group]).median('time'),\
                      lw=2.5,c='orange',label='Group 1')
            axs[idx,idgroup].plot(da_scales['f_scales_norm_ql'].sel(time=time_g2[group]).mean('time'),da_scales_norm[var].sel(height=ih).sel(time=time_g2[group]).median('time'),\
                      lw=2.5,c='b',label='Group 2')
            axs[idx,idgroup].plot(da_scales['f_scales_norm_ql'].sel(time=time_g3[group]).mean('time'),da_scales_norm[var].sel(height=ih).sel(time=time_g3[group]).median('time'),\
                      lw=2.5,c='green',label='Group 3')
            
            ## quartiles        
            axs[idx,idgroup].fill_between(da_scales['f_scales_norm_ql'].sel(time=time_g1[group]).mean('time'),\
                          da_scales_norm[var].sel(height=ih).sel(time=time_g1[group]).chunk(dict(time=-1)).quantile(0.25,dim='time'),\
                          da_scales_norm[var].sel(height=ih).sel(time=time_g1[group]).chunk(dict(time=-1)).quantile(0.75,dim='time'),\
                              color='orange',alpha=0.1)
            axs[idx,idgroup].fill_between(da_scales['f_scales_norm_ql'].sel(time=time_g2[group]).mean('time'),\
                          da_scales_norm[var].sel(height=ih).sel(time=time_g2[group]).chunk(dict(time=-1)).quantile(0.25,dim='time'),\
                          da_scales_norm[var].sel(height=ih).sel(time=time_g2[group]).chunk(dict(time=-1)).quantile(0.75,dim='time'),\
                              color='b',alpha=0.1)
            axs[idx,idgroup].fill_between(da_scales['f_scales_norm_ql'].sel(time=time_g3[group]).mean('time'),\
                          da_scales_norm[var].sel(height=ih).sel(time=time_g3[group]).chunk(dict(time=-1)).quantile(0.25,dim='time'),\
                          da_scales_norm[var].sel(height=ih).sel(time=time_g3[group]).chunk(dict(time=-1)).quantile(0.75,dim='time'),\
                              color='green',alpha=0.1)
                
            ## Dimensional x axis
            # axs[idx,idgroup].plot(f_scales/1000,da_scales_norm[var].sel(height=ih).sel(time=time_g1).median('time'),\
            #           lw=2.5,c='orange',label='Group 1')
            # axs[idx,idgroup].plot(f_scales/1000,da_scales_norm[var].sel(height=ih).sel(time=time_g2).median('time'),\
            #           lw=2.5,c='b',label='Group 2')
            # axs[idx,idgroup].plot(f_scales/1000,da_scales_norm[var].sel(height=ih).sel(time=time_g3).median('time'),\
            #           lw=2.5,c='green',label='Group 3')
                
            axs[idx,idgroup].set_ylim([-0.1,+1.1])
            
            axs[idx,idgroup].set_xscale('log')
            axs[idx,idgroup].axhline(0,c='k',lw=0.5)
            axs[idx,idgroup].axvline(1,c='k',lw=0.5)
            
            if idgroup == 0:
                axs[idx,idgroup].set_ylabel('Flux partition \n at '+str(ih)+' m')
        axs[0,idgroup].legend(loc='lower right')
        axs[2,idgroup].set_xlabel(r'Dimensionless $\frac{\Delta x}{h_b}$ ')
    axs[0,0].set_title('Grouping by rain-rate',fontsize=24)  
    axs[0,1].set_title(r'Grouping by $I_{org}$',fontsize=24)  
    if var[0]=='u':
        plt.suptitle('Zonal component',fontsize=24)
    elif var[0]=='v':
        plt.suptitle('Meridional component',fontsize=24)

    for n, ax in enumerate(axs.flat):
        ax.text(0.08, 0.9, string.ascii_uppercase[n], transform=ax.transAxes, 
                size=13)
    plt.tight_layout()
    # plt.savefig(save_dir+'Figure8_'+var[0]+'_spectra_groups.pdf', bbox_inches="tight")  
##################
#%% ## FIGURE appendix ## spectral plot by groups 
for idgroup, group in enumerate(['iorg']):   
    fig, axs = plt.subplots(3,2,figsize=(12,12))
    for idvar, var in enumerate(['qt_psfw_psf_unres','thl_psfw_psf_unres']):         
        for idx,ih in enumerate(heights_to_plot):  
            
            ## Non Dimensional x axis
            axs[idx,idvar].plot(da_scales['f_scales_norm_ql'].sel(time=time_g1[group]).mean('time'),da_scales_norm[var].sel(height=ih).sel(time=time_g1[group]).median('time'),\
                      lw=2.5,c='orange',label='Group 1')
            axs[idx,idvar].plot(da_scales['f_scales_norm_ql'].sel(time=time_g2[group]).mean('time'),da_scales_norm[var].sel(height=ih).sel(time=time_g2[group]).median('time'),\
                      lw=2.5,c='b',label='Group 2')
            axs[idx,idvar].plot(da_scales['f_scales_norm_ql'].sel(time=time_g3[group]).mean('time'),da_scales_norm[var].sel(height=ih).sel(time=time_g3[group]).median('time'),\
                      lw=2.5,c='green',label='Group 3')
            
            ## quartiles        
            axs[idx,idvar].fill_between(da_scales['f_scales_norm_ql'].sel(time=time_g1[group]).mean('time'),\
                          da_scales_norm[var].sel(height=ih).sel(time=time_g1[group]).chunk(dict(time=-1)).quantile(0.25,dim='time'),\
                          da_scales_norm[var].sel(height=ih).sel(time=time_g1[group]).chunk(dict(time=-1)).quantile(0.75,dim='time'),\
                              color='orange',alpha=0.1)
            axs[idx,idvar].fill_between(da_scales['f_scales_norm_ql'].sel(time=time_g2[group]).mean('time'),\
                          da_scales_norm[var].sel(height=ih).sel(time=time_g2[group]).chunk(dict(time=-1)).quantile(0.25,dim='time'),\
                          da_scales_norm[var].sel(height=ih).sel(time=time_g2[group]).chunk(dict(time=-1)).quantile(0.75,dim='time'),\
                              color='b',alpha=0.1)
            axs[idx,idvar].fill_between(da_scales['f_scales_norm_ql'].sel(time=time_g3[group]).mean('time'),\
                          da_scales_norm[var].sel(height=ih).sel(time=time_g3[group]).chunk(dict(time=-1)).quantile(0.25,dim='time'),\
                          da_scales_norm[var].sel(height=ih).sel(time=time_g3[group]).chunk(dict(time=-1)).quantile(0.75,dim='time'),\
                              color='green',alpha=0.1)
                
            ## Dimensional x axis
            # axs[idx,idgroup].plot(f_scales/1000,da_scales_norm[var].isel(height=ih).sel(time=time_g1).median('time'),\
            #           lw=2.5,c='orange',label='Group 1')
            # axs[idx,idgroup].plot(f_scales/1000,da_scales_norm[var].isel(height=ih).sel(time=time_g2).median('time'),\
            #           lw=2.5,c='b',label='Group 2')
            # axs[idx,idgroup].plot(f_scales/1000,da_scales_norm[var].isel(height=ih).sel(time=time_g3).median('time'),\
            #           lw=2.5,c='green',label='Group 3')
                
            axs[idx,idgroup].set_ylim([-0.1,+1.1])
            
            axs[idx,idvar].set_xscale('log')
            axs[idx,idvar].axhline(0,c='k',lw=0.5)
            axs[idx,idvar].axvline(1,c='k',lw=0.5)
            
            if idgroup == 0:
                axs[idx,idvar].set_ylabel('Flux partition \n at '+str(ih)+' m')
        axs[0,idvar].legend(loc='lower right')
        axs[0,idvar].set_title(var[0:-9]+' Flux',fontsize=24)  
        axs[2,idvar].set_xlabel(r'Dimensionless $\frac{\Delta x}{h_b}$ ')
    plt.suptitle(r'Grouping by '+group,fontsize=24)  
    
    for n, ax in enumerate(axs.flat):
        ax.text(0.08, 0.9, string.ascii_uppercase[n], transform=ax.transAxes, 
                size=13)
    plt.tight_layout()
    # plt.savefig(save_dir+'Figure8_appendix_scalar.pdf', bbox_inches="tight")  
##################

#%% ## FIGURE for Wim
for idgroup, group in enumerate(['iorg']):   
    fig, axs = plt.subplots(3,2,figsize=(18,13))      
    for idgrp, grp in enumerate([1,2,3]):  
        if grp == 1:
            time_to_plot = time_g1[group]
        elif grp ==2:
            time_to_plot = time_g2[group]
        elif grp ==3:
            time_to_plot = time_g3[group]

        for ih in ['100','midCL']:  
            if ih =='100':
                stl = '-'
            else:
                stl='--'
            
            for idvar, var in enumerate(['u_psfw_psf_unres','v_psfw_psf_unres']): 
                ## Non Dimensional x axis
                axs[idgrp,0].plot(da_scales['f_scales_norm_ql'].sel(time=time_to_plot)\
                                    .mean('time'),da_scales_norm[var].sel(height=ih)\
                                        .sel(time=time_to_plot).median('time'),\
                          lw=2,ls=stl,c=col[idvar*3],label=var[0:-15]+' at '+ih)
                    
                    
            for idvar, var in enumerate(['qt_psfw_psf_unres','thl_psfw_psf_unres']): 
                ## Non Dimensional x axis
                axs[idgrp,1].plot(da_scales['f_scales_norm_ql'].sel(time=time_to_plot)\
                                    .mean('time'),da_scales_norm[var].sel(height=ih)\
                                        .sel(time=time_to_plot).median('time'),\
                          lw=2,ls=stl,c=col[(2+idvar)*3],label=var[0:-15]+' at '+ih)

            # ## quartiles        
            # axs[idx,idvar].fill_between(da_scales['f_scales_norm_ql'].sel(time=time_g1[group]).mean('time'),\
            #               da_scales_norm[var].isel(height=ih).sel(time=time_g1[group]).chunk(dict(time=-1)).quantile(0.25,dim='time'),\
            #               da_scales_norm[var].isel(height=ih).sel(time=time_g1[group]).chunk(dict(time=-1)).quantile(0.75,dim='time'),\
            #                   color='orange',alpha=0.1)
            # axs[idx,idvar].fill_between(da_scales['f_scales_norm_ql'].sel(time=time_g2[group]).mean('time'),\
            #               da_scales_norm[var].isel(height=ih).sel(time=time_g2[group]).chunk(dict(time=-1)).quantile(0.25,dim='time'),\
            #               da_scales_norm[var].isel(height=ih).sel(time=time_g2[group]).chunk(dict(time=-1)).quantile(0.75,dim='time'),\
            #                   color='b',alpha=0.1)
            # axs[idx,idvar].fill_between(da_scales['f_scales_norm_ql'].sel(time=time_g3[group]).mean('time'),\
            #               da_scales_norm[var].isel(height=ih).sel(time=time_g3[group]).chunk(dict(time=-1)).quantile(0.25,dim='time'),\
            #               da_scales_norm[var].isel(height=ih).sel(time=time_g3[group]).chunk(dict(time=-1)).quantile(0.75,dim='time'),\
            #                   color='green',alpha=0.1)
                
                
        # plt.ylim([-0.02,+0.02])
            
            axs[idgrp,0].set_xscale('log')
            axs[idgrp,1].set_xscale('log')
            axs[idgrp,0].axhline(0,c='k',lw=0.5)
            axs[idgrp,1].axhline(0,c='k',lw=0.5)
            axs[idgrp,0].axvline(1,c='k',lw=0.5)
            axs[idgrp,1].axvline(1,c='k',lw=0.5)
            

        axs[idgrp,0].set_title('Group '+str(grp),fontsize=24)  
        axs[idgrp,1].set_title('Group '+str(grp),fontsize=24)  
        axs[2,0].set_xlabel(r'Dimensionless $\frac{\Delta x}{h_b}$ ')
        axs[2,1].set_xlabel(r'Dimensionless $\frac{\Delta x}{h_b}$ ')

    axs[0,0].legend(loc='lower right',fontsize=18)
    axs[0,1].legend(loc='lower right',fontsize=18)
    # plt.suptitle(r'Grouping by '+group,fontsize=24)  
    
    for n, ax in enumerate(axs.flat):
        ax.text(0.08, 0.9, string.ascii_uppercase[n], transform=ax.transAxes, 
                size=13)
    plt.tight_layout()

##################
#%% APPENDIX
for idgroup, group in enumerate(['iorg']):   
    fig, axs = plt.subplots(3,3,figsize=(18,13))      
    for idgrp, grp in enumerate([1,2,3]):  
        if grp == 1:
            time_to_plot = time_g1
        elif grp ==2:
            time_to_plot = time_g2
        elif grp ==3:
            time_to_plot = time_g3

        for idx,ih in enumerate(heights_to_plot):  
            
            for idvar, var in enumerate(['qt_psfw_psf_unres','thl_psfw_psf_unres','u_psfw_psf_unres','v_psfw_psf_unres']): 
                ## Non Dimensional x axis
                axs[idx,idgrp].plot(da_scales['f_scales_norm_ql'].sel(time=time_to_plot[group])\
                                    .mean('time'),da_scales_norm[var].sel(height=ih)\
                                        .sel(time=time_to_plot[group]).median('time'),\
                          lw=2,c=col[idvar*2],label=var[0:-15])

            
            # ## quartiles        
            # axs[idx,idvar].fill_between(da_scales['f_scales_norm_ql'].sel(time=time_g1[group]).mean('time'),\
            #               da_scales_norm[var].isel(height=ih).sel(time=time_g1[group]).chunk(dict(time=-1)).quantile(0.25,dim='time'),\
            #               da_scales_norm[var].isel(height=ih).sel(time=time_g1[group]).chunk(dict(time=-1)).quantile(0.75,dim='time'),\
            #                   color='orange',alpha=0.1)
            # axs[idx,idvar].fill_between(da_scales['f_scales_norm_ql'].sel(time=time_g2[group]).mean('time'),\
            #               da_scales_norm[var].isel(height=ih).sel(time=time_g2[group]).chunk(dict(time=-1)).quantile(0.25,dim='time'),\
            #               da_scales_norm[var].isel(height=ih).sel(time=time_g2[group]).chunk(dict(time=-1)).quantile(0.75,dim='time'),\
            #                   color='b',alpha=0.1)
            # axs[idx,idvar].fill_between(da_scales['f_scales_norm_ql'].sel(time=time_g3[group]).mean('time'),\
            #               da_scales_norm[var].isel(height=ih).sel(time=time_g3[group]).chunk(dict(time=-1)).quantile(0.25,dim='time'),\
            #               da_scales_norm[var].isel(height=ih).sel(time=time_g3[group]).chunk(dict(time=-1)).quantile(0.75,dim='time'),\
            #                   color='green',alpha=0.1)
                
                
        # plt.ylim([-0.02,+0.02])
            
            axs[idx,idgrp].set_xscale('log')
            axs[idx,idgrp].axhline(0,c='k',lw=0.5)
            axs[idx,idgrp].axvline(1,c='k',lw=0.5)
            
            if idgrp == 0:
                axs[idx,idgrp].set_ylabel('Flux partition \n at '+str(ih)+' m')
        axs[0,idgrp].set_title('Group '+str(grp),fontsize=24)  
        axs[2,idgrp].set_xlabel(r'Dimensionless $\frac{\Delta x}{h_b}$ ')
    axs[0,0].legend(loc='lower right',fontsize=18)
    plt.suptitle(r'Grouping by '+group,fontsize=24)  
    
    for n, ax in enumerate(axs.flat):
        ax.text(0.08, 0.9, string.ascii_uppercase[n], transform=ax.transAxes, 
                size=13)
    plt.tight_layout()
    plt.savefig(save_dir+'Figure8_appendix_allVar.pdf', bbox_inches="tight")  
##################

#%% ## FIGURE 12 NEW ## PROFILES_all groups
z_cut = 0
klp      = 38
fil_size = round(150/(2*klp),1) 
for idgroup, sel_time in enumerate([time_g1['iorg'],time_g2['iorg'],time_g3['iorg']]):
    fig, axs = plt.subplots(2,3,figsize=(12,13))
    for idvar, var in enumerate(['u','v']):
        ## wind
        idx = 0
        profiles[var].sel(time=sel_time).mean('time').plot(y='z',\
                           c='k',lw=4,ax=axs[idvar,idx])
        axs[idvar,idx].fill_betweenx(profiles.z.values,\
                           np.quantile(profiles[var].sel(time=sel_time).values,0.25,0),\
                           np.quantile(profiles[var].sel(time=sel_time).values,0.75,0),\
                              color='grey',alpha=0.2)
        axs[idvar,idx].axvline(0,c='k',lw=1)
        axs[idvar,idx].axhline(200,c='k',lw=1)
        axs[idvar,idx].axhline(cl_base.sel(time=sel_time).mean(),c='k',lw=1)
        axs[idvar,idx].axhline(midCL.sel(time=sel_time)['z'].mean(),c='k',lw=1)
        axs[idvar,idx].set_ylim([0,4000])
        if var == 'u':
            axs[idvar,idx].set_xlim([-14,-2])
            axs[idvar,idx].plot([-13.9] * len(profiles['z']),profiles['z'].where((profiles[var+'wt'].sel(time=sel_time).mean('time')\
                             * profiles['d'+var+'_dz'].sel(time=sel_time)\
                                 .mean('time'))>0).values,c='r',lw=5)
        if var == 'v':
            axs[idvar,idx].set_xlim([-3,1])    
            axs[idvar,idx].plot([-2.99] * len(profiles['z']),profiles['z'].where((profiles[var+'wt'].sel(time=sel_time).mean('time')\
                             * profiles['d'+var+'_dz'].sel(time=sel_time)\
                                 .mean('time'))>0).values,c='r',lw=5)
        
        axs[idvar,idx].set_title("Wind $\overline{"+var+"}$",fontsize =25)
        axs[idvar,idx].set_xlabel(r'ms$^{-1}$')
        axs[idvar,idx].set_ylabel(r'z (m)')
        
        
        
        
        ## flux 
        idx = 1
        # mean
        da_scales_prof[var+"w_pf"].sel(klp=klp,time=sel_time).where(da_scales_prof['height']>z_cut)\
                .mean('time').\
            plot(y='height',lw=3,ls='-',c='g',\
                 label='Total \n resolved',ax=axs[idvar,idx],zorder=2)
        da_scales_prof[var+"_pfw_pf"].sel(klp=klp,time=sel_time)\
                .mean('time').\
            plot(y='height',lw=2,ls='-',c='brown',marker='+',markevery=8,\
                 label='Up-filter',ax=axs[idvar,idx])
        da_scales_prof[var+"_psfw_psf"].sel(klp=klp,time=sel_time).where(da_scales_prof['height']>z_cut)\
                .mean('time').\
            plot(y='height',lw=2,ls='-',c='mediumpurple',marker='o',markevery=8,markerfacecolor='none',\
                 label='Sub-filter',ax=axs[idvar,idx])
        # profiles[var+"wr"].sel(time=sel_time).mean('time').\
        #     plot(y='z',ls='--',c='g',label='Total from prof',ax=axs[idvar,idx])  
        profiles[var+"ws"].sel(time=sel_time).mean('time').\
            plot(y='z',lw=1,ls='--',c='g',label='Unresolved',ax=axs[idvar,idx])  
                
        axs[idvar,idx].axvline(0,c='k',lw=1)
        axs[idvar,idx].set_ylim([0,4000])
        axs[idvar,idx].axhline(200,c='k',lw=1)
        axs[idvar,idx].axhline(cl_base.sel(time=sel_time).mean(),c='k',lw=1)
        axs[idvar,idx].axhline(midCL.sel(time=sel_time)['z'].mean(),c='k',lw=1)
        axs[idvar,idx].yaxis.set_visible(False) 
        if var == 'u':
            axs[idvar,idx].set_xlim([-0.02,0.09])
            axs[idvar,idx].legend(fontsize=18)
        if var =='v':
            axs[idvar,idx].set_xlim([-0.021,0.035])
            axs[idvar,idx].set_xticks([-0.01,0.02])
        axs[idvar,idx].set_title("Flux $\overline{"+var+"'w'}$",fontsize =25)
        axs[idvar,idx].set_xlabel(r'm$^{2}$s$^{-2}$')
        
        ## divergence
        idx = 2
        # from flux to tendency
        axs[idvar,idx].plot(-acc_time*((da_scales_prof[var+"w_pf"].diff('height')/np.diff(da_scales_prof['height'].values))\
                                       .where(da_scales_prof['height']>z_cut)\
                                               # - tend_to_plot[var+'tenddif'+samp]\
                                                   .sel(klp=klp,time=sel_time).mean('time')).values,\
                                    (da_scales_prof['height'].values[1:] + da_scales_prof['height'].values[:-1])/2,\
                                        ls='-',c='g',lw=3,label='Total \n resolved')
        axs[idvar,idx].plot(-acc_time*(da_scales_prof[var+"_pfw_pf"].diff('height')/np.diff(da_scales_prof['height'].values))\
                            #.where(da_scales_prof['height']>z_cut)\
                            .sel(klp=klp,time=sel_time).mean('time').values,\
                            (da_scales_prof['height'].values[1:] + da_scales_prof['height'].values[:-1])/2,\
                                ls='-',c='brown',lw=2,marker='+',markevery=8,\
                                    label='Up-filter')
        axs[idvar,idx].plot(-acc_time*(da_scales_prof[var+"_psfw_psf"].diff('height')/np.diff(da_scales_prof['height'].values))\
                                       .where(da_scales_prof['height']>z_cut)\
                                       # - tend_to_plot[var+'tenddif'+samp]\
                                           .sel(klp=klp,time=sel_time).mean('time').values,\
                            (da_scales_prof['height'].values[1:] + da_scales_prof['height'].values[:-1])/2,\
                                ls='-',c='mediumpurple',lw=2,marker='o',markevery=8,markerfacecolor='none',\
                                    label='Sub-filter')
        # axs[idvar,idx].plot(-acc_time*(profiles[var+"wr"].diff('z')/np.diff(profiles['z'])).sel(time=sel_time).mean('time').values,\
        #                     (profiles['z'].values[1:] + profiles['z'].values[:-1])/2,\
        #                         ls='--',c='m',lw=2.5,label='Total from prof')
        # ## plot directly tendencies as from DALES 
        # axs[idvar,idx].plot(unit*acc_time*(tend_to_plot[var+'tendadv'+samp]).\
        #         sel(time=sel_time).mean('time'),tend_to_plot.height,c='g',lw=2.5,label='Adv')  
        axs[idvar,idx].plot(acc_time*(\
                                            samptend[var+'tenddifall']).\
                sel(time=sel_time).mean('time'),samptend.z,c='g',ls='--',lw=1,label='Unresolved')         
            
        axs[idvar,idx].axvline(0,c='k',lw=1)
        axs[idvar,idx].set_ylim([0,4000])
        axs[idvar,idx].axhline(200,c='k',lw=1)
        axs[idvar,idx].axhline(cl_base.sel(time=sel_time).mean(),c='k',lw=1)
        axs[idvar,idx].axhline(midCL.sel(time=sel_time)['z'].mean(),c='k',lw=1)
        axs[idvar,idx].yaxis.set_visible(False) 
        axs[idvar,idx].set_xlim([-0.22,0.48])
        axs[idvar,idx].set_title(r"Flux divergence -$\frac{\partial}{\partial{z}} \overline{"+var+"'w'}$",fontsize =25)
        axs[idvar,idx].set_xlabel(r'ms$^{-2}$')
        
    for n, ax in enumerate(axs.flat):
        ax.text(0.91, 0.95, string.ascii_uppercase[n], transform=ax.transAxes, 
                    size=20)
    plt.tight_layout()

    plt.savefig(save_dir+'Figure12_profiles_all'+str(idgroup+1)+'_f'+str(fil_size)+'.pdf', bbox_inches="tight")  


#%%
# for idgroup, sel_time in enumerate([time_g1['iorg'],time_g2['iorg'],time_g3['iorg']]):
#     for idvar, var in enumerate(['u','v']):
#         plt.figure()
#         (profiles[var+'wt'].sel(time=sel_time).mean('time') * profiles['d'+var+'_dz'].sel(time=sel_time).mean('time')).plot(y='z')
#         # (profiles[var+'wt'] * profiles['d'+var+'_dz']).sel(time=sel_time).mean('time').plot(y='z')
#         plt.axvline(0,c='k')
#         plt.ylim(height_lim)
#         plt.title('Group '+str(idgroup+1)+' '+var,fontsize=20)
#         plt.xlim([-0.00015,None])

#         profiles.where((profiles[var+'wt'].sel(time=sel_time).mean('time') * profiles['d'+var+'_dz'].sel(time=sel_time).mean('time'))>0)
        

#%% ## FIGURE 9 ## Flux profils by group 
klp=38
## Profiles
fig, axs = plt.subplots(2,3,figsize=(12,13))
for idvar, var in enumerate(['u','v']):
    for idx, sel_time in enumerate([time_g1['iorg'],time_g2['iorg'],time_g3['iorg']]):

        ## mean
        da_scales_prof[var+"w_pf"].sel(klp=klp,time=sel_time).mean('time').\
            plot(y='height',lw=3,ls='-',c='g',\
                 label='Total \n resolved',ax=axs[idvar,idx],zorder=2)
        da_scales_prof[var+"_pfw_pf"].sel(klp=klp,time=sel_time).mean('time').\
            plot(y='height',lw=2,ls='-',c='brown',marker='+',markevery=8,\
                 label='Up-filter',ax=axs[idvar,idx])
        da_scales_prof[var+"_psfw_psf"].sel(klp=klp,time=sel_time).mean('time').\
            plot(y='height',lw=2,ls='-',c='mediumpurple',marker='o',markevery=8,markerfacecolor='none',\
                 label='Sub-filter',ax=axs[idvar,idx])
            
        # profiles[var+"wr"].sel(time=sel_time).mean('time').\
        #     plot(y='z',ls='--',c='g',label='Total from prof',ax=axs[idvar,idx])  
        profiles[var+"ws"].sel(time=sel_time).mean('time').\
            plot(y='z',lw=1,ls='--',c='g',label='Unresolved',ax=axs[idvar,idx])  
            
        ### HARMONIE
        # (harm_clim_avg[var+'flx_conv']+harm_clim_avg[var+'flx_turb']).\
        #     sel(time=sel_time,method='nearest').mean('time').\
        #         plot(y='z',c='b',ax=axs[idvar,idx],label ='HARMONIE \n parameterized')

        ## quartiles        
        axs[idvar,idx].fill_betweenx(da_scales_prof.height,\
                          da_scales_prof[var+"w_pf"].sel(klp=klp,time=sel_time).quantile(0.25,dim='time'),\
                          da_scales_prof[var+"w_pf"].sel(klp=klp,time=sel_time).quantile(0.75,dim='time'),\
                              color='g',alpha=0.1)
            
        # axs[idvar,idx].fill_betweenx(da_scales_prof.height,\
        #                   da_scales_prof[var+"_pfw_pf"].sel(klp=klp,time=sel_time).quantile(0.25,dim='time'),\
        #                   da_scales_prof[var+"_pfw_pf"].sel(klp=klp,time=sel_time).quantile(0.75,dim='time'),\
        #                       color='brown',alpha=0.1)
            
        # axs[idvar,idx].fill_betweenx(da_scales_prof.height,\
        #                   da_scales_prof[var+"_psfw_psf"].sel(klp=klp,time=sel_time).quantile(0.25,dim='time'),\
        #                   da_scales_prof[var+"_psfw_psf"].sel(klp=klp,time=sel_time).quantile(0.75,dim='time'),\
        #                       color='mediumpurple',alpha=0.1)

        axs[idvar,idx].axvline(0,c='k',lw=0.5)
        axs[0,idx].set_xlim([-0.015,0.09])
        axs[1,idx].set_xlim([-0.028,0.041])
        axs[idvar,idx].set_ylim([-1,4000])
        axs[idvar,idx].set_title(var+'$\'w\'$ Group '+str(idx+1),fontsize=21)
        
        axs[idvar,idx].axhline(200,c='k',ls='-',lw=0.5)
        axs[idvar,idx].axhline(650,c='k',ls='-',lw=0.5)
        axs[idvar,idx].axhline(1500,c='k',ls='-',lw=0.5)
        
        if idx >0:
            axs[idvar,idx].yaxis.set_visible(False) 
        axs[0,idx].set_xlabel(None)
        axs[1,idx].set_xlabel(r'$m^2 s^{-2}$')
axs[0,0].set_ylabel(r'z ($m$)')
axs[1,0].set_ylabel(r'z ($m$)')
axs[0,0].legend(fontsize=16)
# plt.xlabel(r''+var+' momentum flux [$m^2 / s^2$]')
# plt.ylabel('Z [m]')
# plt.title('Partitioning of momentum flux \n Filter scale = 2.5 km')
    
for n, ax in enumerate(axs.flat):
    ax.text(0.05, 0.95, string.ascii_uppercase[n], transform=ax.transAxes, 
                size=20)
plt.tight_layout()
# plt.savefig(save_dir+'Figure9_flux_profiles_groups.pdf', bbox_inches="tight")  
##################
#%% ## FIGURE 10 ## Resolved tendency term  
## budget 
cond_sampl = ['all']
## be careful with accumulatedd tendency... maybe resample hourly?
tend_to_plot   = samptend.sel(time=slice(temp_hrs[0],temp_hrs[1]))
tend_to_plot = tend_to_plot.rename({'z':'height'})
h_clim_to_plot = harm_clim_avg.sel(time=slice(samptend.time[0]-np.timedelta64(1,'h'),samptend.time[-1]))
fig, axs = plt.subplots(2,3,figsize=(12,13))
for idvar, var in enumerate(['u','v']):
    for idx, sel_time in enumerate([time_g1['iorg'],time_g2['iorg'],time_g3['iorg']]):
        sel_time = sel_time.sel(time=sel_time.time.dt.minute == 00)
        for samp in cond_sampl:
            unit = 1
            ## from flux to tendency
            axs[idvar,idx].plot(-acc_time*(da_scales_prof[var+"w_pf"].diff('height')/np.diff(da_scales_prof['height'].values)\
                                           # - tend_to_plot[var+'tenddif'+samp]\
                                               ).sel(time=sel_time).mean('time').values[0,:],\
                                (da_scales_prof['height'].values[1:] + da_scales_prof['height'].values[:-1])/2,\
                                    ls='-',c='g',lw=3,label='Total \n resolved')
            axs[idvar,idx].plot(-acc_time*(da_scales_prof[var+"_pfw_pf"].diff('height')/np.diff(da_scales_prof['height'].values))\
                                .sel(time=sel_time).mean('time').values[0,:],\
                                (da_scales_prof['height'].values[1:] + da_scales_prof['height'].values[:-1])/2,\
                                    ls='-',c='brown',lw=2,marker='+',markevery=8,\
                                        label='Up-filter')
            axs[idvar,idx].plot(-acc_time*(da_scales_prof[var+"_psfw_psf"].diff('height')/np.diff(da_scales_prof['height'].values)\
                                           # - tend_to_plot[var+'tenddif'+samp]\
                                               ).sel(time=sel_time).mean('time').values[0,:],\
                                (da_scales_prof['height'].values[1:] + da_scales_prof['height'].values[:-1])/2,\
                                    ls='-',c='mediumpurple',lw=2,marker='o',markevery=8,markerfacecolor='none',\
                                        label='Sub-filter')


                
                
            # axs[idvar,idx].plot(-acc_time*(profiles[var+"wr"].diff('z')/np.diff(profiles['z'])).sel(time=sel_time).mean('time').values,\
            #                     (profiles['z'].values[1:] + profiles['z'].values[:-1])/2,\
            #                         ls='--',c='m',lw=2.5,label='Total from prof')
            
            
            # ## plot directly tendencies as from DALES 
            # axs[idvar,idx].plot(unit*acc_time*(tend_to_plot[var+'tendadv'+samp]).\
            #         sel(time=sel_time).mean('time'),tend_to_plot.height,c='g',lw=2.5,label='Adv')  
            axs[idvar,idx].plot(unit*acc_time*(\
                                                tend_to_plot[var+'tenddif'+samp]).\
                    sel(time=sel_time).mean('time'),tend_to_plot.height,c='g',ls='--',lw=1,label='Unresolved')  
                
                
        axs[idvar,idx].set_title(r'$\frac{\overline{'+var+'\'w\'}}{dz}$ - Group '+str(idx+1),fontsize=21)
        axs[idvar,idx].axvline(0,c='k',lw=0.5)
        axs[idvar,idx].set_xlim([-0.2,+0.49])        
        axs[idvar,idx].set_ylim(height_lim)
        axs[idvar,idx].set_xlabel(r'($m/s$ / hour)',fontsize=18)
        axs[idvar,idx].axhline(200,c='k',ls='-',lw=0.5)
        axs[idvar,idx].axhline(650,c='k',ls='-',lw=0.5)
        axs[idvar,idx].axhline(1500,c='k',ls='-',lw=0.5)
        if idx >0:
            axs[idvar,idx].yaxis.set_visible(False) 
        axs[0,idx].set_xlabel(None)
axs[0,0].legend(frameon=True,loc='upper right',fontsize=16)
axs[0,0].set_ylabel(r'z ($m$)')
axs[1,0].set_ylabel(r'z ($m$)')

for n, ax in enumerate(axs.flat):
    ax.text(0.08, 0.95, string.ascii_uppercase[n], transform=ax.transAxes, 
                size=20)
plt.tight_layout()
plt.savefig(save_dir+'Figure10_momfluxdivergence_reconstructed.pdf', bbox_inches="tight")  
##################
#%% ## FIGURE 11 ## Mom budget by group
fig, axs = plt.subplots(2,3,figsize=(12,13))
for idvar, var in enumerate(['u','v']):
    for idx, sel_time in enumerate([time_g1['iorg'],time_g2['iorg'],time_g3['iorg']]):
        sel_time = sel_time.sel(time=sel_time.time.dt.minute == 00)
        for samp in cond_sampl:
            unit = 1         
            ## plot directly tendencies as from DALES 
            axs[idvar,idx].plot(unit*acc_time*tend_to_plot[var+'tendls'+samp].\
                    sel(time=sel_time).mean('time'),tend_to_plot['height'],c='k',label='D. LS',lw=2)     
            axs[idvar,idx].plot(unit*acc_time*tend_to_plot[var+'tendtot'+samp].\
                    sel(time=sel_time).mean('time'),tend_to_plot['height'],c='deeppink',label='D. Tot',lw=2)
            axs[idvar,idx].plot(unit*acc_time*(tend_to_plot[var+'tendtot'+samp] - tend_to_plot[var+'tendls'+samp]).\
                    sel(time=sel_time).mean('time'),tend_to_plot['height'],c='g',label='D. Tot - LS')
            # axs[idvar,idx].plot(unit*acc_time*(tend_to_plot[var+'tendadv'+samp] + tend_to_plot[var+'tenddif'+samp]).\
            #         sel(time=sel_time).mean('time'),tend_to_plot['height'],c='g',label='D. Adv \n+ dif')

        c=1
        ## HARMONIE cy43 
        # axs[idvar,idx].plot(unit*acc_time*h_clim_to_plot['dt'+var+'_dyn'].\
        #             sel(time=sel_time).mean('time'),h_clim_to_plot['z'],c='k',ls='--',lw=2,label='H. dyn')         
        # axs[idvar,idx].plot(unit*acc_time*(h_clim_to_plot['dt'+var+'_phy']+h_clim_to_plot['dt'+var+'_dyn']).\
        #             sel(time=sel_time).mean('time'),h_clim_to_plot['z'],c='deeppink',ls='--',lw=2,label='H. Tot')
        # axs[idvar,idx].plot(unit*acc_time*h_clim_to_plot['dt'+var+'_phy'].\
        #             sel(time=sel_time).mean('time'),h_clim_to_plot['z'],c='g',ls='--',lw=2,label='H. phy')

        if var == 'u' or var == 'v':
            axs[idvar,idx].plot(unit*acc_time*h_clim_to_plot['dt'+var+'_turb'].\
                sel(time=sel_time).mean('time'),h_clim_to_plot['z'],c='c',ls='--',lw=1,label='H. turb')
            axs[idvar,idx].plot(unit*acc_time*h_clim_to_plot['dt'+var+'_conv'].\
                sel(time=sel_time).mean('time'),h_clim_to_plot['z'],c='b',ls='--',lw=1,label='H. conv')
    

        axs[idvar,idx].set_title(r'$\frac{d\overline{'+var+'}}{dt}$ - Group '+str(idx+1),fontsize=21)
        axs[idvar,idx].axvline(0,c='k',lw=0.5)
        axs[idvar,idx].set_xlim([-0.45,+0.45])        
        axs[idvar,idx].set_ylim(height_lim)
        axs[idvar,idx].set_xlabel(r'($m/s$ / hour)',fontsize=18)
        axs[idvar,idx].axhline(200,c='k',ls='-',lw=0.5)
        axs[idvar,idx].axhline(650,c='k',ls='-',lw=0.5)
        axs[idvar,idx].axhline(1500,c='k',ls='-',lw=0.5)
        if idx >0:
            axs[idvar,idx].yaxis.set_visible(False) 
        axs[0,idx].set_xlabel(None)
axs[0,0].legend(frameon=False,loc='upper right')
axs[0,0].set_ylabel(r'z ($m$)')
axs[1,0].set_ylabel(r'z ($m$)')
for n, ax in enumerate(axs.flat):
    ax.text(0.08, 0.95, string.ascii_uppercase[n], transform=ax.transAxes, 
                size=13)
plt.tight_layout()
plt.savefig(save_dir+'Figure11_mom_budget.pdf', bbox_inches="tight")  
##################
# 
#%% ## FIGURE 11 bis ##
### tmser of momentum budget 
fig, axs = plt.subplots(2,1,figsize=(20,16))
layer =   'CLtop' # [0,1500]
rol = 8

if layer =='CLtop':
    dales_to_plot   = samptend.where(samptend.z<hc_ql).mean('z')
    scale_tend      =(da_scales_prof.diff('height')/np.diff(da_scales_prof['height']))\
        .where(da_scales_prof.height<hc_ql).mean('height')
else:
    dales_to_plot   = samptend.sel(z=slice(layer[0],layer[1])).mean('z')\
        # .sel(time=slice(np.datetime64('2020-02-02'),np.datetime64('2020-02-11')))
        
    scale_tend = (da_scales_prof.diff('height')/np.diff(da_scales_prof['height']))\
        .sel(height=slice(layer[0],layer[1])).mean('height')\
        # .sel(time=slice(np.datetime64('2020-02-02'),np.datetime64('2020-02-11')))  
        
# h_clim_to_plot = harm_clim_avg.sel(z=slice(layer[0],layer[1])).mean('z')\
    # .sel(time=slice(np.datetime64('2020-02-02'),np.datetime64('2020-02-11')))
    
  

for idx, var in enumerate(['u','v']):
    ## DALES 
    (acc_time*dales_to_plot.rolling(time=rol*4).mean()[var+'tendtotall']).\
        plot(c='deeppink',lw=2.5,label='D. Tot',ax=axs[idx])
    (acc_time*dales_to_plot.rolling(time=rol*4).mean()[var+'tendlsall']).\
        plot(c='k',lw=2.5,label='D. LS',ax=axs[idx])
    (acc_time*dales_to_plot.rolling(time=rol*4).mean()[var+'tendphyall']).\
        plot(c='g',lw=2.5,label='D. Tot - LS',ax=axs[idx])
        
    ((-acc_time*scale_tend[var+'_psfw_psf'])\
        +(acc_time*dales_to_plot[var+'tenddifall'])).rolling(time=rol*2).mean()\
        .plot(c='mediumpurple',marker='o',lw=2.5,\
              markevery=15,markerfacecolor='none',\
                  label=r'D. Sub-filter ($\Delta x=2.5 km$)',ax=axs[idx])
    # (-acc_time*scale_tend[var+'_pfw_pf']).rolling(time=rol*2).mean()\
    #     .plot(c='orange',ls='-',label='DALES: Up-filter',ax=axs[idx])
        
    # (-acc_time*(scale_tend[var+'_psfw_psf']+scale_tend[var+'_pfw_pf'])\
    #     +(acc_time*dales_to_plot[var+'tenddifall'])).rolling(time=rol*2).mean()\
    #     .plot(c='m',ls='-',label='DALES: SUM',ax=axs[idx])
    
    ## HARMONIE cy43 clim
    # (acc_time*(h_clim_to_plot['dt'+var+'_dyn']+h_clim_to_plot['dt'+var+'_phy'])\
    #  .rolling(time=rol).mean()).plot(c='r',ls=':',label='HAR: Tot',ax=axs[idx])
    # (acc_time*h_clim_to_plot['dt'+var+'_dyn'].rolling(time=rol).mean()).plot(c='k',ls=':',label='HAR: Dyn',ax=axs[idx])
    # (acc_time*(dales_to_plot[var+'tendlsall']+h_clim_to_plot['dt'+var+'_phy'])\
    #  .rolling(time=rol).mean()).plot(c='deeppink',ls=':',label='H. Tot',ax=axs[idx])
    # (acc_time*dales_to_plot[var+'tendlsall'].rolling(time=rol*4).mean()).plot(c='k',ls=':',label='H. Dyn',ax=axs[idx])
    # (acc_time*h_clim_to_plot['dt'+var+'_phy'].rolling(time=rol).mean()).plot(c='g',ls=':',label='H. Phy',ax=axs[idx])

    axs[idx].axhline(0,c='k',lw=1)
    axs[idx].set_ylabel(r'Tendency ($m s^{-1} hour^{-1}$)')
    # axs[idx].set_ylim([-0.9,0.82])
    axs[idx].set_ylim([-0.6,0.41])
    for idgroup, sel_time in enumerate([time_g1['iorg'],time_g3['iorg']]):
        # axs[idx].scatter(sel_time,[-0.89] * len(sel_time),\
        #                      c=['orange','green'][idgroup])
            
        for ii in sel_time:
            axs[idx].fill_betweenx([-1,1],ii-np.timedelta64(15,'m'),\
                            ii+np.timedelta64(14,'m'),\
                                color=['orange','green'][idgroup],alpha=0.1)
    
    if layer == 'CLtop':
        axs[idx].set_title('Mean '+var+' tendency below cloud top',fontsize=30)
    else:        
        axs[idx].set_title('Mean '+var+' tendency between '+str(layer[0])+' and '+str(layer[1])+' m',fontsize=30)
    axs[idx].set_xlim([srt_plot,end_time])

    #####
    for day in np.arange(srt_plot,end_plot):
        axs[idx].axvline(x=day,c='k',lw=1)
        
axs[0].legend(ncol=2,fontsize=20)
axs[0].set_xlabel(None)
axs[0].xaxis.set_visible(False) 
axs[1].set_xlabel(None)

plt.tight_layout()
for n, ax in enumerate(axs):
    ax.text(0.95, 0.95, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=20)
# plt.savefig(save_dir+'Figure_mom_budget_tmser.pdf', bbox_inches="tight")
##################
#%% ## FIGURE 11 tris ## vertically integrated tendency 
### tmser of momentum budget 
fig, axs = plt.subplots(2,1,figsize=(20,16))
# layer = [0,750]
layer =   'CLtop' # [0,1500]
rol = 8
klp      = 38
fil_size = round(150/(2*klp),1) 

if layer =='CLtop':
    dales_to_plot   = samptend.where(samptend.z<hc_ql).sum('z')
    scale_tend      =(da_scales_prof.diff('height')/np.diff(da_scales_prof['height']))\
        .sel(klp=klp).where(da_scales_prof.height<hc_ql).sum('height')
else:
    dales_to_plot   = samptend.sel(z=slice(layer[0],layer[1])).sum('z')\
        # .sel(time=slice(np.datetime64('2020-02-02'),np.datetime64('2020-02-11')))
        
    scale_tend = (da_scales_prof.diff('height')/np.diff(da_scales_prof['height']))\
        .sel(klp=klp).sel(height=slice(layer[0],layer[1])).sum('height')

for idx, var in enumerate(['u','v']):
    ## DALES 
    (acc_time*dales_to_plot.rolling(time=rol*4).mean()[var+'tendtotall']).\
        plot(c='deeppink',lw=2.5,label='Total',ax=axs[idx])
    (acc_time*dales_to_plot.rolling(time=rol*4).mean()[var+'tendlsall']).\
        plot(c='k',lw=2.5,label='Large scale',ax=axs[idx])
    (acc_time*dales_to_plot.rolling(time=rol*4).mean()[var+'tendphyall']).\
        plot(c='g',lw=2.5,label='Total - Large scale',ax=axs[idx])
        
    ((-acc_time*scale_tend[var+'_psfw_psf'])\
        +(acc_time*dales_to_plot[var+'tenddifall'])).rolling(time=rol*2).mean()\
        .plot(c='mediumpurple',marker='o',lw=2.5,\
              markevery=15,markerfacecolor='none',\
                  label=r'Sub-filter ($\Delta x='+str(fil_size)+' km$)',ax=axs[idx])
    # (-acc_time*scale_tend[var+'_pfw_pf']).rolling(time=rol*2).mean()\
    #     .plot(c='orange',ls='-',label='DALES: Up-filter',ax=axs[idx])
        
    # (-acc_time*(scale_tend[var+'_psfw_psf']+scale_tend[var+'_pfw_pf'])\
    #     +(acc_time*dales_to_plot[var+'tenddifall'])).rolling(time=rol*2).mean()\
    #     .plot(c='m',ls='-',label='DALES: SUM',ax=axs[idx])
    
    ## HARMONIE cy43 clim
    # (acc_time*(h_clim_to_plot['dt'+var+'_dyn']+h_clim_to_plot['dt'+var+'_phy'])\
    #  .rolling(time=rol).mean()).plot(c='r',ls=':',label='HAR: Tot',ax=axs[idx])
    # (acc_time*h_clim_to_plot['dt'+var+'_dyn'].rolling(time=rol).mean()).plot(c='k',ls=':',label='HAR: Dyn',ax=axs[idx])
    # (acc_time*(dales_to_plot[var+'tendlsall']+h_clim_to_plot['dt'+var+'_phy'])\
    #  .rolling(time=rol).mean()).plot(c='deeppink',ls=':',label='H. Tot',ax=axs[idx])
    # (acc_time*dales_to_plot[var+'tendlsall'].rolling(time=rol*4).mean()).plot(c='k',ls=':',label='H. Dyn',ax=axs[idx])
    # (acc_time*h_clim_to_plot['dt'+var+'_phy'].rolling(time=rol).mean()).plot(c='g',ls=':',label='H. Phy',ax=axs[idx])

    axs[idx].axhline(0,c='k',lw=1)
    axs[idx].set_ylabel(r'Tendency ($m s^{-1} hour^{-1}$)',fontsize=30)
    axs[idx].set_ylim([-40,31])
    for idgroup, sel_time in enumerate([time_g1['iorg'],time_g3['iorg']]):
        # axs[idx].scatter(sel_time,[-39.5] * len(sel_time),\
        #                      c=['orange','green'][idgroup])
        for ii in sel_time:
            axs[idx].fill_betweenx([-100,+100],ii-np.timedelta64(15,'m'),\
                            ii+np.timedelta64(14,'m'),\
                                color=['orange','green'][idgroup],alpha=0.1)
    
    if layer == 'CLtop':
        axs[idx].set_title('Integrated '+var+' tendency below cloud top',fontsize=35)
    else:        
        axs[idx].set_title('Integrated '+var+' tendency between '+str(layer[0])+' and '+str(layer[1])+' m',fontsize=35)
    axs[idx].set_xlim([srt_plot,end_time])
    axs[idx].tick_params(axis='both', which='major', labelsize=28)
    #####
    for day in np.arange(srt_plot,end_plot):
        axs[idx].axvline(x=day,c='k',lw=1)
        
axs[0].legend(ncol=2,fontsize=22)
axs[0].set_xlabel(None)
axs[0].xaxis.set_visible(False) 
axs[1].set_xlabel(None)

plt.tight_layout()
for n, ax in enumerate(axs):
    ax.text(0.95, 0.93, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=30)
plt.tight_layout()
plt.savefig(save_dir+'Figure_integrated_mom_budget_tmser.pdf', bbox_inches="tight")
##################
#%% FIGURE 13 BUDGET CYCLE AND HYSTERESIS 
layer = [0,200]
acc_time = 3600*1
klp=38
day_interval=['2020-02-02','2020-02-09']

dales_to_plot   = samptend.sel(z=slice(layer[0],layer[1])).mean('z')\
    .sel(time=slice(day_interval[0],day_interval[1]))
scale_tend = (da_scales_prof.diff('height')/np.diff(da_scales_prof['height']))\
    .sel(height=slice(layer[0],layer[1])).mean('height').\
        sel(time=slice(day_interval[0],day_interval[1]))
dales_temp = acc_time*dales_to_plot.groupby(dales_to_plot.time.dt.hour).mean()

fig, axs = plt.subplots(2,2,figsize=(20,15),gridspec_kw={'width_ratios': [1, 2]})
idy = 1
for idx,var in enumerate(['u','v']):
    ## DALES
    dales_temp[var+'tendlsall'].plot(c='k',lw=6, label='Large-scale forcing',ax=axs[idx,idy])
    dales_temp[var+'tendphyall'].plot(c='g',lw=6,label='Subgrid + resolved eddies',ax=axs[idx,idy])
    (-acc_time*scale_tend[var+'_pfw_pf'].sel(klp=klp)).groupby(scale_tend.time.dt.hour).mean()\
        .plot(c='brown',ls='-',lw=6,marker='+',label='Mesoscale flows',ax=axs[idx,idy])
    ax2 = axs[idx,idy].twinx()  
    profiles[var].sel(time=slice(day_interval[0],day_interval[1]),\
                      z=slice(layer[0],layer[1])).mean('z')\
        .groupby('time.hour').mean().plot(color='b',\
                                                  ls='--',lw=3,label= 'Zonal wind',ax=ax2)
    ax2.tick_params(axis='y', colors='b')
    if var =='u':
        ax2.set_ylim([-12.5,-9])
        ax2.set_ylabel(r'Zonal wind ($m s^{-1}$)',color='b',fontsize=30)
    if var =='v':
        ax2.set_ylim([-3,+0.5])
        ax2.set_ylabel(r'Meridional wind ($m s^{-1}$)',color='b',fontsize=30)

    axs[idx,idy].axhline(0,c='k',lw=0.5)
    axs[idx,idy].set_title('')
    # axs[idx,idy].set_title('Mean '+var+' tendency below '+str(layer[1])+' m',fontsize=35)

    axs[idx,idy].tick_params(axis='both', which='major', labelsize=28)
    ax2.tick_params(axis='both', which='major', labelsize=28)

    axs[idx,idy].set_xlim([-0.75, 0.3])
    axs[idx,idy].set_ylabel(None)
    # axs[idx,idy].set_ylabel(r'Tendency ($m s^{-1} h^{-1}$)',fontsize=30)
    axs[idx,idy].set_xlim([0, 23]) 
    axs[idx,idy].set_ylim([-0.75, 0.6])
        
axs[1,idy].legend(loc='lower right',ncol=1,fontsize=28,bbox_to_anchor=(0.9,-0.8))
axs[0,idy].set_xlabel(None)
axs[1,idy].set_xlabel('Hour in local time',fontsize=30)
########
########

idy = 0

x  = (3600*samptend).sel(z=slice(layer[0],layer[1])).mean('z')
x1 = tend_daily.sel(z=slice(layer[0],layer[1])).mean('z')


##################
for idx,var in enumerate(['u','v']):
        
    colors = cm.jet(np.linspace(0, 1, 9))
    for idcol,ii in enumerate(x1.time[1:]):
        axs[idx,idy].scatter(x1[var+'tendlsall'].sel(time=ii),\
                          x1[var+'tendtotall'].sel(time=ii),\
                              marker='s',c=colors[idcol-1],s=140,label=str(ii.values)[5:10])
        if idcol%2:
            axs[idx,idy].scatter(x[var+'tendlsall'].sel(time=ii.dt.strftime('%Y-%m-%d').values),\
                          x[var+'tendtotall'].sel(time=ii.dt.strftime('%Y-%m-%d').values),\
                              c=colors[idcol-1],alpha=1,s=5)

    
    axs[idx,idy].plot([-1.1, 0.8], [-1.1, 0.8],c='grey',lw=1)
    axs[idx,idy].plot([-1.1, 0.8], [0,0],c='grey',lw=0.5)
    axs[idx,idy].plot([0,0], [-1.1, 0.8],c='grey',lw=0.5)
    
    axs[idx,idy].set_xlim([-1.05, 0.3])
    axs[idx,idy].set_ylim([-0.75, 0.6])
    # axs[idy].set_xlim([-0.6, 0.4])
    axs[idx,idy].tick_params(axis='both', which='major', labelsize=28)
    
    # plt.ylabel('Divergence duwt_dz ()')
    axs[idx,idy].set_ylabel(r'Total tendency ($m s^{-1} h^{-1}$)',fontsize=30)
    
# axs[idx,0].set_title('Zonal momentum budget',size=25)
# axs[idx,1].set_title('Meridional momentum budget',size=25)
axs[1,idy].set_xlabel(r'Large scale tendency ($m s^{-1} h^{-1}$)',fontsize=30)

axs[1,idy].legend(loc='lower left',ncol=2,fontsize=23, bbox_to_anchor=(0.01,-0.9))
plt.suptitle('Mean of the lower '+str(layer[1])+' m',fontsize=35)

plt.tight_layout()
# plt.savefig(save_dir+'Figure13_mosaic.pdf', bbox_inches="tight")

#%% Diurnal cycle 
## COMPOSITE
layer = [0,200]
acc_time = 3600*1
klp=38
day_interval=['2020-02-02','2020-02-09']

dales_to_plot   = samptend.sel(z=slice(layer[0],layer[1])).mean('z')\
    .sel(time=slice(day_interval[0],day_interval[1]))
h_clim_to_plot = harm_clim_avg.sel(z=slice(layer[0],layer[1])).mean('z')\
    .sel(time=slice(day_interval[0],day_interval[1]))
    
scale_tend = (da_scales_prof.diff('height')/np.diff(da_scales_prof['height']))\
    .sel(height=slice(layer[0],layer[1])).mean('height').\
        sel(time=slice(day_interval[0],day_interval[1]))

dales_temp = acc_time*dales_to_plot.groupby(dales_to_plot.time.dt.hour).mean()
fig, axs = plt.subplots(2,1,figsize=(19,15))
for idx,var in enumerate(['v','u']):
    ## DALES
    # dales_temp[var+'tendtotall'].plot(c='r',lw=4,label='DALES: Tot',ax=axs[idx])
    dales_temp[var+'tendlsall'].plot(c='k',lw=6, label='Large-scale forcing',ax=axs[idx])
    dales_temp[var+'tendphyall'].plot(c='g',lw=6,label='Subgrid + resolved eddies',ax=axs[idx])
    # ((-acc_time*scale_tend[var+'_psfw_psf'].sel(klp=klp)).groupby(scale_tend.time.dt.hour).mean()\
    #     +dales_temp[var+'tenddifall'])\
    #     .plot(c='m',marker='o',lw=2,markerfacecolor='none',label='DALES: Sub-filter',ax=axs[idx])
    (-acc_time*scale_tend[var+'_pfw_pf'].sel(klp=klp)).groupby(scale_tend.time.dt.hour).mean()\
        .plot(c='brown',ls='-',lw=6,marker='+',label='Mesoscale flows',ax=axs[idx])
        
    ax2 = axs[idx].twinx()  
    profiles[var].sel(time=slice(day_interval[0],day_interval[1]),\
                      z=slice(layer[0],layer[1])).mean('z')\
        .groupby('time.hour').mean().plot(color='b',\
                                                  ls='--',lw=3,label= 'Zonal wind',ax=ax2)
    ax2.tick_params(axis='y', colors='b')
    if var =='u':
        ax2.set_ylim([-12.5,-9])
        ax2.set_ylabel(r'Zonal wind ($m s^{-1}$)',color='b',fontsize=30)
    if var =='v':
        ax2.set_ylim([-3,+0.5])
        ax2.set_ylabel(r'Meridional wind ($m s^{-1}$)',color='b',fontsize=30)
            

    # ## HARMONIE cy43 clim
    # acc_time*(h_clim_to_plot['dt'+var+'_dyn']+h_clim_to_plot['dt'+var+'_phy'])\
    #     .groupby(h_clim_to_plot.time.dt.hour).mean().\
    #         plot(c='r',ls=':',label='H.clim cy43: Tot',ax=axs[idx])
    # acc_time*h_clim_to_plot.groupby(h_clim_to_plot.time.dt.hour).mean()\
    #     ['dt'+var+'_dyn'].\
    #         plot(c='k',ls=':',label='H.clim cy43: Dyn',ax=axs[idx])
    # acc_time*h_clim_to_plot.groupby(h_clim_to_plot.time.dt.hour).mean()\
    #     ['dt'+var+'_phy'].\
    #         plot(c='c',ls=':',label='H.clim cy43: Phy',ax=axs[idx]) 

    axs[idx].axhline(0,c='k',lw=0.5)
    axs[idx].set_title('Mean '+var+' tendency below '+str(layer[1])+' m',fontsize=35)

    axs[idx].tick_params(axis='both', which='major', labelsize=28)
    ax2.tick_params(axis='both', which='major', labelsize=28)
    if var == 'thl':
        plt.ylim([-0.00005, 0.00005]) 
        plt.ylabel('Tendency (K /hour)')
    elif var == 'qt':
        plt.ylim([-0.001, 0.001]) 
        plt.ylabel('Tendency (g/kg /hour)')
    else:
        # axs[idx].set_ylim([-0.85,0.8])
        axs[idx].set_ylabel(r'Tendency ($m s^{-1} h^{-1}$)',fontsize=30)
        axs[idx].set_xlim([0, 23]) 
        
axs[0].legend(ncol=2,fontsize=28)
axs[0].set_xlabel(None)
axs[1].set_xlabel('Hour in local time',fontsize=30)
    

plt.tight_layout()
# plt.savefig(save_dir+'Budget_diurnal_cycle.pdf', bbox_inches="tight")
########################
#%% Groups and time of the day
plt.figure(figsize=(15,8))
time_g1['iorg'].dt.hour.plot.hist(color=col_groups[0],alpha=0.3,label='Group 1')
# time_g2['iorg'].dt.hour.plot.hist(color=col_groups[1],alpha=0.3,label='Group 2')
time_g3['iorg'].dt.hour.plot.hist(color=col_groups[2],alpha=0.3,label='Group 3')
plt.legend(fontsize=15)
plt.xlim([0,23])
plt.xlabel('Hour in LT')
plt.ylabel('Count')
plt.title('Distribution of scenes grouped by Iorg',fontsize=25)

#%% Percentage of unresolved flux

plt.figure()
(profiles.vwr).sel(z=200,method='nearest').plot()
(profiles.vws).sel(z=200,method='nearest').plot()
(profiles.vwt).sel(z=200,method='nearest').plot()
plt.ylim([-0.005,0.01])
(profiles.uws/profiles.uwt).sel(time='2020-02-03').sel(z=200,method='nearest').mean()

#########################
#%%
#### CORRELATION PLOTS of tendencies 
layer = [0,200]
color_time = 'p'
model = 'dales'

if model == 'harm':
    y  = 3600*(harm_clim_avg['dt'+var+'_dyn']+harm_clim_avg['dt'+var+'_phy']).sel(z=slice(layer[0],layer[1])).mean('z')
    x  = 3600*(harm_clim_avg).sel(z=slice(layer[0],layer[1])).mean('z')['dt'+var+'_dyn']
    y1 = 3600*(harm_clim_avg['dt'+var+'_dyn']+harm_clim_avg['dt'+var+'_phy']).resample(time='D').mean('time').sel(z=slice(layer[0],layer[1])).mean('z')
    x1 = 3600*(harm_clim_avg.resample(time='D').mean('time')).sel(z=slice(layer[0],layer[1])).mean('z')['dt'+var+'_dyn']
elif model =='dales':        
    # y  = (3600*samptend).sel(z=slice(layer[0],layer[1])).mean('z')[var+'tendtotall']
    x  = (3600*samptend).sel(z=slice(layer[0],layer[1])).mean('z')
    # y1 = tend_daily.sel(z=slice(layer[0],layer[1])).mean('z')[var+'tendtotall']
    x1 = tend_daily.sel(z=slice(layer[0],layer[1])).mean('z')

# y  = profiles.sel(z=slice(layer[0],layer[1])).mean('z')['duwt_dz']
# y1 = profiles_daily.sel(z=slice(layer[0],layer[1])).mean('z')['duwt_dz']

#### FIGURE 1
##################
fig, axs = plt.subplots(1,2,figsize=(21,8))
for idy,var in enumerate(['u','v']):
    if color_time=='hours':
        colors = cm.hsv(np.linspace(0, 1, 24))
        for hour in range(0,24):
            # axs[idy].scatter((3600*samptend[var+'tendlsall']).sel(z=slice(layer[0],layer[1])).mean('z').where(samptend.time.dt.hour==hour,drop=True),\
            #           (3600*samptend[var+'tendtotall']).sel(z=slice(layer[0],layer[1])).mean('z').where(samptend.time.dt.hour==hour,drop=True),\
            #           c=colors[hour],alpha=0.3,s=4)
            # plt.scatter((3600*samptend[var+'tendlsall']).sel(z=slice(layer[0],layer[1])).mean('z').where(samptend.time.dt.hour==hour,drop=True).mean(),\
            #           (3600*samptend[var+'tendtotall']).sel(z=slice(layer[0],layer[1])).mean('z').where(samptend.time.dt.hour==hour,drop=True).mean(),\
            #             label=hour,c=colors[hour],marker='s',s=70)
            axs[idy].text((3600*samptend[var+'tendlsall']).sel(z=slice(layer[0],layer[1])).mean('z').where(samptend.time.dt.hour==hour,drop=True).mean(),\
                      (3600*samptend[var+'tendtotall']).sel(z=slice(layer[0],layer[1])).mean('z').where(samptend.time.dt.hour==hour,drop=True).mean(),\
                        hour,c=colors[hour],fontsize=13,weight='bold')
         
    elif color_time == 'groups':
        for idx,ii in enumerate([time_g1['iorg'],time_g2['iorg'][0],time_g3['iorg']]):
            axs[idy].scatter(x[var+'tendlsall'].sel(time=ii),\
                             x[var+'tendtotall'].sel(time=ii),\
                                 c=col_groups[idx],alpha=0.5,s=7,label='Group '+str(idx+1))
            # plt.scatter(x.sel(time=ii.dt.strftime('%Y-%m-%d').values),\
            #             y.sel(time=ii.dt.strftime('%Y-%m-%d').values),c=col[idx-1],alpha=0.5,s=5)
            
    
        
    else:
        # plt.scatter(x,y,alpha=0.7,s=10)
        colors = cm.jet(np.linspace(0, 1, 9))
        for idx,ii in enumerate(x1.time[1:]):
            axs[idy].scatter(x1[var+'tendlsall'].sel(time=ii),\
                              x1[var+'tendtotall'].sel(time=ii),\
                                  marker='s',c=colors[idx-1],s=140,label=str(ii.values)[5:10])
            if idx%2:
                axs[idy].scatter(x[var+'tendlsall'].sel(time=ii.dt.strftime('%Y-%m-%d').values),\
                              x[var+'tendtotall'].sel(time=ii.dt.strftime('%Y-%m-%d').values),\
                                  c=colors[idx-1],alpha=1,s=5)
        # for idx,ii in enumerate([time_g1['iorg'],time_g3['iorg']]):
        #     axs[idy].scatter(x[var+'tendlsall'].sel(time=ii).mean('time'),\
        #                       x[var+'tendtotall'].sel(time=ii).mean('time'),\
        #                           c=col_groups[idx+idx],alpha=0.9,s=55,marker='s',label='Group '+str(idx+1+idx))
        # for hour in range(0,24):
        #     axs[idy].text(x[var+'tendlsall'].where(samptend.time.dt.hour==hour,drop=True).mean(),\
        #               (x[var+'tendtotall']).where(samptend.time.dt.hour==hour,drop=True).mean(),\
        #                 hour,c=matplotlib.cm.get_cmap('tab20b')(1/23*hour),fontsize=15,weight='light')
    ################## 
    
    axs[idy].plot([-1.1, 0.8], [-1.1, 0.8],c='grey',lw=1)
    axs[idy].plot([-1.1, 0.8], [0,0],c='grey',lw=0.5)
    axs[idy].plot([0,0], [-1.1, 0.8],c='grey',lw=0.5)
    
    if var == 'thl':
        plt.xlim([-0.28, 0.25])
        plt.ylim([-0.4, 0.25]) 
    elif var == 'qt':
        plt.xlim([-0.0005, 0.0003])
        plt.ylim([-0.00003, 0.00003]) 
    else:
        axs[idy].set_xlim([-0.75, 0.3])
        axs[idy].set_ylim([-0.75, 0.3])
        # axs[idy].set_xlim([-0.6, 0.4])
        # axs[idy].set_ylim([-0.6, 0.4])
    
    # plt.ylabel('Divergence duwt_dz ()')
    axs[idy].set_xlabel(r'Large scale tendency ($m s^{-1} h^{-1}$)')
    
axs[0].set_title('Zonal momentum budget',size=25)
axs[1].set_title('Meridional momentum budget',size=25)
axs[0].set_ylabel(r'Total tendency ($m s^{-1} h^{-1}$)')
axs[0].legend(ncol=2,fontsize=23)
plt.suptitle('Mean between '+str(layer[0])+' and '+str(layer[1])+' m',fontsize=22)

plt.tight_layout()
# plt.savefig(save_dir+'Figure13_hysteresis.pdf', bbox_inches="tight")  

################################
################################




#%% PLOT ORG METRICS
# for var in da_org_norm:
#     plt.figure(figsize=(9,5))
#     da_org_norm[var].plot()
    
#     da_org_norm.where(da_org_norm['iorg'] <= da_org_norm['iorg'].quantile(0.25),drop=True).time

x= end_of_code

#%% ## CROSS TERMS 

desktop_dir = '/Users/acmsavazzi/Desktop/'

da_scales['u_pfw_pf_norm'] = da_scales['u_pfw_pf']/(da_scales.uw_pf + da_scales.uw_psf)
da_scales['u_psfw_psf_norm'] = da_scales['u_psfw_psf']/(da_scales.uw_pf + da_scales.uw_psf)
da_scales['v_pfw_pf'] = da_scales['v_pfw_pf']/(da_scales.vw_pf + da_scales.vw_psf)
da_scales['v_psfw_pf'] = da_scales['v_psfw_psf']/(da_scales.vw_pf + da_scales.vw_psf)
 
lev =200

# (da_scales.uw_pf + da_scales.uw_psf).sel(height=200).isel(time=100).plot(label='Tot')
plt.figure()
plt.title('Decomposed momentum flux',fontsize=20)
plt.axhline(1,c='k',label='Tot')
plt.plot(da_scales['f_scales_norm_ql'].mean('time'),da_scales.u_pfw_pf_norm.sel(height=lev).median('time'),label='Up')
# plt.plot(da_scales['f_scales_norm_ql'].mean('time'),da_scales.u_psfw_psf_norm.sel(height=lev).quantile(0.75,'time'),ls='--',label='Sub - 75 percentile')
# plt.plot(da_scales['f_scales_norm_ql'].mean('time'),da_scales.u_psfw_psf_norm.sel(height=lev).quantile(0.25,'time'),ls='--',label='Sub - 25 percentile')

plt.plot(da_scales['f_scales_norm_ql'].mean('time'),da_scales.u_psfw_psf_norm.sel(height=lev).median('time'),label='Sub')
plt.plot(da_scales['f_scales_norm_ql'].mean('time'),(da_scales.u_pfw_pf_norm+da_scales.u_psfw_psf_norm)\
         .sel(height=lev).median('time'),label='Up + Sub')
plt.plot(da_scales['f_scales_norm_ql'].mean('time'),(1 - (da_scales.u_pfw_pf_norm + da_scales.u_psfw_psf_norm))\
         .sel(height=lev).median('time'),label='Cross (residual)',lw=3,c='r')

# da_scales.uw_psf.sel(height=200).isel(time=100).plot()
plt.legend()
plt.xscale('log')
plt.ylabel('Percentage')
plt.xlabel(r'$\Delta x / h_b$')
# plt.savefig(desktop_dir+'Decomposed_Uflux_'+str(lev)+'m.pdf', bbox_inches="tight")



plt.figure()
plt.title('Cross Terms',fontsize=20)
plt.plot(da_scales['f_scales_norm_ql'].mean('time'),(1 - (da_scales.u_pfw_pf_norm + da_scales.u_psfw_psf_norm)).sel(height=lev).mean('time'),label='Mean')
plt.plot(da_scales['f_scales_norm_ql'].mean('time'),(1 - (da_scales.u_pfw_pf_norm + da_scales.u_psfw_psf_norm)).sel(height=lev).median('time'),label='Median',lw=3,c='r')
plt.plot(da_scales['f_scales_norm_ql'].mean('time'),(1 - (da_scales.u_pfw_pf_norm + da_scales.u_psfw_psf_norm)).sel(height=lev).quantile(0.90,'time'),label='90 percentile')
plt.plot(da_scales['f_scales_norm_ql'].mean('time'),(1 - (da_scales.u_pfw_pf_norm + da_scales.u_psfw_psf_norm)).sel(height=lev).quantile(0.10,'time'),label='10 percentile')
# da_scales.uw_psf.sel(height=200).isel(time=100).plot()
plt.legend()
plt.xscale('log')
plt.ylabel('Percentage')
plt.xlabel(r'$\Delta x / h_b$')
# plt.savefig(desktop_dir+'Cross_terms_Uflux_'+str(lev)+'m.pdf', bbox_inches="tight")


#%%

# plt.plot(da_scales['f_scales_norm'].mean('time'),(da_toplot['uw_pf']+da_toplot['uw_psf']).median('time').isel(height=0))
plt.plot(da_scales['f_scales_norm'].mean('time'),(da_toplot['u_pfw_pf']).median('time').isel(height=0))
# plt.plot(da_scales['f_scales_norm'].mean('time'),(da_toplot['u_psfw_psf']+da_toplot['u_pfw_pf']).median('time').isel(height=0))

plt.xscale('log')
#%%

honnert = True
fig, axs = plt.subplots(3,2,figsize=(12,12))
for idcol, var in enumerate(['u_psfw_psf','v_psfw_psf']):
    # normalised y axis
    da_toplot = da_scales_norm
    for idx,ih in enumerate(range(len(da_scales.height[0:3]))):   
        iteration =0
        for day in ['02','03','04','05','06','07','08','09']:
            iteration +=1
            if honnert==True:
                axs[idx,idcol].plot(da_scales['f_scales_norm_ql'].\
                resample(time='8h').mean('time').sel(time='2020-02-'+day),\
                  da_toplot.resample(time='8h').median('time')\
                      [var].isel(height=ih).sel(time='2020-02-'+day).T\
                    ,c=cmap(rgba*iteration),label=day)  
                axs[idx,idcol].axvline(1,c='k',lw=0.5)
            else:
                axs[idx,idcol].plot(f_scales/1000,\
                  da_toplot.resample(time='8h').median('time')\
                      [var].isel(height=ih).sel(time='2020-02-'+day).T\
                    ,c=cmap(rgba*iteration))   
                axs[idx,idcol].axvline(2.5,c='k',lw=0.5)
        axs[idx,idcol].set_xscale('log')
        
        axs[idx,idcol].axhline(0,c='k',lw=0.5)
        
        
        if idcol == 0:
            axs[idx,idcol].set_ylabel('Flux partition \n at '+str(da_scales.height[ih].values)+' m')
        axs[idx,idcol].set_ylim([-0.15,1.15])
        axs[idx,1].yaxis.set_visible(False) 
    
        # axs[0,1].legend()
    if honnert==True:
        axs[2,idcol].set_xlabel(r'Dimensionless $\frac{\Delta x}{h_b}$ ')
    else:
        axs[2,idcol].set_xlabel(r'Filter size $\Delta x$ ($km$)')

axs[0,0].set_title('Zonal momentum flux',fontsize=21)  
axs[0,1].set_title('Meridional momentum flux',fontsize=21)  
    
for n, ax in enumerate(axs.flat):
    ax.text(0.08, 0.9, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=13)
plt.tight_layout()
#%% ## like figure3 but for scalars ##
bottom, top = 0.1, 0.9
left, right = 0.1, 0.8

fig, axs = plt.subplots(2,2,figsize=(22,12), gridspec_kw={'width_ratios': [1,6]})
fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, \
                    hspace=0.15, wspace=0.25)

for idx,var in enumerate(['thv','wthvt']):
    iteration = 0
    profiles[var].mean('time').plot(y='z',c='k',lw=4, label='Mean',ax=axs[idx,0])
    for day in np.unique(profiles.time.dt.day)[:-1]:
        iteration +=1
        profiles[var].sel(time='2020-02-'+str(day).zfill(2)).mean('time')\
            .plot(ax=axs[idx,0],y='z',c=cmap(rgba*iteration),lw=1.5,label='Feb-'+str(day).zfill(2))
   

    if var=='wthlt':
        colmin=-0.15
        colmax=0.03
        axs[1,0].set_xlim([-0.08,0.03])
    elif var =='wthvt':
        colmin=-0.005
        colmax=0.07
    elif var =='wqtt':
        colmin=-0.00005
        colmax=0.00015
    else:
        colmin=None
        colmax=None
    im = (profiles[var]).plot(y='z',vmax=colmax,vmin=colmin,\
          cmap=cm.PiYG_r,norm=mcolors.TwoSlopeNorm(0),ax=axs[idx,1],\
              add_colorbar=True,cbar_kwargs={r'label':'$m^2 s^{-2}$'})
        
    # tmser.zi.plot(x='time',ax=axs[idx,1],c='b',ls='-',label='Boundary layer')
    hc_ql.plot(x='time',ax=axs[idx,1],c='k',ls='-',label='cloud top')
    # hc_thlvw.plot(x='time',ax=axs[idx,1],c='r',ls='--',label='Boundary layer thlw')
    

    axs[idx,0].yaxis.set_visible(True) 
    axs[idx,0].set_ylabel(r'z ($m$)')
    axs[idx,0].set_xlabel(r' X $m s^{-1}$')
    axs[idx,0].axvline(0,c='k',lw=0.5)
    axs[idx,1].yaxis.set_visible(False) 
    axs[idx,0].set_ylim(height_lim)
    axs[idx,1].set_ylim(height_lim)
    axs[idx,1].set_xlim([srt_plot,end_time])
    for day in np.arange(srt_plot,end_plot):
        axs[idx,1].axvline(x=day,c='k',lw=0.5)

axs[0,1].set_title('Total Qt flux',fontsize=25)   
axs[1,1].set_title('Total Thv flux',fontsize=25)   
axs[0,1].xaxis.set_visible(False) 
axs[0,0].legend(fontsize=16)
axs[0,1].legend(fontsize=16,loc='upper left')

axs[1,1].set_xlabel(None)
for n, ax in enumerate(axs.flat):
    ax.text(0.9, 1.05, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=20)
# cbar_ax = fig.add_axes([0.9, 0.15, 0.01, 0.7])  # Left, bottom, width, height.
# cbar = fig.colorbar(im, cax=cbar_ax, extend='both', orientation='vertical')
# cbar.set_label(r'$m^2 s^{-1}$')
plt.tight_layout()

#%% UNDERSTAND FLUXES
# sel_time = time_g1['iorg']
sel_time = da_scales_prof.sel(time=slice('2020-02-03','2020-02-03T00:15')).time
# sel_time = da_scales_prof.sel(time='2020-02-03T01:30').time

# sel_time_prof = profiles.sel(time=slice('2020-02-03T01:15','2020-02-03T01:45')).time

sel_time_prof = sel_time

var ='v'
plt.figure()
if size(sel_time) >1:
    da_scales_prof[var+"w_pf"].sel(time=sel_time).mean('time').\
        plot(y='height',lw=2.5,ls='-',c='r',label='Total')
    
    profiles[var+"wr"].sel(time=sel_time_prof).mean('time').\
        plot(y='z',ls='--',c='r',label='Total from prof')   
else:
    da_scales_prof[var+"w_pf"].sel(time=sel_time).\
        plot(y='height',lw=2.5,ls='-',c='r',label='Total')
    
    profiles[var+"wr"].sel(time=sel_time_prof).\
        plot(y='z',ls='--',c='r',label='Total from prof')         
plt.ylim([100,600])

plt.figure()
if size(sel_time) >1:
    
    profiles[var].sel(time=sel_time_prof).mean('time').\
        plot(y='z',ls='--',c='r',label='Total from prof')   
else: 
    profiles[var].sel(time=sel_time_prof).\
        plot(y='z',ls='--',c='r',label='Total from prof')     
plt.xlim([-2.02,-1.51])
plt.ylim([100,600])
#%%
# percentage of sub-filter contribution as a contour, for filter of 2.5 km
da_scales_prof_norm = da_scales_prof/(da_scales_prof.uw_pf +da_scales_prof.uw_psf)
fig, axs = plt.subplots(2,1,figsize=(12,7))
for idvar, var in enumerate(['u','v']):
    da_scales_prof_norm[var+"_psfw_psf"].plot(x='time',vmin=-0.5,vmax=1.5,ax=axs[idvar])
    # da_scales_prof[var+"_psfw_psf"].plot(x='time',ax=axs[idvar])
    axs[idvar].axhline(200,c='k',ls='-',lw=0.5)
    axs[idvar].axhline(650,c='k',ls='-',lw=0.5)
    axs[idvar].axhline(1500,c='k',ls='-',lw=0.5)
axs[0].xaxis.set_visible(None)

#%% Wind profiles per group 
fig, axs = plt.subplots(1,4,figsize=(19,10))
for idx,var in enumerate(['u','v','thl','qt']):
    for idgroup,sel_time in enumerate([time_g1['iorg'],time_g2['iorg'],time_g3['iorg']]):
        profiles[var].sel(time=sel_time).mean('time').plot(y='z',\
                           c=col_groups[idgroup],lw=4, label='Group '+str(idgroup+1),ax=axs[idx])
            
        axs[idx].fill_betweenx(profiles.z.values,\
                           np.quantile(profiles[var].sel(time=sel_time).values,0.25,0),\
                           np.quantile(profiles[var].sel(time=sel_time).values,0.75,0),\
                              color=col_groups[idgroup],alpha=0.2)

       
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
    axs[idx].axhline(200,c='k',ls='-',lw=0.5)
    axs[idx].axhline(650,c='k',ls='-',lw=0.5)
    axs[idx].axhline(1500,c='k',ls='-',lw=0.5)
plt.tight_layout()
for n, ax in enumerate(axs):
    ax.text(0.03, 0.97, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=13)

#%% boxplot by groups
# The box extends from the lower to upper quartile values of the data,
# with a line at the median. 
# The whiskers extend from the box to show the range of the data

da_to_plot = da_scales_norm.where(abs(da_scales.f_scales_norm_ql-1)<=\
                                  abs(da_scales.f_scales_norm_ql-1).min('klp').max(),\
                                      drop=True)    
ih = 1500
fig, axs = plt.subplots(2,1,figsize=(12,7))
for idx, var in enumerate(['u_psfw_psf','v_psfw_psf']):
    if 'lab' in locals(): del lab
    if 'x_ax' in locals(): del x_ax
    iteration=0                           
    for sel_time in enumerate([time_g1['iorg'],time_g2['iorg'],time_g3['iorg']]):
        iteration+=1
        axs[idx].boxplot(da_to_plot[var].sel(time=sel_time[1])\
                        .sel(height=ih,method='nearest').mean('klp').values,\
                            positions=[iteration],\
                    whis=1.8,showfliers=False,showmeans=True,meanline=False,widths=0.25,\
                        medianprops=dict(color="r", lw=2))   
        
    axs[idx].axhline(0,c='k',lw=0.5)
    axs[idx].set_ylabel('Flux partition \n at '+str(ih)+' m')
    axs[idx].set_ylim([-1,1.7])

axs[0].set_title('Zonal momentum flux - ' +r'Mesh $\frac{\Delta x}{h_b} = 1$',fontsize=21) 
axs[1].set_title('Meridional momentum flux - ' +r'Mesh $\frac{\Delta x}{h_b} = 1$',fontsize=21) 



for n, ax in enumerate(axs.flat):
    ax.text(0.97, 0.9, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=13)
plt.tight_layout()


#%% SPECTRA normalised by different boundary layers
rgba = 1/3
fig, axs = plt.subplots(3,2,figsize=(12,12))
for idcol, var in enumerate(['u_psfw_psf','v_psfw_psf']):
    # normalised y axis
    da_toplot = da_scales_norm
    for idx,ih in enumerate(range(len(da_scales.height[0:3]))):   
        iteration =0
        for day in ['08']:
        #     iteration +=1
            # axs[idx,idcol].plot(da_scales['f_scales_norm'].\
            # resample(time='4h').mean('time').sel(time='2020-02-'+day),\
            #   da_toplot.resample(time='4h').median('time')\
            #       [var].isel(height=ih).sel(time='2020-02-'+day).T\
            #     ,label=day)  
        #     axs[idx,idcol].set_prop_cycle(None)
        #     axs[idx,idcol].plot(da_scales['f_scales_norm_ql'].\
        #     resample(time='4h').mean('time').sel(time='2020-02-'+day),\
        #       da_toplot.resample(time='4h').median('time')\
        #           [var].isel(height=ih).sel(time='2020-02-'+day).T\
        #         ,ls='--',label=day)  
        
            ## quartiles     
            axs[idx,idcol].plot(da_scales['f_scales_norm'].\
            sel(time='2020-02-'+day).mean('time'),\
              da_toplot\
                  [var].isel(height=ih).sel(time='2020-02-'+day).median('time').T\
                ,label='Hb: min thl ',c='r')  
            axs[idx,idcol].fill_between(da_scales['f_scales_norm'].sel(time='2020-02-'+day).mean('time'),\
                          da_scales_norm[var].isel(height=ih).sel(time='2020-02-'+day).quantile(0.25,dim='time'),\
                          da_scales_norm[var].isel(height=ih).sel(time='2020-02-'+day).quantile(0.75,dim='time'),\
                              color='red',alpha=0.1)
                
            axs[idx,idcol].plot(da_scales['f_scales_norm_ql'].\
            sel(time='2020-02-'+day).mean('time'),\
              da_toplot\
                  [var].isel(height=ih).sel(time='2020-02-'+day).median('time').T\
                ,label='Hb: ql=0',c='b')  
            axs[idx,idcol].fill_between(da_scales['f_scales_norm_ql'].sel(time='2020-02-'+day).mean('time'),\
                          da_scales_norm[var].isel(height=ih).sel(time='2020-02-'+day).quantile(0.25,dim='time'),\
                          da_scales_norm[var].isel(height=ih).sel(time='2020-02-'+day).quantile(0.75,dim='time'),\
                              color='blue',alpha=0.1)
            axs[idx,idcol].axvline(1,c='k',lw=0.5)
        axs[idx,idcol].set_xscale('log')
        

        
        if idcol == 0:
            axs[idx,idcol].set_ylabel('Flux partition \n at '+str(da_scales.height[ih].values)+' m')
        axs[idx,idcol].set_ylim([-0.15,1.15])
        axs[idx,1].yaxis.set_visible(False) 
    
        axs[0,1].legend()
    if honnert==True:
        axs[2,idcol].set_xlabel(r'Dimensionless $\frac{\Delta x}{h_b}$ ')
    else:
        axs[2,idcol].set_xlabel(r'Filter size $\Delta x$ ($km$)')

axs[0,0].set_title('Zonal momentum flux',fontsize=21)  
axs[0,1].set_title('Meridional momentum flux',fontsize=21)  
    
for n, ax in enumerate(axs.flat):
    ax.text(0.08, 0.9, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=13)
plt.tight_layout()

#%% Countergradient transport
for idx,var in enumerate(['u','v']): 
    plt.figure(figsize=(19,6))
    (profiles[var+'wt'] * profiles['d'+var+'_dz']).plot(x='time',vmin=-0.00005)
    plt.title('Counter-gradient transport: '+var+'w * d'+var+'_dz',fontsize=22)
    plt.ylim([0,4000])
    for day in np.arange(srt_plot,end_plot):
        plt.axvline(x=day,c='k',lw=0.5)
#%%
hrs_to_plot=profiles

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
        
#%% CONTOUR for VARIABLES (any model)
# profiles['uwr_sign']=profiles.uwr*(-1* np.sign(profiles.u))
# profiles['vwr_sign']=profiles.vwr*(-1* np.sign(profiles.v))

for var in ['u','qt','uwr','thl']:
    unit = 1
    vmin = None
    vmax = None
    if var == 'qt': 
        unit = 1
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
    for day in np.arange(srt_plot,end_plot):
        plt.axvline(x=day,c='k',lw=0.5)
#%% profiles  
var ='thl'  
plt.figure(figsize=(5,10))
(profiles['thl'].\
                    resample(time='1h').mean('time').sel(time=slice('2020-02-03T15','2020-02-03T20'))).plot.line(y='z')
plt.axhline(2500,c='k',lw=0.5)
plt.ylim([0,5000])
plt.legend('')



        
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
    ### SURFACE LATENT HEAT FLUX 
    plt.figure(figsize=(15,6))
    plt.plot(tmser.time, ls_surf['rho'].mean() * tmser.wq * Lv,lw=2.5,c=col[3],label='DALES')
    if comp_experiments:
        plt.plot(tmser_isurf5.time, ls_surf['rho'].mean() * tmser_isurf5.wq * Lv,c=col[5],lw=0.7,label='DALES exp')
    # harm_hind_avg['300'].LE.plot()
    harm_clim_avg.hfls.mean(dim=['x','y']).plot(lw=2.5,c=col[0],label='HARMONIE cy43')
    # xr.plot.scatter(Meteor,'time','LHF_bulk_mast',alpha = 0.6,s=12,c=col[2],label='Meteor')
    # xr.plot.scatter(Meteor,'time','LHF_EC_mast',alpha = 0.4,s=12,label='EC')
    plt.xlabel('time')
    plt.ylabel('LH (W/m2)')
    plt.title('Surface latent heat flux',size=20)
    plt.xlim(temp_hrs)
    plt.axvspan(srt_time,srt_time + np.timedelta64(2, 'h'), alpha=0.2, color='grey')
    plt.legend(fontsize=15)
    for day in np.arange(srt_time,end_time):
        plt.axvline(x=day,c='k',lw=0.5)
    # plt.savefig(save_dir+'LatentHeat_srf.pdf', bbox_inches="tight")
        
    ### SURFACE SENSIBLE HEAT FLUX
    plt.figure(figsize=(15,6))
    plt.plot(tmser.time, rho * tmser.wtheta * cp,lw=2.5,c=col[3],label='DALES')
    if comp_experiments:
        plt.plot(tmser_isurf5.time, rho * tmser_isurf5.wtheta * cp,c=col[5],lw=0.9,label='DALES exp')
    # harm_hind_avg['300'].H.plot(c=col[0],lw=2,label='HARMONIE_cy40 hind')
    harm_clim_avg.hfss.mean(dim=['x','y']).plot(lw=2.5,c=col[0],label='HARMONIE cy43')
    # xr.plot.scatter(Meteor,'time','SHF_bulk_mast',alpha = 0.6,s=12,c=col[2],label='Meteor')
    # xr.plot.scatter(Meteor,'time','SHF_EC_mast',alpha = 0.4,s=12,label='EC')
    plt.xlabel('time')
    plt.ylabel('SH ($W/m^2$)')
    plt.title('Surface sensible heat flux',size=20)
    plt.xlim(temp_hrs)
    plt.axvspan(srt_time,srt_time + np.timedelta64(2, 'h'), alpha=0.2, color='grey')
    plt.legend(fontsize=15)
    for day in np.arange(srt_time,end_time):
        plt.axvline(x=day,c='k',lw=0.5)
    # plt.savefig(save_dir+'SensHeat_srf.pdf', bbox_inches="tight")
        
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
        title='Domain and Temporal mean'
    elif ii == 'days':
        tend_to_plot   = exp_tend.where(exp_tend.time.dt.strftime('%Y-%m-%dT%H').isin(days),drop=True)
        # h_clim_to_plot = harm_clim_avg.where(harm_clim_avg.time.dt.strftime('%Y-%m-%d').isin(days),drop=True)
        h_clim_to_plot = harm_clim_avg.where(harm_clim_avg.time.isin(tend_to_plot.time,),drop=True)
        tend_to_plot = tend_to_plot.sel(time=h_clim_to_plot.time)
        
        ls_flux_toplot = ls_flux.where(ls_flux.time.dt.strftime('%Y-%m-%d').isin(days),drop=True)
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
#%%
####################################################################### 
#%% Make videos
if make_videos:

#%% FROM DALES
    ii = '2020-02-08T04:00'
    print('Creating images for video')
    var = 'lwp'
    plt.figure()
    cape.sel(time=ii)[var].plot(vmin=0,vmax=0.1,\
                            cmap=plt.cm.Blues_r)
    plt.suptitle(r'$I_{org}$ = ' +str(np.round(da_org['iorg'].sel(time=ii).values,3)))
    
    plt.figure(figsize=(6,9))
    profiles['vwt'].sel(time=ii).plot(y='z',lw=2)
    plt.axhline(cl_base.sel(time=ii),c='k',lw=0.5)
    plt.axhline(hc_ql.sel(time=ii),c='k',lw=0.5)
    plt.ylim([0,4500])
    plt.axvline(0,c='k')
    
    #%%
    plt.figure()
    ax =cape.sel(time=ii)[var].plot(vmin=0,vmax=1,\
                            cmap=plt.cm.Blues_r,x='xt',y='yt',\
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


