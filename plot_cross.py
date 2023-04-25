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
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from matplotlib import cm
import string
from matplotlib.colors import DivergingNorm
import matplotlib.animation as animation
import os
from glob import glob
from datetime import datetime, timedelta
import sys
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

my_source_dir = os.path.abspath('{}/../../../My_source_codes')
sys.path.append(my_source_dir)
from My_thermo_fun import *

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
case       = '20200202_12_clim'
expnr      = '004'
snap_time ='2020-02-03T10:00'  # LT    

data_dir = '/Users/acmsavazzi/Documents/WORK/PhD_Year1/DATA/DALES/DALES_atECMWF/outputs/20200202_12_clim/'
save_dir        = '/Users/acmsavazzi/Documents/WORK/PhD_Year2/Manuscript/Figures/'


srt_time   = np.datetime64('2020-02-02T20')
# srt_time   = np.datetime64('2020-02-08T20')
# srt_time   = np.datetime64('2020-02-05T20')
# srt_time   = np.datetime64('2020-02-07T20')


get_lwp = True

#%%     OPTIONS FOR PLOTTING

# col=['b','r','g','orange','k']
col=['red','coral','maroon','blue','cornflowerblue','darkblue','green','lime','forestgreen','m']
height_lim = [0,3800*0.001]        # in km

#%%                             Import
###############################################################################
###### import horizontal cross
if get_lwp:
    crossxy = xr.open_mfdataset(data_dir+'Exp_'+expnr+'/merged_cape_lwp_rain.004.nc',combine='by_coords',decode_times=False)
    crossxy['time'] = srt_time + crossxy.time.astype("timedelta64[s]") + np.timedelta64(24, 'h')
    
    # make axes in km 
    crossxy['xt']= crossxy.xt * (150/1511)
    crossxy['yt']= crossxy.yt * (150/1511)


###### import vertical cross
crossxz = xr.open_mfdataset(data_dir+'Exp_'+expnr+'/crossxz*.nc',combine='by_coords',decode_times=False)
crossyz = xr.open_mfdataset(data_dir+'Exp_'+expnr+'/crossyz*.nc',combine='by_coords',decode_times=False)
## convert time from seconds to date
crossxz['time'] = srt_time + crossxz.time.astype("timedelta64[s]")
crossyz['time'] = srt_time + crossyz.time.astype("timedelta64[s]")

## from m to km 
crossxz['xt']=crossxz['xt']*0.001
crossxz['yt']=crossxz['yt']*0.001
crossxz['zt']=crossxz['zt']*0.001
crossxz['zm']=crossxz['zm']*0.001
crossyz['yt']=crossyz['yt']*0.001
crossyz['xt']=crossyz['xt']*0.001
crossyz['zt']=crossyz['zt']*0.001
crossyz['zm']=crossyz['zm']*0.001

## interpolate coordinates to single grid (not zm and zt)
crossxz['xm'] = crossxz['xt']
crossyz['ym'] = crossyz['yt']
crossxz = crossxz.assign_coords({"xm": ("xm", crossxz.xm.values)})
crossyz = crossyz.assign_coords({"ym": ("ym", crossyz.ym.values)})

##
crossxz = crossxz.rename({'yt':'y'})
#
crossxz = crossxz.swap_dims({"xt": "x"})
crossxz = crossxz.swap_dims({"xm": "x"})
crossxz['x'] = crossxz['xt'].values
crossxz = crossxz.drop(['xt','xm'])
#
crossxz = crossxz.swap_dims({"zt": "z"})
crossxz = crossxz.swap_dims({"zm": "z"})
crossxz['z'] = crossxz['zt'].values
crossxz = crossxz.drop(['zt','zm'])
# crossxz['z'] = crossxz['z']*1000
##
crossyz = crossyz.rename({'xt':'x'})
#
crossyz = crossyz.swap_dims({"yt": "y"})
crossyz = crossyz.swap_dims({"ym": "y"})
crossyz['y'] = crossyz['yt'].values
crossyz = crossyz.drop(['yt','ym'])
#
crossyz = crossyz.swap_dims({"zt": "z"})
crossyz = crossyz.swap_dims({"zm": "z"})
crossyz['z'] = crossyz['zt'].values
crossyz = crossyz.drop(['zt','zm'])
# crossyz['z'] =  crossyz['z']*1000

#%% import profiles
prof_files      = []
print("Finding output files.")  
for path,subdir,files in os.walk(data_dir):
    if path[-3:-2] == '0': 
        for file in glob(os.path.join(path, 'profiles*.nc')):
            prof_files.append(file)

profiles = xr.open_mfdataset(prof_files, combine='by_coords')


srt_time   = np.datetime64('2020-02-01T20')
# profiles = xr.open_mfdataset(data_dir+'Exp_'+expnr+'/profiles.'+expnr+'.nc', combine='by_coords')
profiles['time'] = srt_time + profiles.time.astype("timedelta64[s]")
profiles.time.attrs["units"] = "Local Time"
profiles['zm'] = profiles['zm'] *0.001
profiles['zt'] = profiles['zt'] *0.001
profiles = profiles.interp(zm=profiles.zt)
profiles = profiles.rename({'zt': 'z'})

# da_scales_prof = xr.open_dataset('/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/DALES/scale_sep_prof_allExp.nc')
# da_scales_prof['time'] = da_scales_prof.time  - np.timedelta64(4, 'h')
# da_scales_prof.time.attrs["units"] = "Local Time"
# da_scales_prof = da_scales_prof.sel(time=snap_time)

#%% recenter the flouwer in the y direction 

crossyz = xr.concat([crossyz,crossyz.sel(y=slice(0,30)).assign_coords(\
                      {"y": (crossyz.sel(y=slice(0,30)).y +(crossyz.y[-1]+crossyz.y[0]).values)})],dim='y')

crossyz = crossyz.sel(y=slice(30,None))
#%%
## find outline of clouds for 'snap_time' 
for section in ['xz','yz']:
    if section =='xz':
        mask = np.nan_to_num((crossxz['ql'].where(crossxz['ql']>0.0001)\
                          .sel(time=snap_time)).values)
    if section =='yz':
        mask = np.nan_to_num((crossyz['ql'].where(crossyz['ql']>0.0001)\
                          .sel(time=snap_time)).values)
    mask[mask > 0] = 3
    kernel = np.ones((4,4))
    C      = ndi.convolve(mask, kernel, mode='constant', cval=0)
    outer  = np.where( (C>=3) & (C<=12 ), 1, 0)
    # add variable cloud contour
    # works only for 1 time stamp 
    if section =='xz':
        crossxz['cloud'] = (('z', 'x'), outer)
    if section =='yz':
        crossyz['cloud'] = (('z', 'y'), outer)



### compute momentum fluxes 
### note that you are using the full vectors u, v. 
### You should be using the fluctuations u' , v' .
### 

crossxz['u'] = crossxz['u'] -6
crossyz['u'] = crossyz['u'] -6

crossxz['u_p'] = crossxz['u'] - profiles.sel(time=snap_time)['u']
crossxz['v_p'] = crossxz['v'] - profiles.sel(time=snap_time)['v']

crossyz['u_p'] = crossyz['u'] - profiles.sel(time=snap_time)['u']
crossyz['v_p'] = crossyz['v'] - profiles.sel(time=snap_time)['v']

###
# crossxz['uw'] = crossxz['u']*crossxz['w']
# crossxz['vw'] = crossxz['v']*crossxz['w']
# #
# crossyz['uw'] = crossyz['u']*crossyz['w']
# crossyz['vw'] = crossyz['v']*crossyz['w']
crossxz['uw'] = crossxz['u_p']*crossxz['w']
crossxz['vw'] = crossxz['v_p']*crossxz['w']
#
crossyz['uw'] = crossyz['u_p']*crossyz['w']
crossyz['vw'] = crossyz['v_p']*crossyz['w']




#%% ##############     PLOTTING       ##############
####################################################

# !!! plot wind anomaly instead 
# need to know the mean wind at each height
# where the mean is of the entire domain, not ust of the trimmed domain.

# coarse = 20
# ## cross-xz
# for var in ['u','w','uw']:
#     if var =='u':
#         v_min = -8
#         temp = crossxz
#     if var == 'w':
#         v_min = -0.7
#         temp = crossxz.coarsen(x=coarse, boundary='trim').mean()

#     if var == 'uw':
#         v_min = -3
#         temp = crossxz.coarsen(x=coarse, boundary='trim').mean()

#     plt.figure()
#     temp[var].sel(time=snap_time).plot(x='x',vmin=v_min)
#     xr.plot.contour(crossxz.coarsen(x=coarse, boundary='trim').mean()\
#                     .sel(time=snap_time)['w'],vmax=0.3,cmap='PiYG_r',alpha=0.5)
#     crossxz['cloud'].where(crossxz['cloud'] > 0).plot(cmap='binary',\
#                                                       add_colorbar=False,vmin=0,vmax=0.5)
#     plt.axvline(crossyz['x'].values,c='k',ls='--',lw=1)
#     plt.axhline(0.2,c='k',ls='--',lw=1)
#     plt.ylim(height_lim)
#     plt.title(snap_time,fontsize=20)

# ## cross-yz
# for var in ['v','w','vw']:
#     if var =='v':
#         v_min = -8
#         temp = crossyz
#     if var == 'w':
#         v_min = -1
#         temp = crossyz.coarsen(y=coarse, boundary='trim').mean()
#     if var == 'uw':
#         v_min = -3
#         temp = crossyz.coarsen(y=coarse, boundary='trim').mean()

#     plt.figure()
#     temp[var].sel(time=snap_time).plot(x='y',vmin=v_min)
#     crossyz['cloud'].where(crossyz['cloud'] > 0).plot(cmap='binary',\
#                                                       add_colorbar=False,vmin=0,vmax=0.5)
#     plt.axvline(crossxz['y'].values,c='k',ls='--',lw=1)
#     plt.axhline(0.2,c='k',ls='--',lw=1)
#     plt.ylim(height_lim)
#     plt.title(snap_time,fontsize=20)

#%%
coarse = 151    # number of grid point in the new grid box 
                # for resolution 100m -> 150 = 15 km
resol = coarse * 150/1511       

v_min = -5

v_min = 0
# # plot the vector field 


# temp = crossxz.sel(x=)

fig, axs = plt.subplots(2,1,figsize=(19,15))
for idx,var in enumerate(['u','v']):
    
    if var == 'u':
        temp = crossxz.coarsen(x=coarse, boundary='trim').mean()
        temp = temp.coarsen(z=1, boundary="trim").mean()
        temp = temp.interp(z=np.linspace(temp.z.min(),temp.z.max(), num=25))
        
        im_1a = crossxz.coarsen(x=1, boundary='trim').mean()['ql'].sel(time=snap_time)\
            .plot(x='x',vmin=v_min,alpha=0.5,ax=axs[idx],cmap='Blues')
        # temp.sel(time=snap_time).plot.quiver('x','z','u','w',vmin=-0.5)
        im_1b = temp.sel(time=snap_time).plot.streamplot('x','z','u_p','w',hue='uw',vmin=-0.001,\
                             density=[0.6, 0.6],\
                            linewidth=3.5,arrowsize=4.5,\
                        arrowstyle='fancy',cmap='PiYG_r',ax=axs[idx])
        crossxz['cloud'].where(crossxz['cloud'] > 0).plot(cmap='binary',\
                                add_colorbar=False,vmin=0,vmax=0.5,ax=axs[idx])
        

        axs[idx].set_xlim([35,100])
        axs[idx].set_ylim([0,3])
        axs[idx].axvline(crossyz['x'].values,c='k',ls='--',lw=1.5)
        axs[idx].axhline(0.2,c='k',ls='--',lw=1.5)
    
        axs[idx].set_title('Zonal wind at y= '+str(temp.y.round(1).values)+' km' ,\
              fontsize =40)
        cbar = im_1a.colorbar
        cbar.remove()
        cbar = im_1b.colorbar
        cbar.remove()
        axs[idx].set_xlabel(r'x ($km$)',fontsize=30)
    
    if var == 'v':
        temp = crossyz.coarsen(y=coarse, boundary='trim').mean()
        temp = temp.coarsen(z=1, boundary="trim").mean()
        temp = temp.interp(z=np.linspace(temp.z.min(),temp.z.max(), num=25))

        # temp = crossxz.sel(x=)
        im_2a = crossyz.coarsen(y=1, boundary='trim').mean()['v_p'].sel(time=snap_time)\
            .plot(x='y',vmin=v_min,alpha=0.5,ax=axs[idx])
        # temp.sel(time=snap_time).plot.quiver('x','z','u','w',vmin=-0.5)
        im_2b =temp.sel(time=snap_time).plot.streamplot('y','z','v_p','w',hue='vw' ,vmin=-0.001,\
                                density=[0.9, 0.8],\
                        linewidth=2.5,arrowsize=1.5,\
                    arrowstyle='fancy',cmap='PiYG_r',ax=axs[idx])
        crossyz['cloud'].where(crossyz['cloud'] > 0).plot(cmap='binary',\
                                    add_colorbar=False,vmin=0,vmax=0.5,ax=axs[idx])
    
        axs[idx].set_xlim([40,160])
        axs[idx].axvline(crossxz['y'].values,c='k',ls='--',lw=1.5)
        axs[idx].axhline(0.2,c='k',ls='--',lw=1.5)
        # plt.axvline(150, c='k',lw=1,ls='--')
        axs[idx].set_title('Meridional wind at x= '+str(temp.x.round(1).values)+' km',\
                  fontsize=40)
        cbar = im_2a.colorbar
        cbar.remove()
        cbar = im_2b.colorbar
        cbar.remove()
        axs[idx].set_xlabel(r'y ($km$)',fontsize=30)
    
    axs[idx].set_ylabel(r'z ($km$)',fontsize=30)
    axs[idx].tick_params(axis='both', which='major', labelsize=30)
    # axs[idx].set_xlabel(fontsize=26)
    
cbar_ax = fig.add_axes([1.05, 0.15, 0.05, 0.7])
cbar = fig.colorbar(im_1a, cax=cbar_ax)
cbar.ax.tick_params(labelsize=30)
cbar.set_label(r'Wind anomaly  ($ms^{-1}$)', fontsize=35)

# plt.suptitle('Coarsened to '+str(np.round(resol,1))+'km',fontsize=40)
for n, ax in enumerate(axs.flat):
    ax.text(0.9, 1.05, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=30)
plt.tight_layout()
# plt.savefig(save_dir+'Figure12_streamflow.pdf', bbox_inches="tight")  


#%% For Pier
coarse = 125    # number of grid point in the new grid box 
                # for resolution 100m -> 150 = 15 km
resol = coarse * 150/1511       
v_min = 0.001

fig, axs = plt.subplots(1,1,figsize=(19,8))
var='v'
temp = crossyz.coarsen(y=coarse, boundary='trim').mean()
temp = temp.coarsen(z=1, boundary="trim").mean()
temp = temp.interp(z=np.linspace(temp.z.min(),temp.z.max(), num=25))

# temp = crossxz.sel(x=)
im_2a = (1000*crossyz.coarsen(y=1, boundary='trim').mean()['ql']).sel(time=snap_time)\
    .plot(x='y',vmin=v_min,vmax= 0.99,alpha=0.7,cmap='Blues',add_colorbar=False)
# temp.sel(time=snap_time).plot.quiver('x','z','u','w',vmin=-0.5)
im_2b =temp.sel(time=snap_time).plot.streamplot('y','z','v_p','w',\
                        density=[0.6, 0.6],\
                linewidth=3.5,arrowsize=4.5,\
            arrowstyle='fancy',color='indianred')
crossyz['cloud'].where(crossyz['cloud'] > 0).plot(cmap='binary',\
                            add_colorbar=False,vmin=0,vmax=0.5)
plt.ylim([0,3])
plt.xlim([81,151])
plt.xlabel(r'y ($km$)',fontsize=30)
plt.ylabel(r'z ($km$)',fontsize=30)
plt.tick_params(axis='both', which='major', labelsize=30)
axs.set_title('')
cbar_ax = fig.add_axes([1.05, 0.15, 0.02, 0.75])
cbar = fig.colorbar(im_2a, cax=cbar_ax)
cbar.ax.tick_params(labelsize=30)
cbar.set_label(r'Liquid specific humidity'+"\n"+'($gkg^{-1}$)', fontsize=35)
plt.tight_layout()

# plt.savefig('/Users/acmsavazzi/Documents/WORK/PhD_Year3/Figures/'+'Flower_flow_res'+str(round(resol,1))+'.pdf', bbox_inches="tight")  

#%% plto lwp

if get_lwp:
    fig, axs = plt.subplots(1,1)
    im = crossxy['lwp'].sel(time=snap_time).plot(cmap=cm.Blues_r,\
                                 vmin=0,vmax=0.1,add_colorbar=False)
        
    axs.set_aspect('equal')
    axs.set_yticks([0, 50, 100, 150])
    axs.set_xticks([0, 50, 100, 150])
    axs.axvline(x=crossyz.x,ymin=(81/150),ymax=(151/150),c='indianred',ls='-',lw=2.5)
    axs.set_title('')
    plt.xlabel(r'x ($km$)')
    plt.ylabel(r'y ($km$)')
    plt.tight_layout()
    plt.savefig('/Users/acmsavazzi/Documents/WORK/PhD_Year3/Figures/'+'Flower_lwp.png',dpi=400, bbox_inches="tight")  

#%% Fluxes on cross

plt.figure(figsize=(8,11))
crossxz.sel(time=snap_time)['uw'].mean('x').plot(y='z',label='Mean of entire section')

# crossxz.sel(x=slice(40,110),time=snap_time)['uw'].mean('x').plot(y='z',label='Mean of  section')
profiles.sel(time=snap_time[:-6]).mean('time')['uwt'].plot(y='z',label=snap_time[5:-6])
profiles.sel(time=snap_time)['uwt'].plot(y='z',lw=2,label=snap_time[5:])
profiles.mean('time')['uwt'].plot(y='z',c='k',lw=3,label='All days')
# crossxz.sel(time=snap_time)['uw'].mean('x').plot(y='z',label='Mean of entire section')
plt.axvline(0,c='k',lw=0.5)
plt.legend()
plt.ylim([0,4.5])





plt.figure(figsize=(8,11))
crossyz.sel(y=slice(85,150),time=snap_time)['vw'].mean('y').plot(y='z',label='Mean of  section')

# crossyz.sel(time=snap_time)['vw'].mean('y').plot(y='z',label='Mean of entire section')
profiles.sel(time=snap_time[:-6]).mean('time')['vwt'].plot(y='z',label=snap_time[5:-6])
profiles.sel(time=snap_time)['vwt'].plot(y='z',lw=2,label=snap_time[5:])
profiles.mean('time')['vwt'].plot(y='z',c='k',lw=3,label='All days')
# crossxz.sel(time=snap_time)['uw'].mean('x').plot(y='z',label='Mean of entire section')
plt.axvline(0,c='k',lw=0.5)
plt.legend()
plt.ylim([0,4.5])


#%%
print('end.')


