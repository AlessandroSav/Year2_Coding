#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 17:06:33 2022

@author: acmsavazzi
"""
#%% EUREC4A_intercomparison
# Here some figures for the EUREC4A website are produced

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
import geopy
import geopy.distance
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

#%%
save_dir = '/Users/acmsavazzi/Documents/WORK/PhD_Year2/EUREC4A-MIP/'
import shapely.geometry as sgeom
from cartopy.geodesic import Geodesic


srm_in  = [10, -48, 20, -61]
srm_out = [0, -20, 30, -76]
srm_typ = [6,-35, 24, -63]

##
les_centre = [13.3,-57.7]
HALO_centre = [13.3,-57.7]
Dx = geopy.distance.distance(kilometers = 200)
Dy = geopy.distance.distance(kilometers = 125)

lat_max = Dy.destination(point=les_centre, bearing=0)
lat_min = Dy.destination(point=les_centre, bearing=180)

lon_max = Dx.destination(point=les_centre, bearing=270)
lon_min = Dx.destination(point=les_centre, bearing=90)




les     = [lat_min[0], lon_min[1], lat_max[0], lon_max[1]]

proj=ccrs.PlateCarree()
coast = cartopy.feature.NaturalEarthFeature(\
        category='physical', scale='50m', name='coastline',
        facecolor='none', edgecolor='k')
    
gd = Geodesic()
geoms = []
HALO_circle = gd.circle(lon=HALO_centre[1], lat=HALO_centre[0], radius=111000.)
geoms.append(sgeom.Polygon(HALO_circle))
les_domain = gd.circle(lon=HALO_centre[1], lat=HALO_centre[0], radius=111000.)
geoms.append(sgeom.Polygon(HALO_circle))


srt_time   = np.datetime64('2020-02-02T00')
end_time   = np.datetime64('2020-02-12T00')

#%% vertical spacing
class Grid:
    def __init__(self, kmax, dz0):
        self.kmax = kmax
        self.dz0  = dz0
    
        self.z = np.zeros(kmax)
        self.dz = np.zeros(kmax)
        self.zsize = None

    def plot(self):
        plt.figure()
        plt.title('zsize = {0:.1f} m'.format(self.zsize), loc='left')
        plt.plot(self.dz, self.z, '-x')
        plt.xlabel('dz (m)')
        plt.ylabel('z (m)')
class Grid_linear_stretched(Grid):
    def __init__(self, kmax, dz0, alpha):
        Grid.__init__(self, kmax, dz0)
    
        self.dz[:] = dz0 * (1 + alpha)**np.arange(kmax)
        zh         = np.zeros(kmax+1)
        zh[1:]     = np.cumsum(self.dz)
        self.z[:]  = 0.5 * (zh[1:] + zh[:-1])
        self.zsize = zh[-1]
grid = Grid_linear_stretched(kmax=150, dz0=20, alpha=0.012)
grid.plot()
#%% Relaxation profile 
def func(x, a, b, c, lev_max_change = 2400,end = 3600*6):
    y = b * (np.pi/2+np.arctan(a* np.pi/2*(1-x/lev_max_change)))
    y = end + y**c
    
    
    # plot
    plt.figure(figsize=(6,9))
    plt.axhline(lev_max_change,c='k',lw=1,ls=':')
    plt.axvline(end/3600/24,c='r',lw=1,ls=':',label='6 hours')
    plt.axvline(5,c='r',lw=1,ls='--',label='5 days')
    plt.plot(y/3600/24,x)
    # plt.xlim([10e3,10e6])
    plt.ylim([0,4000])
    plt.xscale('log')
    plt.legend()
    plt.xlabel('Days')
    plt.ylabel('Height (m)')
    plt.title('Relaxation time scale',fontsize=24)
    plt.savefig(save_dir+'relaxation_time_EUREC4A_intercomparison.pdf', bbox_inches="tight")
    return y

nudgefac = func(grid.z,a=2,b=3,c=6,lev_max_change=2400)
def logic(index,first_line=4):
    levels=150
    if ((index-3)%levels+3 == 0) or ((index-2)%levels+3 == 0) or (index<first_line):
       return True
    return False
#%% Read forcings

levels = grid.kmax

era5_forcing_dir ='/Users/acmsavazzi/Documents/WORK/PhD_Year2/EUREC4A-MIP/codes/'
era5_forcing =xr.open_mfdataset(era5_forcing_dir+'les_input_eurec4a.nc')


base_dir        = '/Users/acmsavazzi/Documents/WORK/PhD_Year1/DATA/DALES/'
harmonie_forcing_dir = base_dir  + 'DALES/Cases/EUREC4A/20200202_12_300km_clim/'

####     ls_flux.inp    ####
# hrs_inp = ((end_time - srt_time)/np.timedelta64(1, 'h')).astype(int)+1
with open(harmonie_forcing_dir+'ls_flux.inp.001') as f:
    hrs_inp = 0
    for line in f:
        if '(s)' in line: hrs_inp = 0
        else: hrs_inp += 1  # number of timesteps in input files 
        if "z (m)" in line: break
    hrs_inp -= 3
# first read the surface values
print("Reading input surface fluxes.")
colnames = ['time','wthl_s','wqt_s','th_s','qt_s','p_s']
ls_surf = pd.read_csv(harmonie_forcing_dir+'ls_flux.inp.001',header = 3,nrows=hrs_inp,\
                     names=colnames,index_col=False,delimiter = " ")
ls_surf.set_index(['time'], inplace=True)
ls_surf = ls_surf.to_xarray()
ls_surf['time'] = srt_time + ls_surf.time.astype("timedelta64[s]")
ls_surf.time.attrs["units"] = "UTC"
# ls_surf=ls_surf.sel(time=slice(srt_plot,end_time))

# second read the profiles
print("Reading input forcing profiles.")
colnames = ['z','u_g','v_g','w_ls','dqtdx','dqtdy','dqtdt','dthldt','dudt','dvdt']
skip = 0
with open(harmonie_forcing_dir+'ls_flux.inp.001') as f:
    for line in f:
        if line and line != '\n':
            skip += 1
        else:
            break
ls_flux    = pd.read_csv(harmonie_forcing_dir+'ls_flux.inp.001',\
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
ls_flux.time.attrs["units"] = "UTC"
ls_surf['T_s'] = calc_T(ls_surf['th_s'],ls_surf['p_s'])


ls_flux = ls_flux.resample(time='1H').mean()

#%%
plt.figure(figsize=(11,5))
era5_forcing.ps.plot(lw=2.5,label='ERA5')
ls_surf.p_s.plot(label='HARMONIE')
plt.legend()

plt.figure(figsize=(11,5))
era5_forcing.sst.plot(lw=2.5,label='ERA5 sst')
ls_surf.T_s.plot(label='HARMONIE')
era5_forcing.ts.plot(label='ERA5 skin')
plt.legend()
#%%
layer=[100,300]
var='v'

plt.figure(figsize=(11,5))
era5_forcing['dt'+var+'_advec'].sel(z=slice(layer[0],layer[1])).mean('z').plot(lw=2.5,label='ERA5')
ls_flux['d'+var+'dt'].sel(z=slice(layer[0],layer[1])).mean('z').plot(label='HARMONIE')
plt.legend()
plt.title(var+' tendency between '+str(layer)+' m',fontsize=20)


plt.figure(figsize=(11,5))
era5_forcing['dt'+var+'_advec'].sel(z=slice(layer[0],layer[1])).mean('z').rolling(time=8, center=True).mean().plot(lw=2.5,label='ERA5')
ls_flux['d'+var+'dt'].sel(z=slice(layer[0],layer[1])).mean('z').rolling(time=8, center=True).mean().plot(label='HARMONIE')
plt.legend()
plt.title(var+' tendency between '+str(layer)+' m',fontsize=20)

plt.figure(figsize=(11,5))
(era5_forcing['dt'+var+'_advec'].sel(z=slice(layer[0],layer[1])).mean('z')-ls_flux['d'+var+'dt'].sel(z=slice(layer[0],layer[1])).mean('z')).plot(lw=2.5,label='ERA5 - HARMONIE')
plt.legend()
plt.title(var+' tendency between '+str(layer)+' m',fontsize=20)


#%%
### LARGE ###  
plt.figure()
# ax =cape.sel(time=ii)[var].plot(vmin=0,vmax=1,\
#                     cmap=plt.cm.Blues_r,x='lon',y='lat',\
#                     subplot_kws=dict(projection=proj))
ax = plt.axes(projection=proj)
ax.add_feature(coast, lw=2, zorder=7)
ax.add_geometries(geoms, crs=proj,ls = ':', edgecolor='r', facecolor='None')
## Inner domain for SRM
ax.plot([srm_in[1],srm_in[3]],[srm_in[0],srm_in[0]],c='b',ls='-')
ax.plot([srm_in[1],srm_in[3]],[srm_in[2],srm_in[2]],c='b',ls='-')
ax.plot([srm_in[1],srm_in[1]],[srm_in[0],srm_in[2]],c='b',ls='-')
ax.plot([srm_in[3],srm_in[3]],[srm_in[0],srm_in[2]],c='b',ls='-')
## Max domain for SRM
ax.plot([srm_out[1],srm_out[3]],[srm_out[0],srm_out[0]],c='brown',ls='-')
ax.plot([srm_out[1],srm_out[3]],[srm_out[2],srm_out[2]],c='brown',ls='-')
ax.plot([srm_out[1],srm_out[1]],[srm_out[0],srm_out[2]],c='brown',ls='-')
ax.plot([srm_out[3],srm_out[3]],[srm_out[0],srm_out[2]],c='brown',ls='-')
## Typical domain for SRM
ax.plot([srm_typ[1],srm_typ[3]],[srm_typ[0],srm_typ[0]],c='b',ls='--')
ax.plot([srm_typ[1],srm_typ[3]],[srm_typ[2],srm_typ[2]],c='b',ls='--')
ax.plot([srm_typ[1],srm_typ[1]],[srm_typ[0],srm_typ[2]],c='b',ls='--')
ax.plot([srm_typ[3],srm_typ[3]],[srm_typ[0],srm_typ[2]],c='b',ls='--')

## LES domain
ax.plot([les[1],les[3]],[les[0],les[0]],c='g',ls='-')
ax.plot([les[1],les[3]],[les[2],les[2]],c='g',ls='-')
ax.plot([les[1],les[1]],[les[0],les[2]],c='g',ls='-')
ax.plot([les[3],les[3]],[les[0],les[2]],c='g',ls='-')
plt.xlim([-80,-15])
plt.ylim([-1,34])
gl = ax.gridlines(crs=proj, draw_labels=True)
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.savefig(save_dir+'map_EUREC4A_intercomparison.pdf')

### ZOOM ###
plt.figure()
# ax =cape.sel(time=ii)[var].plot(vmin=0,vmax=1,\
#                     cmap=plt.cm.Blues_r,x='lon',y='lat',\
#                     subplot_kws=dict(projection=proj))
ax = plt.axes(projection=proj)
ax.add_feature(coast, lw=2, zorder=7)
ax.add_geometries(geoms, crs=proj,ls = ':', edgecolor='r', facecolor='None')
## Inner domain for SRM
ax.plot([srm_in[1],srm_in[3]],[srm_in[0],srm_in[0]],c='b',ls='-')
ax.plot([srm_in[1],srm_in[3]],[srm_in[2],srm_in[2]],c='b',ls='-')
ax.plot([srm_in[1],srm_in[1]],[srm_in[0],srm_in[2]],c='b',ls='-')
ax.plot([srm_in[3],srm_in[3]],[srm_in[0],srm_in[2]],c='b',ls='-')
## Max domain for SRM
ax.plot([srm_out[1],srm_out[3]],[srm_out[0],srm_out[0]],c='brown',ls='-')
ax.plot([srm_out[1],srm_out[3]],[srm_out[2],srm_out[2]],c='brown',ls='-')
ax.plot([srm_out[1],srm_out[1]],[srm_out[0],srm_out[2]],c='brown',ls='-')
ax.plot([srm_out[3],srm_out[3]],[srm_out[0],srm_out[2]],c='brown',ls='-')
## Typical domain for SRM
ax.plot([srm_typ[1],srm_typ[3]],[srm_typ[0],srm_typ[0]],c='b',ls='--')
ax.plot([srm_typ[1],srm_typ[3]],[srm_typ[2],srm_typ[2]],c='b',ls='--')
ax.plot([srm_typ[1],srm_typ[1]],[srm_typ[0],srm_typ[2]],c='b',ls='--')
ax.plot([srm_typ[3],srm_typ[3]],[srm_typ[0],srm_typ[2]],c='b',ls='--')

## LES domain
ax.plot([les[1],les[3]],[les[0],les[0]],c='g',ls='-')
ax.plot([les[1],les[3]],[les[2],les[2]],c='g',ls='-')
ax.plot([les[1],les[1]],[les[0],les[2]],c='g',ls='-')
ax.plot([les[3],les[3]],[les[0],les[2]],c='g',ls='-')
plt.xlim([-64,-34])
plt.ylim([7,25])
# plt.xlim([-60,-55])
# plt.ylim([12,15])
gl = ax.gridlines(crs=proj, draw_labels=True)
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.savefig(save_dir+'map_EUREC4A_intercomparison_zoom.pdf')

### LEMs ###
plt.figure()
# ax =cape.sel(time=ii)[var].plot(vmin=0,vmax=1,\
#                     cmap=plt.cm.Blues_r,x='lon',y='lat',\
#                     subplot_kws=dict(projection=proj))
ax = plt.axes(projection=proj)
ax.add_feature(coast, lw=2, zorder=7)
ax.add_geometries(geoms, crs=proj,ls = ':', edgecolor='r', facecolor='None',lw=2)

## LES domain
ax.plot([les[1],les[3]],[les[0],les[0]],c='g',ls='-',lw=3)
ax.plot([les[1],les[3]],[les[2],les[2]],c='g',ls='-',lw=3)
ax.plot([les[1],les[1]],[les[0],les[2]],c='g',ls='-',lw=3)
ax.plot([les[3],les[3]],[les[0],les[2]],c='g',ls='-',lw=3)
plt.xlim([-60.1,-55.1])
plt.ylim([11.7,15])
gl = ax.gridlines(crs=proj, draw_labels=True)
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.savefig(save_dir+'map_EUREC4A_intercomparison_LEMs.pdf')

### SRMs ###
plt.figure()
# ax =cape.sel(time=ii)[var].plot(vmin=0,vmax=1,\
#                     cmap=plt.cm.Blues_r,x='lon',y='lat',\
#                     subplot_kws=dict(projection=proj))
ax = plt.axes(projection=proj)
ax.add_feature(coast, lw=2, zorder=7)
ax.add_geometries(geoms, crs=proj,ls = ':', edgecolor='r', facecolor='None',lw=1)
## Inner domain for SRM
ax.plot([srm_in[1],srm_in[3]],[srm_in[0],srm_in[0]],c='b',ls='-',lw=3)
ax.plot([srm_in[1],srm_in[3]],[srm_in[2],srm_in[2]],c='b',ls='-',lw=3)
ax.plot([srm_in[1],srm_in[1]],[srm_in[0],srm_in[2]],c='b',ls='-',lw=3)
ax.plot([srm_in[3],srm_in[3]],[srm_in[0],srm_in[2]],c='b',ls='-',lw=3)
## Max domain for SRM
ax.plot([srm_out[1],srm_out[3]],[srm_out[0],srm_out[0]],c='brown',ls='-',lw=3)
ax.plot([srm_out[1],srm_out[3]],[srm_out[2],srm_out[2]],c='brown',ls='-',lw=3)
ax.plot([srm_out[1],srm_out[1]],[srm_out[0],srm_out[2]],c='brown',ls='-',lw=3)
ax.plot([srm_out[3],srm_out[3]],[srm_out[0],srm_out[2]],c='brown',ls='-',lw=3)
## Typical domain for SRM
ax.plot([srm_typ[1],srm_typ[3]],[srm_typ[0],srm_typ[0]],c='b',ls='--')
ax.plot([srm_typ[1],srm_typ[3]],[srm_typ[2],srm_typ[2]],c='b',ls='--')
ax.plot([srm_typ[1],srm_typ[1]],[srm_typ[0],srm_typ[2]],c='b',ls='--')
ax.plot([srm_typ[3],srm_typ[3]],[srm_typ[0],srm_typ[2]],c='b',ls='--')

plt.xlim([-80,-15])
plt.ylim([-1,34])
gl = ax.gridlines(crs=proj, draw_labels=True)
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.savefig(save_dir+'map_EUREC4A_intercomparison_SRMs.pdf')



