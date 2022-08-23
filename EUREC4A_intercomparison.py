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

#%%
save_dir = '/Users/acmsavazzi/Documents/WORK/PhD_Year2/EMD-MIP/'
import shapely.geometry as sgeom
from cartopy.geodesic import Geodesic


srm_in  = [12, -48, 22, -60]
srm_out = [0, -20, 30, -76]
srm_typ = [10,-35, 24, -61]

##
les_centre = [13.3,-57.5]
HALO_centre = [13.3,-57.7]
Dx = geopy.distance.distance(kilometers = 150)
Dy = geopy.distance.distance(kilometers = 75)

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


