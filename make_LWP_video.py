#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:18:35 2023

@author: acmsavazzi
"""

#%% DMAKE VIDEO LWP 

import numpy as np
import xarray as xr
import os
from glob import glob
import matplotlib.pyplot as plt
from matplotlib import cm
import cartopy
import cartopy.crs as ccrs
import sys


#%%
cape_merg_files = []
print("Finding output files.")  
for path,subdir,files in os.walk(Output_dir):
    if path[-3:] in expnr: 
        for file in glob(os.path.join(path, 'merged_cape*')):
            cape_merg_files.append(file)

#%%
####     merged_cape.nc    ####
cape_merg_files.sort()
cape   = xr.open_mfdataset(cape_merg_files[1:], combine='by_coords',decode_times=False)
cape['time'] = srt_time + cape.time.astype("timedelta64[s]")
cape.time.attrs["units"] = "Local Time"
cape = cape.sel(time=slice(srt_plot,end_time))
#%% FROM DALES
ii = '2020-02-03T10:55'
print('Creating images for video')
var = 'lwp'
plt.figure()
cape.sel(time=ii)[var].plot(vmin=0,vmax=0.1,\
                        cmap=plt.cm.Blues_r)
plt.suptitle(r'$I_{org}$ = ' +str(np.round(da_org['iorg'].sel(time=ii).values,2)))