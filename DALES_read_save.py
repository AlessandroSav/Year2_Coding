#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 16:16:49 2021

@author: acmsavazzi
"""

#%% HARMONIE_read_save.py



#%%                             Libraries
###############################################################################
from cmath import exp
from heapq import merge
import numpy as np
import xarray as xr
import netCDF4
import os
from glob import glob
import sys
from datetime import datetime, timedelta
from netCDF4 import Dataset

my_source_dir = os.path.abspath('{}/../../../My_source_codes')
sys.path.append(my_source_dir)
from My_thermo_fun import *

#%% initial 
exp_nr=['004']
## running on Local
base_dir = '/Users/acmsavazzi/Documents/WORK/PhD_Year1/DATA/DALES/DALES_atECMWF/outputs/'
## running on staffumbrella
# base_dir = 
## running on VrLab
# read_dir  = os.path.abspath('{}/../')+'/'
# write_dir = os.path.abspath('{}/../average_150km/')+'/'
## running on Mounted staffumbrella
# read_dir   = '/Users/acmsavazzi/Documents/Mount/'
# harmonie_dir   = base_dir+'Raw_Data/HARMONIE/BES_harm43h22tg3_fERA5_exp0/2020/'

dales_exp_dir   = base_dir  + '20200202_12_clim/'
#%% Open files
#
#
for path,subdir,files in sorted(os.walk(dales_exp_dir)):
    if path[-3:] in exp_nr: 
        file = os.path.join(path, 'merged_fielddump*.nc')):
        merged_wind = xr.open_mfdataset(file,decode_times=False)


cross_u = fielddump['u'].sel(yt=1200,zt=slice(0,5500),xm=slice(100,1200))
cross_v = fielddump['v'].sel(ym=1200,zt=slice(0,5500),xt=slice(100,1200))
cross_w = fielddump['w'].sel(yt=1200,zm=slice(0,5500),xt=slice(100,1200))

cross_u.to_netcdf(path+'/crossXZ_u.nc')
cross_v.to_netcdf(path+'/crossXZ_v.nc')
cross_w.to_netcdf(path+'/crossXZ_w.nc')

#%%         # Read in model level outputs

### Import raw Harmonie data
# This is too slow... need to find a better way. 
if harm_3d:
    print("Reading HARMONIE raw outputs.") 
    ### 3D fields

