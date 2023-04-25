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
dales_exp_dir   = base_dir  + 'DALES_atECMWF/outputs/20200202_12_clim'
Output_dir      = base_dir  + 'DALES_atECMWF/outputs/20200202_12_clim/'

#SAVE DIRECTORY 
save_dir        = '/Users/acmsavazzi/Documents/WORK/PhD_Year2/Manuscript/Figures/'

### times to read and to plot 
srt_time   = np.datetime64('2020-02-02')
end_time   = np.datetime64('2020-02-11')
temp_hrs   = [np.datetime64('2020-02-02'),np.datetime64('2020-02-11')]
hours = srt_time,srt_time + [np.timedelta64(2, 'h'),np.timedelta64(48, 'h'),\
                             np.timedelta64(108, 'h'),np.timedelta64(144, 'h')]
    
make_videos       = False   

#%%     OPTIONS FOR PLOTTING
# col=['b','r','g','orange','k']
col=['red','coral','maroon','blue','cornflowerblue','darkblue','green','lime','forestgreen','m']
height_lim = [0,4000]        # in m

z_plot = 200 #m
f_size = 2500 # m

proj=ccrs.PlateCarree()
coast = cartopy.feature.NaturalEarthFeature(\
        category='physical', scale='50m', name='coastline',
        facecolor='none', edgecolor='r')

#%%                             Import
##############################################################################
#%% Import organisatio metrics
da_org      = xr.open_dataset('/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/DALES/df_org_allExp.nc')
## exclude the first hour
da_org = da_org.isel(time=slice(11,-1))
da_org_norm = (da_org - da_org.min()) / (da_org.max() - da_org.min())
#%% Import cross sections
cross_files = []
prof_files      = []
cross_scalar_files = []
print("Finding output files.")  
for path,subdir,files in os.walk(dales_exp_dir):
    if path[-3:] in expnr: 
        for file in glob(os.path.join(path, 'profiles*.nc')):
            prof_files.append(file)
        for file in glob(os.path.join(path, 'cross_field'+str(z_plot)+'*'+str(f_size)+'m*.nc')):
            cross_files.append(file)
        for file in glob(os.path.join(path, 'cross_field_scalar'+str(z_plot)+'*'+str(f_size)+'m*.nc')):
            cross_scalar_files.append(file)
cross_files.sort()
da_cross = xr.open_mfdataset(cross_files, combine='by_coords')
da_cross_scalar = xr.open_mfdataset(cross_scalar_files, combine='by_coords')

da_cross['x'] = da_cross_scalar['x']
da_cross['y'] = da_cross_scalar['y']

da_cross = xr.merge([da_cross,da_cross_scalar])
del da_cross_scalar

da_cross['x'] = da_cross['x']*0.001
da_cross['y'] = da_cross['y']*0.001

####     profiles.nc    ####    
print("Reading DALES profiles.")      
profiles = xr.open_mfdataset(prof_files, combine='by_coords')
profiles['time'] = srt_time + profiles.time.astype("timedelta64[s]") #- np.timedelta64(4, 'h')
#remove last time step because it is a midnight of the day after
profiles = profiles.sel(time=slice(srt_time,end_time))
# interpolate half level to full level
profiles = profiles.interp(zm=profiles.zt)
profiles = profiles.rename({'zt':'z'})
#profiles.time.attrs["units"] = "Local Time"
profiles.time.attrs["units"] = "UTC"


unresolved = profiles.interp(z=z_plot)
unresolved = unresolved.interp(time=da_cross.time)

#%% Import LWP


#%% New variables 
da_cross['u_p'] = da_cross['u_pf'] + da_cross['u_psf']
da_cross['v_p'] = da_cross['v_pf'] + da_cross['v_psf']
da_cross['w_p'] = da_cross['w_pf'] + da_cross['w_psf']
da_cross['wspd'] = np.sqrt(da_cross['u']**2 + da_cross['v']**2)
da_cross['wspd_p'] = da_cross['wspd'] - da_cross['wspd'].mean(dim=['x','y'])



#%% grouping

time_g={}



time_g['1'] = da_cross.where(da_org['iorg'] <= \
                            da_org['iorg'].quantile(0.25),drop=True).time
time_g['3'] = da_cross.where(da_org['iorg'] >= \
                            da_org['iorg'].quantile(0.75),drop=True).time

time_g['2'] = da_cross.where(np.logical_not(da_cross.time.\
                        isin(xr.concat((time_g['1'],time_g['3']),'time'))),drop=True).time

##
for group in ['1','2','3']:
    time_g[group] = time_g[group].where(time_g[group].isin(da_cross.time),drop=True)

#%%                         PLOTTING
###############################################################################
print("Plotting.") 
#%% ## FIGURE 4 ##
# snapshots 3x3 figure
### flower ### sugar ### gravel ###
#LWP  x    ###  x    ###   x    ###
#u    x    ###  x    ###   x    ###
#uw   x    ###  x    ###   x    ###
snap_time =['2020-02-03T14','2020-02-06T13','2020-02-08T13']
#snap_time =['2020-02-03T14','2020-02-03T17','2020-02-03T21']
var = 'v'
fig, axs = plt.subplots(2,3,figsize=(12,12))
for idx,ii in enumerate(snap_time):
    # LWP
    
    # U field
    im =(da_cross[var+'_pf']+da_cross[var+'_psf']).sel(time=ii).plot(ax=axs[0,idx]\
                               ,cmap=cm.PiYG_r,vmin=-2,vmax=2,add_colorbar=False)
    axs[0,idx].set_aspect('equal')
    # axs[0,idx].suptitle(ii)
    axs[0,idx].set_title(r''+ii+'\n '+var+' wind anomaly')
    axs[0,idx].xaxis.set_visible(False) 
    

    # flux 
    im_2 = da_cross[var+'w_p'].sel(time=ii).plot(ax=axs[1,idx],\
                               cmap=cm.PiYG_r,vmin=-0.4,vmax=0.4,add_colorbar=False)
    axs[1,idx].set_aspect('equal')
    axs[1,idx].set_title(var+' momentum flux')
    
axs[0,0].set_yticks([0, 500, 1000, 1500])
axs[1,0].set_yticks([0, 500, 1000, 1500])
axs[1,0].set_xticks([0, 500, 1000, 1500])
axs[1,1].set_xticks([0, 500, 1000, 1500])
axs[1,2].set_xticks([0, 500, 1000, 1500])
axs[0,1].yaxis.set_visible(False) 
axs[0,2].yaxis.set_visible(False) 
axs[1,1].yaxis.set_visible(False) 
axs[1,2].yaxis.set_visible(False) 

cbar_ax = fig.add_axes([0.95, 0.55, 0.01, 0.3])
fig.colorbar(im, cax=cbar_ax)

cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.3])
fig.colorbar(im_2, cax=cbar_ax)
#%%

snap_time ='2020-02-03T14'

da_cross['uw_pf'] = da_cross['u_pfw_pf']
da_cross['uw_psf'] = da_cross['uw_p'] - da_cross['u_pfw_pf']

da_cross['vw_pf'] = da_cross['v_pfw_pf']
da_cross['vw_psf'] = da_cross['vw_p'] - da_cross['v_pfw_pf']

da_cross['thlw_pf'] = da_cross['thl_pfw_pf']
da_cross['thlw_psf'] = da_cross['thlw_p'] - da_cross['thl_pfw_pf']

da_cross['qtw_pf'] = da_cross['qt_pfw_pf']
da_cross['qtw_psf'] = da_cross['qtw_p'] - da_cross['qt_pfw_pf']

#%%  Where are the regions of mesoscale w>0 ?
snap_time ='2020-02-03T14'

fig, axs = plt.subplots(3,3,figsize=(12,12))
# for idx,ii in enumerate(['v','w','vw']):
for idx,ii in enumerate(['thl','qt','qtw']):
# for idx,ii in enumerate(['qt','w','qtw']):

    if ii == 'u' or ii =='v':
        vbar=[-3,3]
        bar =True
    elif ii== 'qt':
        vbar=[-0.0012,0.0012]
        bar=True
    elif ii=='qtw':
        vbar=[-0.00018,0.00018]
        bar=True
    elif ii == 'thl':
        vbar=[-1.2,1.2]
        bar=True
    else:
        vbar=[-0.35,0.35]
        vbar=[-0.2,0.2]
        bar=False
        
    # Net field
    im =(da_cross[ii+'_pf']+da_cross[ii+'_psf']).sel(time=snap_time).plot(ax=axs[0,idx]\
                               ,cmap=cm.PiYG_r,vmin=vbar[0],vmax=vbar[1],\
                                   add_colorbar=False)
    axs[0,idx].set_aspect('equal')
    # axs[0,idx].suptitle(ii)
    axs[0,idx].set_title(r''+str(np.datetime64(snap_time) - np.timedelta64(4, 'h'))+'\n'+ii+' ${\prime}$ at '+str(z_plot)+'m')
    axs[0,idx].xaxis.set_visible(False) 

    # Up-filter
    im_1 =(da_cross[ii+'_pf']).sel(time=snap_time).plot(ax=axs[1,idx]\
                               ,cmap=cm.PiYG_r,vmin=vbar[0],vmax=vbar[1],add_colorbar=False)
    axs[1,idx].set_aspect('equal')
    # axs[0,idx].suptitle(ii)
    axs[1,idx].set_title(r'Up-filter')
    axs[1,idx].xaxis.set_visible(False) 
    

    # Sub-filter
    if ii[0:2] == 'qt':
            im_2 = da_cross[ii+'_psf'].sel(time=snap_time).plot(ax=axs[2,idx],\
                               cmap=cm.PiYG_r,vmin=vbar[0],vmax=vbar[1],add_colorbar=False)
    else:
        im_2 = da_cross[ii+'_psf'].sel(time=snap_time).plot(ax=axs[2,idx],\
                               cmap=cm.PiYG_r,vmin=-0.4,vmax=0.4,add_colorbar=False)
    axs[2,idx].set_aspect('equal')
    axs[2,idx].set_title('Sub-filter')
    axs[2,idx].set_xticks([0, 50, 100, 150])
    
axs[0,0].set_yticks([50, 100, 150])
axs[1,0].set_yticks([50, 100, 150])
axs[2,0].set_yticks([50, 100, 150])


axs[0,1].yaxis.set_visible(False) 
axs[0,2].yaxis.set_visible(False) 
axs[1,1].yaxis.set_visible(False) 
axs[1,2].yaxis.set_visible(False) 
axs[2,1].yaxis.set_visible(False) 
axs[2,2].yaxis.set_visible(False) 

cbar_ax = fig.add_axes([0.93, 0.42, 0.01, 0.42])
fig.colorbar(im, cax=cbar_ax)

cbar_ax = fig.add_axes([0.93, 0.14, 0.01, 0.2])
fig.colorbar(im_2, cax=cbar_ax)
#%% FOR LOUISE
snap_time ='2020-02-03T14'

fig, axs = plt.subplots(3,2,figsize=(8,10))
for idx,ii in enumerate(['uw','vw']):
    if ii == 'u' or ii =='v':
        vbar=[-3,3]
        bar =True
    else:
        vbar=[-0.4,0.4]
        # vbar=[-0.2,0.2]
        bar=False
        
    # Net field
    im =(da_cross[ii+'_pf']+da_cross[ii+'_psf']+ unresolved[ii+'s']).sel(time=snap_time).plot(ax=axs[0,idx]\
                               ,cmap=cm.PiYG_r,vmin=vbar[0],vmax=vbar[1],\
                                   add_colorbar=False)
    im =(da_cross[ii+'_psf']+ unresolved[ii+'s']).sel(time=snap_time).plot(ax=axs[1,idx]\
                               ,cmap=cm.PiYG_r,vmin=vbar[0],vmax=vbar[1],\
                                   add_colorbar=False)
    im =(da_cross[ii+'_pf']).sel(time=snap_time).plot(ax=axs[2,idx]\
                               ,cmap=cm.PiYG_r,vmin=vbar[0],vmax=vbar[1],\
                                   add_colorbar=False)
    axs[0,idx].set_aspect('equal')
    axs[1,idx].set_aspect('equal')
    axs[2,idx].set_aspect('equal')
    axs[0,idx].set_title(r''+ii+'$^{\prime}$',fontsize=22)
    axs[1,idx].set_title(r''+ii+' $^{\prime}_{SF}$',fontsize=22)
    axs[2,idx].set_title(r''+ii+' $^{\prime}_{UF}$',fontsize=22)

    axs[0,idx].xaxis.set_visible(False) 
    axs[1,idx].xaxis.set_visible(False) 

plt.suptitle(str(np.datetime64(snap_time) - np.timedelta64(4, 'h'))+' at '+str(z_plot)+'m')



axs[0,1].yaxis.set_visible(False)    
axs[1,1].yaxis.set_visible(False)  
axs[2,1].yaxis.set_visible(False)  
axs[0,0].set_yticks([500, 1000, 1500])
axs[1,0].set_yticks([500, 1000, 1500])
axs[2,0].set_yticks([500, 1000, 1500])
cbar_ax = fig.add_axes([0.98, 0.15, 0.01, 0.7])
fig.colorbar(im, cax=cbar_ax, label= r'$m^2 s^{-2}$')
plt.tight_layout()
# plt.savefig('/Users/acmsavazzi/Documents/WORK/PhD_Year2/Figures/Mom_fluxes_200m.pdf', bbox_inches="tight")  
# plt.savefig('/Users/acmsavazzi/Documents/WORK/PhD_Year2/Figures/Mom_fluxes_200m.png',dpi=300, bbox_inches="tight")  


#%% sample on updraft / downdraft 
updraft   = 1.1  #m/s
downdraft = -1.1 #m/s
snap_time =['2020-02-03T14','2020-02-06T13','2020-02-07T14']
var='u'

fig, axs = plt.subplots(2,len(snap_time),figsize=(6,6))
for idx,ii in enumerate(snap_time):

    # (da_cross[var+'_pf'] + da_cross[var+'_psf']).where((da_cross['w_pf'] + \
    #                   da_cross['w_psf'])>=updraft).sel(time=ii).plot(\
    #                  cmap=cm.PiYG_r,ax=axs[0,idx],add_colorbar=False,vmin=-5,vmax=+5)
                                                                     
    (da_cross[var+'_pf']).where((da_cross['w_pf'] + \
                      da_cross['w_psf'])>=updraft).sel(time=ii).plot(\
                     cmap=cm.PiYG_r,ax=axs[0,idx],add_colorbar=False,vmin=-2,vmax=+2)
                                                                     
    axs[0,idx].set_title(r''+ii+'\n Updraft')
    axs[0,idx].set_aspect('equal')

    # (da_cross['w_pf'] + da_cross['w_psf']).where((da_cross['w_pf'] + \
    #       da_cross['w_psf'])>=updraft).sel(time=ii).plot(ax=axs[1,idx])

    # im = (da_cross[var+'_pf'] + da_cross[var+'_psf']).where((da_cross['w_pf'] + \
    #                   da_cross['w_psf'])<=downdraft).sel(time=ii).plot(\
    #                cmap=cm.PiYG_r, ax=axs[1,idx],add_colorbar=False,vmin=-5,vmax=+5)
                                                                       
    im = (da_cross[var+'_pf']).where((da_cross['w_pf'] + \
                      da_cross['w_psf'])<=downdraft).sel(time=ii).plot(\
                   cmap=cm.PiYG_r, ax=axs[1,idx]\
                       ,add_colorbar=False,vmin=-2,vmax=+2)                                                                       
                                                                       
    axs[1,idx].set_title(r'Downdraft')
    axs[1,idx].set_aspect('equal')
    
    if idx >0:
        axs[0,idx].yaxis.set_visible(False) 
        axs[1,idx].yaxis.set_visible(False) 
    axs[0,idx].xaxis.set_visible(False) 
    axs[0,idx].set_yticks([ 500, 1000])
    axs[1,idx].set_yticks([ 500, 1000])
    axs[1,idx].set_xticks([ 500, 1000])

cbar_ax = fig.add_axes([0.95, 0.2, 0.01, 0.6])
fig.colorbar(im, cax=cbar_ax,label='up-filter '+var+' anomaly')

#%% sample on mesoscale updraft 
updraft   = 1  #m/s
downdraft = -1 #m/s

c_min =-1
c_max=1
var='v'
fig, axs = plt.subplots(2,len(snap_time),figsize=(6,6))
for idx,ii in enumerate(snap_time):

    (da_cross[var+'_psf'] ).where((da_cross['w_psf'] \
                      )>=updraft).sel(time=ii).plot(\
                      cmap=cm.PiYG_r,ax=axs[0,idx],add_colorbar=False,vmin=c_min,vmax=c_max)
    axs[0,idx].set_title(r''+ii+'\n small Updraft')
    axs[0,idx].set_aspect('equal')
    
    # (da_cross['wspd_p']).where((da_cross['w_pf'] \
    #                  )>=updraft).sel(time=ii).plot(\
    #                  cmap=cm.PiYG_r,ax=axs[0,idx],add_colorbar=False,vmin=-3,vmax=+3)
    # axs[0,idx].set_title(r''+ii+'\n wspd on meso Updraft')
    # axs[0,idx].set_aspect('equal')

    # (da_cross['w_pf'] + da_cross['w_psf']).where((da_cross['w_pf'] + da_cross['w_psf'])>=updraft).sel(time=ii).plot(ax=axs[1,idx])
    # axs[1,idx].set_title(r'Downdraft')
    # axs[1,idx].set_aspect('equal')

    im = (da_cross[var+'_psf'] ).where((da_cross['w_psf'] \
                      )<=downdraft).sel(time=ii).plot(\
                   cmap=cm.PiYG_r, ax=axs[1,idx],add_colorbar=False,vmin=c_min,vmax=c_max)
    axs[1,idx].set_title(r' small Downdraft')
    axs[1,idx].set_aspect('equal')
    
    if idx >0:
        axs[0,idx].yaxis.set_visible(False) 
        axs[1,idx].yaxis.set_visible(False) 
    axs[0,idx].xaxis.set_visible(False) 
    axs[0,idx].set_yticks([ 500, 1000])
    axs[1,idx].set_yticks([ 500, 1000])
    axs[1,idx].set_xticks([ 500, 1000])

cbar_ax = fig.add_axes([0.95, 0.2, 0.01, 0.6])
fig.colorbar(im, cax=cbar_ax,label='sub-filter '+var+' anomaly')

#%% wind speed fields
updraft   = 1  #m/s
downdraft = -0.5 #m/s
snap_time =['2020-02-09T13','2020-02-09T14','2020-02-09T16']

c_min =-2
c_max=2
var='wspd'
fig, axs = plt.subplots(2,len(snap_time),figsize=(6,6))
for idx,ii in enumerate(snap_time):

    (da_cross[var+'_p']).where((da_cross['w_psf'] \
                      )>=updraft).sel(time=ii).plot(\
                      cmap=cm.PiYG_r,ax=axs[0,idx],add_colorbar=False,vmin=-3,vmax=+3)
    axs[0,idx].set_title(r''+ii+'\n small Updraft')
    axs[0,idx].set_aspect('equal')

    im = (da_cross[var+'_p'] ).where((da_cross['w_psf'] \
                      )<=downdraft).sel(time=ii).plot(\
                   cmap=cm.PiYG_r, ax=axs[1,idx],add_colorbar=False,vmin=c_min,vmax=c_max)
    axs[1,idx].set_title(r' small Downdraft')
    axs[1,idx].set_aspect('equal')
    
    if idx >0:
        axs[0,idx].yaxis.set_visible(False) 
        axs[1,idx].yaxis.set_visible(False) 
    axs[0,idx].xaxis.set_visible(False) 
    axs[0,idx].set_yticks([ 500, 1000])
    axs[1,idx].set_yticks([ 500, 1000])
    axs[1,idx].set_xticks([ 500, 1000])
cbar_ax = fig.add_axes([0.95, 0.2, 0.01, 0.6])
fig.colorbar(im, cax=cbar_ax,label= var+' anomaly')

#%% Scatter plot
var= 'u'
ii = '2020-02-06T13'
plt.figure()
da_cross.sel(time=ii).plot.scatter('w_pf',var+'_pf',alpha=0.1,label='Up-filter',s=1)
da_cross.sel(time=ii).plot.scatter('w_psf',var+'_psf',alpha=0.1,label='Sub-filter',s=1)
# da_cross.sel(time=ii).plot.scatter('w_p',var+'_p',alpha=0.1,label='Net',s=1)
plt.axvline(0,c='k',lw=1)
plt.axhline(0,c='k',lw=1)
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.legend()
plt.xlabel('w_p (m/s)')
plt.ylabel(var+'_p (m/s)')

#%% Prob Density Functions 
ii = '2020-02-06T13'
for ii in time_g.keys():
    plt.figure()
    # xr.plot.hist(da_cross['w_p'].sel(time=time_g[ii]),bins=30,range=[-2,2],alpha=0.5,label='w_p')
    # xr.plot.hist(da_cross['u_p'].sel(time=time_g[ii]),bins=35,range=[-2,2],alpha=0.5,label='u_p')
    # # xr.plot.hist(da_cross['uw_p'].sel(time=time_g[ii]),bins=40,range=[-1.5,1.5],alpha=0.5,label='uw_p')
    # plt.legend()
    # plt.axvline(x=0,c='k',lw=1)
    plt.suptitle('Group '+ii)
    # plt.yscale('log')



# plt.hist2d(da_cross['w_p'].sel(time=ii).values,da_cross['u_p'].sel(time=ii).values)


    plt.hexbin(da_cross['w_p'].sel(time=time_g[ii]),da_cross['u_p'].sel(time=time_g[ii]),\
              cmap='Blues')
        
    plt.xlabel('w_p')
    plt.ylabel('u_p')
    plt.ylim([-3,3])
    plt.xlim([-2,2])
    plt.axvline(0,c='k',lw=1)
    plt.axhline(0,c='k',lw=1)

#%%
from scipy.stats import gaussian_kde
#%%
x=da_cross['w_p'].sel(time=ii).values.ravel()
y=da_cross['u_p'].sel(time=ii).values.ravel()


# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=2)
plt.show()

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


#%%

#%%
print('end.')


