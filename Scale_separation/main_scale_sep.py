#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 13:13:10 2022

@author: acmsavazzi
"""

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import gc
import sys
sys.path.insert(1, '/Users/acmsavazzi/Documents/WORK/PhD_Year2/Coding/Scale_separation/')
from functions import *
from dataloader import DataLoaderDALES
import argparse
import xarray as xr

#%%

mod = 'dales'
casenr = '001'
# lp = '/home/hp200321/data/botany-6-768/runs/Run_40'
# lp = '/Users/acmsavazzi/Documents/WORK/PhD_Year1/DATA/DALES/DALES/Experiments/20200209_10/Exp_006'
# lp = '/Users/acmsavazzi/Documents/WORK/PhD_Year1/DATA/DALES/DALES_atECMWF/outputs/20200202_12_clim/Exp_'+casenr
lp =  '/Users/acmsavazzi/Documents/Mount/Raw_Data/Les/Eurec4a/20200202_12_clim/Exp_'+casenr

save_data_dir   = '/Users/acmsavazzi/Documents/WORK/PhD_Year1/DATA/DALES/DALES_atECMWF/outputs/20200202_12_clim/Exp_'+casenr

itmin = 1
itmax = 24
di    = 8       # delta time for plots 
izmin = 0
izmax = 95
store = True
klps = [30,]         ## halfh the number of grids after coarsening 
#domain size from namotions
xsize      =  150000 # m
ysize      =  150000 # m
cu         = -6 # m/s
vu         = 0 # m/s


##### NOTATIONS
# _av = domain average 
# _p  = domain perturbation (prime)
# sf  = sub filter scale 
# f   = filter scale

# t   = total grid laevel 
# m   = middle of the grid 
#%%
if mod == 'dales':
    dl = DataLoaderDALES(lp,casenr=casenr)
    
time = dl.time
zt = dl.zt
zm = dl.zm
xt = dl.xt
xm = dl.xm
yt = dl.yt
ym = dl.ym
time1d = dl.time1d

# FIXME temporary hardcoding of dx/dy for data that does not have xf/yf as variables
# dx = np.diff(xt)[0]
# dy = np.diff(yt)[0] # Assumes uniform horizontal spacing
dx = xsize/xt.size  # in metres
dy = ysize/yt.size  # in metres
##########  All these dz to be checked !!!
# Vertical differences
dzt = np.zeros(zm.shape)
dzt[:-1] = np.diff(zm) # First value is difference top 1st cell and surface
dzt[-1] = dzt[-2]

dzm = np.zeros(zt.shape)
dzm[1:] = np.diff(zt) # First value is difference mid 1st cell and mid 1st cell below ground
dzm[0] = 2*zt[1]

plttime = np.arange(itmin, itmax, di)
plttime = np.array([1,3])
ztlim = zt[izmin:izmax]
zmlim = zm[izmin:izmax]

dztlim = dzt[izmin:izmax-1]
dzmlim = dzm[izmin:izmax-1]

##########
##########
# import averaged variables 
# profiles = xr.open_mfdataset(lp+'/profiles.'+casenr+'.nc')
# profiles['time'] = np.datetime64('2020-02-02') + profiles.time.astype("timedelta64[s]")
# samptend = xr.open_mfdataset(lp+'/samptend.'+casenr+'.nc' )
# ##########
##########

if (int(casenr) % 2) == 0:
    start_d = int(casenr)//2 +1
    start_h = 12
else:
    start_d = int(casenr)//2 +2
    start_h = 0
time = np.array(time,dtype='timedelta64[s]') + (np.datetime64('2020-02-'+str(start_d).zfill(2)+'T'+str(start_h).zfill(2)+':30'))
##################
############
###### SOMEHOW TIME IS WRONG and needs this 
time = time -np.timedelta64(12,'h') - np.timedelta64(30,'m')
############
##################



# initialie variables for scale separation
u_p_avtime          = np.zeros((plttime.size,izmax-izmin))
v_p_avtime          = np.zeros((plttime.size,izmax-izmin))
w_p_avtime          = np.zeros((plttime.size,izmax-izmin))
u_pf_avtime         = np.zeros((len(klps),plttime.size,izmax-izmin))
v_pf_avtime         = np.zeros((len(klps),plttime.size,izmax-izmin))
w_pf_avtime         = np.zeros((len(klps),plttime.size,izmax-izmin))
u_pfw_pf_avtime     = np.zeros((len(klps),plttime.size,izmax-izmin))
u_psfw_psf_avtime   = np.zeros((len(klps),plttime.size,izmax-izmin))
v_pfw_pf_avtime     = np.zeros((len(klps),plttime.size,izmax-izmin))
v_psfw_psf_avtime   = np.zeros((len(klps),plttime.size,izmax-izmin))
    
#%% Loop in time
for i in range(len(plttime)):
    print('Processing time step', i+1, '/', len(plttime))
    
    # 3D fields
    # qt = dl.load_qt(plttime[i], izmin, izmax)
    wm1 = dl.load_wm(plttime[i], izmin, izmax)
    wm2 = dl.load_wm(plttime[i],izmin+1,izmax+1)
    # thlp = dl.load_thl(plttime[i], izmin, izmax)
    # qlp = dl.load_ql(plttime[i], izmin, izmax)
    u = dl.load_u(plttime[i], izmin, izmax) + cu
    # v = dl.load_v(plttime[i], izmin, izmax) + vu
    w = (wm1 + wm2)*0.5 ### grid is stretched !!! # from w at midlevels caclculate w at full levels
    
    # averages and perturbations 
    u_av  = np.mean(u,axis=(1,2))
    # v_av  = np.mean(v,axis=(1,2))
    w_av  = 0
    u_p   = u - u_av[:,np.newaxis,np.newaxis]
    # v_p   = v - v_av[:,np.newaxis,np.newaxis]
    w_p   = w - w_av
    
    
    for k in range(len(klps)):
        print('Processing scale', k+1, '/', len(klps))
        klp=klps[k]
        #
        if klp > 0:
            f_scale = xsize/(klp*2)  # m
        elif klp == 0:
            f_scale = xsize
        else: print('Warning: Cutoff wavenumber for lw-pass filter smaller than 0.')
        
        # Mask for low-pass filtering
        circ_mask = np.zeros((xt.size,xt.size))
        rad = getRad(circ_mask)
        circ_mask[rad<=klp] = 1
    
        #filtered U
        u_pf  = lowPass(u_p, circ_mask)
        u_psf = u_p - u_pf
        #filtered V
        # v_pf = lowPass(v_p, circ_mask)
        # v_psf = v_p - v_pf
        #filtered W total level
        w_pf  = lowPass(w_p, circ_mask)
        w_psf = w_p - w_pf
        
        #### Fluxes at full level      
        # uw_p = (u_pf + u_psf) * (w_pf + w_psf)   
        
        # filtered and sub-filtered fluxes without the cross-terms
        u_pfw_pf   = u_pf  * w_pf 
        u_psfw_psf = u_psf * w_psf
        # v_pfw_pf   = v_pf  * w_pf 
        # v_psfw_psf = v_psf * w_psf
    
        # uu_p  = u_p * u_p
        # uv_p  = u_p * v_p
        uw_p  = u_p * w_p
        # vv_p  = v_p * v_p
        # vw_p  = v_p * w_p
        # # filtered fluxes
        # uu_pf = lowPass(uu_p, circ_mask)
        # uv_pf   = lowPass(uv_p, circ_mask)
        # uw_pf = lowPass(uw_p, circ_mask)
        # vv_pf   = lowPass(vv_p, circ_mask)
        # vw_pf   = lowPass(vw_p, circ_mask)
        # # subgrid fluxes
        # uu_psf = uu_p - uu_pf
        # uv_psf    = uv_p - uv_pf
        # uw_psf = uw_p - uw_pf
        # vv_psf    = vv_p - vv_pf
        # vw_psf    = vw_p - vw_pf    
        
        u_p_avtime[i,:] = np.mean(u_p,axis=(1,2))
        # v_p_avtime[i,:] = np.mean(v_p,axis=(1,2))
        w_p_avtime[i,:] = np.mean(w_p,axis=(1,2))
        
        u_pf_avtime[k,i,:] = np.mean(u_pf,axis=(1,2))
        # v_pf_avtime[k,i,:] = np.mean(v_pf,axis=(1,2))
        w_pf_avtime[k,i,:] = np.mean(w_pf,axis=(1,2))
        
        u_pfw_pf_avtime[k,i,:]   = np.mean(u_pfw_pf,axis=(1,2))
        u_psfw_psf_avtime[k,i,:] = np.mean(u_psfw_psf,axis=(1,2))
        # v_pfw_pf_avtime[k,i,:]   = np.mean(v_pfw_pf,axis=(1,2))
        # v_psfw_psf_avtime[k,i,:] = np.mean(v_psfw_psf,axis=(1,2))
        
        
        
        #### Momentum fluxes divergence 
        
        # # calculate at mid level and total level    
        # um   = (u[1:,:,:] + u[:-1,:,:])*0.5    # from w at midlevels caclculate w at full levels
        # vm   = (v[1:,:,:] + v[:-1,:,:])*0.5    # from w at midlevels caclculate w at full levels
        # wm   = wm1[1:,:,:]
        # um_x = (np.roll(u,-1,axis=2) + u) * 0.5    # np.roll is fine because of periodic boundaries
        # vm_y = (np.roll(v,-1,axis=1) + v) * 0.5   
        
        # ## Zonal component
        # ududx = um_x * (np.roll(u,-1,axis=2) - u)/dx
        # udvdx = um_x * (np.roll(v,-1,axis=2) - v)/dx
        # ## Meridional component
        # vdudy = vm_y * (np.roll(u,-1,axis=1) - u)/dy
        # vdvdy = vm_y * (np.roll(v,-1,axis=1) - v)/dy
        # ## Verticval component
        # wdudz  = wm * (u[1:,:,:] - u[:-1,:,:])/dztlim[:,np.newaxis,np.newaxis]
        # wdvdz  = wm * (v[1:,:,:] - v[:-1,:,:])/dztlim[:,np.newaxis,np.newaxis]
        # # filtered momentum fluxes
        # ududxf = lowPass(ududx, circ_mask)
        # udvdxf = lowPass(udvdx, circ_mask)
        # vdudyf = lowPass(vdudy, circ_mask)
        # vdvdyf = lowPass(vdvdy, circ_mask)
        # wdudzf = lowPass(wdudz, circ_mask)
        # wdvdzf = lowPass(wdvdz, circ_mask)
        # # subgrid momentum fluxes
        # ududxp = ududx - ududxf
        # udvdxp = udvdx - udvdxf
        # vdudyp = vdudy - vdudyf
        # vdvdyp = vdvdy - vdvdyf
        # wdudzp = wdudz - wdudzf
        # wdvdzp = wdvdz - wdvdzf
        
        # # !!! duwdz is different from wdudz !!!
        # # Alternatively differenciate the flux 
        # duudz  = (uu[1:,:,:] - uu[:-1,:,:])/dzmlim[:,np.newaxis,np.newaxis]
        # duvdz  = (uv[1:,:,:] - uv[:-1,:,:])/dzmlim[:,np.newaxis,np.newaxis]
        # duwdz  = (uw[1:,:,:] - uw[:-1,:,:])/dzmlim[:,np.newaxis,np.newaxis]
        # dvvdz  = (vv[1:,:,:] - vv[:-1,:,:])/dzmlim[:,np.newaxis,np.newaxis]
        # dvwdz  = (vw[1:,:,:] - vw[:-1,:,:])/dzmlim[:,np.newaxis,np.newaxis]
    
        ## Full advective tendency at half level 
        # total
        # utend_adv = (ududx[1:,:,:] + ududx[:-1,:,:])*0.5 + \
        #             (vdudy[1:,:,:] + vdudy[:-1,:,:])*0.5 + \
        #             wdudz
        # # filtered
        # utend_advf = (ududxf[1:,:,:] + ududxf[:-1,:,:])*0.5 + \
        #              (vdudyf[1:,:,:] + vdudyf[:-1,:,:])*0.5 + \
        #              wdudzf
        # # subgrid
        # utend_advp = (ududxp[1:,:,:] + ududxp[:-1,:,:])*0.5 + \
        #              (vdudyp[1:,:,:] + vdudyp[:-1,:,:])*0.5 + \
        #              wdudzp
        
    
    
        # up_av_time[i,:] = np.mean(u_psf,axis=(1,2))
        # vp_av_time[i,:] = np.mean(vp,axis=(1,2))
        # uf_av_time[i,:] = np.mean(u_pf,axis=(1,2))
        # vf_av_time[i,:] = np.mean(vf,axis=(1,2))
        # mean tendencies
        # utend_adv_av_time[i,:] = np.mean(utend_adv,axis=(1,2))
        # vtend_adv_av_time[i,:] = np.mean(vtend_adv,axis=(1,2))
        
        gc.collect()
        
        # Scale decompose & contributions FIXME need to make Galilean invariant
        # for winds
        # upf_l, upf_c, upf_r = scaleDecomposeFlux(wff , wfp, upf, upp, circ_mask)
        # vpf_l, vpf_c, vpf_r = scaleDecomposeFlux(wff , wfp, vpf, vpp, circ_mask)

if store:       
    # df = xr.DataArray(u_pf_avtime, coords=[('klp',klps),('time', time), ('z', ztlim)])
    print('Saving data...')
    np.save(lp+'/scale_time.npy',time[plttime])
    np.save(lp+'/scale_plttime.npy',plttime)
    np.save(lp+'/scale_zt.npy',ztlim)
    np.save(lp+'/scale_klps.npy',klps)
    
    np.save(lp+'/scale_u.npy',u_p_avtime)
    # np.save(lp+'/scale_v.npy',v_p_avtime)
    np.save(lp+'/scale_w.npy',w_p_avtime)
    
    np.save(lp+'/scale_u_pf.npy',u_pf_avtime)
    # np.save(lp+'/scale_v_pf.npy',v_pf_avtime)
    np.save(lp+'/scale_w_pf.npy',w_pf_avtime)
    np.save(lp+'/scale_u_pfw_pf.npy',u_pfw_pf_avtime)
    np.save(lp+'/scale_u_psfw_psf.npy',u_psfw_psf_avtime)
    # np.save(lp+'/scale_v_pfw_pf.npy',v_pfw_pf_avtime)
    # np.save(lp+'/scale_v_psfw_psf.npy',v_psfw_psf_avtime)
    
    # np.save(lp+'/up_wp.npy',uw_p)
    # np.save(lp+'/vp_wp.npy',vw_p)
    
        
 #%%       
z_plot = 8

fig,axs = plt.subplots(nrows=2,sharex=True,figsize=(6,8))
sc0 = axs[0].imshow(u[z_plot,:,:],
                    aspect='auto',cmap='RdYlBu_r',
                    vmin=np.min(u[z_plot,:,:]*0.75),
                    # vmax=-np.min(u[z_plot,:,:]*0.5),
                    vmax=-1.5,
                    extent=[0,150,0,150])

axs[0].invert_yaxis()
# axs[0].set_yticklabels(np.arange(0,150,(ysize/len(yt))/100))
axs[0].set_ylabel(r'y [km]')
axs[0].title.set_text('Zonal wind field. Height:'+str(int(ztlim[z_plot].values))+' m')
pos0 = axs[0].get_position()
cbax0 = fig.add_axes([0.95, pos0.ymin, 0.01, pos0.height])
cb0 = fig.colorbar(sc0, cax=cbax0)

cb0.ax.set_ylabel(r"u [$m^2 s^{-1}$]", rotation=90, labelpad=15,fontsize = 18)

sc1 = axs[1].imshow(uw_p[z_plot,:,:],
                   aspect='auto',cmap='RdYlBu_r',
                   vmin=np.min(uw_p[z_plot,:,:]*0.01),
                   vmax=-np.min(uw_p[z_plot,:,:]*0.01),
                   extent=[0,150,0,150])
axs[1].invert_yaxis()
axs[1].set_xlabel(r"x[km]")
axs[1].set_ylabel(r'y [km]')
axs[1].title.set_text('Momentum flux UW. Height: '+str(int(ztlim[z_plot].values))+' m')
pos1 = axs[1].get_position()
cbax1 = fig.add_axes([0.95, pos1.ymin, 0.01, pos1.height])
cb1 = fig.colorbar(sc1, cax=cbax1)
cb1.ax.set_ylabel(r"u'w' [$m^2 s^{-1} /h$]", rotation=90, labelpad=15,fontsize = 18)


save_dir = '/Users/acmsavazzi/Documents/WORK/PhD_Year2/Figures/'
plt.savefig(save_dir+'field_u_uw_'+str(int(ztlim[z_plot].values))+'m_'+np.datetime_as_string(time[plttime[i]], unit='m')+'.pdf', bbox_inches="tight")


#%% Domain average 
plt.figure()
plt.plot(np.mean(uw_p,axis=(1,2)),ztlim[:],label='total')
plt.plot(np.mean(u_pfw_pf ,axis=(1,2)),ztlim[:],label='FS')
plt.plot(np.mean(u_psfw_psf,axis=(1,2)),ztlim[:],label='SFS')
profiles.uwr.sel(zm=ztlim,method='nearest').sel(time=time[plttime[-1]],method='nearest').plot(y='zt',ls=':',label='profiles')
plt.plot(np.mean(u_pfw_pf + u_psfw_psf  ,axis=(1,2)),ztlim[:],c='k',ls= ':',alpha=0.5,label='sum')
plt.legend()
plt.xlim([0,0.004])
plt.title('Mean UW, scale: '+str(round(f_scale/1000, 2))+' km')

#%%
### advective term 
# plt.figure()
# # samptend.utendadvall.isel(time=11*2+1).plot(y='zt',label='samptend')
# plt.plot(np.mean(utend_advf,axis=(1,2)),zmlim[1:],label='FS')
# plt.plot(np.mean(utend_advp,axis=(1,2)),ztlim[1:],label='SFS')
# plt.legend()
# plt.xlabel('advect tenency [m/s2]')
# plt.title( 'Scale: '+str(f_scale)+' m')



