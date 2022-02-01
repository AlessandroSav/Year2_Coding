from netCDF4 import Dataset
import numpy as np
import os
from glob import glob

# Find files and number of processors
exp_dir='/Users/acmsavazzi/Documents/WORK/PhD_Year1/DATA/DALES/DALES/Experiments/20200209_10/Exp_009/'
files=[]
nprocx = 1
nprocy = 1

files = glob(os.path.join(exp_dir, 'fielddump*.nc'))
for file in files:
    nprocx = max(nprocx,int(file[-14:-11])+1)
    nprocy = max(nprocy,int(file[-10:-7])+1)
    iexpnr = file[-6:-3]

# Find Nx and Ny
nx = np.zeros(nprocx,dtype=int)
for i in range(nprocx):
    file = os.path.join(exp_dir,'fielddump.'+str(i).zfill(3)+'.000.'+iexpnr+'.nc')
    nc = Dataset(file,'r')
    nx[i]=np.size(nc.dimensions['xt'])
    Nz = np.size(nc.dimensions['zt'])
    Nt = np.size(nc.dimensions['time'])
    if i==0:
        try:
            nc.variables['sv002']
            lrain = True
        except KeyError:
            lrain = False
    nc.close()
ny = np.zeros(nprocy,dtype=int)
for i in range(nprocy):
    file = os.path.join(exp_dir,'fielddump.000.'+str(i).zfill(3)+'.'+iexpnr+'.nc')
    nc = Dataset(file,'r')
    ny[i]=np.size(nc.dimensions['yt'])
    nc.close()
Ny = np.sum(ny)
Nx = np.sum(nx)
print(Nt,Nz,Ny,Nx)

# Create netcdf
nc = Dataset(os.path.join(exp_dir,'merged_field.'+iexpnr+'.nc'), 'w', format='NETCDF4')
nc.createDimension('time', None)
nc.createDimension('z', Nz-1)
nc.createDimension('y', Ny-1)
nc.createDimension('x', Nx-1)
nc.createDimension('zt', Nz)
nc.createDimension('yt', Ny)
nc.createDimension('xt', Nx)
nc.createDimension('zm', Nz)
nc.createDimension('ym', Ny)
nc.createDimension('xm', Nx)
tnc = nc.createVariable('time', 'f4', ('time',))
tnc.long_name = 'Time'
tnc.units = 's'
tnc.FillValue = -999.
znc = nc.createVariable('z', 'f4', ('z',))
znc.long_name = 'Centre Vertical displacement'
znc.units = 'm'
ync = nc.createVariable('y', 'f4', ('y',))
ync.long_name = 'South-North displacement'
ync.units = 'm'
xnc = nc.createVariable('x', 'f4', ('x',))
xnc.long_name = 'West-East displacement'
xnc.units = 'm'
ztnc = nc.createVariable('zt', 'f4', ('zt',))
ztnc.long_name = 'Centre Vertical displacement'
ztnc.units = 'm'
ytnc = nc.createVariable('yt', 'f4', ('yt',))
ytnc.long_name = 'Centre South-North displacement'
ytnc.units = 'm'
xtnc = nc.createVariable('xt', 'f4', ('xt',))
xtnc.long_name = 'Centre West-East displacement'
xtnc.units = 'm'
zmnc = nc.createVariable('zm', 'f4', ('zm',))
zmnc.long_name = 'Half Vertical displacement'
zmnc.units = 'm'
ymnc = nc.createVariable('ym', 'f4', ('ym',))
ymnc.long_name = 'Half South-North displacement'
ymnc.units = 'm'
xmnc = nc.createVariable('xm', 'f4', ('xm',))
xmnc.long_name = 'Half West-East displacement'
xmnc.units = 'm'
umnc    = nc.createVariable('um', 'f4', ('time','zt','yt','xm'))
umnc.long_name = 'Half West-East velocity'
umnc.units = 'm/s'
umnc.FillValue = -999.
vmnc    = nc.createVariable('vm', 'f4', ('time','zt','ym','xt'))
vmnc.long_name = 'Half South-North velocity'
vmnc.units = 'm/s'
vmnc.FillValue = -999.
wmnc    = nc.createVariable('wm', 'f4', ('time','zm','yt','xt'))
wmnc.long_name = 'Half South-North velocity'
wmnc.units = 'm/s'
wmnc.FillValue = -999.
utnc    = nc.createVariable('ut', 'f4', ('time','z','y','x'))
utnc.long_name = 'Centre West-East velocity'
utnc.units = 'm/s'
utnc.FillValue = -999.
vtnc    = nc.createVariable('vt', 'f4', ('time','z','y','x'))
vtnc.long_name = 'Centre South-North velocity'
vtnc.units = 'm/s'
vtnc.FillValue = -999.
wtnc    = nc.createVariable('wt', 'f4', ('time','z','y','x'))
wtnc.long_name = 'Centre South-North velocity'
wtnc.units = 'm/s'
wtnc.FillValue = -999.
thlnc    = nc.createVariable('thl', 'f4', ('time','zt','yt','xt'))
thlnc.long_name = 'Liquid potential temperature'
thlnc.units = 'K'
thlnc.FillValue = -999.
thldiffnc    = nc.createVariable('thldiff', 'f4', ('time','zt','yt','xt'))
thldiffnc.long_name = 'Perturbation from slab average Liquid potential temperature'
thldiffnc.units = 'K'
thldiffnc.FillValue = -999.
qtnc    = nc.createVariable('qt', 'f4', ('time','zt','yt','xt'))
qtnc.long_name = 'Total water specific humidity'
qtnc.units = '1e-5kg/kg'
qtnc.FillValue = -999.
qlnc    = nc.createVariable('ql', 'f4', ('time','zt','yt','xt'))
qlnc.long_name = 'Liquid water specific humidity'
qlnc.units = '1e-5kg/kg'
qlnc.FillValue = -999.
if lrain:
    qrnc    = nc.createVariable('qr', 'f4', ('time','zt','yt','xt'))
    qrnc.long_name = 'Rain water specific humidity'
    qrnc.units = 'kg/kg'
    qrnc.FillValue = -999.
    nrnc    = nc.createVariable('nr', 'f4', ('time','zt','yt','xt'))
    nrnc.long_name = 'Rain droplet concentration'
    nrnc.units = 'kg/kg'
    nrnc.FillValue = -999.


# Read dimensions
xt = np.zeros(Nx)
xm = np.zeros(Nx)
yt = np.zeros(Ny)
ym = np.zeros(Ny)
xse = np.concatenate(([0,],np.cumsum(nx)))
yse = np.concatenate(([0,],np.cumsum(ny)))
for i in range(nprocx):
    file = os.path.join(exp_dir,'fielddump.'+str(i).zfill(3)+'.'+str(0).zfill(3)+'.'+iexpnr+'.nc')
    ncd = Dataset(file,'r')
    xt[xse[i]:xse[i+1]] = ncd.variables['xt'][:]
    xm[xse[i]:xse[i+1]] = ncd.variables['xm'][:]
    ncd.close()
for j in range(nprocy):
    file = os.path.join(exp_dir,'fielddump.'+str(0).zfill(3)+'.'+str(j).zfill(3)+'.'+iexpnr+'.nc')
    ncd = Dataset(file,'r')
    yt[yse[j]:yse[j+1]] = ncd.variables['yt'][:]
    ym[yse[j]:yse[j+1]] = ncd.variables['ym'][:]
    zt = ncd.variables['zt'][:]
    zm = ncd.variables['zm'][:]
    ncd.close()
x = xt[:-1]; y = yt[:-1]; z = zt[:-1]
znc[:]=z; ync[:]=y; xnc[:]=x
ztnc[:]=zt; ytnc[:]=yt; xtnc[:]=xt
zmnc[:]=zm; ymnc[:]=ym; xmnc[:]=xm
# Read variables
um = np.zeros([Nz,Ny,Nx])
vm = np.zeros([Nz,Ny,Nx])
wm = np.zeros([Nz,Ny,Nx])
thl = np.zeros([Nz,Ny,Nx])
thldiff = np.zeros([Nz,Ny,Nx])
qt  = np.zeros([Nz,Ny,Nx])
ql  = np.zeros([Nz,Ny,Nx])
if lrain:
    qr  = np.zeros([Nz,Ny,Nx])
    nr  = np.zeros([Nz,Ny,Nx])
for it in range(Nt):
    for i in range(nprocx):
        for j in range(nprocy):
            file = os.path.join(exp_dir,'fielddump.'+str(i).zfill(3)+'.'+str(j).zfill(3)+'.'+iexpnr+'.nc')  
            ncd = Dataset(file,'r')
            t  = ncd.variables['time'][it]
            um[:,yse[j]:yse[j+1],xse[i]:xse[i+1]] = ncd.variables['u'][it,:,:,:]
            vm[:,yse[j]:yse[j+1],xse[i]:xse[i+1]] = ncd.variables['v'][it,:,:,:]
            wm[:,yse[j]:yse[j+1],xse[i]:xse[i+1]] = ncd.variables['w'][it,:,:,:]
            thl[:,yse[j]:yse[j+1],xse[i]:xse[i+1]] = ncd.variables['thl'][it,:,:,:]
            thldiff = thl[:,:,:]-np.mean(thl,axis=(1,2))[:,None,None]
            qt[:,yse[j]:yse[j+1],xse[i]:xse[i+1]] = ncd.variables['qt'][it,:,:,:]
            ql[:,yse[j]:yse[j+1],xse[i]:xse[i+1]] = ncd.variables['ql'][it,:,:,:]
            if lrain:
                qr[:,yse[j]:yse[j+1],xse[i]:xse[i+1]] = ncd.variables['sv001'][it,:,:,:]
                nr[:,yse[j]:yse[j+1],xse[i]:xse[i+1]] = ncd.variables['sv002'][it,:,:,:]
            ncd.close()
    ut = (um[:-1,:-1,1:]+um[:-1,:-1,:-1])/2.
    vt = (vm[:-1,1:,:-1]+vm[:-1,:-1,:-1])/2.
    wt = (wm[1:,:-1,:-1]+wm[:-1,:-1,:-1])/2.
    tnc[it]=t
    utnc[it,:,:,:]=ut; umnc[it,:,:,:]=um
    vtnc[it,:,:,:]=vt; vmnc[it,:,:,:]=vm
    wtnc[it,:,:,:]=wt; wmnc[it,:,:,:]=wm
    thlnc[it,:,:,:]=thl
    thldiffnc[it,:,:,:]=thldiff
    qtnc[it,:,:,:] = qt
    qlnc[it,:,:,:] = ql
    if lrain:
        qrnc[it,:,:,:] = qr
        nrnc[it,:,:,:] = nr
    print(str((it+1)/Nt*100)+'% done')
nc.close()
