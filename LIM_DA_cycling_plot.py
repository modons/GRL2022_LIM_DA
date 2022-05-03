#
# Originator: Greg Hakim
#             December 2020
#
# plot results from DA experiments in LIM_DA_cycling.py
#

savefigs = True
#savefigs = False

import warnings
import sys,os
import numpy as np
import yaml
import datetime
import LIM_utils_new as LIM_utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.util import add_cyclic_point

# load results for standard R and multivariate regression R
DA_checks_all_standard = np.load('DA_results_standard_paper.npy',allow_pickle='TRUE').item()
LIM_forecasts_standard = DA_checks_all_standard['LIM_forecasts']

DA_checks_all_multi = np.load('DA_results_linreg_multi_paper.npy',allow_pickle='TRUE').item()
LIM_forecasts_multireg = DA_checks_all_multi['LIM_forecasts']
lags = LIM_forecasts_multireg['lags']

obinds = DA_checks_all_multi['obinds']
obinds_atmos = DA_checks_all_multi['obinds_atmos']
obinds_ocean = DA_checks_all_multi['obinds_ocean']
print(obinds_atmos.keys())
print(LIM_forecasts_multireg.keys())
print(LIM_forecasts_multireg['coupled'].keys())

# two figures for the paper (run twice; tas and tos): violin plots for ob-based verification
#var = 'tas'; var_label = 'T$_{2m}$'
var = 'tos'; var_label = 'SST'

fig, ax = plt.subplots()

kk = -1
xlabels = []
for Rtype in [DA_checks_all_standard,DA_checks_all_multi]:
    kk +=1

    print(kk,' working on ',Rtype)
    DA_results = Rtype['DA_results']
    
    for DA_expt in DA_results.keys():
        print(DA_expt)
        if var == 'tos' and DA_expt == 'atmos':
            continue
        elif var != 'tos' and DA_expt == 'ocean':
            continue

        kk +=1
        if DA_expt == 'atmos':
            iobinds = list(range(obinds_atmos[var][0],obinds_atmos[var][1]+1,1))
            fc = 'green'
            DA_expt_name = DA_expt
        elif DA_expt == 'ocean':
            iobinds = list(range(obinds_ocean[var][0],obinds_ocean[var][1]+1,1))
            fc = 'green'
            DA_expt_name = DA_expt
        else:
            iobinds = list(range(obinds[var][0],obinds[var][1]+1,1))
            if DA_expt == 'coupled':
                fc = 'red'
                DA_expt_name = 'SCDA'
            else:
                fc = 'blue'
                DA_expt_name = 'WCDA'
                
        plotvar = DA_results[DA_expt]['HBHTpR'][iobinds] 
        plotvar_check = np.diag(Rtype[DA_expt]['HBHTpR_check'])[iobinds]
        cratio = plotvar_check/plotvar
        print(var+' ratio mean:',np.mean(cratio))
        print(var+' ratio median:',np.median(cratio))
        pos = [kk]
        vplot = ax.violinplot(cratio, pos,points=20, widths=0.7, showmeans=False,
                    showextrema=False, showmedians=True, bw_method=0.5)
        xlabels.append(DA_expt_name)
        # fix colors
        for pc in vplot['bodies']:
            pc.set_facecolor(fc)            
        vp = vplot['cmedians']
        vp.set_edgecolor('k')
        
ax.set_xticks([1,2,3,5,6,7])
ax.set_xticklabels(xlabels)
plt.title(var_label+'   HBH$^T$ + R calibration')
xlim = ax.get_xlim()
ax.plot(xlim,[1,1],'k-',linewidth=1)
if var == 'tos':
    ax.text(1.5,0.2,'control-R')
    ax.text(5.3,0.2,'regression-R')
    plt.ylim([0.1,2.5])
elif var == 'tas':
    ax.text(1.5,0.2,'control-R')
    ax.text(5.3,0.2,'regression-R')
    ax.text(1.5,0.2,'control-R')
    ax.text(5.3,0.2,'regression-R')
    plt.ylim([0.1,2.5])
    
plt.tight_layout()

if savefigs: plt.savefig('paper_HBHTpR_'+var+'.png',dpi=300)

# two-panel figure for paper: global-mean MSE in the control, and %change for WCDA & SCDA
fig, (ax1, ax2) = plt.subplots(2, 1)

# top panel: MSE of the control
var = 'tas'
ref = LIM_forecasts_multireg['atmos']['mse_gm_save'][var]
# this is for the legend only
ax1.plot(lags,ref,'g-',label='SST')
ax1.plot(lags,ref,'w-')
ax1.plot(lags,ref,'g--',label='T$_{2m}$')
ax12 = ax1.twinx()
var = 'tos'
ref = LIM_forecasts_multireg['ocean']['mse_gm_save'][var]
ax12.plot(lags,ref,'g-',label='SST')

# bottom panel: MSE %change
var = 'tos'
ref = LIM_forecasts_multireg['ocean']['mse_gm_save'][var]
expt = LIM_forecasts_multireg
ax2.plot(lags,100*(expt['coupled']['mse_gm_save'][var]-ref)/ref,'r-',linestyle='-',label='SST SCDA')
ax2.plot(lags,100*(expt['uncoupled']['mse_gm_save'][var]-ref)/ref,'b-',linestyle='-',label='SST WCDA')
var = 'tas'
ref = LIM_forecasts_multireg['atmos']['mse_gm_save'][var]
expt = LIM_forecasts_multireg
ax2.plot(lags,100*(expt['coupled']['mse_gm_save'][var]-ref)/ref,'r--',label='T$_{2m}$ SCDA')
ax2.plot(lags,100*(expt['uncoupled']['mse_gm_save'][var]-ref)/ref,'b--',label='T$_{2m}$ WCDA')

xlim = ax2.get_xlim()
ax2.plot([lags[0],lags[-1]],[0,0],'k-',linewidth=1)

for ax in fig.get_axes():
    ax.label_outer()
    
ax1.set_ylabel('MSE T$_{2m}$ (K$^2$)')
ax12.set_ylabel('MSE SST (K$^2$)')
ax2.set_ylabel('% change')
ax2.set_xlabel('lag (days)')
ax1.legend(bbox_to_anchor=(0.95, 0.55), loc='upper right')
ax2.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.tight_layout()
plt.title('global-mean MSE')

if savefigs: plt.savefig('paper_MSE_gm_new.png',dpi=300)

# load observations to plot locations on the maps below
ofile = DA_checks_all_multi['ofile']
print('reading observation information from: ',ofile)
obs_allvars = np.load(ofile,allow_pickle='TRUE').item()
for var in obs_allvars['obvars']:
    print(var,' nobs = ',obs_allvars[var]['nobs'])

# two-panel figure for paper: maps of MSE in the control, for tas and tos
ilag = 10
cmap = 'jet'
print('plotting lag=',lags[ilag])
proj = ccrs.Robinson(central_longitude=-90.)
fig,ax = plt.subplots(1,2,figsize=(8,8),subplot_kw=dict(projection=proj))
fig.tight_layout(w_pad=0.5,h_pad=-25.) # w_pad:horizontal; h_pad:vertical
var = 'tas'
lat_2d = LIM_forecasts_standard['coupled']['var_info'][var]['lat_2d']
lon_2d = LIM_forecasts_standard['coupled']['var_info'][var]['lon_2d']
nlon = LIM_forecasts_standard['coupled']['var_info'][var]['nlon']
nlat = LIM_forecasts_standard['coupled']['var_info'][var]['nlat']
pdat = LIM_forecasts_multireg['atmos']['all_save_var'][var][ilag,:]
pdat = np.reshape(pdat,[nlat,nlon])
pdat_wrap,lon_wrap = add_cyclic_point(pdat,coord=lon_2d[0,:], axis=1)
cs = ax[0].pcolormesh(lon_wrap,lat_2d[:,0],pdat_wrap,transform=ccrs.PlateCarree(),cmap=cmap,shading='nearest')
fig.colorbar(cs,ax=ax[0],orientation='horizontal',pad =0.01,shrink=0.5)
# add observation locations
obvec_lat = obs_allvars[var]['obvec_lat']
obvec_lon = obs_allvars[var]['obvec_lon']
ax[0].scatter(obvec_lon,obvec_lat,s=5,c='w',transform=ccrs.PlateCarree())
#
var = 'tos'
lat_2d = LIM_forecasts_standard['coupled']['var_info'][var]['lat_2d']
lon_2d = LIM_forecasts_standard['coupled']['var_info'][var]['lon_2d']
nlon = LIM_forecasts_standard['coupled']['var_info'][var]['nlon']
nlat = LIM_forecasts_standard['coupled']['var_info'][var]['nlat']
pdat = LIM_forecasts_multireg['ocean']['all_save_var'][var][ilag,:]
pdat = np.reshape(pdat,[nlat,nlon])
pdat_wrap,lon_wrap = add_cyclic_point(pdat,coord=lon_2d[0,:], axis=1)
cs = ax[1].pcolormesh(lon_wrap,lat_2d[:,0],pdat_wrap,transform=ccrs.PlateCarree(),cmap=cmap,shading='nearest')
fig.colorbar(cs,ax=ax[1],orientation='horizontal',pad =0.01,shrink=0.5,ticks=[0,1,2,3,4])
ax[1].add_feature(cartopy.feature.LAND, zorder=2, edgecolor='black',facecolor='black')
# add observation locations
obvec_lat = obs_allvars[var]['obvec_lat']
obvec_lon = obs_allvars[var]['obvec_lon']
ax[1].scatter(obvec_lon,obvec_lat,s=5,c='w',transform=ccrs.PlateCarree())

# make all of the grid visible in a subplot
ax[0].coastlines()
ax[1].coastlines()
ax[0].set_global()
ax[0].set_title('MSE T$_{2m}$ (K$^2$)')
ax[1].set_title('MSE SST (K$^2$)')
if savefigs: plt.savefig('paper_MSE_map_lag_'+str(lags[ilag])+'_control.png',bbox_inches='tight',dpi=300)

# four panel figure for paper: maps of %change in MSE from the control, for tas and tos
cmap = 'bwr'
fig,ax = plt.subplots(2,2,figsize=(8,8),subplot_kw=dict(projection=proj))
fig.tight_layout(w_pad=0.5,h_pad=-25.) # w_pad:horizontal; h_pad:vertical
var = 'tas'
maxv =  40.
minv = -maxv
lat_2d = LIM_forecasts_standard['coupled']['var_info'][var]['lat_2d']
lon_2d = LIM_forecasts_standard['coupled']['var_info'][var]['lon_2d']
nlon = LIM_forecasts_standard['coupled']['var_info'][var]['nlon']
nlat = LIM_forecasts_standard['coupled']['var_info'][var]['nlat']
ref = LIM_forecasts_multireg['atmos']['all_save_var'][var][ilag,:]
test = LIM_forecasts_multireg['uncoupled']['all_save_var'][var][ilag,:]
pdat = 100.*(test-ref)/ref
pdat = np.reshape(pdat,[nlat,nlon])
pdat_wrap,lon_wrap = add_cyclic_point(pdat,coord=lon_2d[0,:], axis=1)
cs = ax[0,0].pcolormesh(lon_wrap,lat_2d[:,0],pdat_wrap,transform=ccrs.PlateCarree(),cmap=cmap,shading='nearest',vmin=minv,vmax=maxv)
xl = ax[0,0].get_xbound()
ax[0,0].text(1.3*xl[0],0,'WCDA',fontweight='bold')
#
test = LIM_forecasts_multireg['coupled']['all_save_var'][var][ilag,:]
pdat = 100.*(test-ref)/ref
pdat = np.reshape(pdat,[nlat,nlon])
pdat_wrap,lon_wrap = add_cyclic_point(pdat,coord=lon_2d[0,:], axis=1)
cs = ax[1,0].pcolormesh(lon_wrap,lat_2d[:,0],pdat_wrap,transform=ccrs.PlateCarree(),cmap=cmap,shading='nearest',vmin=minv,vmax=maxv)
fig.colorbar(cs,ax=ax[1,0],orientation='horizontal',pad =0.01,shrink=0.5,ticks=np.arange(-40,50,20))
xl = ax[1,0].get_xbound()
ax[1,0].text(1.3*xl[0],0,'SCDA',fontweight='bold')
#
var = 'tos'
maxv =  40.
minv = -maxv
lat_2d = LIM_forecasts_standard['coupled']['var_info'][var]['lat_2d']
lon_2d = LIM_forecasts_standard['coupled']['var_info'][var]['lon_2d']
nlon = LIM_forecasts_standard['coupled']['var_info'][var]['nlon']
nlat = LIM_forecasts_standard['coupled']['var_info'][var]['nlat']
ref = LIM_forecasts_multireg['ocean']['all_save_var'][var][ilag,:]
test = LIM_forecasts_multireg['uncoupled']['all_save_var'][var][ilag,:]
pdat = 100.*(test-ref)/ref
pdat = np.reshape(pdat,[nlat,nlon])
pdat_wrap,lon_wrap = add_cyclic_point(pdat,coord=lon_2d[0,:], axis=1)
cs = ax[0,1].pcolormesh(lon_wrap,lat_2d[:,0],pdat_wrap,transform=ccrs.PlateCarree(),cmap=cmap,shading='nearest',vmin=minv,vmax=maxv)
#
test = LIM_forecasts_multireg['coupled']['all_save_var'][var][ilag,:]
pdat = 100.*(test-ref)/ref
pdat = np.reshape(pdat,[nlat,nlon])
pdat_wrap,lon_wrap = add_cyclic_point(pdat,coord=lon_2d[0,:], axis=1)
cs = ax[1,1].pcolormesh(lon_wrap,lat_2d[:,0],pdat_wrap,transform=ccrs.PlateCarree(),cmap=cmap,shading='nearest',vmin=minv,vmax=maxv)
#
ax[0,1].add_feature(cartopy.feature.LAND, zorder=2, edgecolor='black',facecolor='black')
ax[1,1].add_feature(cartopy.feature.LAND, zorder=2, edgecolor='black',facecolor='black')
ax[0,0].coastlines()
ax[0,1].coastlines()
ax[1,0].coastlines()
ax[1,1].coastlines()
#
fig.colorbar(cs,ax=ax[1,1],orientation='horizontal',pad =0.01,shrink=0.5,ticks=np.arange(-40,50,20))
fig.tight_layout()
ax[0,0].set_title('%change T$_{2m}$')
ax[0,1].set_title('%change SST')
if savefigs: plt.savefig('paper_MSE_map_lag_'+str(lags[ilag])+'_pchange.png',bbox_inches='tight',pad_inches=0.25,dpi=300)

