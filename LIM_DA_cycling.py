#
# Originator: Greg Hakim
#             December 2020
#
# perform DA experiments to assess assimilation impact on analyses and forecasts. 
#

"""
------------------------------
start: user defined parameters
------------------------------
"""

# set the observations for all experiments 
#obnet = 'atmos'
#obnet = 'ocean'
obnet = 'all'

# list of DA experiments ('uncoupled'=WCDA; 'coupled'=SCDA)
DA_expts = ['coupled','atmos','ocean','uncoupled']

# set the DA cycling time 
DA_dt = 1

# error type ('standard'=truncation error; 'linreg_' options remove bits that covary with state)
#error_type = 'standard'
error_type = 'linreg_multi'

# save the O-F statistics and forecasts to a file, with the label defined below
#save_stats = False
save_stats = True
file_label = error_type+'_paper'

# use netcdf preprocessed files
use_nc = True
#use_nc = False

# set the (uncorrelated) observation error here in a dictionary (dimensional, std)
ofac = 0.
oberr_dict = {}
oberr_dict['tas'] = ofac*1.0 # K
oberr_dict['tos'] = ofac*1.0 # K
oberr_dict['rlut'] = ofac*1.0 # K
oberr_dict['ua_850hPa'] = ofac*1.0 # K
oberr_dict['va_850hPa'] = ofac*1.0 # K

# choose obs either in EOF space or on a lat-lon grid
#ob_grid = 'eof'
ob_grid = 'latlon'

# specify the LIM file
lim_fname = './LIMd_CFSR_tas_tos_rlut_ua_850hPa_va_850hPa_1979_2010_ntrunc400_nmodes30.npy'

# specify the observation file, and the companion state file
ofile = './obs_global_equalarea_tas_rlut_ua_850hPa_va_850hPa_tos_CFSR_20040101_20101231.npy'
sfile = './state_CFSR_20040101_20101231.npy'

# specify files containing the definitions for H, R, and N
ofile_prep = './obs_operators_global_equalarea_tas_rlut_ua_850hPa_va_850hPa_tos_CFSR_19790101_20031231.npy'

# define atmos and ocean variables 
vars_atmos = ['tas','rlut','ua_850hPa','va_850hPa']
vars_ocean = ['tos']

"""
------------------------------
end: user defined parameters
------------------------------
"""
import warnings
warnings.filterwarnings("ignore")
import sys,os
import numpy as np
import xarray as xr
import datetime
import yaml
import LIM_utils_new as LIM_utils
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.util import add_cyclic_point

# seed an instance of the random number generator (this is an object than can be passed to functions)
rng = np.random.default_rng(2021)

# read the observations
obs_allvars = np.load(ofile,allow_pickle='TRUE').item()

# read the truncated state that corresponds to the observations
P_state = np.load(sfile,allow_pickle='TRUE').item() # dimensional!

# assign variables to observation list based on obnet choice
obvars_atmos = ['tas','rlut','ua_850hPa','va_850hPa']
obvars_ocean = ['tos']
obvars_all = obvars_atmos + obvars_ocean

# timestamp this dataset
when_created = datetime.datetime.now()

# load the LIM (from LIM_train.py)
print('loading LIM file ',lim_fname)
LIMd = np.load(lim_fname,allow_pickle='TRUE').item()
scale_fac = LIMd['scale_fac']
limvars = LIMd['limvars']
nEOF = LIMd['nEOF']
nmodes = LIMd['nmodes']
ivars = LIMd['ivars']
nvars = len(limvars)

# set up the resolvant
Gt = np.matmul(np.matmul(LIMd['vec'],np.diag(np.exp(LIMd['lam']*DA_dt))),LIMd['veci'])

# make variable-index look-up dictionary for atmos and ocean LIMs, like ivars does for all
ivars_atmos = {}
ivars_ocean = {}
asi = 0
osi = 0
k = -1
EOF_files = {}
for var in limvars:
    k +=1
    nnmodes = nmodes[k]
    EOF_files[var] = './CFSR_'+var+'_1979_2010_ntrunc400_mavg5.npy'

    if var in vars_ocean: # can generalize if needed
        oei = osi + nnmodes
        ivars_ocean[var] = list(range(osi,osi+nnmodes,1))
        osi = oei
    elif var in vars_atmos:
        aei = asi + nnmodes
        ivars_atmos[var] = list(range(asi,asi+nnmodes,1))
        asi = aei
    else:
        print('ERROR! this variable is not assigned to atmos or ocean')

# set the observation operators and observation error covariance
if ob_grid == 'eof':
    print('using identity obs in EOF space...')
    Htmp = np.zeros([ndof,ndof])
    Rdiag = np.ones(ndof)
    si = 0
    k = -1
    for var in limvars:
        k +=1
        nnmodes = nmodes[k]

        ei = si + nnmodes
        print('working on...',var,si,si+nnmodes)
        if var != 'tos': 
            if obnet != 'ocean':
                Htmp[si:si+nnmodes,si:si+nnmodes] = np.eye(nnmodes)
        else:
            if obnet != 'atmos':
                Htmp[si:si+nnmodes,si:si+nnmodes] = np.eye(nnmodes)

        si = ei
        
        # fill in the diagonal observation error variance
        oberr = (oberr_dict[var])**2
        Rdiag[si:si+nnmodes,si:si+nnmodes] = oberr*Rdiag[si:si+nnmodes,si:si+nnmodes]
        
    # remove the empty rows (null obs) (https://stackoverflow.com/questions/11188364/remove-zero-lines-2-d-numpy-array)
    H = Htmp[~np.all(Htmp == 0,axis=0)]

    # diagonal ob-error covariance matrix
    nobs = H.shape[0]
    R = np.diag(Rdiag)

elif ob_grid == 'latlon':

    # observing network is specified on a lat-lon grid
    print('using identity obs in lat-lon space from '+ofile)
    obs_allvars = np.load(ofile,allow_pickle='TRUE').item()
    
    print('using calibrated H, R, and N from '+ofile_prep)
    ob_prep = np.load(ofile_prep,allow_pickle='TRUE').item()
    
    # assign the observation variables
    print("observing network = obvars_%s" % obnet)
    exec("obvars = obvars_%s" % obnet)

    # load index info
    obinds = ob_prep['obinds']
    obinds_atmos = ob_prep['obinds_atmos']
    obinds_ocean = ob_prep['obinds_ocean']
    iatmos_obs = ob_prep['iatmos_obs']
    iocean_obs = ob_prep['iocean_obs']
    iatmos = ob_prep['iatmos']
    iocean = ob_prep['iocean']
    
    si = 0
    k = -1
    first_var = True
    Rdiag = []
    for var in limvars:
        k +=1
        nnmodes = nmodes[k]

        ei = si + nnmodes

        # note: obs_allvars has already been scaled
        # yh is the observation resolved in the truncated lat-lon space
        if var in obvars:
            nobs_var = obinds[var][1]-obinds[var][0]+1
            print('working on...',var,si,si+nnmodes,nobs_var)
            if first_var:
                allobs = obs_allvars[var]['obs_full']
                yh = obs_allvars[var]['obs_full'] - obs_allvars[var]['obs_trunc_error']
                first_var = False
            else:
                allobs = np.concatenate((allobs,obs_allvars[var]['obs_full']),axis=0)
                yh = np.concatenate((yh,obs_allvars[var]['obs_full']- obs_allvars[var]['obs_trunc_error']),axis=0)

            # fill in the (optional) diagonal observation error variance ("instrument" error)
            oberr = (oberr_dict[var])**2
            Rdiag = Rdiag + (oberr*np.ones(nobs_var)).tolist()        
            print('Rdiag:',nobs_var,len(Rdiag),allobs.shape,yh.shape)

        # move on to next limvar
        si = ei

    nobs = allobs.shape[0]
    print('nobs = ',nobs)
    n_obtimes = allobs.shape[1]
    print('number of times with observations=',n_obtimes)

else:
    raise Exception('no valid observing grid')

print(np.trace(ob_prep['R']))
print(np.trace(ob_prep['Rn']))
print(np.trace(ob_prep['Rn_one']))
print(np.trace(ob_prep['Rn'])/np.trace(ob_prep['R']))

# initialize calibrated LIM noise and training covariance
Noise = ob_prep['Noise']
C_0 = ob_prep['C_0']

# select H & R here; do so consistently as they are paired
if error_type == 'standard':
    H = ob_prep['H']
    R = ob_prep['R']
elif error_type == 'linreg_multi':
    H = ob_prep['newH']
    R = ob_prep['Rn']
else:
    raise('this type of ob error not supported!')
    
#
# atmosphere and ocean specific pieces of N, H, R, and G
#
Gatmos = Gt[iatmos,:][:,iatmos]
Gocean = Gt[iocean,:][:,iocean]
print('G:',Gt.shape,Gatmos.shape,Gocean.shape)
Natmos = Noise[iatmos,:][:,iatmos]
Nocean = Noise[iocean,:][:,iocean]
print('Noise:',Noise.shape,Natmos.shape,Nocean.shape)
# recall that H and R have different indices (observations, not state)
Hatmos = H[iatmos_obs,:][:,iatmos]
Hocean = H[iocean_obs,:][:,iocean]
print('H:',H.shape,Hatmos.shape,Hocean.shape)
Ratmos = R[iatmos_obs,:][:,iatmos_obs]
Rocean = R[iocean_obs,:][:,iocean_obs]
print('R:',R.shape,Ratmos.shape,Rocean.shape)
# filter observations & Rdiag
allobs_all = np.copy(allobs)
allobs_atmos = np.copy(allobs[iatmos_obs,:])
allobs_ocean = np.copy(allobs[iocean_obs,:])
Rdiag_all = np.array(Rdiag)
Rdiag_atmos = np.array(Rdiag)[iatmos_obs]
Rdiag_ocean = np.array(Rdiag)[iocean_obs]
# save the originals since I recycle the names below
Rall = np.copy(R)
Hall = np.copy(H)
Gall = np.copy(Gt)
Nall = np.copy(Noise)

"""
loop over DA experiments.

the first step involves a forecast from the initialized state. the k=0 index 
therefore applies to the *update* at the time of the first forecast, which is 
also the time of the first observations. the last time is the time of the last 
observations.
"""

# set up the resolvant
Gt = np.matmul(np.matmul(LIMd['vec'],np.diag(np.exp(LIMd['lam']*DA_dt))),LIMd['veci'])
Gatmos = Gt[iatmos,:][:,iatmos]
Gocean = Gt[iocean,:][:,iocean]

# set the number of DA cycles (in units of DA_dt)
nDA = n_obtimes
#nDA = 100

DA_results = {}
for DA_expt in DA_expts:
    print('DA case: ',DA_expt)

    # initialize according to the DA experiment
    if DA_expt == 'atmos':
        # atmos LIM and atmos obs only
        B = C_0[iatmos,:][:,iatmos]
        H = Hatmos
        R = np.copy(Ratmos) + np.diag(Rdiag_atmos)
        G = Gatmos
        Noise = Natmos
        allobs = allobs_atmos
        Rdiag = Rdiag_atmos
    elif DA_expt == 'ocean':
        # ocean LIM and ocean obs only
        B = C_0[iocean,:][:,iocean]
        #B = Bconv
        H = Hocean
        R = np.copy(Rocean) + np.diag(Rdiag_ocean)
        G = Gocean
        Noise = Nocean
        allobs = allobs_ocean
        Rdiag = Rdiag_ocean
    elif DA_expt == 'uncoupled':
        # coupled LIM and separate atmos and ocean DA (WCDA)
        B = C_0
        H = Hall
        G = Gall
        Noise = Nall
        allobs = allobs_all
        Rdiag = Rdiag_all
        R = np.copy(Rall) + np.diag(Rdiag_all)
    elif DA_expt == 'coupled':
        # coupled LIM and coupled DA (SCDA)
        B = C_0
        H = Hall
        G = Gall
        Noise = Nall
        allobs = allobs_all
        Rdiag = Rdiag_all
        R = np.copy(Rall) + np.diag(Rdiag_all)
    else:
        print('not a valid experiment')
    
    # number of observations
    nobs = H.shape[0]
    # number of state variables
    ndof = H.shape[1]
    
    print('done with setup',nobs,Rdiag.shape,nDA)

    # observations for all DA times
    tims = tuple(np.arange(0,nDA*DA_dt,DA_dt))
    Y = np.take(allobs,tims,axis=1)
    print('Y:',Y.shape)
    
    A = np.copy(B)
    xbm = np.zeros(ndof)
    xam = np.zeros(ndof)
    xbm_save = np.zeros([ndof,nDA])
    xam_save = np.zeros([ndof,nDA])
    
    dob = np.zeros([nobs,nDA])
    doa = np.zeros([nobs,nDA])
    dab = np.zeros([nobs,nDA])
    HBHT = np.zeros(nobs)
    HBHTpR = np.zeros(nobs)
    HAHT = np.zeros(nobs)
    
    for k in range(nDA):
        if np.mod(k,100)==0: print('DA cycle ',k,'= day',k*DA_dt)
        y = Y[:,k*DA_dt]
        # forecast
        xbm = np.matmul(G,xam) 
        B = np.real(np.matmul(np.matmul(G,A),G.T)) + Noise
        # analysis
        if DA_expt == 'uncoupled':
            # unpack the atmos and ocean covariance matrices
            Batmos = B[iatmos,:][:,iatmos]
            Bocean = B[iocean,:][:,iocean]
            xam_atmos,Aatmos = LIM_utils.kalman_update(xbm[iatmos],Batmos,Hatmos,y[iatmos_obs],R[iatmos_obs,:][:,iatmos_obs])
            xam_ocean,Aocean = LIM_utils.kalman_update(xbm[iocean],Bocean,Hocean,y[iocean_obs],R[iocean_obs,:][:,iocean_obs])
            # repack the analysis state vector
            xam = np.zeros(ndof)
            xam[iatmos] = xam_atmos
            xam[iocean] = xam_ocean
            # repack the atmos and ocean matrices into the full state matrix
            A = np.zeros([ndof,ndof])
            kk = -1
            for ki in iatmos:
                kk +=1
                A[iatmos,ki] = Aatmos[:,kk]
            kk = -1
            for ki in iocean:
                kk +=1
                A[iocean,ki] = Aocean[:,kk] 
        else:
            xam,A = LIM_utils.kalman_update(xbm,B,H,y,R)
        
        # save the state
        xbm_save[:,k] = xbm
        xam_save[:,k] = xam
        
        # desrosiers verification
        if DA_expt == 'uncoupled':
            dob[iatmos_obs,k] = y[iatmos_obs] - np.matmul(Hatmos,xbm[iatmos])
            dob[iocean_obs,k] = y[iocean_obs] - np.matmul(Hocean,xbm[iocean])
            doa[iatmos_obs,k] = y[iatmos_obs] - np.matmul(Hatmos,xam[iatmos])
            doa[iocean_obs,k] = y[iocean_obs] - np.matmul(Hocean,xam[iocean])
            dab[iatmos_obs,k] = np.matmul(Hatmos,xam[iatmos]) - np.matmul(Hatmos,xbm[iatmos])
            dab[iocean_obs,k] = np.matmul(Hocean,xam[iocean]) - np.matmul(Hocean,xbm[iocean])
        else:
            dob[:,k] = y - np.matmul(H,xbm)
            doa[:,k] = y - np.matmul(H,xam)
            dab[:,k] = np.matmul(H,xam) - np.matmul(H,xbm)

    # only check converged versions of A and B
    if DA_expt == 'uncoupled':
        # only save the diagonal elements of these
        HBHT[iatmos_obs] = np.diag(np.matmul(Hatmos,np.matmul(B[iatmos,:][:,iatmos],Hatmos.T)))
        HAHT[iatmos_obs] = np.diag(np.matmul(Hatmos,np.matmul(A[iatmos,:][:,iatmos],Hatmos.T)))
        HBHTpR[iatmos_obs] = HBHT[iatmos_obs] + np.diag(R[iatmos_obs,:][:,iatmos_obs])
        #
        HBHT[iocean_obs] = np.diag(np.matmul(Hocean,np.matmul(B[iocean,:][:,iocean],Hocean.T)))
        HAHT[iocean_obs] = np.diag(np.matmul(Hocean,np.matmul(A[iocean,:][:,iocean],Hocean.T)))
        HBHTpR[iocean_obs] = HBHT[iocean_obs] + np.diag(R[iocean_obs,:][:,iocean_obs])
        Rsave = np.zeros(nobs)
        Rsave[iatmos_obs] = np.diag(R[iatmos_obs,:][:,iatmos_obs])
        Rsave[iocean_obs] = np.diag(R[iocean_obs,:][:,iocean_obs])
    else:
        HBHT = np.diag(np.matmul(H,np.matmul(B,H.T)))
        HAHT = np.diag(np.matmul(H,np.matmul(A,H.T)))
        HBHTpR = HBHT + np.diag(R)
        Rsave = np.diag(R)
        
    # save the results (all times, so time series can be plotted)
    DA_results[DA_expt] = {'dob':dob,'doa':doa,'dab':dab}
    DA_results[DA_expt]['HBHT'] = HBHT
    DA_results[DA_expt]['HBHTpR'] = HBHTpR
    DA_results[DA_expt]['HAHT'] = HAHT
    DA_results[DA_expt]['Rsave'] = Rsave
    DA_results[DA_expt]['xbm'] = xbm_save
    DA_results[DA_expt]['A'] = A
    DA_results[DA_expt]['B'] = B
    DA_results[DA_expt]['xam'] = xam_save

# verify all experiments
DA_checks_all = {}

for DA_expt in DA_expts:
    DA_checks = {}
    dob = DA_results[DA_expt]['dob']
    doa = DA_results[DA_expt]['doa']
    dab = DA_results[DA_expt]['dab']
    HAHT = DA_results[DA_expt]['HAHT']
    HBHT = DA_results[DA_expt]['HBHT']
    HBHTpR = DA_results[DA_expt]['HBHTpR']
    Rsave = DA_results[DA_expt]['Rsave']

    # MSE
    dob_mse = np.mean(dob**2,axis=1)
    DA_checks['dob_mse'] = dob_mse
    dob_bias = np.mean(dob,1,keepdims=True)
    DA_checks['dob_bias'] = dob_bias

    # desrosiers checks
    dob = dob - np.mean(dob,1,keepdims=True)
    doa = doa - np.mean(doa,1,keepdims=True)
    dab = dab - np.mean(dab,1,keepdims=True)
    HBHT_check = np.matmul(dab,dob.T)/(nDA-1)
    HBHTpR_check = np.matmul(dob,dob.T)/(nDA-1)
    Rsave_check = np.matmul(doa,dob.T)/(nDA-1)
    HAHT_check = np.matmul(dab,doa.T)/(nDA-1)

    # print trace of checks
    print('HBHT:',np.sum(HBHT),np.trace(HBHT_check))
    print('HBHT + R:',np.sum(HBHTpR),np.trace(HBHTpR_check))
    print('R:',np.sum(R),np.trace(Rsave_check))
    print('HAHT:',np.sum(HAHT),np.trace(HAHT_check))
    print(np.min(np.diag(HBHT_check)))
    print(np.min(np.diag(HBHTpR_check)))
    print(np.min(np.diag(HAHT_check)))
    DA_checks['HBHT_check'] =HBHT_check
    DA_checks['HBHTpR_check'] = HBHTpR_check
    DA_checks['Rsave_check'] = Rsave_check
    DA_checks['HAHT_check'] = HAHT_check
    
    DA_checks_all[DA_expt] = DA_checks

# add data and save to a file
DA_checks_all['obinds'] = obinds
DA_checks_all['obinds_atmos'] = obinds_atmos
DA_checks_all['obinds_ocean'] = obinds_ocean
DA_checks_all['iatmos_obs'] = iatmos_obs
DA_checks_all['iocean_obs'] = iocean_obs
DA_checks_all['iocean'] = iocean
DA_checks_all['iatmos'] = iatmos
DA_checks_all['obvars_all'] = obvars_all
DA_checks_all['obvars_atmos'] = obvars_atmos
DA_checks_all['obvars_ocean'] = obvars_ocean
DA_checks_all['DA_results'] = DA_results
DA_checks_all['EOF_files'] = EOF_files
DA_checks_all['nmodes'] = nmodes
DA_checks_all['sfile'] = sfile
DA_checks_all['ivars'] = ivars
DA_checks_all['scale_fac'] = scale_fac
DA_checks_all['limvars'] = limvars
DA_checks_all['nDA'] = nDA
DA_checks_all['obnet'] = obnet
DA_checks_all['sfile'] = sfile
DA_checks_all['ofile'] = ofile
DA_checks_all['ofile_prep'] = ofile_prep
DA_checks_all['ob_grid'] = ob_grid

"""
begin forecasting experiments 
"""

# make the full state matrix---moved to here to use Ptruth in forecasting
print(P_state.keys())
first = True
for var in limvars:
    if first:
        Ptruth = P_state[var]
        first = False
    else:
        Ptruth = np.concatenate((Ptruth,P_state[var]),axis=0)
        
print(Ptruth.shape)

# generate ocean mask
var = 'tos'
var_tru = P_state[var]
var_var = np.matmul(var_tru,var_tru.T)/(var_tru.shape[1]-1)
infile = EOF_files[var]
if use_nc:
    npfile = LIM_utils.npydict_from_netcdf(infile[:-3]+'nc')
else:
    npfile = np.load(infile,allow_pickle='TRUE').item()

lat_2d = npfile['lat_2d']
lon_2d = npfile['lon_2d']
nlat = lat_2d.shape[0]
nlon = lat_2d.shape[1]
varinfile = npfile['varinfile']
u_field = npfile['u_field'][varinfile].values[:,:nnmodes]
var_latlon = np.einsum('ij,ji->i',np.matmul(u_field,var_var),u_field.T)
var_latlon[var_latlon<1e-12]=np.nan
# make masks that can be used with nan functions:
# ocean_mask: 1 over ocean, nan over land
ocean_mask = np.copy(var_latlon)
ocean_mask[ocean_mask>1e-12]=1.0
ocean_mask[ocean_mask<1e-12]=np.nan

nregions = 16 # number of geographical areas to average for regional means (hardwired)
LIM_forecasts = {}
lags = [0,1,2,3,4,5,6,7,8,9,10,15,20,25,30,40,50]
print('lags:',lags)
LIM_forecasts['lags'] = lags

for DA_expt in DA_expts:
    # set initial conditions
    if DA_expt == 'atmos':
        print('----atmos control----')
        x = DA_checks_all['DA_results']['atmos']['xam']
    elif DA_expt == 'ocean':
        print('----ocean control----')
        x = DA_checks_all['DA_results']['ocean']['xam']
    elif DA_expt == 'cold':
        print('----cold start coupled----')
        x = Ptruth
    else:
        print('----'+DA_expt+'----')
        x = DA_checks_all['DA_results'][DA_expt]['xam']

    # forecasts for all lags---work needed here
    ndof = x.shape[0]
    ntims = x.shape[1]
    nlags = len(lags)
    x_forecast = np.zeros([nlags,ndof,ntims])
    ilag = -1
    for lag in lags:
        ilag +=1
        Gt = np.matmul(np.matmul(LIMd['vec'],np.diag(np.exp(LIMd['lam']*lag))),LIMd['veci'])
        if DA_expt == 'atmos':
            Gt = Gt[iatmos,:][:,iatmos]
        elif DA_expt == 'ocean':
            print('using G ocean for lag ',lag,ilag)
            Gt = Gt[iocean,:][:,iocean]
        # forecast for one lag; save in temp dictionary
        LIMfd_tmp = LIM_utils.LIM_forecast(LIMd,x,[lag],Gin=Gt)
        x_forecast[ilag,:,:] = LIMfd_tmp['x_forecast'][0,:,:]

    # fill the dictionary with the forecasts for all lags
    LIMfd = LIMfd_tmp
    LIMfd['x_forecast'] = x_forecast
    
    var_gm_save = {}
    mse_gm_save = {}
    mse_rm_save = {}
    mse_gm_alltims_save = {}
    all_save_var = {}
    
    # verify each variable
    var_info = {}
    for var in limvars:
        if DA_expt == 'atmos' and var in vars_ocean:
            continue
        elif DA_expt == 'ocean' and var in vars_atmos:
            continue
        elif DA_expt == 'atmos':
            istate = ivars_atmos[var]
        elif DA_expt == 'ocean':
            istate = ivars_ocean[var]
        elif DA_expt == 'cold':
            istate = ivars[var]
        else:
            istate = ivars[var]

        print('verifying '+var)
        nnmodes = len(ivars[var])
        infile = EOF_files[var]
        if use_nc:
            npfile = LIM_utils.npydict_from_netcdf(infile[:-3]+'nc')
        else:
            npfile = np.load(infile,allow_pickle='TRUE').item()

        lat_2d = npfile['lat_2d']
        lon_2d = npfile['lon_2d']
        nlat = lat_2d.shape[0]
        nlon = lat_2d.shape[1]
        var_info[var] = {'lat_2d':lat_2d,'lon_2d':lon_2d,'nlat':nlat,'nlon':nlon}
        varinfile = npfile['varinfile']
        u_field = npfile['u_field'][varinfile].values[:,:nnmodes]
        # apply ocean mask to tos
        if var == 'tos':
            u_field = ocean_mask[:,np.newaxis]*u_field
            
        # cold start forecasts don't need scaling, but everything else does
        if DA_expt == 'cold':
            scaling = 1.0
        else:
            scaling = scale_fac[var]

        var_gm = np.zeros(len(lags))
        mse_gm = np.zeros(len(lags))
        mse_rm = np.zeros([nregions,len(lags)])
        all_save = np.zeros([len(lags),u_field.shape[0]])
        mse_gm_alltims = np.zeros([len(lags),nDA])
        mse_gm_alltims[:] = np.nan
        ilag = -1
        for lag in lags:
            ilag +=1
            print('lag=',lag)
            ftims = tuple(np.arange(lag,nDA*DA_dt,DA_dt))
            verif = np.take(P_state[var],ftims,axis=1)
            nverif = verif.shape[1]
            if lag == 0:
                ferror = scaling*x[istate,:] - verif
            else:
                ferror = scaling*LIMfd['x_forecast'][ilag,istate,:-lag] - verif

            print('ferror shape:',ferror.shape)
            
            # MSE over time, then averaged in space (small storage)
            mse_c = np.matmul(ferror,ferror.T)/(ferror.shape[1]-1)
            mse_latlon = np.einsum('ij,ji->i',np.matmul(u_field,mse_c),u_field.T)
            err_latlon = np.reshape(mse_latlon,[nlat,nlon])
            mse_gm_tmp,_,_ = LIM_utils.global_hemispheric_means(err_latlon,lat_2d[:,0])
            mse_gm[ilag] = mse_gm_tmp
            all_save[ilag,:] = mse_latlon
            rnames,rmean = LIM_utils.PAGES2K_regional_means(err_latlon,lat_2d,lon_2d)
            mse_rm[:,ilag] = rmean[:,0]
            # MSE global mean as a function of time (large storage)
            msevar = np.matmul(u_field,ferror)**2
            tmp = np.reshape(msevar,[nlat,nlon,msevar.shape[1]])
            tmp2 = np.moveaxis(tmp,2,0)
            tmp,_,_ = LIM_utils.global_hemispheric_means(tmp2,lat_2d[:,0])
            mse_gm_alltims[ilag,lag:] = tmp

            # variance
            ferror = ferror - np.mean(ferror,axis=1,keepdims=True)
            evar_c = np.matmul(ferror,ferror.T)/(ferror.shape[1]-1)
            var_latlon = np.einsum('ij,ji->i',np.matmul(u_field,evar_c),u_field.T)
            err_latlon = np.reshape(var_latlon,[nlat,nlon])
            var_gm_tmp,_,_ = LIM_utils.global_hemispheric_means(err_latlon,lat_2d[:,0])
            var_gm[ilag] = var_gm_tmp

        # archive results in dictionaries
        var_gm_save[var] = var_gm
        mse_gm_save[var] = mse_gm
        mse_rm_save[var] = mse_rm
        all_save_var[var] = all_save
        mse_gm_alltims_save[var] = mse_gm_alltims

    # package up for each variable
    LIM_forecasts[DA_expt] = {'mse_rm_save':mse_rm_save,'mse_gm_alltims_save':mse_gm_alltims_save,'var_gm_save':var_gm_save,'mse_gm_save':mse_gm_save,'all_save_var':all_save_var,'var_info':var_info}

# MSE by regional average; add names for the regions
LIM_forecasts['region_names'] = rnames

"""
end forecasting experiments 
"""

# add forecasting results to the existing dictionary and save the whole experiment
DA_checks_all['LIM_forecasts']= LIM_forecasts
if save_stats:
    ofname = 'DA_results_'+file_label+'.npy'
    print('saving DA results here: ',ofname)
    np.save(ofname,DA_checks_all)


