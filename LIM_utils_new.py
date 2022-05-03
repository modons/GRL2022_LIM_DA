"""
Originator: Greg Hakim
            December 2020

Support functions for strongly coupled data assimilaton with a LIM

"""
#
# read netcdf file, and return a dictionary as the code expected from npy
#
def make_vnames(limvars,sep='_'):
    vnames = ''
    for v in limvars:
        vnames = vnames+sep+v
    return vnames


def npydict_from_netcdf(filen):
    """
    read netcdf file that was created from a .npy and return a similar dictionary
    """
    from netCDF4 import Dataset,num2date,date2num,stringtochar
    import json
    import xarray as xr
    
    # take care to restore datasets from dataarrays, and strings from character byte arrays
    ds = Dataset(filen,'r', format='NETCDF4')
    P_tmp = ds.variables['P'][:]
    u_field_tmp = ds.variables['u_field'][:]
    dates_daily = ds.variables['time'][:]
    mavg = ds.variables['mavg'][:]
    config_tmp = ds.variables['config'][:].tostring().decode()
    config = json.loads(config_tmp)
    ntrunc = ds.variables['ntrunc'][:]
    dsource = ds.variables['dsource'][:].tostring().decode()
    lat_2d = ds.variables['lat_2d'][:]
    lon_2d = ds.variables['lon_2d'][:]
    nfourier = ds.variables['nfourier'][:]
    svals_tmp = ds.variables['svals'][:]
    varinfile = ds.variables['varinfile'][:].tostring().decode()
    fvar = ds.variables['fvar'][:]
    when_created = ds.variables['when_created'][:].tostring().decode()

    # fourier coefficients are complex, stored in a float array format. restore complex array here
    tmp = ds.variables['fourier_data'][:]
    fourier_data = tmp[:,:,:,0] +1j*tmp[:,:,:,1]

    # make the xarray datasets
    P =  xr.DataArray(data=P_tmp)
    u_field =  xr.DataArray(data=u_field_tmp).to_dataset(name=varinfile)
    svals =  xr.DataArray(data=svals_tmp).to_dataset(name=varinfile)

    npfile = {}
    npfile['P'] = P
    npfile['u_field'] = u_field
    npfile['dates_daily'] = dates_daily
    npfile['ntrunc'] = ntrunc
    npfile['dsource'] = dsource
    npfile['lat_2d'] = lat_2d
    npfile['lon_2d'] = lon_2d
    npfile['nfourier'] = nfourier
    npfile['fourier_data'] = fourier_data
    npfile['svals'] = svals
    npfile['mavg'] = mavg
    npfile['varinfile'] = varinfile
    npfile['config'] = config
    npfile['fvar'] = fvar
    npfile['when_created'] = when_created

    return npfile

def kalman_update(xbm,B,H,y,R):
    """
    update of mean and covariance using the Kalman filter update equation
    Input:
    * xbm: prior mean
    * B: prior covariance
    * H: observation operator
    * y: vector of observations
    * R: observation error covariance matrix
    Output:
    * xam: posterior mean
    * A: posterior covariance
    """

    import numpy as np
    
    # ob estimates
    ye = np.matmul(H,xbm)
    # innovation covariance
    IC = np.matmul(H,np.matmul(B,H.T)) + R
    # gain
    K = np.matmul(np.matmul(B,H.T),np.linalg.pinv(IC))
    # innovation
    I = y - ye
    # update mean and covariance
    xam = xbm + np.matmul(K,I)
    A = B - np.matmul(np.matmul(K,H),B)
    
    return xam,A


def LIM_forecast(LIMd,x,lags,E=None,truth=None,Gin=None):
    """
    deterministic forecasting experiments for states in x and time lags in lags.

    Inputs:
    * LIMd: a dictionary with LIM attributes
    * x: a state-time matrix for initial conditions and verification ~(ndof,ntims)
    * lags: list of time lags for deterministic forecasts
    * E: the linear map from the coordinates of the LIM to physical (lat,lon) coordinates ~(nx*ny,ndof)
    
    Outputs (in a dictionary):
    * error variance as a function of space and forecast lead time ~(ndof,ntims)
    * the forecast states ~(nlags,ndof,ntims)

    23 December 2021: generalized to allow forecasts for a single initial time (x is a vector)
    """
    import numpy as np
    
    ndof = x.shape[0]
    if isinstance(lags,list):
        nlags = len(lags)
    else:
        nlags = 1
        
    if len(x.shape) == 2:
        ntims = x.shape[1]
        x_predict_save = np.zeros([nlags,ndof,ntims])
    else:
        ntims = 0
        x_predict_save = np.zeros([nlags,ndof])

    if E != None:
        nx = E.shape[0]    
        error = np.zeros([nx,nlags])        
    
    for k,t in enumerate(lags):
        print('t=',t,k)
        # make the propagator for this lead time
        if Gin is None:
            Gt = np.matmul(np.matmul(LIMd['vec'],np.diag(np.exp(LIMd['lam']*t))),LIMd['veci'])
        else:
            Gt = Gin
            
        # forecast
        if t == 0 or ntims == 0:
            # need to handle this time separately, or the matrix dimension is off
            x_predict = np.matmul(Gt,x)
            if ntims != 0:
                x_predict_save[k,:,:] = x_predict
            else:
                x_predict_save[k,:] = x_predict                
        elif t>0 and ntims >0:
            x_predict = np.matmul(Gt,x[:,:-t])
            if ntims != 0:
                x_predict_save[k,:,:-t] = x_predict

        if E != None:
            # physical-space fields for forecast and truth for this forecast lead time ~(ndof,ntims)
            X_predict = np.real(np.matmul(E,x_predict))
            X_truth = truth[:,t:]

            # error variance as a function of space and forecast lead time ~(ndof,ntims)
            error[:,k] = np.var(X_predict - X_truth,axis=1,ddof=1)
    
    # return the LIM forecast error dictionary
    LIMfd = {}
    if E != None:
        LIMfd['error'] = error
    LIMfd['x_forecast'] = x_predict_save
        
    return LIMfd


def global_hemispheric_means(field,lat):

    """
     compute global and hemispheric mean valuee for all times in the input (i.e. field) array
     input:  field[ntime,nlat,nlon] or field[nlat,nlon]
             lat[nlat,nlon] in degrees

     output: gm : global mean of "field"
            nhm : northern hemispheric mean of "field"
            shm : southern hemispheric mean of "field"
    """

    import numpy as np
    
    # Originator: Greg Hakim
    #             University of Washington
    #             August 2015
    #
    # Modifications:
    #           - Modified to handle presence of missing values (nan) in arrays
    #             in calculation of spatial averages [ R. Tardif, November 2015 ]
    #           - Enhanced flexibility in the handling of missing values
    #             [ R. Tardif, Aug. 2017 ]

    # set number of times, lats, lons; array indices for lat and lon    
    if len(np.shape(field)) == 3: # time is a dimension
        ntime,nlat,nlon = np.shape(field)
        #print('time sent...',ntime,nlat,nlon)
        lati = 1
        loni = 2
    else: # only spatial dims
        ntime = 1
        nlat,nlon = np.shape(field)
        field = field[None,:] # add time dim of size 1 for consistent array dims
        lati = 1
        loni = 2

    # latitude weighting 
    lat_weight = np.cos(np.deg2rad(lat))
    tmp = np.ones([nlon,nlat])
    W = np.multiply(lat_weight,tmp).T

    # define hemispheres
    eqind = nlat//2

    if lat[0] > 0:
        # data has NH -> SH format
        W_NH = W[0:eqind+1]
        field_NH = field[:,0:eqind+1,:]
        W_SH = W[eqind+1:]
        field_SH = field[:,eqind+1:,:]
    else:
        # data has SH -> NH format
        W_NH = W[eqind:]
        field_NH = field[:,eqind:,:]
        W_SH = W[0:eqind]
        field_SH = field[:,0:eqind,:]

    gm  = np.zeros(ntime)
    nhm = np.zeros(ntime)
    shm = np.zeros(ntime)

    # Check for valid (non-NAN) values & use numpy average function (includes weighted avg calculation) 
    # Get arrays indices of valid values
    indok    = np.isfinite(field)
    indok_nh = np.isfinite(field_NH)
    indok_sh = np.isfinite(field_SH)
    for t in range(ntime):
        if lati == 0:
            # Global
            gm[t]  = np.average(field[indok],weights=W[indok])
            # NH
            nhm[t] = np.average(field_NH[indok_nh],weights=W_NH[indok_nh])
            # SH
            shm[t] = np.average(field_SH[indok_sh],weights=W_SH[indok_sh])
        else:
            # Global
            indok_2d    = indok[t,:,:]
            if indok_2d.any():
                field_2d    = np.squeeze(field[t,:,:])
                gm[t]       = np.average(field_2d[indok_2d],weights=W[indok_2d])
            else:
                gm[t] = np.nan
            # NH
            indok_nh_2d = indok_nh[t,:,:]
            if indok_nh_2d.any():
                field_nh_2d = np.squeeze(field_NH[t,:,:])
                nhm[t]      = np.average(field_nh_2d[indok_nh_2d],weights=W_NH[indok_nh_2d])
            else:
                nhm[t] = np.nan
            # SH
            indok_sh_2d = indok_sh[t,:,:]
            if indok_sh_2d.any():
                field_sh_2d = np.squeeze(field_SH[t,:,:])
                shm[t]      = np.average(field_sh_2d[indok_sh_2d],weights=W_SH[indok_sh_2d])
            else:
                shm[t] = np.nan

    return gm,nhm,shm


def regional_mask(lat,lon,southlat,northlat,westlon,eastlon):

    """
    Given vectors for lat and lon, and lat-lon boundaries for a regional domain, 
    return an array of ones and zeros, with ones located within the domain and zeros outside
    the domain as defined by the input lat,lon vectors.

    Originator: Greg Hakim
                University of Washington
                July 2017

    """

    import numpy as np
    
    nlat = len(lat)
    nlon = len(lon)

    tmp = np.ones([nlon,nlat])
    latgrid = np.multiply(lat,tmp).T
    longrid = np.multiply(tmp.T,lon)

    lab = (latgrid >= southlat) & (latgrid <=northlat)
    # check for zero crossing 
    if eastlon < westlon:
        lob1 = (longrid >= westlon) & (longrid <=360.)
        lob2 = (longrid >= 0.) & (longrid <=eastlon)
        lob = lob1+lob2
    else:
        lob = (longrid >= westlon) & (longrid <=eastlon)

    mask = np.multiply(lab*lob,tmp.T)

    return mask

def PAGES2K_regional_means(field,lat,lon):
    """
    Compute geographical spatial mean values for all times in the input (i.e. field) array. 
    Regions are defined following The PAGES2K Consortium (2013) Nature Geosciences paper, 
    Supplementary Information.

    input:  field[ntime,nlat,nlon] or field[nlat,nlon]
             lat[nlat,nlon] in degrees
             lon[nlat,nlon] in degrees

    output: rm[nregions,ntime] : regional means of "field" where nregions = 7 by definition, but could change

    uses functions: regional_mask()

    Originator: Greg Hakim
                University of Washington
                July 2017
    
    Revisions:
					7 September 2018: added Greenland area average (GJH)     
    """

    import numpy as np
    
    # print debug statements
    #debug = True
    debug = False
    
    # number of geographical regions (default, as defined in PAGES2K(2013) paper
    nregions = 16
    
    # set number of times, lats, lons; array indices for lat and lon    
    if len(np.shape(field)) == 3: # time is a dimension
        ntime,nlat,nlon = np.shape(field)
    else: # only spatial dims
        ntime = 1
        nlat,nlon = np.shape(field)
        field = field[None,:] # add time dim of size 1 for consistent array dims

    if debug:
        print('field dimensions...')
        print(np.shape(field))

    # define regions as in PAGES paper

    # lat and lon range for each region (first value is lower limit, second is upper limit)
    rlat = np.zeros([nregions,2]); rlon = np.zeros([nregions,2])
    reg_names = []
    
    # 1. Global
    reg_names.append('Global')
    rlat[0,0] = -90.; rlat[0,1] = 90.
    rlon[0,0] = 0.;   rlon[0,1] = 360.
    # 2. N. Hemisphere
    reg_names.append('Northern Hemisphere')
    rlat[1,0] = 0.; rlat[1,1] = 90.
    rlon[1,0] = 0.; rlon[1,1] = 360.
    # 3. S. Hemisphere
    reg_names.append('Southern Hemisphere')
    rlat[2,0] = -90.; rlat[2,1] = 0.
    rlon[2,0] = 0.;   rlon[2,1] = 360.
    # 4. N. Hemisphere extratropics
    reg_names.append('extratropical Northern Hemisphere')
    rlat[3,0] = 30.; rlat[3,1] = 90.
    rlon[3,0] = 0.;  rlon[3,1] = 360.
    # 5. S. Hemisphere extratropics
    reg_names.append('extratropical Southern Hemisphere')
    rlat[4,0] = -90.; rlat[4,1] = -30.
    rlon[4,0] = 0.;   rlon[4,1] = 360.
    # 6. Tropics
    reg_names.append('Tropics')
    rlat[5,0] = -30.; rlat[5,1] = 30.
    rlon[5,0] = 0.;   rlon[5,1] = 360.
    # 7. Arctic: north of 60N 
    reg_names.append('Arctic')
    rlat[6,0] = 60.; rlat[6,1] = 90.
    rlon[6,0] = 0.;  rlon[6,1] = 360.
    # 8. Europe: 35-70N, 10W-40E
    reg_names.append('Europe')
    rlat[7,0] = 35.;  rlat[7,1] = 70.
    rlon[7,0] = 350.; rlon[7,1] = 40.
    # 9. Asia: 23-55N; 60-160E (from map)
    reg_names.append('Asia')
    rlat[8,0] = 23.; rlat[8,1] = 55.
    rlon[8,0] = 60.; rlon[8,1] = 160.
    # 10. North America 1 (trees):  30-55N, 75-130W 
    reg_names.append('North America')
    rlat[9,0] = 30.; rlat[9,1] = 55.
    rlon[9,0] = 55.; rlon[9,1] = 230.
    # 11. South America: Text: 20S-65S and 30W-80W
    reg_names.append('South America')
    rlat[10,0] = -65.; rlat[10,1] = -20.
    rlon[10,0] = 280.; rlon[10,1] = 330.
    # 12. Australasia: 110E-180E, 0-50S 
    reg_names.append('Australasia')
    rlat[11,0] = -50.; rlat[11,1] = 0.
    rlon[11,0] = 110.; rlon[11,1] = 180.
    # 13. Antarctica: south of 60S (from map)
    reg_names.append('Antarctica')
    rlat[12,0] = -90.; rlat[12,1] = -60.
    rlon[12,0] = 0.;   rlon[12,1] = 360.
    # 14. Greenland
    reg_names.append('Greenland')
    rlat[13,0] = 59.;  rlat[13,1] = 85.
    rlon[13,0] = 285.; rlon[13,1] = 350.
    # 15. NE Atlantic
    reg_names.append('NE Atlantic')
    rlat[14,0] = 50.;  rlat[14,1] = 75.
    rlon[14,0] = 285.; rlon[14,1] = 25.
    # 16. Nino3.4
    reg_names.append('Nino3.4')
    rlat[15,0] = -5.;  rlat[15,1] = 5.
    rlon[15,0] = 190.; rlon[15,1] = 240.
    # ...add other regions here...
    
    # latitude weighting 
    lat_weight = np.cos(np.deg2rad(lat))
    tmp = np.ones([nlat,nlon])
    W = np.multiply(lat_weight,tmp)
    if debug:
        print('W dimensions:',W.shape)
        
    rm  = np.zeros([nregions,ntime])

    # loop over regions
    for region in range(nregions):

        if debug:
            print('region='+str(region))
            print(rlat[region,0],rlat[region,1],rlon[region,0],rlon[region,1])
            
        # regional weighting (ones in region; zeros outside)
        mask = regional_mask(lat[:,0],lon[0,:],rlat[region,0],rlat[region,1],rlon[region,0],rlon[region,1])
        if debug:
            print('mask dimensions',mask.shape)

        # this is the weight mask for the regional domain    
        Wmask = np.multiply(mask,W)

        # make sure data starts at South Pole
        if lat[0,0] > 0:
            # data has NH -> SH format; reverse
            field = np.flipud(field)

        # Check for valid (non-NAN) values & use numpy average function (includes weighted avg calculation) 
        # Get arrays indices of valid values
        indok    = np.isfinite(field)
        for t in range(ntime):
            indok_2d = indok[t,:,:]
            field_2d = np.squeeze(field[t,:,:])            
            if len(Wmask[indok_2d]) > 0 and np.max(Wmask) >0.:
                rm[region,t] = np.average(field_2d[indok_2d],weights=Wmask[indok_2d])
            else:
                rm[region,t] = np.nan
                
    return reg_names, rm
