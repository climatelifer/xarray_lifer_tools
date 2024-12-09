
import xarray as xr 
import numpy as np 
import scipy.stats as sts 
import pymannkendall as mk

# Trend calculation as Sen's slope in xarray 

def xr_mktrend(da,dim = 'time',sig = 0.05):
    '''
    This function compute some paramters for statistical trend in time series.    

    Parameters
    ----------
    da: xarray.DataArray 
        DataArray of variable 
    sig: float
        significance level (0.05 is the default)    
    
    Returns
    -------
    xarray.Dataset with 
        trend: string
            Tells the trend (increasing, decreasing or no trend)
        h: bool
            True (if trend is present) or False (if the trend is absence)
        p: float
            p-value of the significance test
        z: float
            normalized test statistics
        Tau: float 
            Kendall Tau
        s: float
            Mann-Kendal's score
        var_s: float
            Variance S
        slope: float
            Theil-Sen estimator/slope
        intercept: float
            intercept of Kendall-Theil Robust Line
     
    '''
    dx = xr.apply_ufunc(mk.trend_free_pre_whitening_modification_test,da,
                        input_core_dims=[[dim]],
                        output_core_dims=[[] for _ in range(9)],
                        kwargs={'alpha' : sig},
                        vectorize=True)
    ds = xr.Dataset()
    ds['trend'] = dx[0]
    ds['h'] = dx[1]
    ds['p'] = dx[2]
    ds['z'] = dx[3]
    ds['tau'] = dx[4]
    ds['s'] = dx[5]
    ds['var'] = dx[6]
    ds['slope'] = dx[7]
    ds['intercept'] = dx[8]
    
    return ds

# Pearson correlation and pvalue for xarray 

def pcorr(x,y,skipnan = False):
    '''
    This function compute pearson correlation for two time series.    

    Parameters
    ----------
    x: numpy.array  
        Array of variable 1 
    y: numpy.array
        Array of variable 2 
    skipna: bool 
        skip the nan values in both arrays (False is the default)       
    
    Returns
    -------
    r:  float
	pearson correlation value
    '''     
    if not skipnan:        
        xx = x[~np.isnan(x) & ~np.isnan(y)]
        yy = y[~np.isnan(x) & ~np.isnan(y)]
    else: 
        xx = x 
        yy = y 
    return sts.pearsonr(xx,yy)

### Pearson Correlation for xarray (this function also returns p-value)
def xr_person(dx,dy,dim = 'time',skipnan = False):
    r,p = xr.apply_ufunc(pcorr,dx,dy,
                         input_core_dims=[[dim],[dim]],                         
                         output_core_dims=[[],[]],
                         kwargs={'skipnan':skipnan},
                         vectorize=True)
    ds = xr.Dataset()
    ds['r'] = r
    ds['pvalue'] = p
    return ds 



### Test of differnces
def xr_diff_test(dx,dy,dim = 'time',test = 'T'):
    if test == 'T':
        test_func = sts.mannwhitneyu
    elif test == 'U':
        test_func = sts.ttest_ind    
    else:
        print('The test shoud be T or U')
    
    ds = xr.Dataset()
    stat,pvalue = xr.apply_ufunc(test_func,dx,dy,
                                 input_core_dims = [[dim],[dim]],
                                 output_core_dims = [[] for _ in range(2)],
                                 vectorize = True,
                                 )     
    ds['stat'] = stat
    ds['pvalue'] = pvalue
    return ds


# crop xarray dataset in spatial dimension
def crop_ds(ds,west=-85,east=-60,north=-10,south=-60,lon_name= 'lon',lat_name = 'lat'):
    return ds.where((ds[lon_name]>=west)&(ds[lon_name]<=east)&(ds[lat_name]>=south)&(ds[lat_name]<=north),drop = True)

# convert longitude coordinate 0 to 360 into -180 to 180
def lon360to180(ds,lon_name = 'lon'):
    ds.coords[lon_name] = (ds.coords[lon_name] + 180) % 360 - 180
    return ds 
# Remove seasonal cycle from a dataset
def remove_seasonal_cycle(ds,time_dim = 'time'):
    gb = ds.groupby(f'{time_dim}.month')
    ds_anom = gb - gb.mean(dim=time_dim)
    return ds_anom
