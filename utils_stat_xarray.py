
import xarray as xr 
import pymannkendall as mk

# Trend calculation as Sen's slope in xarray 

def xr_mktrend(da,dim = 'time',sig = 0.05):
    '''
    This function compute some paramters for statitical trend in time series.    

    Parameters
    ----------
    da: xarray.DataArray 
        DataArray of variable 
    alpha: float
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
