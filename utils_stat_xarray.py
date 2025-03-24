
import xarray as xr 
import numpy as np 
import scipy.stats as sts 
import pymannkendall as mk
import matplotlib.pyplot as plt 
import seaborn as sns

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
def lon360to180(ds,lon_name = 'lon',order = False):
    ds.coords[lon_name] = (ds.coords[lon_name] + 180) % 360 - 180
    if order == True:
         ds = ds.sortby(ds[lon_name])
    else:
        pass
    return ds 

# Remove seasonal cycle from a dataset
def remove_seasonal_cycle(ds,time_dim = 'time'):
    gb = ds.groupby(f'{time_dim}.month')
    ds_anom = gb - gb.mean(dim=time_dim)
    return ds_anom

# Select season
def select_season(ds,season = 'AMJJAS'):
    if season == 'AMJJAS':
        months = [4,5,6,7,8,9]
    elif season == 'ONDJFM':
        months = [10,11,12,1,2,3]
    elif season == 'JJA':
        months = [6,7,8]
    elif season == 'DJF':
        months = [12,1,2]
    elif season == 'MAM':
        months = [3,4,5]
    elif season == 'SON':
        months = [9,10,11]
    return ds.sel(time = ds.time.dt.month.isin(months))

# Statitics 
def rmse(ds_model,ds_ref,dim = 'time'):
    return np.sqrt(((ds_model-ds_ref)**2).mean(dim))

def bias(ds_model,ds_ref,dim = 'time'):
    return ds_model.mean(dim)-ds_ref.mean(dim)

# Linear regression trend

# linear regression trend
def linr(y):
    x = np.arange(len(y))
    return sts.linregress(x = x,y = y)

def xr_linregress(ds,dim = 'year'):
    da = xr.apply_ufunc(linr,
                        ds,
                        input_core_dims = [[dim]],
                        output_core_dims = [[] for _ in range(5)],
                        vectorize = True                        
                        )
    dx = xr.Dataset()
    dx['slope'] = da[0]
    dx['intercept'] = da[1]
    dx['pearson_r'] = da[2]
    dx['p_value'] = da[3]
    
    return dx

# Matrix of trend and plot
def trend_matrix(ds,dim = 'year'):
    years = ds[dim].values
    nx = len(years)
    matrix_trend = np.zeros((nx,nx))
    matrix_pvalue = np.zeros((nx,nx))
    xd = ds.mean(dim = ['x','y'])
    for i in np.arange(0,nx):
        for j in np.arange(1,nx+1)[::-1]:        
            if i>j-1 or i==j-1:
                matrix_trend[i,j-1] = np.nan
                matrix_pvalue[i,j-1] = np.nan                
            else:                
                dsp = xr_linregress(xd.isel(year = slice(i,j)),dim = dim)
                matrix_trend[i,j-1] = dsp.slope.values
                matrix_pvalue[i,j-1] = dsp.p_value.values
    df_trend_matrix = pd.DataFrame(matrix_trend,index = xd[dim].values,columns=xd[dim].values)
    df_pval_matrix = pd.DataFrame(matrix_pvalue,index = xd[dim].values,columns=xd[dim].values)
    return df_trend_matrix, df_pval_matrix

def plot_trend_matrix(dftrend,dfpval,title,vmin,vmax,vcenter,sig = 0.05):
    fig,ax = plt.subplots(dpi = 300,figsize = (6,6),facecolor = 'w')
    sns.heatmap(dftrend,cmap = plt.cmap.BrBG,center =vcenter,ax = ax,linecolor='w',linewidths=0.6,vmin = vmin,vmax = vmax,cbar_kws=dict(extend = 'both'))
    ax.pcolor(np.arange(len(dfpval.columns)+1), 
            np.arange(len(dfpval.index)+1), dfpval[dfpval<=sig], hatch='/', alpha=0.)    
    ax.set_ylabel('Begining Year')
    ax.set_xlabel('End Year')
    ax.grid(True, which="minor", color="w", linewidth=2)
    ax.tick_params(which="minor", left=False, bottom=False)
    ax.set_title(title)
    #return fig