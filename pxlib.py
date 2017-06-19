import os, sys
import iris
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import iris.quickplot as qplt
import matplotlib.ticker as ticker
import iris.plot as iplt
from mycolormaps import getcolors
from iris.util import *
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cf_units as unit
import datetime

mpl.rc('font', family='Times New Roman') 
import math

def nice_number(value, round_=False):
    '''nice_number(value, round_=False) -> float'''
    exponent = math.floor(math.log(value, 10))
    fraction = value / 10 ** exponent

    if round_:
        if fraction < 1.5: nice_fraction = 1.
        elif fraction < 3.: nice_fraction = 2.
        elif fraction < 7.: nice_fraction = 5.
        else: niceFraction = 10.
    else:
        if fraction <= 1: nice_fraction = 1.
        elif fraction <= 2: nice_fraction = 2.
        elif fraction <= 5: nice_fraction = 5.
        else: nice_fraction = 10.

    return nice_fraction * 10 ** exponent

def nice_bounds(axis_start, axis_end, num_ticks=10):
    '''
    nice_bounds(axis_start, axis_end, num_ticks=10) -> tuple
    @return: tuple as (nice_axis_start, nice_axis_end, nice_tick_width)
    '''
    axis_width = axis_end - axis_start
    min_axis = axis_start
    max_axis = axis_end
    
    if axis_width == 0:
        nice_tick = 0
    else:
        nice_range = nice_number(axis_width)
        nice_tick = nice_number(nice_range / (num_ticks - 1), round_=True)
        axis_start = math.floor(axis_start / nice_tick) * nice_tick
        axis_end = math.ceil(axis_end / nice_tick) * nice_tick
        #print 'Nice tick = %s' % nice_tick, min_axis, axis_start, axis_start - nice_tick
        #print 'Nice tick = %s' % nice_tick, max_axis, axis_end
        marks = np.arange(axis_start, axis_end + nice_tick, nice_tick)
        
        if min_axis > marks[0]:
            marks[0] = min_axis  
            marks = marks[1:]
        if max_axis < marks[-1]:
            marks[-1] = max_axis
            marks = marks[:-1]
        return marks
        

def map_plot(var2plot, ax, title=None, clevs=None, \
             pltname='iris_test_plot.ps', colorReverse=False, \
             minorTicks=False, gridLines = True):
    # Plot Lat-Lon maps
    minlon = min(var2plot.coord('longitude').points)
    maxlon = max(var2plot.coord('longitude').points)
    cenlon = minlon + (maxlon - minlon) / 2.
    
    minlat = min(var2plot.coord('latitude').points)
    maxlat = max(var2plot.coord('latitude').points)
    
    if clevs == None: 
        clevs = np.linspace(round(var2plot.data.min()), round(var2plot.data.max()), 11)
    
    #fig = plt.subplot(subplot)#figsize=(8, 3), dpi=100)
    #print subplot
    #proj = ccrs.PlateCarree(central_longitude=cenlon)
    #ax = plt.axes(projection=proj)
    #ax.projection = ax1.projection
    #ax.coastlines = ax1.coastlines
    #ax.set_xticks = ax1.set_xticks
    #ax.set_yticks = ax1.set_yticks
    #ax =  ax(projection=proj)
    
    if colorReverse:
        cf = iplt.contourf(var2plot, clevs, cmap=getcolors('ncl_default', colorReverse=True), extend='both')
    else:
        cf = iplt.contourf(var2plot, clevs, cmap=getcolors('ncl_default'), extend='both')

    
    #ax.set_xlim([minlon, maxlon])
    ax.coastlines()
    #ax.set_global()
    plt.title(var2plot.long_name)
    
    xmarks = nice_bounds(minlon, maxlon) 
    ymarks = nice_bounds(minlat, maxlat) 
    #print minlon, maxlon
    #print xmarks, ymarks
    
    # Major ticks with labels
    ax.set_xticks(xmarks, crs=ccrs.PlateCarree())
    ax.set_yticks(ymarks, crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    
    if gridLines:
        for xm in xmarks:
            x = [xm, xm]
            y = [minlat, maxlat]
            ax.plot(x,y, transform = ccrs.PlateCarree(),  color='grey', linewidth = 0.5, linestyle=':')
        for ym in ymarks:
            x = [minlon, maxlon]
            y = [ym, ym]
            ax.plot(x,y, transform = ccrs.PlateCarree(),  color='grey', linewidth = 0.5, linestyle=':')
             
    # Minor ticks without labels
    if minorTicks:
            
        nxmarks = len(xmarks)
        xminor = np.array([np.linspace(xmarks[i],xmarks[i+1],11) for i in range(nxmarks-1)])
        xminor = xminor.flatten()
        
        nymarks = len(ymarks)
        yminor = np.array([np.linspace(ymarks[i],ymarks[i+1],11) for i in range(nymarks-1)])
        yminor = yminor.flatten()

        for xm in xminor:
            x = [xm, xm]
            xminorHeight = np.abs((maxlat-minlat)/100.)
            y = [minlat, minlat+xminorHeight]
            ax.plot(x,y, transform = ccrs.PlateCarree(),  color='k', linewidth = 0.5)
        
    #ax.gridlines(crs = proj, xlocs = xmarks, ylocs = ymarks)
    
    #labels = ax.get_xticklabels()
    #plt.setp(labels, rotation=0, fontsize=10)
    #labels = ax.get_yticklabels()
    #plt.setp(labels, rotation=0, fontsize=10)
    
    fig = plt.gcf()
    plt_ax = plt.gca()
    left, bottom, width, height = plt_ax.get_position().bounds
    first_plot_left = plt_ax.get_position().bounds[0]
    
    # the width of the colorbar should now be simple
    #width = left - first_plot_left + width * 0.9
    cwidth = width*0.025  # 2.5% of the plot width
    print width, cwidth
    
    # Add axes to the figure, to place the colour bar
    #colorbar_axes = fig.add_axes([first_plot_left + 0.0375, bottom + 0.1, width, 0.05])
    colorbar_axes = fig.add_axes([left+width, bottom, cwidth, height])
    # Add the colour bar
    cbar = plt.colorbar(cf, colorbar_axes, orientation='vertical')
    cbar.ax.tick_params(labelsize=10) 
    #plt.show()
    #return ax
    #plt.savefig(pltname)#,bbox_inches='tight')
    
    #plt.close()
    
def sym_levels(cube):
    '''
    Generates 11 contour levels symmetric about zero  
    '''
    mn = cube.data.min()
    mx = cube.data.max()
    if mn <= 0.:
        mnx = max([-mn, mx])
        levs = np.linspace(-mnx, mnx, 11)
    else:
        levs = np.linspace(mn, mx, 11)
    return(levs)
        
#def find_nearest(array, value):
#    idx = (np.abs(array-value)).argmin()
#    return array[idx]

def find_nearest(array, values):
    indices = np.abs(np.subtract.outer(array, values)).argmin(0)
    return array[indices]
    
def splitTime(date):
    '''Split date string into yyyy, mm, dd integers'''
    d = str(date)
    year = int(d[:4])
    month = int(d[4:6])
    day = int(d[-2:])
    return year, month, day
    
def subset(cube, **args):
    #print cube, args
    # Usage:
    # ux = subset(u, latitude=2.09931, longitude=[40.5,60], pressure_level=850)
    # ux = subset(u, latitude=2.09931, longitude=40.5, pressure_level=850, time=20141013)
    # ux = subset(u, latitude=2.09931, longitude=[40.5,60], pressure_level=850, time=[20141013, 20141016])
    constraint = ''
    for k, key in enumerate(args.keys()):
        sub_range = args[key]
        if key=='time' or key == 'forecast_time':
            #print sub_range
            #print 'Subsetting in time '
            time = cube.coord(key)
            time_units = cube.coord(key).units
            
            if isinstance(sub_range, list):
                # Iris way of doing time constraints
                # suggested by Bill Little (28/11/2014)
                start_year, start_month, start_day = splitTime(sub_range[0])
                end_year, end_month, end_day = splitTime(sub_range[1])
                
                start_dt = datetime.datetime(start_year, start_month, start_day)
                start_offset = time_units.date2num(start_dt)

                end_dt = datetime.datetime(end_year, end_month, end_day)
                end_offset = time_units.date2num(end_dt)
                constraint += '%s = lambda cell: %s <= cell.point <= %s' % (key, start_offset, end_offset)                
            else:
                start_year, start_month, start_day = splitTime(sub_range)
                start_dt = datetime.datetime(start_year, start_month, start_day)
                start_offset = time_units.date2num(start_dt)
                
                nodes = [float(n) for n in time.points]
                nodes = np.around(nodes, decimals=10)
                node = np.around(float(start_offset), decimals=10)
                xnode = find_nearest(nodes, node)
                
                constraint += '%s = lambda cell: np.around(float(cell.point),decimals=2) == %s' % (key, xnode)                
        else:
            if isinstance(sub_range, list):
                #print 'Looking for intersection'
                constraint += '%s = lambda cell: %s <=cell<= %s' % (key, sub_range[0], sub_range[1])       
            else:
                # Fix the decimal point for the ease of comparison
                nodes = cube.coord(key).points
                nodes = np.around(nodes, decimals=4)
                node = np.around(sub_range, decimals=4)

                xnode = find_nearest(nodes, node)
                constraint += '%s = lambda cell: cell == %s' % (key, xnode )        
        if k < len(args.keys()) - 1:
            constraint += ', '        
    print 'Constraining: \n %s' %constraint
    print '-'*100
    self = eval("cube.extract(iris.Constraint(%s))" % constraint)
    return self

def compute_differential(cube, r, axisname, Cyclic=False):
    
    R = cube.coord(axisname).points
    if not isinstance(r, list):
        R = [r * i for i, xr in enumerate(R)]
    else:
        R = r
        
    coordnames = [cc.name() for cc in cube.coords()]
    axis_ind = np.where(coordnames == np.array(axisname))[0]
    deriv = cube.copy()
    #print 'Computing centered differences along axis = %s, coord = %s...' \
    #      % (axis_ind[0], coordnames[axis_ind[0]])
    # Mid-points
    for i in np.arange(1, len(R) - 1):
        if axis_ind == 0:
            deriv.data[i] = (cube.data[i + 1] - cube.data[i - 1]) \
            / (R[i + 1] - R[i - 1])
        if axis_ind == 1:
            deriv.data[:, i] = (cube.data[:, i + 1] - cube.data[:, i - 1]) \
            / (R[i + 1] - R[i - 1])
        if axis_ind == 2:
            deriv.data[:, :, i] = (cube.data[:, :, i + 1] - cube.data[:, :, i - 1])\
             / (R[i + 1] - R[i - 1])
        if axis_ind == 3:
            deriv.data[:, :, :, i] = (cube.data[:, :, :, i + 1] - cube.data[:, :, :, i - 1]) \
            / (R[i + 1] - R[i - 1])
            
    # End points
    if axis_ind == 0:
        deriv.data[0] = (cube.data[1] - cube.data[0]) / (R[1] - R[0])
        deriv.data[-1] = (cube.data[-1] - cube.data[-2]) / (R[-1] - R[-2])
    if axis_ind == 1:
        deriv.data[:, 0] = (cube.data[:, 1] - cube.data[:, 0]) / (R[1] - R[0])
        deriv.data[:, -1] = (cube.data[:, -1] - cube.data[:, -2]) / (R[-1] - R[-2])
    if axis_ind == 2:
        deriv.data[:, :, 0] = (cube.data[:, :, 1] - cube.data[:, :, 0]) / (R[1] - R[0])
        deriv.data[:, :, -1] = (cube.data[:, :, -1] - cube.data[:, :, -2]) / (R[-1] - R[-2])
    if axis_ind == 3:
        deriv.data[:, :, :, 0] = (cube.data[:, :, :, 1] - cube.data[:, :, :, 0]) / (R[1] - R[0])
        deriv.data[:, :, :, -1] = (cube.data[:, :, :, -1] - cube.data[:, :, :, -2]) / (R[-1] - R[-2])
    
    return deriv
    
def center_finite_diff(cube, axisname, Cyclic=False):
    R = 6378388. # Radius of the earth
    deg2rad = 0.0174533 # pi/180.
    dcube = cube.copy()
    
    print 'Computing centered differences along [%s] axis...' % axisname
    if axisname == 'latitude':
        lats = cube.coord(axisname).points
        dlat = (lats[1] - lats[0]) * deg2rad # convert to radians
        dy = R * np.sin(dlat)  # constant at this latitude
        dcube = compute_differential(cube, dy, axisname)
         
    if axisname == 'longitude':
        lats = cube.coord('latitude').points
        lons = cube.coord('longitude').points
        dlon = (lons[1] - lons[0]) * deg2rad  # convert to radians
        for la, lat in enumerate(lats):
            dx = R * np.cos(deg2rad * lat) * dlon
            dum = compute_differential(cube[:, :, la, :], dx, axisname)
            dcube.data[:, :, la, :] = dum.data 
            
    if axisname == 'time':
        dcube = compute_differential(cube, 24 * 3600., axisname)
         
    if (axisname == 'pressure_level') or (axisname == 'level') or (axisname == 'pressure'):
        P = cube.coord(axisname).points
        P = list(P * 100.)
        dcube = compute_differential(cube, P, axisname)
        
    return dcube

def layerthick(arch_p_lvl, p_s, p_t, t_2_b):
    nalvl = len(arch_p_lvl)
    if t_2_b:
        arch_p_lvl_B_2_T = arch_p_lvl[::-1]
    else:
        arch_p_lvl_B_2_T = arch_p_lvl
        
    p = np.zeros(2 * nalvl + 1)
    #print 'Python---', p.size
    p[1:2 * nalvl:2] = arch_p_lvl_B_2_T[0:nalvl]
    for j in np.arange(3, 2 * nalvl - 3, 2):
        if not ((p[j - 2] > p[j]) and (p[j] > p[j + 2])):
            raise Exception("Archived pressure levels must\
             decrease monotonically.")
    
    # Compute intermediate half-levels from archived pressure levels,
    p[2:2 * nalvl - 1:2] = 0.5 * (p[1:2 * nalvl - 2:2] + p[3:2 * nalvl:2])
    # Assign fixed pressure at bottom of column,
    p[0] = p_s  
    # Assign fixed pressure at top of column,
    p[2 * nalvl] = p_t 
    if p[0] < p[1] :
        p[0] = 101300.
    if p[2 * nalvl] > p[2 * nalvl - 1]:
        p[2 * nalvl], p[2 * nalvl - 1]
        raise Exception("Either p_o is not greater than the bottom\
                        archived pressure, or p_t is not less \
                        than the top archived pressure.")
    #if not ( ( p[0] > p_s ) and ( p_s > p[2*nalvl] ) ):
    #    raise Exception("The surface pressure p_s does not lie\
    #                     between p_s and p_t."
    # Compute layer thicknesses,
    #dp = p[0:2*nalvl-2:2] - p[2:2*nalvl:2] in fortran
    
    dp = p[0:2 * nalvl - 1:2] - p[2:2 * nalvl + 1:2]
    
    #for i in range(nalvl):
    #    print '%i plev=%s  dp = %s' %(i,arch_p_lvl[i], dp[i])
    
    if any(n < 0 for n in dp):
        raise Exception("Layer thicknesses must be nonnegative.")
    
    return dp
    
def vertical_integral(cube, axisname, ps, t_2_b=False):
    """
    #integrand        # Quantity to be vertically integrated 
    #arch_p_lvl       # Pressure value for each archived pressure level, hPa.
    #p_s              # surface pressure fields in hPa
    #t_2_b            # If integrand and arch_p_lvl are ordered top
                      # to bottom then T_2_B must be assigned the
                      # logical value .TRUE.
    """
    arch_p_lvl = cube.coord(axisname).points
    print min(arch_p_lvl), max(arch_p_lvl)
    print "Will convert to Pa by x 100."
    coordnames = [cc.name() for cc in cube.coords()]
    axis_ind = np.where(coordnames == np.array(axisname))[0]
    
    integrand = cube.data
    arch_p_lvl = [lvl*100. for lvl in arch_p_lvl]
    ps *= 100. 
    
    p_t = min(arch_p_lvl) 
    dp_cube = cube.copy()
    ntime, nlev, nlat, nlon = cube.shape
    for nt in range(ntime):
        print 'vertical integration time %i/%i' % (nt, ntime)
        for la in range(nlat):
            for lo in range(nlon):
                dp_cube.data[nt, :, la, lo] = layerthick(arch_p_lvl, ps.data[nt, la, lo], p_t, t_2_b)
    cube = cube * dp_cube
    vint = cube.collapsed('pressure_level', iris.analysis.SUM)\
                / dp_cube.collapsed('pressure_level', iris.analysis.SUM)
    return vint
    
def test():
    datadir = '/data/local/hadpx/obs_data/ERAInt/MSE_budget'
    m_file = os.path.join(datadir, 'MSE_lev_2000_daily.nc')
    m = iris.load(m_file)[0]
    dmdt = center_finite_diff(m, 'time')
    dmdp = center_finite_diff(m, 'pressure_level')
    dmdy = center_finite_diff(m, 'latitude')
    dmdx = center_finite_diff(m, 'longitude')
    
    #qplt.plot(dmdt.extract(subset({'pressure_level':[850], 'latitude':[0], 'longitude':[90]})))
    #qplt.plot(dmdy.extract(subset({'pressure_level':[850], 'latitude':[0], 'longitude':[90]})))
    #qplt.plot(dmdx.extract(subset({'pressure_level':[850], 'latitude':[0], 'longitude':[90]})))
    qplt.plot(dmdp.extract(subset({'pressure_level':[850], 'latitude':[0], 'longitude':[90]})))
    plt.show()
    #dmdx = derivative(m,'latitude')
    #dmdx = derivative(m,'longitude')
    
    #m = m.extract(subset({'pressure_level':[1000], 'latitude':[-10,20], 'longitude':[0,360]}))
    #print m
def test1():
    datadir = '/project/MJO_GCSS/SEA_Data/ERAINT'
    m_file = os.path.join(datadir, 'hus_1998_2012_dmean.nc')
    m = iris.load(m_file)[0]
    dmdp = center_finite_diff(m, 'longitude')
    print dmdp
    ddum = dmdp.extract(subset({'latitude':[0], 'longitude':[90]}))
    qplt.plot(ddum[10])
    plt.show()
def vint_test():
    datadir = '/data/local/hadpx/obs_data/ERAInt/MSE_budget'
    m_file = os.path.join(datadir, 'MSE_lev_2000_daily.nc')
    m = iris.load_cube(m_file)
    ps_file = os.path.join(datadir, 'PSFC_sfc_2000_daily.nc')
    ps = iris.load_cube(ps_file) 
    print m
    #qplt.contourf(ps[0])
    #plt.show()
    #s = m.extract(subset({'pressure_level':[850], \
    #                      'latitude':[0], \
    #                      'longitude':[90]}))
    #print s
    #sys.exit()
    vint = vertical_integral(m[0:1], 'pressure_level', ps / 100., t_2_b=False)
    #out_file = os.path.join(datadir, 'MSE_vint_2000_daily.nc')
    #iris.save(vint , out_file)
    
    #dp = iris.load_cube(out_file)
    
    print vint
    fig = plt.figure()
    proj = ccrs.PlateCarree(central_longitude=180)
    ax = fig.add_subplot(111, projection=proj)
    map_plot(vint[0], ax)
    plt.show()
    
    
if __name__ == '__main__':
    vint_test()
