#
# python code for some calculations related to the dynamic tropopause (DT)  -
# DT pressure, DT potential temperature, 330K PV, 
# and a cross-section of PV at the latitude where the tropopause is lowest -
# all based on the GFS analysis available online.  As the data is accessed
# online, the program can take a while to run.
#
# the date and lat-lon range can be set below
#
# (poorly) coded by Mathew Barlow
# initial release: 14 Nov 2017
# last updated: 30 Nov 2017
#
# this code has *not* been extensively tested and has been 
# awkwardly translated from other coding languages, so if you find
# any errors or have any suggestions or improvements, including for
# the plotting, please let me know at Mathew_Barlow@uml.edu . Thanks!
#
# Support from NSF AGS-1623912 is gratefully acknowledged
#

import os
import numpy as np
import netCDF4
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from scipy.ndimage import gaussian_filter
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from datetime import datetime, timedelta

# VALUES TO SET *************************************************
# set date, lat-lon range, and PV-value definition of tropopause
mydate='20200609'
myhour='00'
(lat1,lat2)=(-75,-10)
(lon1,lon2)=(-50,50)
tpdef=2   # definition of tropopause in PVU
create_casedir = True
download_data = True
save_data = False
gen_directory = 'images'


# CREATE WORKING DIR ********************************************
if create_casedir == True:
    try:
        os.mkdir(mydate)
    except:
        pass
    os.chdir(mydate)
else:
    os.chdir(gen_directory)
    
for timestep in range(0,42,2):
#****************************************************************
    
    #constants
    re=6.37e6
    g=9.81
    cp=1004.5
    r=2*cp/7
    kap=r/cp
    omega=7.292e-5
    pi=3.14159265
    
    # open dataset, retreive variables, close dataset    
    if download_data == True:
        url='http://nomads.ncep.noaa.gov:80/dods/gfs_0p25/gfs'+\
        mydate+'/gfs_0p25_'+myhour+'z'#'z_anl'
        file = netCDF4.Dataset(url)
        
        print('Acquiring dimension variables....')
        lon_in  = file.variables['lon'][:]
        lonNEG = np.where(lon_in >= 180)[0]
        lonPOS = np.where(lon_in < 180)[0]
        lon_reshape = np.concatenate((lonNEG,lonPOS))
        lon_in = np.concatenate((lon_in[lonNEG]-360,lon_in[lonPOS]))
        
        lat_in  = file.variables['lat'][:]
        lev = file.variables['lev'][:]
        time_in = file.variables['time'][:]
        
        
        print('Acquiring 2PV....')
        pres2pv_in = file.variables['pres2pv'][timestep,:,:]
        pres2pv_in = pres2pv_in[:,lon_reshape]
        print('Acquiring Temperature....')
        t_in = file.variables['tmpprs'][timestep,:,:,:]
        t_in = t_in[:,:,lon_reshape]
        print('Acquiring U-wind....')
        u_in = file.variables['ugrdprs'][timestep,:,:,:]
        u_in = u_in[:,:,lon_reshape]
        print('Acquiring V-wind....')
        v_in = file.variables['vgrdprs'][timestep,:,:,:]
        v_in = v_in[:,:,lon_reshape]
        print('Acquiring Geopotential height....')
        hgt_in = file.variables['hgtprs'][timestep,:,:,:]
        hgt_in = hgt_in[:,:,lon_reshape]
        print('Acquiring Mean sea level pressure....')
        mslp_in = file.variables['prmslmsl'][timestep,:,:]
        mslp_in = mslp_in[:,lon_reshape]
        print('Saving data for....')	
        if save_data == True: 
            np.savez('GFS_data_' + str(timestep*3) + 'hr.npz', lat_in=lat_in, lon_in=lon_in, lev=lev, time_in=time_in, timestep=timestep, pres2pv_in=pres2pv_in, t_in=t_in, u_in=u_in, v_in=v_in, hgt_in=hgt_in, mslp_in=mslp_in)
        file.close()
    
    elif download_data == False: 
        file=np.load('GFS_data_' + str(timestep*3) + 'hr.npz')
        lat_in = file['lat_in']
        lon_in = file['lon_in']
        lev = file['lev']
        time_in = file['time_in']
        pres2pv_in = file['pres2pv_in']
        t_in = file['t_in']
        u_in = file['u_in']
        v_in = file['v_in']
        hgt_in = file['hgt_in']
        mslp_in = file['mslp_in']
    
    # get datetime
    time = datetime(1,1,1,0,0,0)+timedelta(days=time_in[timestep]-2)
    # get date for plotting
    fdate=time.strftime('%d %b %Y %H:%M')
    print('for time: ' + fdate)
    
    # get array indices for lat-lon range
    # specified above
    iy1 = np.argmin( np.abs( lat_in - lat1 ) )
    iy2 = np.argmin( np.abs( lat_in - lat2 ) ) 
    ix1 = np.argmin( np.abs( lon_in - lon1 ) )
    ix2 = np.argmin( np.abs( lon_in - lon2 ) )  
    
    # select specified lat-lon range
    t=t_in[:,iy1:iy2,ix1:ix2]
    lon=lon_in[ix1:ix2]
    lat=lat_in[iy1:iy2]
    u=u_in[:,iy1:iy2,ix1:ix2]
    v=v_in[:,iy1:iy2,ix1:ix2]
    hgt=hgt_in[:,iy1:iy2,ix1:ix2]
    mslp=mslp_in[iy1:iy2,ix1:ix2]
    mslp = mslp/100
    pres2pv=pres2pv_in[iy1:iy2,ix1:ix2]
    
    #Subset arrays for plotting
    hgt300 = hgt[np.where(lev==300)[0][0],:,:]
    
    # some prep work for derivatives
    xlon,ylat=np.meshgrid(lon,lat)
    dlony,dlonx=np.gradient(xlon)
    dlaty,dlatx=np.gradient(ylat)
    dx=re*np.cos(ylat*pi/180)*dlonx*pi/180
    dy=re*dlaty*pi/180
    
    # define potential temperature and Coriolis parameter
    theta=t*(1.E5/(lev[:,np.newaxis,np.newaxis]*100))**kap
    f=2*omega*np.sin(ylat*pi/180)
    
    # calculate derivatives
    # (np.gradient can handle 1D uneven spacing,
    # so build that in for p, but do dx and dy 
    # external to the function since they are 2D)
    ddp_theta=np.gradient(theta,lev*100,axis=0)
    ddx_theta=np.gradient(theta,axis=2)/dx
    ddy_theta=np.gradient(theta,axis=1)/dy
    ddp_u=np.gradient(u,lev*100,axis=0)
    ddp_v=np.gradient(v,lev*100,axis=0)
    ddx_v=np.gradient(v,axis=2)/dx
    ddy_ucos=np.gradient(u*np.cos(ylat*pi/180),axis=1)/dy
    
    # calculate contributions to PV and PV
    absvort=ddx_v-(1/np.cos(ylat*pi/180))*ddy_ucos+f
    pv_one=g*absvort*(-ddp_theta)
    pv_two=g*(ddp_v*ddx_theta-ddp_u*ddy_theta)
    pv=pv_one+pv_two
    pv_PVU = pv*1000000
    
    # calculate pressure of tropopause, Fortran-style (alas!)
    # as well as potential temperature (theta) and height
    #
    # starting from 10hPa and working down, to avoid
    # more complicated vertical structure higher up
    #
    nx=ix2-ix1+1
    ny=iy2-iy1+1
    nz=lev.size
    nzs=np.argwhere(lev==10.0)[0,0]
    tp=np.empty((ny-1,nx-1))*np.nan   # initialize as undef
    tp_full=np.empty((ny-1,nx-1))*np.nan   # initialize as undef
    tp_theta=np.empty((ny-1,nx-1))*np.nan   # initialize as undef
    tp_hgt=np.empty((ny-1,nx-1))*np.nan   # initialize as undef
    
    for ix in range(0,nx-1):
        for iy in range(0,ny-1):
            for iz in range(nzs,0,-1):
                if pv[iz,iy,ix]/1e-6<=tpdef:
                    if np.isnan(tp[iy,ix]):
#                        tp[iy,ix]=(
#                        (lev[iz]*(pv[iz+1,iy,ix]-tpdef*1e-6)
#                        -lev[iz+1]*(pv[iz,iy,ix]-tpdef*1e-6))/
#                        (pv[iz+1,iy,ix]-pv[iz,iy,ix])
#                        )
#        
                        tp_theta[iy,ix]=(
                        ((lev[iz]-tp[iy,ix])*theta[iz+1,iy,ix]+
                        (tp[iy,ix]-lev[iz+1])*theta[iz,iy,ix])/
                        (lev[iz]-lev[iz+1])
                        )
                        
                        tp_hgt[iy,ix]=(
                        ((lev[iz]-tp[iy,ix])*hgt[iz+1,iy,ix]+
                        (tp[iy,ix]-lev[iz+1])*hgt[iz,iy,ix])/
                        (lev[iz]-lev[iz+1])
                        )
            pv_temp=np.empty((nz))*np.nan   # initialize as undef
            for iz in range(nz-2,0,-1):
                pv_temp[iz] = pv_PVU[iz,iy,ix]
            pv_temp[np.isnan(pv_temp)]=0   
            pv_temp_lev = lev[np.where(pv_temp<=-2)]
            tp_full[iy,ix] = max(pv_temp_lev)
            pv_temp_lev = pv_temp_lev[pv_temp_lev>=100]
            if pv_temp_lev.shape[0] == 0:
                tp[iy,ix] = 100
            elif np.where(pv_temp_lev[:-1] - pv_temp_lev[1:] > 50)[0].size == 0:
                tp[iy,ix] = max(pv_temp_lev)
            else:    
                tp[iy,ix] = min(pv_temp_lev[np.where(pv_temp_lev[:-1] - pv_temp_lev[1:] > 50)[0]+1])
    
    ytp_low,xtp_low = np.where(tp_full-tp!=0)
    tp_low = tp_full*np.nan
    tp_low[ytp_low,xtp_low] = tp_full[ytp_low,xtp_low] 
	
    # calculate PV on the 310K isentropic surface
    # (also not in a pythonic way)
    nx=ix2-ix1+1
    ny=iy2-iy1+1
    nz=lev.size
    pv310=np.empty((ny-1,nx-1))*np.nan   # initialize as undef
    for ix in range(0,nx-1):
        for iy in range(0,ny-1):
            for iz in range(nz-2,0,-1):
                if theta[iz,iy,ix]>=310:
                    if theta[iz-1,iy,ix]<=310:
                        if np.isnan(pv310[iy,ix]):
                            pv310[iy,ix]=(
                            ((310-theta[iz-1,iy,ix])*pv[iz,iy,ix]+
                            (theta[iz,iy,ix]-310)*pv[iz-1,iy,ix])/
                            (theta[iz,iy,ix]-theta[iz-1,iy,ix])
                            )	
	
    # calculate PV on the 320K isentropic surface
    # (also not in a pythonic way)
    nx=ix2-ix1+1
    ny=iy2-iy1+1
    nz=lev.size
    pv320=np.empty((ny-1,nx-1))*np.nan   # initialize as undef
    for ix in range(0,nx-1):
        for iy in range(0,ny-1):
            for iz in range(nz-2,0,-1):
                if theta[iz,iy,ix]>=320:
                    if theta[iz-1,iy,ix]<=320:
                        if np.isnan(pv320[iy,ix]):
                            pv320[iy,ix]=(
                            ((320-theta[iz-1,iy,ix])*pv[iz,iy,ix]+
                            (theta[iz,iy,ix]-320)*pv[iz-1,iy,ix])/
                            (theta[iz,iy,ix]-theta[iz-1,iy,ix])
                            )	
    
    # calculate PV on the 330K isentropic surface
    # (also not in a pythonic way)
    nx=ix2-ix1+1
    ny=iy2-iy1+1
    nz=lev.size
    pv330=np.empty((ny-1,nx-1))*np.nan   # initialize as undef
    for ix in range(0,nx-1):
        for iy in range(0,ny-1):
            for iz in range(nz-2,0,-1):
                if theta[iz,iy,ix]>=330:
                    if theta[iz-1,iy,ix]<=330:
                        if np.isnan(pv330[iy,ix]):
                            pv330[iy,ix]=(
                            ((330-theta[iz-1,iy,ix])*pv[iz,iy,ix]+
                            (theta[iz,iy,ix]-330)*pv[iz-1,iy,ix])/
                            (theta[iz,iy,ix]-theta[iz-1,iy,ix])
                            )
                            
    # calculate PV on the 340K isentropic surface
    # (also not in a pythonic way)
    nx=ix2-ix1+1
    ny=iy2-iy1+1
    nz=lev.size
    pv340=np.empty((ny-1,nx-1))*np.nan   # initialize as undef
    for ix in range(0,nx-1):
        for iy in range(0,ny-1):
            for iz in range(nz-2,0,-1):
                if theta[iz,iy,ix]>=340:
                    if theta[iz-1,iy,ix]<=340:
                        if np.isnan(pv340[iy,ix]):
                            pv340[iy,ix]=(
                            ((340-theta[iz-1,iy,ix])*pv[iz,iy,ix]+
                            (theta[iz,iy,ix]-340)*pv[iz-1,iy,ix])/
                            (theta[iz,iy,ix]-theta[iz-1,iy,ix])
                            )
                            
    # calculate PV on the 350K isentropic surface
    # (also not in a pythonic way)
    nx=ix2-ix1+1
    ny=iy2-iy1+1
    nz=lev.size
    pv350=np.empty((ny-1,nx-1))*np.nan   # initialize as undef
    for ix in range(0,nx-1):
        for iy in range(0,ny-1):
            for iz in range(nz-2,0,-1):
                if theta[iz,iy,ix]>=350:
                    if theta[iz-1,iy,ix]<=350:
                        if np.isnan(pv350[iy,ix]):
                            pv350[iy,ix]=(
                            ((350-theta[iz-1,iy,ix])*pv[iz,iy,ix]+
                            (theta[iz,iy,ix]-350)*pv[iz-1,iy,ix])/
                            (theta[iz,iy,ix]-theta[iz-1,iy,ix])
                            )
                       
    
    # slight smoothing of result
    # (appears to work better than smoothing u,v,t first)
    tp=gaussian_filter(tp,sigma=1)
    tp_theta=gaussian_filter(tp_theta,sigma=1)
    pv310=gaussian_filter(pv310,sigma=1)
    pv320=gaussian_filter(pv320,sigma=1)
    pv330=gaussian_filter(pv330,sigma=1)
    pv340=gaussian_filter(pv340,sigma=1)
    pv350=gaussian_filter(pv350,sigma=1)
    
    # identify latitude of lowest tropopause
    tp_temp = tp*1
    tp_temp[tp_temp>650]=100
    maxloc=np.argwhere(tp_temp==np.amax(tp_temp))
    latmax=lat[maxloc[0,0]]
    
    
    # now make some plots - these badly need to be improved
    states = NaturalEarthFeature(category='cultural', 
        scale='50m', facecolor='none', 
        name='admin_1_states_provinces_shp')
    
    # Steup contour level arrays
    clevs_PV=np.arange(-12,8,1)
    clevs_mslp=np.arange(900,1040,4)
    clevs_hgt300=np.arange(8000,10500,100)
        
    
    plt.figure(num=1,figsize=(12, 8))
    
    ax = plt.axes(projection=ccrs.PlateCarree( ))
    ax.set_extent([lon1,lon2,lat1,lat2],crs=ccrs.PlateCarree())
#    plt.contour(lon,lat,pv350/1e-6,clevs_PV,transform=ccrs.PlateCarree(),
#        colors='grey',linewidths=0.25)
    plt.contour(lon,lat,pv350/1e-6,levels=[-2],transform=ccrs.PlateCarree(),
        colors='blue',linewidths=2)
    cp=plt.contourf(lon,lat,pv350/1e-6,clevs_PV,transform=ccrs.PlateCarree(),
        cmap='RdBu_r')
    cbar = plt.colorbar(cp, ticks=clevs_PV, orientation='horizontal')
    cbar.set_label('PVU')
    cs_mslp = plt.contour(lon,lat,mslp,clevs_mslp,transform=ccrs.PlateCarree(),
        colors='black',linewidths=0.5)
    plt.clabel(cs_mslp, clevs_mslp,fontsize='smaller', fmt='%1.0f')
    plt.contour(lon,lat,hgt300,clevs_hgt300,transform=ccrs.PlateCarree(),
        colors='black',linewidths=0.75,linestyles='dotted')
    
    #ax.add_feature(states, linewidth=0.8, color='gray')
    ax.coastlines('50m', linewidth=2,color='gray')
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.FixedLocator(np.arange(lon1,lon2+10,10))
    gl.ylocator = mticker.FixedLocator(np.arange(lat1,lat2+5,5))
    plt.title('Potential Vorticity on the 350K Surface\n'+fdate+'Z')
    #plt.show()
    
    pngname = 'PV_350K_' + str(timestep) + '.png'
    plt.savefig(pngname)
    plt.clf()   # Clear figure
    plt.close()
    
    plt.figure(num=2,figsize=(12, 8))
    
    ax = plt.axes(projection=ccrs.PlateCarree( ))
    ax.set_extent([lon1,lon2,lat1,lat2],crs=ccrs.PlateCarree())
#    plt.contour(lon,lat,pv340/1e-6,clevs_PV,transform=ccrs.PlateCarree(),
#        colors='grey',linewidths=0.5)
    plt.contour(lon,lat,pv340/1e-6,levels=[-2],transform=ccrs.PlateCarree(),
        colors='blue',linewidths=2)
    cp=plt.contourf(lon,lat,pv340/1e-6,clevs_PV,transform=ccrs.PlateCarree(),
        cmap='RdBu_r')
    cbar = plt.colorbar(cp, ticks=clevs_PV, orientation='horizontal')
    cbar.set_label('PVU')
    cs_mslp = plt.contour(lon,lat,mslp,clevs_mslp,transform=ccrs.PlateCarree(),
        colors='black',linewidths=0.5)
    plt.clabel(cs_mslp, clevs_mslp,fontsize='smaller', fmt='%1.0f')
    plt.contour(lon,lat,hgt300,clevs_hgt300,transform=ccrs.PlateCarree(),
        colors='black',linewidths=1.5,linestyles='dotted')
    
    
    #ax.add_feature(states, linewidth=0.8, color='gray')
    ax.coastlines('50m', linewidth=2,color='gray')
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.FixedLocator(np.arange(lon1,lon2+10,10))
    gl.ylocator = mticker.FixedLocator(np.arange(lat1,lat2+5,5))
    plt.title('Potential Vorticity on the 340K Surface\n'+fdate+'Z')
    
    pngname = 'PV_340K_' + str(timestep) + '.png'
    plt.savefig(pngname)
    plt.clf()   # Clear figure
    plt.close()
    
    plt.figure(num=3,figsize=(12, 8))
    
    ax = plt.axes(projection=ccrs.PlateCarree( ))
    ax.set_extent([lon1,lon2,lat1,lat2],crs=ccrs.PlateCarree())
#    plt.contour(lon,lat,pv330/1e-6,clevs_PV,transform=ccrs.PlateCarree(),
#        colors='grey',linewidths=0.5)
    plt.contour(lon,lat,pv330/1e-6,levels=[-2],transform=ccrs.PlateCarree(),
        colors='blue',linewidths=2)
    cp=plt.contourf(lon,lat,pv330/1e-6,clevs_PV,transform=ccrs.PlateCarree(),
        cmap='RdBu_r')
    cbar = plt.colorbar(cp, ticks=clevs_PV, orientation='horizontal')
    cbar.set_label('PVU')
    cs_mslp = plt.contour(lon,lat,mslp,clevs_mslp,transform=ccrs.PlateCarree(),
        colors='black',linewidths=0.5)
    plt.clabel(cs_mslp, clevs_mslp,fontsize='smaller', fmt='%1.0f')
    plt.contour(lon,lat,hgt300,clevs_hgt300,transform=ccrs.PlateCarree(),
        colors='black',linewidths=1.5,linestyles='dotted')
    
    #ax.add_feature(states, linewidth=0.8, color='gray')
    ax.coastlines('50m', linewidth=2,color='gray')
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.FixedLocator(np.arange(lon1,lon2+10,10))
    gl.ylocator = mticker.FixedLocator(np.arange(lat1,lat2+5,5))
    plt.title('Potential Vorticity on the 330K Surface\n'+fdate+'Z')
    
    pngname = 'PV_330K_' + str(timestep) + '.png'
    plt.savefig(pngname)
    plt.clf()   # Clear figure
    plt.close()
    
    plt.figure(num=4,figsize=(12, 8))
    
    ax = plt.axes(projection=ccrs.PlateCarree( ))
    ax.set_extent([lon1,lon2,lat1,lat2],crs=ccrs.PlateCarree())
    plt.contour(lon,lat,pv320/1e-6,levels=[-2],transform=ccrs.PlateCarree(),
        colors='blue',linewidths=2)
    cp=plt.contourf(lon,lat,pv320/1e-6,clevs_PV,transform=ccrs.PlateCarree(),
        cmap='RdBu_r')
    cbar = plt.colorbar(cp, ticks=clevs_PV, orientation='horizontal')
    cbar.set_label('PVU')
    cs_mslp = plt.contour(lon,lat,mslp,clevs_mslp,transform=ccrs.PlateCarree(),
        colors='black',linewidths=0.5)
    plt.clabel(cs_mslp, clevs_mslp,fontsize='smaller', fmt='%1.0f')
    plt.contour(lon,lat,hgt300,clevs_hgt300,transform=ccrs.PlateCarree(),
        colors='black',linewidths=1.5,linestyles='dotted')
    
    #ax.add_feature(states, linewidth=0.8, color='gray')
    ax.coastlines('50m', linewidth=2,color='gray')
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.FixedLocator(np.arange(lon1,lon2+10,10))
    gl.ylocator = mticker.FixedLocator(np.arange(lat1,lat2+5,5))
    plt.title('Potential Vorticity on the 320K Surface\n'+fdate+'Z')
    
    pngname = 'PV_320K_' + str(timestep) + '.png'
    plt.savefig(pngname)
    plt.clf()   # Clear figure
    plt.close()
	
    plt.figure(num=5,figsize=(12, 8))
    
    ax = plt.axes(projection=ccrs.PlateCarree( ))
    ax.set_extent([lon1,lon2,lat1,lat2],crs=ccrs.PlateCarree())
#    plt.contour(lon,lat,pv330/1e-6,clevs_PV,transform=ccrs.PlateCarree(),
#        colors='grey',linewidths=0.5)
    plt.contour(lon,lat,pv310/1e-6,levels=[-2],transform=ccrs.PlateCarree(),
        colors='blue',linewidths=2)
    cp=plt.contourf(lon,lat,pv310/1e-6,clevs_PV,transform=ccrs.PlateCarree(),
        cmap='RdBu_r')
    cbar = plt.colorbar(cp, ticks=clevs_PV, orientation='horizontal')
    cbar.set_label('PVU')
    cs_mslp = plt.contour(lon,lat,mslp,clevs_mslp,transform=ccrs.PlateCarree(),
        colors='black',linewidths=0.5)
    plt.clabel(cs_mslp, clevs_mslp,fontsize='smaller', fmt='%1.0f')
    plt.contour(lon,lat,hgt300,clevs_hgt300,transform=ccrs.PlateCarree(),
        colors='black',linewidths=1.5,linestyles='dotted')
    
    #ax.add_feature(states, linewidth=0.8, color='gray')
    ax.coastlines('50m', linewidth=2,color='gray')
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.FixedLocator(np.arange(lon1,lon2+10,10))
    gl.ylocator = mticker.FixedLocator(np.arange(lat1,lat2+5,5))
    plt.title('Potential Vorticity on the 310K Surface\n'+fdate+'Z')
    
    pngname = 'PV_310K_' + str(timestep) + '.png'
    plt.savefig(pngname)
    plt.clf()   # Clear figure
    plt.close()
       
    # plot of DT pressure
    plt.figure(num=6,figsize=(12, 8))
    
    ax = plt.axes(projection=ccrs.PlateCarree( ))
    ax.set_extent([lon1,lon2,lat1,lat2],crs=ccrs.PlateCarree())
    clevs=np.arange(100,700,50)
    #plt.contour(lon,lat,tp_mb,clevs,transform=ccrs.PlateCarree(),colors='black',
    #linewidths=0.5)
    cp=plt.contourf(lon,lat,tp,clevs,transform=ccrs.PlateCarree(),cmap='RdPu',extend='max')
    gl = ax.gridlines(draw_labels=True)
    #plt.contour(lon,lat,ylat,[latmax],transform=ccrs.PlateCarree(),colors='white',
    #linewidths=1,linestyles='dashed')
    cbar = plt.colorbar(cp, ticks=clevs, orientation='horizontal')
    cbar.set_label('hPa')
    #ax.add_feature(states, linewidth=0.8, color='gray')
    ax.coastlines('50m', linewidth=2,color='gray')
    
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.FixedLocator(np.arange(lon1-360,lon2-360+10,10))
    gl.ylocator = mticker.FixedLocator(np.arange(lat1,lat2+5,5))
    plt.title('Dynamic Tropopause/Stratospheric Intrusion Depth (2PVU)\n'+myhour+'Z '+fdate)
    
    pngname = 'DynamicTropopause_' + str(timestep) + '.png'
    plt.savefig(pngname)
    plt.clf()   # Clear figure
    plt.close()
    
    # plot of DT pressure
    plt.figure(num=7,figsize=(12, 8))
    
    ax = plt.axes(projection=ccrs.PlateCarree( ))
    ax.set_extent([lon1,lon2,lat1,lat2],crs=ccrs.PlateCarree())
    clevs=np.arange(100,1050,50)
    #plt.contour(lon,lat,tp_mb,clevs,transform=ccrs.PlateCarree(),colors='black',
    #linewidths=0.5)
    cp=plt.contourf(lon,lat,tp_full,clevs,transform=ccrs.PlateCarree(),cmap='GnBu',extend='max')
    gl = ax.gridlines(draw_labels=True)
    #plt.contour(lon,lat,ylat,[latmax],transform=ccrs.PlateCarree(),colors='white',
    #linewidths=1,linestyles='dashed')
    cbar = plt.colorbar(cp, ticks=clevs, orientation='horizontal')
    cbar.set_label('hPa')
    #ax.add_feature(states, linewidth=0.8, color='gray')
    ax.coastlines('50m', linewidth=2,color='gray')
    
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.FixedLocator(np.arange(lon1-360,lon2-360+10,10))
    gl.ylocator = mticker.FixedLocator(np.arange(lat1,lat2+5,5))
    plt.title('Lowest level high-PV (2PVU)\n'+myhour+'Z '+fdate)
    
    pngname = 'LowestHighPV_' + str(timestep) + '.png'
    plt.savefig(pngname)
    plt.clf()   # Clear figure
    plt.close()
    
    
    for j in range(0,7):
        plotlat=np.argwhere(lat==-25-(5*j))[0,0]
        
        plt.figure(num=8+j,figsize=(12, 8))
        # P-lon cross-section of PV at latitude
        # of lowest tropopause
        ax = plt.axes()
        
        pv_smooth=gaussian_filter(pv,sigma=1)
        pv_smooth = pv_smooth*1000000
        pv_smooth[pv_smooth>10] = -25
        theta_smooth=gaussian_filter(theta,sigma=1)
        
        plt.ylim(lev[0],lev[19])
        #plt.yscale('log')
        clevs=np.arange(-20,0,2)
        plt.contour(lon,lev[0:21],pv_smooth[0:21,plotlat,:],np.append(clevs,-1.5),
            colors='black')
        cp=plt.contourf(lon,lev[0:21],pv_smooth[0:21,plotlat,:],clevs,
            cmap='RdPu',extend='min')
        clevs2=np.arange(260,490,10)
        plt.contour(lon,lev[0:21],theta_smooth[0:21,plotlat,:],[330],
            colors='blue',linewidths=1.2)
        cs=plt.contour(lon,lev[0:21],theta_smooth[0:21,plotlat,:],clevs2,
            colors='blue',linewidths=0.5)
        
        plt.clabel(cs,inline=1,fontsize=8,fmt='%4.0f')
        cbar = plt.colorbar(cp, ticks=clevs, orientation='horizontal')
        cbar.set_label('PVU')
        
        plt.title('Longitudinal Cross-section of PV (shading) and '+r'$\theta$'+
            ' (blue contours) at '+ str(int(abs(lat[plotlat]))) +'S\n'+fdate+'Z')
        
        #plt.show()
        pngname = '2D_CrossSection_' + str(int(abs(lat[plotlat]))) + 'S_' + str(timestep) + '.png'
        plt.savefig(pngname)
        plt.clf()   # Clear figure
        plt.close()