import os
import numpy as np
import scipy as sp
import xarray as xr
import xesmf as xe
import pandas as pd
import cartopy.crs as crs
import matplotlib.pyplot as plt

import dataBuoy as dbuoy
import dataGHRSST as dSST
from params import dataPath, rhoi, rhow, Rt

dataDir = "MEaSUREs2/"
	
def download():
	savePath = dataPath + dataDir + 'Raw/'
	if not os.path.isdir(savePath):
		os.makedirs(savePath)

	from download.MEaSUREs2 import main as downloadMain
	downloadMain(savePath=savePath)

def getData():
	fileName = 'selected.pkl'
	savePath = dataPath + dataDir + 'Pickle/'

	if not os.path.isdir(savePath):
		os.makedirs(savePath)

	if os.path.isfile(savePath + fileName):
		print("MEaSUREs2 LFP")
		return pd.read_pickle(savePath + fileName)
	else:
		latSlice = slice(-80,-77)
		lonSlice = slice(160, 200)

		savePath = dataPath + dataDir + 'Raw/'
		ds = xr.open_dataset(savePath + "antarctica_ice_velocity_450m_v2.nc")
		ds = ds.isel(x=slice(4000,8000), y=slice(7000,10000))
		ds = ds.drop_vars(["ERRX", "ERRY", "CNT"])
		return ds

		dsSST = dSST.getData(2010).sel(lat=latSlice, lon=lonSlice)["T"]
		ds_ = xr.Dataset( {
				'lat': (['lat'], dsSST.lat.values, {'units': 'degrees_north'}),
				'lon': (['lon'], dsSST.lon.values, {'units': 'degrees_east' }) })
		dsSST.close()
		ds_['ui'] = xe.Regridder(ds, ds_, 'nearest_s2d')(ds['VX'], keep_attrs=True)
		ds_['vi'] = xe.Regridder(ds, ds_, 'nearest_s2d')(ds['VY'], keep_attrs=True)
		ds_['svi'] = xe.Regridder(ds, ds_, 'nearest_s2d')(ds['STDY'], keep_attrs=True)
		ds_['sui'] = xe.Regridder(ds, ds_, 'nearest_s2d')(ds['STDX'], keep_attrs=True)

		pd.to_pickle(ds_.load(), dataPath + dataDir + 'Pickle/' + fileName)
		return ds_

def getDataBand(ite=5):
	ds = getData()
	ds["mask"] = getMask(ite=ite)["mask"]
	ds["S"] = (('lat', "lon"), Rt**2*np.cos(np.meshgrid(ds.lon, ds.lat)[1]*np.pi/180)*(0.01*np.pi/180)**2)
	vi = ((-ds.S*ds.vi).where(ds.mask).sum('lat')) / (ds.S.where(ds.mask).sum('lat'))
	ui = ((-ds.S*ds.ui).where(ds.mask).sum('lat')) / (ds.S.where(ds.mask).sum('lat'))
	svi = ((-ds.S*ds.svi).where(ds.mask).sum('lat')) / (ds.S.where(ds.mask).sum('lat'))
	sui = ((-ds.S*ds.sui).where(ds.mask).sum('lat')) / (ds.S.where(ds.mask).sum('lat'))

	res = xr.Dataset({"u":ui, "v":vi, "su":sui, "sv":svi})
	return res

def plotExemple():
	ds = getData()
	fig = plt.figure(figsize=(16,9), constrained_layout=True)
	ax = fig.add_subplot(projection=crs.SouthPolarStereo(central_longitude=180))
	cs = ax.pcolormesh(ds["lon"], ds["lat"], (ds["vi"]**2 + ds["ui"]**2)**0.5, transform=crs.PlateCarree(), cmap="viridis", vmin=900, vmax=1100, zorder=2)
	mask = dSST.getMask(50)
	ax.contour(mask.lon, mask.lat, mask["mask"], transform=crs.PlateCarree(), cmap="viridis", zorder=3)
	mask = getMask(ite=3)
	ax.contour(mask.lon, mask.lat, mask["mask"], transform=crs.PlateCarree(), cmap="autumn", zorder=4)
	mask = getMask(ite=5)
	ax.contour(mask.lon, mask.lat, mask["mask"], transform=crs.PlateCarree(), cmap="autumn", zorder=4)
	buoy_ = dbuoy.getData()
	df1 = getDataBand(ite=3)
	df2 = getDataBand(ite=5)
	for buoyName in buoy_.index:
		ax.scatter(buoy_.loc[buoyName, "lon"], buoy_.loc[buoyName, "lat"], transform=crs.PlateCarree(), s=50, zorder=10, label=buoyName)
		print(buoyName, (df1.u.sel(lon=buoy_.loc[buoyName, "lon"], method="nearest").values**2+df1.v.sel(lon=buoy_.loc[buoyName, "lon"], method="nearest").values**2)**0.5, (df2.u.sel(lon=buoy_.loc[buoyName, "lon"], method="nearest").values**2+df2.v.sel(lon=buoy_.loc[buoyName, "lon"], method="nearest").values**2)**0.5)
	ax.legend()
	plt.colorbar(cs)

	plt.show()

def getMask(ite=20):
	ds = getData()
	mask = np.isnan(ds.vi)
	border = sp.ndimage.sobel(mask, 0)
	mask = np.logical_not(sp.ndimage.binary_erosion(np.logical_not(border), iterations=ite, mask=np.logical_not(mask))) 
	mask[:ite, :] = False
	mask[:172, 4001-ite:] = False
	mask[:282, :ite] = False
	mask = np.logical_and(mask, np.logical_not(border))
	
	ds = ds.drop_vars(["ui", "vi"])
	ds['mask'] = (("lat", "lon"), mask)
	ds['border'] = (("lat", "lon"), border)
	return ds


