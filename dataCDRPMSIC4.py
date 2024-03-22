import os
import inspect as it
import subprocess as sp
import xarray as xr
import cartopy.crs as crs
import matplotlib.pyplot as plt

from params import dataPath, figPath, cmap, lim

dataDir = "CDRPMSIC4/" #Directory in dataPath where this dataset is stored

def download(): #Download NSIDC sea ice concentration data
	"""
	Download NSIDC monthly sea ice concentration data around the RIS from 2002 to 2023 to the dataPath/dataDir. Skip already downloaded files. 
	"""
	savePath = dataPath + dataDir + 'Raw/'
	os.makedirs(savePath, exist_ok=True)

	cmd = f"wget -nc -P {savePath} --ftp-user=anonymous -i ./download/CDRPMSIC4_url.txt"
	sp.run(cmd, shell=True)

def getData(): #Process data to make it consistent with other data sets
	"""
	Regroup the monthly NSIDC sea ice concentration (C) data into a single xarray object, with coordinates latitude (lat), longitude (lon), time (time). 
	"""
	filePath = dataPath + dataDir + 'Raw/'
	Files = [filePath+file for file in sorted(os.listdir(filePath)) if file[-3:] in ".nc"]
	coord = xr.open_dataset(Files[0])
	ds = xr.open_mfdataset(Files[1:], combine='nested', concat_dim='tdim' )
	ds = ds.sortby(ds.time)
	ds["lon"] = coord.longitude
	ds["lat"] = coord.latitude
	ds["mask"] = coord.landmask
	ds = ds.swap_dims({"tdim":"time",})
	ds = ds.isel(x=slice(100,200), y=slice(200,300))
	ds = ds.rename({"cdr_seaice_conc_monthly":"C"})
	ds = ds.drop_vars(["nsidc_bt_seaice_conc_monthly", "nsidc_nt_seaice_conc_monthly", "projection", \
		"qa_of_cdr_seaice_conc_monthly", "stdev_of_cdr_seaice_conc_monthly", "xgrid", "ygrid", "mask"])
	ds["lon"] = ds["lon"] % 360 
	ds = ds.assign_coords(lon=ds.lon, lat=ds.lat)
	ds = ds.drop_vars(["x","y"])
	return ds

def plotExample(var="C", time=0): #Plot example map of the raw data
	ds = getData()
	dd = ds.isel(time=time)[var]
	fig = plt.figure(figsize=(16,9), constrained_layout=True)
	ax = fig.add_subplot(projection=crs.SouthPolarStereo(central_longitude=180))
	cs = ax.pcolormesh(dd["lon"], dd["lat"], dd, transform=crs.PlateCarree(), cmap=cmap[var], vmin=lim[var][0], vmax=lim[var][1])
	fig.colorbar(cs)
	plt.show()

def plotAll(): #Plot map data at every time
	figTitle = it.stack()[0][3]
	sufigPath = figPath + "CDRPMSIC4/"
	os.makedirs(sufigPath, exist_ok=True)
	ds = getData()

	for time in ds.time:
		dd = ds.sel(time=time)["C"]
		fig = plt.figure(figsize=(12,12), constrained_layout=True)
		ax = fig.add_subplot(projection=crs.SouthPolarStereo(central_longitude=180))
		cs = ax.pcolormesh(dd["lon"], dd["lat"], dd, transform=crs.PlateCarree(), vmin=0, vmax=1.3)
		fig.colorbar(cs)
		fig.canvas.manager.full_screen_toggle()
		title = f"{figTitle}_{time.values}.png".replace("-01T00:00:00.000000000","")
		fig.savefig(sufigPath + title, dpi=300)
		plt.close(fig=fig)
