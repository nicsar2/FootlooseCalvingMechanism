import os
import sys
import time as tm
import datetime as dt
import numpy as np
import scipy as sp
import pandas as pd
import subprocess as sb
import xarray as xr
import cartopy.crs as crs
import matplotlib.pyplot as plt

from params import dataPath, figPath, minLat, maxLat, minLon, maxLon, Rt

dataDir = "GHRSST/" #Directory in dataPath where this dataset is stored

def download():#Download ERA5 wind data
	"""
	Download GHRSST sea surface temperature data from 2003 to 2023 to the dataPath in params.py. After downloading a daily file, it is cropped to the area of interes to save space. For each day, the method verify was already downloaded, if it's the case the file is skipped.
	"""
	savePath = dataPath + dataDir + 'Raw/'
	os.makedirs(savePath,  exist_ok=True)
	minYear, maxYear = 2003, 2022
	for _ in range(3):
		for year in range(minYear, maxYear+1):
			for month in range(1, 13):
				nbDays = getNumberDays(year,month)
				for day in range(1, nbDays+1):
					if not os.path.isfile(f"{savePath}{year}{month:02d}{day:02d}090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1_sliced.nc"):
						print(f"    Download {year}-{month:02d}-{day:02d}")
						cmd = f"""podaac-data-subscriber -c MUR-JPL-L4-GLOB-v4.1 -d {savePath} -sd {year}-{month:02d}-{day:02d}T00:00:00Z -ed {year}-{month}-{day:02d}T12:00:00Z --process "python3 dataGHRSST.py " """
						sb.run(cmd, shell=True)
						if os.path.isfile(savePath + ".update__MUR-JPL-L4-GLOB-v4.1"):
							os.remove(savePath + ".update__MUR-JPL-L4-GLOB-v4.1")
						tm.sleep(1)
					else:
						print(f"    Keep {year}-{month:02d}-{day:02d}")

	for file in os.listdir(savePath):
		if file[-3:]==".nc" and file[-9:]!="sliced.nc":
			os.remove(savePath+file)

def getData(year): #Process data to make it consistent with other data sets
	fileName = f'avg{year}.pkl'
	savePath = dataPath + dataDir + 'Pickle/'
	os.makedirs(savePath, exist_ok=True)

	if os.path.isfile(savePath + fileName):
		print(f"GHRSST Load From Pickle ~ {year}")
		ds = pd.read_pickle(savePath + fileName)

	else:
		filePath = dataPath + dataDir + 'Raw/'
		Files = [filePath+file for file in sorted(os.listdir(filePath)) \
			if ("sliced.nc" in file and file.split("/")[-1][:4] == str(year))]
		ds = xr.open_mfdataset(Files, combine='by_coords', \
			chunks="auto", \
			compat = 'override', coords = 'minimal')
		ds = ds.sortby(ds.time)
		ds = ds.drop_vars(["mask", "analysis_error"])
		ds = ds.resample(time='1ME', label="left", loffset=dt.timedelta(days=1)).mean()
		ds["analysed_sst"] -= 273.15
		ds = ds.rename({"analysed_sst":"T", "sea_ice_fraction":"C"})

		print(f"GHRSST Make Pickle ~ {year}")
		os.makedirs(savePath, exist_ok=True)
		pd.to_pickle(ds.load(), savePath + fileName)

	ds["S"] = getSurface()
	ds["mask"] = getMask(ite=50)["mask"]
	return ds

def getDataVar(var, latSlice=None, lonSlice=None, months=None, years=None, maskIte=50): #Subselection of the data to reduce RAM usage
	ds = []

	if not isinstance(var, list):
		var = [var]

	if latSlice is None:
		latSlice = slice(-90, 90)

	if lonSlice is None:
		lonSlice = slice(0, 360)

	if months is None:
		months = range(1, 13)

	if years is None:
		years = range(2003, 2023)

	for year in years:
		ds.append(getData(year)[var].sel(time=[np.datetime64(f'{year}-{month:02d}-01') for month in months ], lat=latSlice, lon=lonSlice))
	ds = xr.concat(ds, dim="time")
	ds["S"] = getSurface()
	ds["mask"] = getMask(ite=maskIte)["mask"]
	return ds

def getNumberDays(year, month): #Get number of days in specific month/year
	if((month==2) and ((year%4==0)  or ((year%100==0) and (year%400==0)))):
	    nbDays = 29
	elif(month==2):
	    nbDays = 28
	elif(month==1 or month==3 or month==5 or month==7 or month==8 or month==10 or month==12) :
	    nbDays = 31
	else:
	    nbDays = 30
	return nbDays

def getSurface(): #Get grid box surface
	filePath = dataPath + dataDir + 'Raw/'
	ds = xr.open_dataset([filePath+file for file in sorted(os.listdir(filePath)) if "sliced.nc" in file][0])
	ds["S"] = (('lat', "lon"), Rt**2*np.cos(np.meshgrid(ds.lon, ds.lat)[1]*np.pi/180)*(0.01*np.pi/180)**2)
	ds = ds["S"]
	return ds

def getMask(ite=50): #Get binary erosion mask of RIS
	filePath = dataPath + dataDir + 'Raw/'
	ds = xr.open_dataset([filePath+file for file in sorted(os.listdir(filePath)) if "sliced.nc" in file][0])
	mask = ds.analysed_sst.to_masked_array()[0]
	mask = np.where(mask == mask[0,0].copy(), False, True)
	border = sp.ndimage.sobel(mask, 0)

	border[:, :1470] = False
	border[:, -880:] = False
	border[1300:, :] = False

	mask = np.logical_not(sp.ndimage.binary_erosion(np.logical_not(border), iterations=ite, mask=mask)) 
	mask[-ite-1:, :] = False
	mask[:, :1470] = False
	mask[:, -880:] = False
	
	ds = ds.rename({"mask":"shelf"})
	ds = ds.drop_vars("sea_ice_fraction")
	ds = ds.drop_vars("analysis_error")
	ds = ds.drop_vars("analysed_sst")
	ds['border'] = (("lat", "lon"), border)
	ds['mask'] = (("lat", "lon"), mask)
	ds = ds.drop_dims("time")

	return ds

def plotExample(): #Plot example map of the raw data
	ds = getData(year=2010)
	dd = ds.isel(time=0)
	fig = plt.figure(figsize=(16,9), constrained_layout=True)
	ax = fig.add_subplot(projection=crs.SouthPolarStereo(central_longitude=180))
	cs = ax.pcolormesh(dd["lon"], dd["lat"], dd["T"], transform=crs.PlateCarree())
	plt.colorbar(cs)
	plt.show()

def plotMask(ite=None): #Plot the mask
	if ite is None:
		mask = getMask()
	else:
		mask = getMask(ite=ite)

	fig = plt.figure(figsize=(16,9), constrained_layout=True)
	ax = fig.add_subplot(projection=crs.SouthPolarStereo(central_longitude=180))
	cs = ax.pcolormesh(mask["lon"], mask["lat"], mask["mask"], transform=crs.PlateCarree())
	plt.colorbar(cs)
	plt.show()

def figAll(): #Plot map data at every time
	subfigPath = figPath+"GHRSST_PlotAll/"
	os.makedirs(subfigPath, exist_ok=True)
	filePath = dataPath + dataDir + 'Raw/'
	Files = [filePath+file for file in sorted(os.listdir(filePath)) if "sliced.nc" in file][::10][440:]

	for file in Files:
		print(file)
		ds = xr.open_dataset(file)
		ds = ds.drop_vars(["mask", "analysis_error"])
		ds["analysed_sst"] -= 273.15
		ds = ds.rename({"analysed_sst":"T", "sea_ice_fraction":"C"})

		time = ds.time.values[0]
		ds = ds.isel(time=0)

		fig = plt.figure(figsize=(16,9), constrained_layout=True)
		ax = fig.add_subplot(projection=crs.SouthPolarStereo(central_longitude=180))
		ax.pcolormesh(ds["lon"], ds["lat"], ds["T"], transform=crs.PlateCarree())
		title = f"{time}"
		fig.savefig(f"{subfigPath}{title}.png", dpi=300)
		plt.close(fig=fig)

if __name__=="__main__":
	if len(sys.argv)==2:
		filePath = sys.argv[1]
		ds = xr.open_dataset(filePath)
		ds = ds.sel(lat=slice(minLat,maxLat))
		ds.coords['lon'] = ds.coords['lon'] % 360 
		ds = ds.sortby(ds["lon"])
		ds = ds.sel(lon=slice(minLon, maxLon))
		for var in ["dt_1km_data", "sst_anomaly"]:
			if var in ds.keys():
				ds = ds.drop_vars(var)
		ds.to_netcdf(filePath.replace(".nc","_sliced.nc"))
		os.remove(filePath)
