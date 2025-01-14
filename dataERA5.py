import os
import cdsapi
import numpy as np
import xarray as xr
import cartopy.crs as crs
import matplotlib.pyplot as plt

from params import dataPath, minLon, maxLon, Rt, cmap, lim

dataDir = "ERA5/" #Directory in dataPath where this dataset is stored

def download(): #Download ERA5 wind data
	savePath = dataPath + dataDir + 'Raw/'
	os.makedirs(savePath, exist_ok=True)
	fileName = "ERA5_raw.grib"

	if not os.path.isfile(savePath + fileName):
		dataset = "reanalysis-era5-pressure-levels-monthly-means"
		request = {
		    "product_type": ["monthly_averaged_reanalysis"],
		    "variable": [
		        "u_component_of_wind",
		        "v_component_of_wind"
		    ],
		    "pressure_level": ["950", "975", "1000"],
		    "year": [
		        "1991", "1992", "1993",
		        "1994", "1995", "1996",
		        "1997", "1998", "1999",
		        "2000", "2001", "2002",
		        "2003", "2004", "2005",
		        "2006", "2007", "2008",
		        "2009", "2010", "2011",
		        "2012", "2013", "2014",
		        "2015", "2016", "2017",
		        "2018", "2019", "2020",
		        "2021", "2022", "2023",
		        "2024"
		    ],
		    "month": [
		        "01", "02", "03",
		        "04", "05", "06",
		        "07", "08", "09",
		        "10", "11", "12"
		    ],
		    "time": ["00:00"],
		    "data_format": "grib",
		    "download_format": "unarchived",
		    "area": [-63, -180, -89, 180]
		}

		client = cdsapi.Client()
		client.retrieve(dataset, request, savePath + fileName).download()


	else:
		print("ERA5 data skipped: already downloaded")

def getData(): #Process data to make it consistent with other data sets
	savePath = dataPath + dataDir + 'Raw/'
	ds = xr.open_dataset(savePath + "ERA5_raw.grib", engine="cfgrib")
	ds.coords['longitude'] = ds.coords['longitude'] % 360 
	ds = ds.sortby(ds.longitude)
	ds = ds.sortby(ds.latitude)
	ds = ds.sel(longitude=slice(minLon, maxLon))
	ds = ds.sel(isobaricInhPa=1000)
	ds = ds.rename({"longitude":"lon", "latitude":"lat"})
	ds = ds.drop_vars(["number", "step","isobaricInhPa", "valid_time"])
	ds["w"] = (ds["u"]**2 + ds["v"]**2)**0.5
	ds['S'] = (('lat', "lon"), Rt**2*np.cos(np.meshgrid(ds.lon, ds.lat)[1]*np.pi/180)*(0.25*np.pi/180)**2)
	return ds

def plotExample(var, time=0):#Plot example map of the raw data
	ds = getData()
	dd = ds.isel(time=time)[var]
	fig = plt.figure(figsize=(16,9), constrained_layout=True)
	ax = fig.add_subplot(projection=crs.SouthPolarStereo(central_longitude=180))
	cs = ax.pcolormesh(dd["lon"], dd["lat"], dd, transform=crs.PlateCarree(), cmap=cmap[var], vmin=lim[var][0], vmax=lim[var][1])
	fig.colorbar(cs)
	plt.show()
