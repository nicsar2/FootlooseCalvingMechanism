import os
import xesmf as xe
import xarray as xr
import pandas as pd
import dataERA5 as dERA
import dataGHRSST as dSST
import dataCDRPMSIC4 as dSIC
import physics as ps

from params import dataPath #Directory in dataPath where this dataset is stored

dataDir = "_WaveMelt/"

def getData(latSlice=None, lonSlice=None, months=None, years=None, maskIte=50, keepVar=False): #Combine SST, SIC, wind data to compute frontal wave-induced melt rate
	dsSST = dSST.getDataVar("T", latSlice=latSlice, lonSlice=lonSlice, months=months, years=years, maskIte=maskIte)
	dsSIC = dSIC.getData().sel(time=[time for time in dsSST.time.values])
	dsERA = dERA.getData().sel(time=[time for time in dsSST.time.values])

	ds = xr.Dataset( {
			'lat': (['lat'], dsSST.lat.values, {'units': 'degrees_north'}),
			'lon': (['lon'], dsSST.lon.values, {'units': 'degrees_east' }) })
	ds['T'] = dsSST['T']
	ds['S'] = dsSST['S']
	ds['maskFront'] = dsSST['mask']
	ds['C'] = xe.Regridder(dsSIC, ds, 'nearest_s2d')(dsSIC['C'], keep_attrs=False)
	ds['w'] = xe.Regridder(dsERA, ds, 'nearest_s2d')(dsERA['w'], keep_attrs=False)
	ds['Me'] = ps.Me(ds)*365.25*24*3600
	ds["maskSIC"] = (ds.C < 2).isel(time=0)
	if not keepVar:
		ds = ds.drop_vars(["C", "w"])

	return ds, dsSIC, dsERA

def getDataBand(): #Melt data 2D as function of lon/time
	fileName = 'data.pkl'
	savePath = dataPath + dataDir

	os.makedirs(savePath, exist_ok=True)

	if os.path.isfile(savePath + fileName):
		print("GHRSST Load From Pickle ~ dataBand")
		return pd.read_pickle(savePath + fileName)
	else:
		os.makedirs(savePath, exist_ok=True)
		latSlice = slice(-79,-76)
		lonSlice = slice(160, 205)
		months = range(1, 13)
		ds = getData(latSlice=latSlice, lonSlice=lonSlice, months=months, maskIte=50)[0]
		S = ds.S.where(ds.maskSIC).where(ds.maskFront).sum('lat')
		ds = ((ds.S*ds.Me).where(ds.maskSIC).where(ds.maskFront).sum('lat')) / S
		ds["S"] = S
		print("GHRSST Make Pickle ~ dataBand")
		pd.to_pickle(ds.load(), savePath + fileName)
		return ds

