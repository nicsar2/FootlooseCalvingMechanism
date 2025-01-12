import os
import sys
sys.path.append('/home/sartore/Documents/Other/esa-snappy')
import numpy as np
import xarray as xr
import pyproj as pp
import pandas as pd
import cartopy.crs as crs
import cartopy.feature as crf
import matplotlib.pyplot as plt

import dataBuoy as dbuoy
import dataIceLines as dIL
from params import dataPath
print("\n\n")

dataDir = "Sentinel1/" #Directory in dataPath where this dataset is stored
nbCore = 2

def download(data="all"):
	if data=="all" or data=="raw":
		import sentinelsat as ss
		from sentinelsat import SentinelAPI
		help(ss.SentinelAPI)
		api = ss.SentinelAPI(user='sh-6b7599e8-16f7-4c89-a9d2-6b1dfa2b9b18',
							password="ZolidJNYY3bj1mngXPCGwRDWVJNieRtQ",
							api_url="https://apihub.dataspace.copernicus.eu",
							show_progressbars=True,
							timeout = 20)

		footprint = ss.geojson_to_wkt(ss.read_geojson('./download/sentinel1.geojson'))
		help(api.query)
		products = api.query(footprint,
							date=("20160101", "20240101"),
							limit=3,
							)
		api.download_all(products)

	elif data=="all" or data=="front":
		import subprocess as ss
		savePath = dataPath + dataDir + "Pickle/"
		os.makedirs(savePath, exist_ok=True)

		if not os.path.isfile(savePath + "RossFrontPosition.csv"):
			cmd = f"wget -nc -P {savePath}  -i ./download/Sentinel1Front_url.txt"
			ss.run(cmd, shell=True)

def getData(skipNotProcessed=False):
	import esa_snappy as es
	savePathRaw = dataPath + dataDir + "Raw/"
	savePathPreprocess = dataPath + dataDir + "Preprocess/"
	os.makedirs(savePathRaw, exist_ok=True)
	os.makedirs(savePathPreprocess, exist_ok=True)

	data = []
	for file in os.listdir(savePathRaw):
		rawfilePath = savePathRaw + file
		ncfilePath = savePathPreprocess + file.replace(".SAFE",".nc")

		time = file.split("_")[4][0:8]
		if not os.path.isfile(ncfilePath):
			if skipNotProcessed:
				continue
			try:
				print(f"{file} EXTRACTION")
				xr.DataArray().to_netcdf(ncfilePath)
				product = es.ProductIO.readProduct(rawfilePath)
				w = product.getBand("Intensity_HH").getRasterWidth()
				h = product.getBand("Intensity_HH").getRasterHeight()

				band_data = np.zeros(w * h, np.float32)
				product.getBand("Intensity_HH").readPixels(0, 0, w, h, band_data)
				band_data.shape = h, w

				lon = np.zeros(band_data.shape)
				lat = np.zeros(band_data.shape)

				for x in range(w):
					if x%250==0:
						print(f"{x:05d} / {w}")
					for y in range(h):
						coor = getLatLonFromBand(product, x, y)
						lat[y,x] = coor[0]
						lon[y,x] = coor[1]

				data_array = xr.DataArray(
					band_data,
					dims=("y", "x"),
					coords={"lon": (("y", "x"), lon), "lat": (("y", "x"), lat)},
					name="variable")
				
				os.remove(ncfilePath)
				data_array.to_netcdf(ncfilePath)
				product.closeIO()
				product.closeProductWriter()
				product.closeProductReader()
				del product
				data_array.close()
				print(f"{file} SAVED")
			except Exception as eee:
				os.remove(ncfilePath)
				print(file, "ERROR\n", eee, "\n\n")
			exit()
			
		else:
			print(f"{file} LFX")
		data.append({"time":time, "data":xr.open_dataset(ncfilePath), "file":file})

	data = pd.DataFrame.from_dict(data)			
	return data

def getLatLonFromBand(product, x, y):
	import esa_snappy as es
	geopos = product.getSceneGeoCoding().getGeoPos(es.PixelPos(x, y), None)
	latitude = geopos.getLat()
	longitude = geopos.getLon()
	del geopos
	return latitude, longitude

def makeFrontPositionManual():
	def onclick(event):
		if event.inaxes is not None:
			ax = event.inaxes
			proj = crs.PlateCarree()
			lon, lat = proj.transform_point(event.xdata, event.ydata, ax.projection)
			click_positions.append((lat, lon))
			print(lat, lon)
	
	il = dIL.getData()
	sent = getData(skipNotProcessed=True)
	buoy = dbuoy.getData()

	filePath = dataPath + dataDir + "Pickle/buoyFrontPosition.pkl"
	failPath = dataPath + dataDir + "Pickle/failed.pkl"

	if os.path.isfile(filePath):
		data = pd.read_pickle(filePath)
	else:
		data = pd.DataFrame(columns=["file", "buoy", "time", "lat", "lon"])

	if os.path.isfile(failPath):
		fail = pd.read_pickle(failPath)
	else:
		fail = pd.DataFrame(columns=["file", "buoy"])

	for sentID in sent.index:
		fileName = sent.loc[sentID, "file"]
		time = sent.loc[sentID, "time"]

		for buoyNumber, buoyName in enumerate(buoy.index):
			print(f"{sentID*3+buoyNumber+1:03d} / {3*len(sent)}  {time}", end="   ")

			# print("---", len(data.loc[(data.file==fileName) & (data.buoy==buoyName)]), len(fail.loc[(fail.file==fileName) & (fail.buoy==buoyName)]), end="")
			if len(data.loc[(data.file==fileName) & (data.buoy==buoyName)]) == 1:
				print("POINT Skipped: already done")
				# continue

			if len(fail.loc[(fail.file==fileName) & (fail.buoy==buoyName)]) == 1:
				print("POINT Skipped: already failed")
				continue

			# print('\n\n')
			# print(sent.loc[sentID, "data"])
			# print('\n\n')

			Lon = sent.loc[sentID, "data"]["lon"].values[::1,::1]%360
			Lat = sent.loc[sentID, "data"]["lat"].values[::1,::1]
			Var = sent.loc[sentID, "data"]["variable"].values[::1,::1]

			blat = buoy.loc[buoyName, "lat"]
			blon = buoy.loc[buoyName, "lon"]
			
			distToPoint = np.maximum(np.abs(Lon - blon), np.abs(Lat - blat))
			([idx], [idy]) = np.where(distToPoint == np.min(distToPoint))

			nb = 800
			Lat = Lat[max(idx-nb,0):min(idx+nb, len(Lat)-1), max(idy-nb,0):min(idy+nb, len(Lat)-1)]
			Lon = Lon[max(idx-nb,0):min(idx+nb, len(Lon)-1), max(idy-nb,0):min(idy+nb, len(Lon)-1)]
			Var = Var[max(idx-nb,0):min(idx+nb, len(Var)-1), max(idy-nb,0):min(idy+nb, len(Var)-1)]

			click_positions = []
			lineToAdd = []
			fig = plt.figure(figsize=(16,9), constrained_layout=True)
			fig.canvas.mpl_connect('button_press_event', onclick)
			ax = fig.add_subplot(projection=crs.SouthPolarStereo(central_longitude=180))
			ax.set_extent([blon-0.3, blon+0.3, blat-0.05, blat+0.20])
			ax.pcolormesh(Lon, Lat, Var, transform=crs.PlateCarree(), vmin=10000, vmax=5000000)
			il.loc[il.subflag==buoyName, "gpd"].values[0].plot(ax=ax, color="tab:orange", transform=crs.PlateCarree(), zorder=11 )
			ax.coastlines(zorder=6)
			ax.add_feature(crf.LAND, facecolor=[74/255,47/255,12/255], zorder=2)
			ax.gridlines( draw_labels=True, dms=True, x_inline=False, y_inline=False, rotate_labels=False, zorder=12)
			ax.stock_img()
			ax.set_title(f"1{buoyName}, {sent.loc[sentID, 'time']}")
			plt.show()
			plt.close(fig=fig)
			sent.loc[sentID, "data"].close()

			if len(click_positions) != 0:
				lineToAdd.append({"file":fileName, "buoy":buoyName, "time":time, "lat":click_positions[-1][0], "lon":click_positions[-1][1]})
				data = data.append(lineToAdd)
				# pd.to_pickle(data, filePath)
				print("POINT SAVED")

			else:
				fail = pd.read_pickle(failPath)
				fail = fail.append([{"file":fileName, "buoy":buoyName}], ignore_index=True)
				# pd.to_pickle(fail, failPath)
				print("NO POINT SAVED")

def getFrontPositionManual():
	filePath = dataPath + dataDir + "Pickle/buoyFrontPosition.pkl"
	data = pd.read_pickle(filePath)
	data.time = pd.to_datetime(data.time)
	data.sort_values(["buoy", "time"], ignore_index=True, inplace=True)
	return data

def getTransect(date, lat, lon, U, V):
	data = getData()[date]
	geod = pp(ellps="WGS84")
	Dist = range(0, 20000, 10)
	azimuth = np.degrees(np.arctan(U/V))
	line =  np.array([geod.fwd(lons=lon, lats=lat, az=azimuth, dist=dist) for dist in Dist])[:,[0,1]]
	
	d = data[date]
	alt = []
	for i in range(len(line)):
		abslat = np.abs(data.lat-line[i,1])
		abslon = np.abs(data.lon-line[i,0])
		c = np.maximum(abslon, abslat)
		([yloc], [xloc]) = np.where(c == np.min(c))
		point_ds = d.sel(x=xloc, y=yloc).variable.values
		alt.append(point_ds)

	return Dist, alt

def main():
	getData()
	# makeFrontPositionManual()

if __name__ == '__main__':
	main()
