import os
import inspect as it
import cartopy.feature as crf
import matplotlib.pyplot as plt
import cartopy.crs as crs

import dataSentinel1 as dt
import dataBuoy as dbuoy
import physics as ps

from params import figPath
os.makedirs(figPath, exist_ok=True)
codeName = __file__.split("/")[-1].replace(".py","")[3:]

plt.rc('font', size=24)         # controls default text sizes
plt.rc('axes', titlesize=24)    # fontsize of the axes title
plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=21)   # fontsize of the tick labels
plt.rc('ytick', labelsize=21)   # fontsize of the tick labels
plt.rc('legend', fontsize=24)   # legend fontsize
plt.rc('figure', titlesize=24)  # fontsize of the figure title
figSupName = "SENTINEL1_"

def overimposeAll():
	data = dt.getData(skipNotProcessed=True)
	fig = plt.figure(figsize=(16,9), constrained_layout=True)
	ax = fig.add_subplot(projection=crs.SouthPolarStereo(central_longitude=180))
	for date in data.keys():
		d = data[date]
		ax.pcolormesh(d["lon"][::15,::15], d["lat"][::15,::15], d["variable"][::15,::15], transform=crs.PlateCarree(), vmin=10000, vmax=5000000, alpha=0.3)
		d.close()
	ax.coastlines(zorder=6)
	ax.add_feature(crf.LAND, facecolor=[74/255,47/255,12/255], zorder=2)
	ax.gridlines( draw_labels=True, dms=True, x_inline=False, y_inline=False, rotate_labels=False, zorder=7)
	ax.stock_img()
	ax.set_extent([165, 197, -80.5, -74])
	fig.savefig("plotAllOverimpose.png", dpi=500)
	fig.canvas.manager.full_screen_toggle()
	figTitle = f"{codeName}_{it.stack()[0][3]}"
	fig.savefig(f"{figPath}{figTitle}.png", dpi=300)
	plt.close(fig=fig)

def separatedPhoto():
	data = dt.getData(skipNotProcessed=True)
	buoy = dbuoy.getData().loc["DR03"]
	for idx in data.index:
		date = data.loc[idx, "time"]
		print(date)
		figTitle = f"{codeName}_{it.stack()[0][3]}_{date}"
		if os.path.isfile(f"{figPath}{figTitle}.png"):
			continue
		try:
			d = data.loc[idx, "data"]
			fig = plt.figure(figsize=(16,9), constrained_layout=True)
			ax = fig.add_subplot(projection=crs.SouthPolarStereo(central_longitude=180))
			ax.pcolormesh(d["lon"][::20,::20], d["lat"][::20,::20], d["variable"][::20,::20], transform=crs.PlateCarree(), vmin=10000, vmax=5000000)
			d.close()
			ax.scatter(buoy["lon"], buoy["lat"], marker="x", s=50, color="r", transform=crs.PlateCarree())
			ax.coastlines(zorder=6)
			ax.add_feature(crf.LAND, facecolor=[74/255,47/255,12/255], zorder=2)
			ax.gridlines( draw_labels=True, dms=True, x_inline=False, y_inline=False, rotate_labels=False, zorder=7)
			ax.stock_img()
			ax.set_extent([165, 197, -80.5, -74])
			fig.savefig(f"{figPath}{figTitle}.png", dpi=300)
			plt.close(fig=fig)
		except Exception as eee:
			print(date, '\n', d, '\n',eee,"\n\n\n")

def frontPositionAtBuoy():
	figName = "Figure_A4"
	plt.rc('xtick', labelsize=20)   # fontsize of the tick labels
	data = dt.getFrontPositionManual().sort_values("time", ignore_index=True)
	buoyData = dbuoy.getData()
	color = {"DR01":"green", "DR02":"tab:orange", "DR03":"tab:cyan"}

	fig = plt.figure(figsize=(26,9), constrained_layout=True)
	gs = fig.add_gridspec(1, 3)

	for i, buoyName in enumerate(buoyData.index):
		buoy = buoyData.loc[buoyName]
		time = data.loc[data.buoy==buoyName, "time"].values
		d = ps.sphericalDst(lat1=buoy["lat"], lon1=buoy["lon"], lat2=data.loc[data.buoy==buoyName, "lat"].values, lon2=data.loc[data.buoy==buoyName, "lon"].values)

		ax = fig.add_subplot(gs[i])
		ax.plot(time, (d-d[0])/1e3, "+", color=color[buoyName], markersize=15)
		ax.set_title(buoyName)
		ax.set_xlabel("Time (year)")
		if i==0:
			ax.set_ylabel("Distance (km)")
		ax.set_ylim(-0.1,10)

	plt.rc('xtick', labelsize=21)   # fontsize of the tick labels
	fig.savefig(f"{figPath}{figName}.png", dpi=300)
	plt.close(fig=fig)

def main():
	frontPositionAtBuoy()
	exit()
	d = dt.getFrontPositionManual()
	d = d[d.buoy=="DR01"]
	buoy = dbuoy.getData()
	print(buoy)
	buoy = buoy.loc["DR01"]
	Xsent = d["time"].values
	Ysent = ps.sphericalDst(lat1=buoy["lat"], lon1=buoy["lon"], lat2=d["lat"].values, lon2=d["lon"].values)
	plt.plot(Xsent, Ysent)
	plt.show()
	print(d)

if __name__ == '__main__':
	main()