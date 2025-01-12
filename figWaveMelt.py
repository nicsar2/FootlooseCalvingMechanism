import os
import numpy as np
import inspect as it
import colorcet as cc
import scipy as sp
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as crf

import dataWaveMelt as dt 

from params import figPath, cmap
os.makedirs(figPath, exist_ok=True)

plt.rc('font', size=24)         # controls default text sizes
plt.rc('axes', titlesize=24)    # fontsize of the axes title
plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=21)   # fontsize of the tick labels
plt.rc('ytick', labelsize=21)   # fontsize of the tick labels
plt.rc('legend', fontsize=24)   # legend fontsize
plt.rc('figure', titlesize=24)  # fontsize of the figure title
figSupName = "WaveMelt_"


def compareRegridData(): #Map plots at every time of the SIC and wind speed raw data with the reggridded to the SST grid.
	figTitle = it.stack()[0][3]
	subfigPath = figPath + figTitle + "/"
	os.makedirs(subfigPath, exist_ok=True)

	latSlice = slice(-82,-69)
	lonSlice = slice(160, 210)
	months = range(1,13)
	maskIte = 50

	for year in range(2003, 2023):
		for month in months:
			ds, dsC, dsU = dt.getData(latSlice=latSlice, lonSlice=lonSlice, months=[month], years=[year], maskIte=maskIte, keepVar=True)
			ds = ds.isel(time=0)
			dsU = dsU.isel(time=0).sel(lat=latSlice, lon=lonSlice)
			dsC = dsC.isel(time=0)

			fig = plt.figure(figsize=(16,9), constrained_layout=True)
			gs = fig.add_gridspec(2, 1)
			ax1 = fig.add_subplot(gs[0,0], projection=crs.SouthPolarStereo(central_longitude=180))
			ax2 = fig.add_subplot(gs[1,0], projection=crs.SouthPolarStereo(central_longitude=180))
			ax1.pcolormesh(ds.lon, ds.lat, ds.C, transform=crs.PlateCarree(), vmin=0, vmax=1)
			ax2.pcolormesh(dsC.lon, dsC.lat, dsC.C, transform=crs.PlateCarree(), vmin=0, vmax=1)
			fig.canvas.manager.full_screen_toggle()
			title = f"C_{year}_{month}.png"
			fig.savefig(f"{subfigPath}{title}.png", dpi=300)
			plt.close(fig=fig)

			fig = plt.figure(figsize=(16,9), constrained_layout=True)
			gs = fig.add_gridspec(2, 1)
			ax1 = fig.add_subplot(gs[0,0], projection=crs.SouthPolarStereo(central_longitude=180))
			ax2 = fig.add_subplot(gs[1,0], projection=crs.SouthPolarStereo(central_longitude=180))
			ax1.pcolormesh(ds.lon, ds.lat, ds.w, transform=crs.PlateCarree(), vmin=0, vmax=10)
			ax2.pcolormesh(dsU.lon, dsU.lat, dsU.w, transform=crs.PlateCarree(), vmin=0, vmax=10)
			fig.canvas.manager.full_screen_toggle()
			title = f"U_{year}_{month}.png"
			fig.savefig(f"{subfigPath}{title}.png", dpi=300)
			plt.close(fig=fig)

def mapsSnapshots(): #Map plots at every time of SST, SIC, wind speed, and melt rate.
	figTitle = it.stack()[0][3]
	subfigPath = figPath + figTitle + "/"
	os.makedirs(subfigPath, exist_ok=True)

	latSlice = slice(-82,-69)
	lonSlice = slice(160, 210)
	months = range(1,13)
	maskIte = 50

	for year in range(2003, 2023):
		for month in months:
			ds, dsC, dsU = dt.getData(latSlice=latSlice, lonSlice=lonSlice, months=[month], years=[year], maskIte=maskIte)
			ds = ds.isel(time=0)
			dsC = dsC.isel(time=0)
			dsU = dsU.isel(time=0)

			fig = plt.figure(figsize=(13,9), constrained_layout=True)
			gs = fig.add_gridspec(2, 2)
			ax1 = fig.add_subplot(gs[0,0], projection=crs.SouthPolarStereo(central_longitude=180))
			ax2 = fig.add_subplot(gs[0,1], projection=crs.SouthPolarStereo(central_longitude=180))
			ax3 = fig.add_subplot(gs[1,0], projection=crs.SouthPolarStereo(central_longitude=180))
			ax4 = fig.add_subplot(gs[1,1], projection=crs.SouthPolarStereo(central_longitude=180))

			cs = ax1.pcolormesh(ds.lon, ds.lat, ds.T, zorder=3, transform=crs.PlateCarree(), vmin=-2, vmax=1, cmap=cmap["T"])
			cb = fig.colorbar(cs) 
			cb.set_label("SST (°C)")
			ax1.coastlines(zorder=7)
			ax1.add_feature(crf.LAND, facecolor=[74/255,47/255,12/255], zorder=6)
			ax1.set_extent([163,202,-82,-70])
			ax1.contour(ds.lon, ds.lat, ds.maskFront, transform=crs.PlateCarree(), cmap="inferno", zorder=4)
			cb.ax.tick_params(axis='y')
			ax1.annotate("a)", xy=(163.8, -70.5), color="white", xycoords=crs.PlateCarree()._as_mpl_transform(ax1), ha="left", va="top", zorder=99)

			cs = ax2.pcolormesh(dsC.lon, dsC.lat, dsC.C, zorder=3, transform=crs.PlateCarree(), vmin=0, vmax=1, cmap=cmap["C"])
			cb = fig.colorbar(cs) 
			cb.set_label("Sea ice concentration")
			ax2.coastlines(zorder=7)
			ax2.add_feature(crf.LAND, facecolor=[74/255,47/255,12/255], zorder=6)
			ax2.contour(ds.lon, ds.lat, ds.maskFront, transform=crs.PlateCarree(), cmap="inferno", zorder=4)
			ax2.set_extent([163,202,-82,-70])
			cb.ax.tick_params(axis='y')
			ax2.annotate("b)", xy=(163.8, -70.5), color="white", xycoords=crs.PlateCarree()._as_mpl_transform(ax2), ha="left", va="top", zorder=99)

			cs = ax3.pcolormesh(dsU.lon, dsU.lat, dsU.w, zorder=3, transform=crs.PlateCarree(), vmin=0, vmax=5, cmap=cmap["w"])
			cb = fig.colorbar(cs) 
			cb.set_label("Wind speed (m/s)")
			ax3.coastlines(zorder=7)
			ax3.add_feature(crf.LAND, facecolor=[74/255,47/255,12/255], zorder=6)
			ax3.quiver(dsU.lon.values[::5], dsU.lat.values[::3],\
				dsU.u.values[::3,::5], dsU.v.values[::3,::5],\
				transform=crs.PlateCarree(), zorder=5, color="w", scale=100, width=5e-3)
			ax3.contour(ds.lon, ds.lat, ds.maskFront, transform=crs.PlateCarree(), cmap="inferno", zorder=4)
			ax3.set_extent([163,202,-82,-70])
			cb.ax.tick_params(axis='y')
			ax3.annotate("c)", xy=(163.8, -70.5), color="white", xycoords=crs.PlateCarree()._as_mpl_transform(ax3), ha="left", va="top", zorder=99)

			cs = ax4.pcolormesh(ds.lon, ds.lat, ds.Me, zorder=3, transform=crs.PlateCarree(), vmin=0, vmax=500, cmap=cmap["Me"])
			cb = fig.colorbar(cs) 
			cb.set_label("Melt rate (m/y)")
			ax4.coastlines(zorder=7)
			ax4.add_feature(crf.LAND, facecolor=[74/255,47/255,12/255], zorder=6)
			ax4.contour(ds.lon, ds.lat, ds.maskFront, transform=crs.PlateCarree(), cmap="inferno", zorder=4)
			ax4.set_extent([163,202,-82,-70])
			cb.ax.tick_params(axis='y')
			ax4.annotate("d)", xy=(163.8, -70.5), color="white", xycoords=crs.PlateCarree()._as_mpl_transform(ax4), ha="left", va="top", zorder=99)

			fig.canvas.manager.full_screen_toggle()
			title = f"{year}_{month}.png"
			fig.savefig(f"{subfigPath}{title}.png", dpi=300)
			plt.close(fig=fig)

def mapsWinterMean(): #Map plot of temporal January mean for SST, SIC, wind speed, and melt rate.
	figTitle = "Figure_3"

	latSlice = slice(-82,-69)
	lonSlice = slice(160, 210)
	months = [1]
	maskIte = 50

	ds, dsC, dsU = dt.getData(latSlice=latSlice, lonSlice=lonSlice, months=months, maskIte=maskIte)

	fig = plt.figure(figsize=(13,9), constrained_layout=True)
	gs = fig.add_gridspec(2, 2)
	ax1 = fig.add_subplot(gs[0,0], projection=crs.SouthPolarStereo(central_longitude=180))
	ax2 = fig.add_subplot(gs[0,1], projection=crs.SouthPolarStereo(central_longitude=180))
	ax3 = fig.add_subplot(gs[1,0], projection=crs.SouthPolarStereo(central_longitude=180))
	ax4 = fig.add_subplot(gs[1,1], projection=crs.SouthPolarStereo(central_longitude=180))

	cs = ax1.pcolormesh(ds.lon, ds.lat, ds.T.mean("time"), zorder=3, transform=crs.PlateCarree(), vmin=-2, vmax=1, cmap=cmap["T"])
	cb = fig.colorbar(cs) 
	cb.set_label("SST (°C)")
	ax1.coastlines(zorder=7)
	ax1.add_feature(crf.LAND, facecolor=[74/255,47/255,12/255], zorder=6)
	ax1.set_extent([163, 202,-82,-70])
	cb.ax.tick_params(axis='y')
	ax1.contour(ds.lon, ds.lat, ds.maskFront, transform=crs.PlateCarree(), cmap="inferno", zorder=4)
	ax1.annotate("a)", xy=(163.8, -70.5), color="white", xycoords=crs.PlateCarree()._as_mpl_transform(ax1), ha="left", va="top", zorder=99)
	
	cs = ax2.pcolormesh(dsC.lon, dsC.lat, dsC.C.mean("time"), zorder=3, transform=crs.PlateCarree(), vmin=0, vmax=1, cmap=cmap["C"])
	cb = fig.colorbar(cs) 
	cb.set_label("Sea ice concentration")
	ax2.coastlines(zorder=7)
	ax2.add_feature(crf.LAND, facecolor=[74/255,47/255,12/255], zorder=6)
	ax2.set_extent([163,202,-82,-70])
	cb.ax.tick_params(axis='y')
	ax2.contour(ds.lon, ds.lat, ds.maskFront, transform=crs.PlateCarree(), cmap="inferno", zorder=4)
	ax2.annotate("b)", xy=(163.8, -70.5),color="white", xycoords=crs.PlateCarree()._as_mpl_transform(ax2), ha="left", va="top", zorder=99)

	cs = ax3.pcolormesh(dsU.lon, dsU.lat, dsU.w.mean("time"), zorder=3, transform=crs.PlateCarree(), vmin=0, vmax=5, cmap=cmap["w"])
	cb = fig.colorbar(cs) 
	cb.set_label("Wind speed (m/s)")
	ax3.coastlines(zorder=7)
	ax3.add_feature(crf.LAND, facecolor=[74/255,47/255,12/255], zorder=6)
	ax3.quiver(dsU.lon.values[::5], dsU.lat.values[::3],\
		dsU.u.mean("time").values[::3,::5], dsU.v.mean("time").values[::3,::5],\
		transform=crs.PlateCarree(), zorder=5, color="w", scale=100, width=5e-3)
	ax3.set_extent([163,202,-82,-70])
	cb.ax.tick_params(axis='y')
	ax3.contour(ds.lon, ds.lat, ds.maskFront, transform=crs.PlateCarree(), cmap="inferno", zorder=4)
	ax3.annotate("c)", xy=(163.8, -70.5),color="white", xycoords=crs.PlateCarree()._as_mpl_transform(ax3), ha="left", va="top", zorder=99)

	cs = ax4.pcolormesh(ds.lon, ds.lat, ds.Me.mean("time"), zorder=3, transform=crs.PlateCarree(), vmin=0, vmax=500, cmap=cmap["Me"])
	cb = fig.colorbar(cs) 
	cb.set_label("Melt rate (m/y)")
	ax4.coastlines(zorder=7)
	ax4.add_feature(crf.LAND, facecolor=[74/255,47/255,12/255], zorder=6)
	ax4.contour(ds.lon, ds.lat, ds.maskFront, transform=crs.PlateCarree(), cmap="inferno", zorder=4)
	ax4.set_extent([163,202,-82,-70])
	cb.ax.tick_params(axis='y')
	ax4.annotate("d)", xy=(163.8, -70.5),color="white", xycoords=crs.PlateCarree()._as_mpl_transform(ax4), ha="left", va="top", zorder=99)

	fig.canvas.manager.full_screen_toggle()
	title = f"{figTitle}"
	fig.savefig(f"{figPath}{title}.png", dpi=300)
	plt.close(fig=fig)

def climatology(): #Climatology of the spatial averaged melt
	figName = "Figure_8"
	fig = plt.figure(figsize=(12, 9), constrained_layout=True)
	ax = fig.add_subplot()

	d = dt.getDataBand(meltEq="C1")
	d = ((d.S*d).sum('lon')) / d.S.sum('lon')
	dd = d.groupby("time.month")
	d2 = dt.getDataBand(meltEq="C3")
	d2 = ((d2.S*d2).sum('lon')) / d2.S.sum('lon')
	dd2 = d2.groupby("time.month")

	meanTime = [dd.mean("time").sel(month=m) for m in [8,9,10,11,12,1,2,3,4,5,6,7]]
	stdTime = [dd.std("time").sel(month=m) for m in [8,9,10,11,12,1,2,3,4,5,6,7]]

	ax.plot(range(1,13), meanTime,"-o", color="tab:blue", zorder=5,markersize=11, label="Clim. melt (revised)",  linewidth=2)
	ax.plot(range(1,13), [dd2.mean("time").sel(month=m) for m in [8,9,10,11,12,1,2,3,4,5,6,7]],":",color="gray", zorder=5, label="Clim. melt (original)")
	ax.fill_between(range(1,13), [meanTime[i] - stdTime[i] for i in range(len(stdTime))], [meanTime[i] + stdTime[i] for i in range(len(stdTime))], color="tab:blue", alpha=0.1, zorder=3)

	ax.plot([1,12], [d.mean("time")]*2,color="tab:blue", linewidth=3)
	ax.plot([1,12], [d2.mean("time")]*2,":",color="gray")

	ax.text(8.5, d2.mean('time')+2, f"Ann. mean melt: {d2.mean('time'):.0f} m/yr", fontsize=16, color="gray")
	ax.text(8.5, d.mean('time')-9, f"Ann. mean melt: {d.mean('time'):.0f} m/yr", fontsize=16,color="tab:blue")
	ax.set_ylabel(r"Melt rate (m/yr)")
	ax.set_ylim(0, 350)
	ax.set_xticks([1, 4, 7, 10, 12])
	ax.set_yticks([0, 100, 200, 300])
	ax.legend(fontsize=22)
	ax.set_xticklabels(["August", "November", "February", "May", "July"])
	
	fig.canvas.manager.full_screen_toggle()
	fig.savefig(f"{figPath}{figName}.png", dpi=300)
	plt.close(fig=fig)

def alongFrontMeltPerMonths(): #Climatology of the melt along RIS for every months
	figTitle = "Figure_A5"
	fig = plt.figure(figsize=(12,9), constrained_layout=True)
	ax = fig.add_subplot()

	monthLabels = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr",\
		5:"May", 6:"Jun", 7:"Jul", 8:"Aug",\
		9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}

	d = dt.getDataBand()
	d = d.groupby("time.month").mean("time")
	col = cc.cm.CET_R1(np.linspace(0,1,12))
	for i, month in enumerate([7,8,9,10,11,12,1,2,3,4,5,6]):
		X = d.lon.values
		Y = d.sel(month=month).values
		mask = np.logical_not(np.isnan(Y))
		X, Y = X[mask], Y[mask]
		
		x = np.linspace(X.min(), X.max(), 20000)
		y = np.interp(x, X, Y, right=np.nan)
		b, a = sp.signal.butter(3, 0.003)
		y = sp.signal.filtfilt(b, a, y)
		
		ax.plot(x, y, color=col[i], label=monthLabels[month])

	ax.legend()
	ax.set_xlabel("Longitude (°)")
	ax.set_ylabel(r"Melt ($m\,yr^{-1}$)")
	fig.canvas.manager.full_screen_toggle()
	title = f"{figTitle}."
	fig.savefig(f"{figPath}{title}.png", dpi=300)
	plt.close(fig=fig)

