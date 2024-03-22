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
	figTitle = it.stack()[0][3]

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
	title = f"{figTitle}.png"
	fig.savefig(f"{figPath}{title}.png", dpi=300)
	plt.close(fig=fig)

def climatology(): #Climatology of the spatial averaged melt
	figTitle = it.stack()[0][3]

	maskIte = 50
	latSlice = slice(-79,-76)
	lonSlice = slice(160, 205)
	months = [8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7]
	years = range(2003, 2023)

	ds, _, _ = dt.getData(latSlice=latSlice, lonSlice=lonSlice, months=months, years=years, maskIte=maskIte)

	Me = ((ds.S*ds.Me).where(ds.maskFront).where(ds.maskSIC).sum(['lat','lon'])) / (ds.S.where(ds.maskFront).where(ds.maskSIC).sum(['lat','lon']))
	MeClim = {month:[] for month in months}

	for month in months:
		MeClim[month] += Me.sel(time=[np.datetime64(f'{year}-{month:02d}-01') for year in years]).values.tolist()

	fig = plt.figure(figsize=(12,9), constrained_layout=True)
	ax = fig.add_subplot()
	ax.plot(range(1,13), [np.mean(MeClim[m]) for m in months],"-bs", zorder=5, label="Melt")

	mean = []
	for m in months:
		mean += MeClim[m]
	mean = np.mean(mean)
	ax.plot([1,12],[mean]*2,"-r",label=f"Mean: {mean:.0f} m/yr")

	ax.fill_between(range(1,13), [np.min(MeClim[m]) for m in months], [np.max(MeClim[m]) for m in months], color="grey", alpha=0.2, zorder=3, label="Min-Max")
	ax.fill_between(range(1,13), [np.mean(MeClim[m]) - np.std(MeClim[m]) for m in months]	, [np.mean(MeClim[m]) + np.std(MeClim[m]) for m in months], color="grey", alpha=0.2, zorder=3, label="Standard deviation")

	ax.set_xlabel("Time",size=25)
	ax.set_ylabel("Melt rate (m/yr)",size=25)
	ax.set_ylim(0, 350)
	ax.set_xticks([1, 3, 5, 7, 9, 11])
	ax.legend()
	ax.set_xticklabels(["August", "October", "December", "February", "April", "June"])
	
	fig.canvas.manager.full_screen_toggle()
	title = f"{figTitle}.png"
	fig.savefig(f"{figPath}{title}.png", dpi=300)
	plt.close(fig=fig)

def alongFrontMeltPerMonths(): #Climatology of the melt along RIS for every months
	figTitle = it.stack()[0][3]
	fig = plt.figure(figsize=(12,9), constrained_layout=True)
	ax = fig.add_subplot()

	d = dt.getDataBand()
	d = d.groupby("time.month").mean("time")
	col = cc.cm.CET_R1(np.linspace(0,1,12))
	for i,month in enumerate([7,8,9,10,11,12,1,2,3,4,5,6]):
		X = d.lon.values
		Y = d.sel(month=month).values
		mask = np.logical_not(np.isnan(Y))
		X, Y = X[mask], Y[mask]
		
		x = np.linspace(X.min(), X.max(), 20000)
		y = np.interp(x, X, Y, right=np.nan)
		b, a = sp.signal.butter(3, 0.003)
		y = sp.signal.filtfilt(b, a, y)
		
		ax.plot(x, y, color=col[i], label=month)

	ax.legend()
	ax.set_xlabel("Longitude (°)")
	ax.set_ylabel(r"Melt ($m.yr^{-1}$)")
	fig.canvas.manager.full_screen_toggle()
	title = f"{figTitle}_.png"
	fig.savefig(f"{figPath}{title}.png", dpi=300)
	plt.close(fig=fig)

def combinedClimatology(): #FIGURE 7: climatology + alongFrontMeltPerMonths combined
	figTitle = it.stack()[0][3]
	fig = plt.figure(figsize=(20, 9), constrained_layout=True)
	gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1], wspace=0.010)
	ax1 = fig.add_subplot(gs[0,0])
	ax2 = fig.add_subplot(gs[0,1])

	d = dt.getDataBand()
	d = d.groupby("time.month").mean("time")
	monthLabels={1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr",\
		5:"May", 6:"Jun", 7:"Jul", 8:"Aug",\
		9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}

	col = plt.cm.jet(np.linspace(0, 1, 12))

	for i, month in enumerate([1,2,3,4,5,6,7,8,9,10,11,12]):
		X = d.lon.values
		Y = d.sel(month=month).values
		S = d.S.values
		mask = np.logical_not(np.isnan(Y))
		X, Y, S = X[mask], Y[mask], S[mask]
		
		x = np.linspace(X.min(), X.max(), 20000)
		y = np.interp(x, X, Y, right=np.nan)
		b, a = sp.signal.butter(3, 0.003)
		y = sp.signal.filtfilt(b, a, y)
		
		ax1.plot(x, y, color=col[i], label=monthLabels[month])
		# ax1.scatter([200], [(Y*S).sum()/(S.sum())], marker="s", color=col[i])

	ax1.legend(ncol=3)
	ax1.set_ylim(0, 500)
	ax1.set_xlabel("Longitude (° E)")
	ax1.set_ylabel(r"Melt rate (m/yr)")
	ax1.set_yticks([0, 100, 200, 300, 400, 500])
	ax1.annotate("a)", xy=(ax1.get_xlim()[0], ax1.get_ylim()[1]), xytext=(3, -4),textcoords='offset points', ha="left", va="top")

	maskIte = 50
	latSlice = slice(-79,-76)
	lonSlice = slice(160, 205)
	months = [8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7]
	years = range(2003, 2023)

	ds, _, _ = dt.getData(latSlice=latSlice, lonSlice=lonSlice, months=months, years=years, maskIte=maskIte)

	Me = ((ds.S*ds.Me).where(ds.maskFront).where(ds.maskSIC).sum(['lat','lon'])) / (ds.S.where(ds.maskFront).where(ds.maskSIC).sum(['lat','lon']))
	MeClim = {month:[] for month in months}

	for month in months:
		MeClim[month] += Me.sel(time=[np.datetime64(f'{year}-{month:02d}-01') for year in years]).values.tolist()

	ax2.plot(range(1,13), [np.mean(MeClim[m]) for m in months],"-k", zorder=5)
	ax2.scatter(range(1,13), [np.mean(MeClim[m]) for m in months], marker="s", color=np.roll(col,5,axis=0), zorder=6, label="Melt")

	mean = []
	for m in months:
		mean += MeClim[m]
	mean = np.mean(mean)
	ax2.plot([1,12],[mean]*2,"-r",label=f"Mean: {mean:.0f} m/yr")

	ax2.fill_between(range(1,13), [np.min(MeClim[m]) for m in months], [np.max(MeClim[m]) for m in months], color="grey", alpha=0.2, zorder=3, label="Min-Max")
	ax2.fill_between(range(1,13), [np.mean(MeClim[m]) - np.std(MeClim[m]) for m in months]	, [np.mean(MeClim[m]) + np.std(MeClim[m]) for m in months], color="grey", alpha=0.2, zorder=3, label="Standard deviation")

	ax2.set_ylim(0, 500)
	ax2.set_xticks([1, 4, 7, 10, 12])
	ax2.set_yticks([0, 100, 200, 300, 400, 500])
	ax2.set_yticklabels(["", "", "", "", "", ""])
	ax2.legend()
	ax2.set_xticklabels(["August", "November", "February", "May", "July"])
	ax2.annotate("b)", xy=(ax2.get_xlim()[0], ax2.get_ylim()[1]), xytext=(3, -4),textcoords='offset points', ha="left", va="top")
	
	fig.canvas.manager.full_screen_toggle()
	title = f"{figTitle}.png"
	fig.savefig(f"{figPath}{title}.png", dpi=300)
	plt.close(fig=fig)
