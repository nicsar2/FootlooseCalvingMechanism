import os
import numpy as np
import scipy as sp
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as crf

import dataWaveMelt as dWVM
import dataMEaSUREs2 as dVEL
import dataICESAT as dISA
import dataBuoy as dbuoy
import dataSentinel1 as dSent

from params import dataPath, Rt 
import physics as ps
from params import figPath
os.makedirs(figPath, exist_ok=True)

dataDir = "Pickles/"

def figCumulativeRates():
	figTitle = "figCumulativeRates"
	dsISA = dISA.getData().sort_values("mTime")

	fig = plt.figure(figsize=(16,9), constrained_layout=True)
	ax = fig.add_subplot()

	Xf,Yf = [], []
	lon = []
	lat = []
	for i in dsISA.index:
		r, f = dsISA.loc[i,"lat"]*Rt*np.pi/180/1000, dsISA.loc[i,"f"]
		if "r0" not in locals():
			r0 = r[np.argmin(f>7)]
		r = r-r0
		Yf.append(r[np.argmin(f>7)])
		Xf.append(dsISA.loc[i,"mTime"])
		lon.append(dsISA.loc[i,"lon"][np.abs(r)==0])
		lat.append(dsISA.loc[i,"lat"][np.abs(r)==0])

	ax.plot(Xf, Yf, "-k", zorder=1)
	ax.scatter(Xf, Yf, c=[Xf[i]-Xf[0] for i in range(len(Xf))], s=250, cmap="jet", zorder=2)

	lon=lon[0][0]
	lat=lat[0][0]
	print(lat, lon)
	d = dVEL.getDataBand()
	vel = d.sel(lon=lon, method="nearest").values
	vel = 1100
	print(vel)
	Xv = np.linspace(np.min(Xf), np.max(Xf),10000)
	Yv = vel*(Xv-Xv[0])/1000
	ax.plot(Xv, Yv, "r")

	ax.set_xlabel("Time")
	ax.set_ylabel("Projected ATD (km)")
	title = f"{figTitle}.png"
	fig.savefig(f"../../Fig/{title}", dpi=300)
	plt.show()

def figCompareBuoyIcelineMeasureMeltBar():
	plt.rc('axes', labelsize=19)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=14)   # fontsize of the tick labels
	plt.rc('ytick', labelsize=14)   # fontsize of the tick labels
	melt = dWVM.getDataBand(meltEq="C1")
	meas = dVEL.getDataBand()
	sent_ = dSent.getFrontPositionManual()
	buoy_ = dbuoy.getData()
	fig = plt.figure(figsize=(16,9), constrained_layout=True)
	gs = fig.add_gridspec(5, 3)


	dylim = 150
	color = {"DR01":"green", "DR02":"tab:orange", "DR03":"tab:cyan"}
	color2 = {"buoy":"red", "meas":"tab:purple", "sent":"blue", "melt":"tab:brown", "iceline":"tab:olive"}
	ax = fig.add_subplot(gs[0,:], projection=crs.SouthPolarStereo(central_longitude=180))
	for buoyName in buoy_.index:
		ax.scatter(buoy_.loc[buoyName, "lon"], buoy_.loc[buoyName, "lat"],color=color[buoyName], transform=crs.PlateCarree(), s=75, zorder=10)
	d = xr.open_dataset(f"{dataPath}Sentinel1/Preprocess/S1B_EW_GRDM_1SDH_20181230T110848_20181230T111003_014270_01A8B2_ED45.nc")
	ax.pcolormesh(d["lon"][::3,::3], d["lat"][::3,::3], xr.where(d["variable"][::3,::3]<=40000,250000000, d["variable"][::3,::3]), transform=crs.PlateCarree(), vmin=30000, vmax=2300000, zorder=8, cmap="gist_earth")
	d = xr.open_dataset(f"{dataPath}Sentinel1/Preprocess/S1A_EW_GRDM_1SDH_20181231T110045_20181231T110149_025268_02CB5F_FBD1.nc")
	ax.pcolormesh(d["lon"][::3,::3], d["lat"][::3,::3], xr.where(d["variable"][::3,::3]<=70000,250000000, d["variable"][::3,::3]), transform=crs.PlateCarree(), vmin=70000, vmax=2300000, zorder=9, cmap="gist_earth")
	d = xr.open_dataset(f"{dataPath}Sentinel1/Preprocess/S1A_EW_GRDM_1SDH_20190103T094628_20190103T094732_025311_02CCE9_8E88.nc")
	ax.pcolormesh(d["lon"][::3,::3], d["lat"][::3,::3], xr.where(d["variable"][::3,::3]<=70000,250000000, d["variable"][::3,::3]), transform=crs.PlateCarree(), vmin=70000, vmax=2300000, zorder=7, cmap="gist_earth")
	ax.set_extent([175, 180, -78.8, -77.5], crs=crs.PlateCarree())
	ax.coastlines(zorder=6)
	ax.set_aspect(1)
	ax.set_adjustable('datalim')
	ax.set_facecolor('white')
	ax.add_feature(crf.LAND, facecolor=[74/255,47/255,12/255], zorder=10)
	gl = ax.gridlines( draw_labels=True, dms=True, x_inline=False, y_inline=False,\
				rotate_labels=False, zorder=12)
	gl.bottom_labels = False
	gl.xlabel_style = {'size': 15, 'color': 'gray'}
	gl.ylabel_style = {'size': 15, 'color': 'gray'}


	ax1 = fig.add_subplot(gs[1:,0])
	buoyName = "DR01"
	buoy = buoy_.loc[buoyName]
	sent = sent_.loc[sent_.buoy==buoyName]
	buoyVel = (buoy["V"]**2 + buoy["U"]**2)**0.5
	buoySig = ((buoy["V"]*buoy["sV"]/buoyVel)**2 + (buoy["U"]*buoy["sU"]/buoyVel)**2)**0.5
	u = meas.u.sel(lon=buoy["lon"], method='nearest').values
	v = meas.v.sel(lon=buoy["lon"], method='nearest').values
	measVel = ((u**2 + v**2 )**0.5)
	measSig = ( (u*meas.su.sel(lon=buoy["lon"], method='nearest').values/measVel)**2 + (v*meas.sv.sel(lon=buoy["lon"], method='nearest').values/measVel)**2 )**0.5

	Xsent = sent["time"].values
	Ysent = ps.sphericalDst(lat1=buoy["lat"], lon1=buoy["lon"], lat2=sent["lat"].values, lon2=sent["lon"].values)
	velSent = sp.optimize.curve_fit(lambda x,a: a*x, (Xsent-np.min(Xsent)) / np.timedelta64(1, 's')/365.25/24/3600, Ysent-Ysent[0])
	XXX = (Xsent-np.min(Xsent)) / np.timedelta64(1, 's')/365.25/24/3600
	YYY = Ysent-Ysent[0]

	velSentRed = []
	for i in range(250000):
		YYY_ = YYY + np.random.normal(loc=0, scale=200, size=XXX.shape)
		velSentRed.append(sp.optimize.curve_fit(lambda x,a: a*x, XXX, YYY_)[0][0])


	sigSent = np.std(velSentRed)
	velSent = velSent[0][0]
	velSentMelt = buoyVel - melt.sel(lon=buoy["lon"], method="nearest").mean("time").values
	meltSig = melt.sel(lon=buoy["lon"], method="nearest").groupby("time.year").mean("time").std("year").values

	ax1.plot([-9,9], [buoyVel]*2, "--", color=color2["buoy"])
	ax1.plot([-9,9], [velSent]*2, "--", color=color2["sent"])
	ax1.annotate(text='', xy=(1,buoyVel), xytext=(1,velSent), arrowprops=dict(arrowstyle='<-'))
	ax1.annotate(text=f'Ablation   {buoyVel-velSent:.1f}' + r' $\mathrm{m}\,\mathrm{yr}^{-1}$', xy=(0.15 , (buoyVel+velSent)/2-1.3), fontsize=14)


	ax1.errorbar(0, buoyVel, yerr=buoySig, color=color2["buoy"], marker=".", markersize=15, capsize=5, linestyle='None')
	ax1.errorbar(0, measVel, yerr=measSig, color=color2["meas"], marker="^", markersize=12, capsize=5, linestyle='None')
	ax1.errorbar(2, velSent, yerr=sigSent, color=color2["sent"], marker=".", markersize=15, capsize=5, linestyle='None')
	pl = ax1.errorbar(2, velSentMelt, yerr=meltSig, color=color2["melt"], marker=".", markersize=15, capsize=0, fillstyle='none', linestyle='None')
	pl[-1][0].set_linestyle("--")
	ax1.set_title(buoyName, fontsize=16)
	ax1.set_xticks([0, 2])
	ax1.set_xticklabels(["Ice flow\nvelocity, V", "Frontal advance\nvelocity, F"], fontsize=18)
	ax1.set_xlim(-1,3)
	ax1.set_ylim(900, 900+dylim)
	ax1.set_ylabel(r"Velocity ($\mathrm{m}\,\mathrm{yr}^{-1}$)")

	ax2 = fig.add_subplot(gs[1:,1])
	buoyName = "DR02"
	buoy = buoy_.loc[buoyName]
	sent = sent_.loc[sent_.buoy==buoyName]
	buoyVel = (buoy["V"]**2 + buoy["U"]**2)**0.5
	buoySig = ((buoy["V"]*buoy["sV"]/buoyVel)**2 + (buoy["U"]*buoy["sU"]/buoyVel)**2)**0.5
	u = meas.u.sel(lon=buoy["lon"], method='nearest').values
	v = meas.v.sel(lon=buoy["lon"], method='nearest').values
	measVel = ((u**2 + v**2 )**0.5)
	measSig = ( (u*meas.su.sel(lon=buoy["lon"], method='nearest').values/measVel)**2 + (v*meas.sv.sel(lon=buoy["lon"], method='nearest').values/measVel)**2 )**0.5

	Xsent = sent["time"].values
	Ysent = ps.sphericalDst(lat1=buoy["lat"], lon1=buoy["lon"], lat2=sent["lat"].values, lon2=sent["lon"].values)
	velSent = sp.optimize.curve_fit(lambda x,a: a*x, (Xsent-np.min(Xsent)) / np.timedelta64(1, 's')/365.25/24/3600, Ysent-Ysent[0])
	XXX = (Xsent-np.min(Xsent)) / np.timedelta64(1, 's')/365.25/24/3600
	YYY = Ysent-Ysent[0]

	velSentRed = []
	for i in range(250000):
		YYY_ = YYY + np.random.normal(loc=0, scale=200, size=XXX.shape)
		velSentRed.append(sp.optimize.curve_fit(lambda x,a: a*x, XXX, YYY_)[0][0])

	sigSent = np.std(velSentRed)
	velSent = velSent[0][0]
	velSentMelt = buoyVel - melt.sel(lon=buoy["lon"], method="nearest").mean("time").values
	meltSig = melt.sel(lon=buoy["lon"], method="nearest").groupby("time.year").mean("time").std("year").values
	
	ax2.plot([-9,9], [buoyVel]*2, "--", color=color2["buoy"])
	ax2.plot([-9,9], [velSent]*2, "--", color=color2["sent"])
	ax2.annotate(text='', xy=(1,buoyVel), xytext=(1,velSent), arrowprops=dict(arrowstyle='<-'))
	ax2.annotate(text=f'{buoyVel-velSent:.1f}' + r' $\mathrm{m}\,\mathrm{yr}^{-1}$', xy=(1.1, (buoyVel+velSent)/2-2.30), fontsize=14)


	ax2.errorbar(0, buoyVel,yerr=buoySig, color=color2["buoy"], marker=".", markersize=15, capsize=5, linestyle='None')
	ax2.errorbar(0, measVel, yerr=measSig, color=color2["meas"], marker="^", markersize=12, capsize=5, linestyle='None')
	ax2.errorbar(2, velSent, yerr=sigSent , color=color2["sent"], marker=".", markersize=15, capsize=5, linestyle='None')
	pl = ax2.errorbar(2, velSentMelt, yerr=meltSig, color=color2["melt"], marker=".", markersize=15, capsize=0, fillstyle='none', linestyle='None')
	pl[-1][0].set_linestyle("--")
	ax2.set_title(buoyName, fontsize=16)
	ax2.set_xticks([0, 2])
	ax2.set_xticklabels(["Ice flow\nvelocity, V", "Frontal advance\nvelocity, F"], fontsize=18)
	ax2.set_xlim(-1,3)
	ax2.set_ylim(1000, 1000+dylim)
	ax2.legend()


	ax3 = fig.add_subplot(gs[1:,2])
	buoyName = "DR03"
	buoy = buoy_.loc[buoyName]
	sent = sent_.loc[sent_.buoy==buoyName]
	buoyVel = (buoy["V"]**2 + buoy["U"]**2)**0.5
	buoySig = ((buoy["V"]*buoy["sV"]/buoyVel)**2 + (buoy["U"]*buoy["sU"]/buoyVel)**2)**0.5
	u = meas.u.sel(lon=buoy["lon"], method='nearest').values
	v = meas.v.sel(lon=buoy["lon"], method='nearest').values
	measVel = ((u**2 + v**2 )**0.5)
	measSig = ( (u*meas.su.sel(lon=buoy["lon"], method='nearest').values/measVel)**2 + (v*meas.sv.sel(lon=buoy["lon"], method='nearest').values/measVel)**2 )**0.5

	velicelRed = []
	for i in range(250000):
		YYY_ = YYY + np.random.normal(loc=0, scale=120, size=XXX.shape)
		velicelRed.append(sp.optimize.curve_fit(lambda x,a: a*x, XXX, YYY_)[0][0])


	Xsent = sent["time"].values
	Ysent = ps.sphericalDst(lat1=buoy["lat"], lon1=buoy["lon"], lat2=sent["lat"].values, lon2=sent["lon"].values)
	velSent = sp.optimize.curve_fit(lambda x,a: a*x, (Xsent-np.min(Xsent)) / np.timedelta64(1, 's')/365.25/24/3600, Ysent-Ysent[0])
	XXX = (Xsent-np.min(Xsent)) / np.timedelta64(1, 's')/365.25/24/3600
	YYY = Ysent-Ysent[0]
	velSentRed = []
	for i in range(250000):
		YYY_ = YYY + np.random.normal(loc=0, scale=200, size=XXX.shape)
		velSentRed.append(sp.optimize.curve_fit(lambda x,a: a*x, XXX, YYY_)[0][0])

	sigSent = np.std(velSentRed)
	velSent = velSent[0][0]
	velSentMelt = buoyVel - melt.sel(lon=buoy["lon"], method="nearest").mean("time").values
	meltSig = melt.sel(lon=buoy["lon"], method="nearest").groupby("time.year").mean("time").std("year").values
	
	ax3.plot([-9,9], [buoyVel]*2, "--", color=color2["buoy"])
	ax3.plot([-9,9], [velSent]*2, "--", color=color2["sent"])
	ax3.annotate(text='', xy=(1,buoyVel), xytext=(1,velSent), arrowprops=dict(arrowstyle='<-'))
	ax3.annotate(text=f'{buoyVel-velSent:.1f}' + r' $\mathrm{m}\,\mathrm{yr}^{-1}$', xy=(1.1, (buoyVel+velSent)/2-1.45), fontsize=14)


	ax3.errorbar(0, buoyVel, yerr=buoySig, color=color2["buoy"], marker=".", markersize=15, label="GPS station, V", capsize=5, linestyle='None')
	ax3.errorbar(0, measVel, yerr=measSig, color=color2["meas"], marker="^", markersize=12, label="MEaSUREs", capsize=5, linestyle='None')
	ax3.errorbar(2, velSent, yerr=sigSent, color=color2["sent"], marker=".", markersize=15, label=r"Sentinel 1, $F^{obs}$", capsize=5, linestyle='None')
	pl = ax3.errorbar(2, velSentMelt, yerr=meltSig, color=color2["melt"], marker=".", markersize=15, label=r"Est. frontal advance, $F^{est}$", capsize=0, fillstyle='none', linestyle='None')
	pl[-1][0].set_linestyle("--")

	handles, labels = ax3.get_legend_handles_labels()
	ax2.legend(handles=handles, labels=labels, fontsize=14)
	ax3.set_title(buoyName, fontsize=16)
	ax3.set_xticks([0, 2])
	ax3.set_xticklabels(["Ice flow\nvelocity, V", "Frontal advance\nvelocity, F"], fontsize=18)
	ax3.set_xlim(-1,3)
	ax3.set_ylim(900, 900+dylim)
	
	ax1.annotate("b)", xy=(ax1.get_xlim()[0], ax1.get_ylim()[1]), xytext=(3, -4), textcoords='offset points', ha="left", va="top", fontsize=16)
	ax2.annotate("c)", xy=(ax2.get_xlim()[0], ax2.get_ylim()[1]), xytext=(3, -4), textcoords='offset points', ha="left", va="top", fontsize=16)
	ax3.annotate("d)", xy=(ax3.get_xlim()[0], ax3.get_ylim()[1]), xytext=(3, -4), textcoords='offset points', ha="left", va="top", fontsize=16)

	import matplotlib as mp
	l1 = mp.patches.FancyArrow(0.520, 0.932, -0.324, -0.139, color=color["DR01"],
                            transform=fig.transFigure, figure=fig)
	fig.patches.append(l1)
	l1 = mp.patches.FancyArrow(0.566, 0.925, -0.053, -0.128, color=color["DR02"],
                            transform=fig.transFigure, figure=fig)
	fig.patches.append(l1)
	l1 = mp.patches.FancyArrow(0.607, 0.872, 0.220, -0.077, color=color["DR03"],
                            transform=fig.transFigure, figure=fig)
	fig.patches.append(l1)

	fig.text(0.072, 0.955, 'a)',
    horizontalalignment='center',
    verticalalignment='center',
    fontsize=18, color='white',
    transform=fig.transFigure)

	title = "Figure_4"
	fig.savefig(f"{figPath}{title}.png", dpi=300)
	plt.close(fig=fig)
