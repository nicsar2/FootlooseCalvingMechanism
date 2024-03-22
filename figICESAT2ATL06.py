import os
import inspect as it
import numpy as np
import scipy as sp
import colorcet as cc
import cartopy.feature as crf
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.pylab as pll
import cartopy.crs as crs

import dataICESAT2ATL06 as dt

from params import figPath, rhotio
os.makedirs(figPath, exist_ok=True)

plt.rc('font', size=24)         # controls default text sizes
plt.rc('axes', titlesize=24)    # fontsize of the axes title
plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=21)   # fontsize of the tick labels
plt.rc('ytick', labelsize=21)   # fontsize of the tick labels
plt.rc('legend', fontsize=24)   # legend fontsize
plt.rc('figure', titlesize=24)  # fontsize of the figure title


def RMProfile(): #Overlap ICESat2 transect ID = 215 with the elastic beam theory
	figTitle = "ICESAT2_"+it.stack()[0][3]
	df = dt.getData()
	i = 215
	dfi = df.loc[i]
	r, f = dt.getNormalizedRF(dfi)

	x = np.linspace(r.min(),r.max(),50000)
	fct = lambda x : dfi.wrm_o/(1 + np.exp(-3*np.pi/4)/2**0.5)*np.cos(x/(2**0.5)/dfi.lw_o)*np.exp(-x/(2**0.5)/dfi.lw_o)
	y = fct(x)
	y = y - y[0] + f[0]
	
	fig = plt.figure(figsize=(16,9), constrained_layout=True)
	ax = fig.add_subplot()
	ax.plot(x,y,"k",label="Beam model")
	ax.plot(r,f,"r-.",label="ICESat-2 transect")

	axi = ax.inset_axes([0.33+0.03, 0.5+0.03, 0.5-0.03, 0.45-0.03])
	axi.fill_between(np.linspace(-10000,x.max(),50000), 2*np.min(-y*rhotio/(1-rhotio)), 0, color="b", alpha=0.5)
	axi.fill_between(x, y-dfi.h0, y, color="c", alpha=0.9)
	axi.plot(x, y-dfi.h0, "k")
	axi.plot(x, y, "k")
	axi.plot([x[0]]*100, np.linspace(np.min(y-dfi.h0), np.max(y), 100), "k")
	axi.set_xlim(1500,-100)
	axi.set_ylim(np.min((y-dfi.h0))*1.1, np.max(y)*1.3)
	axi.plot(r, f, "r-.")
	axi.set_ylim(-10, 60)
	axi.tick_params(labelsize=14)

	ax.set_xlim(1500, -50)
	ax.set_ylim(f[r<3000].min()*0.98, 60)
	ax.set_xlabel("Distance from front (m)")
	ax.set_ylabel("Ice freeboard (m)")
	ax.fill_between(x, 40, y, color="c", alpha=0.9)
	ax.plot([x[0]]*100, np.linspace(np.min(y-dfi.h0), np.max(y), 100), "k")

	ax.legend(handles=[
		mp.lines.Line2D([0], [0], color="k", label="Beam model:"),
		mp.lines.Line2D([0], [0], color="w", label=r"   $\bullet\ l_f = 21$ "+"m"),
		mp.lines.Line2D([0], [0], color="w", label=r"   $\bullet\ E = 1.5$ "+"MPa"),
		mp.lines.Line2D([0], [0], color="r",linestyle="-.", label="ICESat-2 transect"),
		], ncol=1, loc="upper left")

	fig.canvas.manager.full_screen_toggle()
	title = f"{figTitle}"
	fig.savefig(f"{figPath}{title}.png", dpi=300)
	plt.close(fig=fig)

def Overimpose(): #FIGURE 4: overlap of all normalized ICESat2 transects with the beam theory. Additional distribution plots of the observed moat positions and the observed rampart heights.
	figTitle = "ICESAT2_"+it.stack()[0][3]
	df = dt.getData()

	fig = plt.figure(figsize=(20,9), constrained_layout=True)
	gs = fig.add_gridspec(2, 3, width_ratios=[1.4, 1.4, 1], height_ratios=[1,1] )
	ax = fig.add_subplot(gs[:,0:2])
	ax1 = fig.add_subplot(gs[0, 2])
	ax2 = fig.add_subplot(gs[1, 2])
	cmapName = pll.cm.gist_heat_r
	cMin, cMax = 0, 12
	cNbPoints = int(1e8)
	cmap = cmapName(np.linspace(0, 1, cNbPoints))
	Index = df[df.flag==2].index
	nbRemovedTransect = 0
	Xrm = []
	Wrm = []
	for i in Index:
		if i==3440:
			nbRemovedTransect+=1
			continue
		dfi = df.loc[i]
		r, f = dt.getNormalizedRF(dfi)
		if dfi.wrm_o < 2:
			nbRemovedTransect+=1
			continue
		r /= dfi.lw_o
		f /= dfi.wrm_o/(1 + np.exp(-3*np.pi/4)/2**0.5)
		f -= f[0]-1 
		if f[r<1].max()>1.025:
			nbRemovedTransect+=1
			continue
		color = cmap[min(int(cNbPoints*(dfi.wrm_o-cMin)/(cMax-cMin)), cNbPoints-1)]
		ax.plot(r, f, color=color, alpha=0.15, zorder=4)
		Wrm.append(dfi.wrm_o)
		Xrm.append(dfi.xrm_o)
	print(f"Removed {nbRemovedTransect} / {len(df[df.flag==2])}")
	ax1.hist(Xrm, bins=30, color="tab:red")
	ax1.yaxis.tick_right()
	ax1.yaxis.set_label_position("right")
	ax1.set_ylabel("Count")
	ax1.set_xlabel(r"Moat position: $x_{RM}$ (m)")
	ax2.hist(Wrm, bins=30, color="tab:red")
	ax2.yaxis.tick_right()
	ax2.yaxis.set_label_position("right")
	ax2.set_ylabel("Count")
	ax2.set_xlabel(r"Rampart height: $w_{RM}$ (m)")
	sm = mp.cm.ScalarMappable(cmap=cmapName, norm=mp.colors.Normalize(vmin=cMin, vmax=cMax)) 
	cb = fig.colorbar(sm, ax=ax)
	x = np.linspace(0,20,10000)
	y = np.exp(-x/2**0.5)*np.cos(x/2**0.5)
	ax.plot(x,y,"k", zorder=6, linewidth=3)	
	ax.plot([0,10], [0]*2, "-.", color="gray", zorder=2)
	xrm = 3*np.pi/2/2**0.5
	ax.plot([xrm]*2,   [-0.3, y.min()], ":", color="gray", zorder=2)
	ax.plot([xrm/3]*2, [-0.3, 0.32],    ":", color="gray", zorder=2)
	ax.text(xrm,   -0.45, r"$X_{RM}$", horizontalalignment="center")
	ax.text(xrm/3, -0.45, r"$X_{RM}/3$", horizontalalignment="center")
	ax.set_xlim(10, -0.5)
	ax.set_ylim(-0.5, 3)
	ax.annotate("a)", xy=(ax.get_xlim()[0], ax.get_ylim()[1]), xytext=(3, -4),textcoords='offset points', ha="left", va="top")
	ax1.annotate("b)", xy=(ax1.get_xlim()[0], ax1.get_ylim()[1]), xytext=(3, -4),textcoords='offset points', ha="left", va="top")
	ax2.annotate("c)", xy=(ax2.get_xlim()[0], ax2.get_ylim()[1]), xytext=(3, -4),textcoords='offset points', ha="left", va="top")
	ax.set_xlabel(r"Normalized distance from front: $x/l_w$")
	ax.set_ylabel(r"Normalized deflection: $w/w\left(0\right)$")
	cb.set_label(r"Rampart height: $w_{RM}$ (m)"+"\n")
	fig.canvas.manager.full_screen_toggle()
	title = f'{figTitle}'
	fig.savefig(f"{figPath}{title}.png", dpi=300)
	plt.close(fig=fig)
	
	fig = plt.figure(figsize=(20,9), constrained_layout=True)
	ax = fig.add_subplot()
	cmapName = pll.cm.gist_heat_r
	cMin, cMax = 0, 12
	cNbPoints = int(1e8)
	cmap = cmapName(np.linspace(0, 1, cNbPoints))
	Index = df[df.flag==2].index
	nbRemovedTransect = 0
	for i in Index:
		if i==3440:
			nbRemovedTransect+=1
			continue
		dfi = df.loc[i]
		r, f = dt.getNormalizedRF(dfi)
		if dfi.wrm_o < 2:
			nbRemovedTransect+=1
			continue
		if f[r<dfi.lw_o].max()-f[0]>1.025:
			nbRemovedTransect+=1
			continue
		f -=  dfi.fxrm_o
		color = cmap[min(int(cNbPoints*(dfi.wrm_o-cMin)/(cMax-cMin)), cNbPoints-1)]
		ax.plot(r, f, color=color, alpha=0.15, zorder=4)
	print(nbRemovedTransect)
	ax.set_xlim(1500, -30)
	ax.set_ylim(-2, 20)
	ax.set_xlabel(r"Distance from front: $x$ (m)", fontsize=36)
	ax.set_ylabel(r"Deflection: $w$ (m)", fontsize=36)
	ax.tick_params('both', labelsize=36)
	fig.canvas.manager.full_screen_toggle()
	title = f'{figTitle}_sub'
	fig.savefig(f"{figPath}{title}.png", dpi=300)
	plt.close(fig=fig)

def RMInfo(): #FIGURE 5, 6, S3: series of plots of the various rampart-moat characteristics.
	figTitle = "ICESAT2_"+it.stack()[0][3]
	figPath_ = figPath + "RMInfo/"
	os.makedirs(figPath_, exist_ok=True)

	wrmMin = 1
	df = dt.getData(wrmMin)
	df = df[df.flag==2].sort_values("wrm_o", ignore_index=True)
	epsLin = df.attrs.get("epsLin")
	epsQuad = df.attrs.get("epsQuad")
	mask = df.wrm_o > wrmMin
	mask_ = df.wrm_o <= wrmMin
	H0 = df.h0
	Xrm_o = df.xrm_o
	Lf_s1 = df.lf_s1
	Lf_s2 = df.lf_s2
	Lw_s1 = df.lw_s1
	Lw_s2 = df.lw_s2
	Lw_o = df.lw_o
	Lw_t = df.lw
	Wrm_o = df.wrm_o

	fig = plt.figure(figsize=(16,9), constrained_layout=True)
	ax = fig.add_subplot()
	vmax = 30
	cmap = cc.cm.CET_R1.resampled(vmax*1000)
	newcolors = cmap(np.linspace(0, 1, vmax*1000))
	newcolors[:int(Lf_s2[mask].min()*1000), :] = np.array([97/256, 97/256, 97/256, 0.1])
	newcmap = mp.colors.ListedColormap(newcolors)
	ax.scatter(H0[mask_], Xrm_o[mask_], color="#616161", alpha=0.4)
	cs = ax.scatter(H0[mask], Xrm_o[mask], c=Lf_s2[mask], cmap=newcmap, vmin=0, vmax=vmax, alpha=0.7)
	cb = fig.colorbar(cs, ticks = [0, 1, 5, 10, 15, 20, 25, 30], ax=ax, pad=0.007, label=r"Foot length: $l_f$ (m)")
	H0_ = np.linspace(H0.min(), H0.max(), 10000)
	ax.plot(H0_, dt._xrm_s(H0_, epsLin, 1), "--", color="gray", label=r"$x^{*}_{RM}$ with $h^{*}\sim h$", linewidth=2)
	ax.plot(H0_, dt._xrm_s(H0_, epsQuad, 2), "k--", label=r"$x^{*}_{RM}$ with $h^{*} \sim h^2$", linewidth=2)
	ax.set_xlabel(r"Front thickness: $h$ (m)")
	ax.set_ylabel(r"Moat position: $x_{RM}$ (m)")
	ax.legend()
	ax.set_xlim(90,300)
	ax.set_ylim(0, 750)
	fig.canvas.manager.full_screen_toggle()
	title = f'{figTitle}_h0_xrm_lf'
	fig.savefig(f"{figPath_}{title}.png", dpi=300)
	plt.close(fig=fig)


	fig = plt.figure(figsize=(16,9), constrained_layout=True)
	ax = fig.add_subplot()
	vmax = 15
	cmap = cc.cm.CET_R1.resampled(vmax*1000)
	newcolors = cmap(np.linspace(0, 1, vmax*1000))
	newcolors[:int(Wrm_o[mask].min()*1000), :] = np.array([97/256, 97/256, 97/256, 0.1])
	newcmap = mp.colors.ListedColormap(newcolors)
	ax.scatter(H0[mask_], Xrm_o[mask_], color="#616161", alpha=0.4)
	cs = ax.scatter(H0[mask], Xrm_o[mask], c=Wrm_o[mask], cmap=newcmap, vmin=0, vmax=vmax, alpha=0.7)
	cb = fig.colorbar(cs, ticks = [0, 1, 5, 10, 15], ax=ax, pad=0.007, label=r"Rampart height: $w_{RM}$ (m)")
	H0_ = np.linspace(H0.min(), H0.max(), 10000)
	ax.plot(H0_, dt._xrm_s(H0_, epsLin, 1), "--", color="gray", label=r"$x^{*}_{RM}$ with $h^{*}\sim h$", linewidth=2)
	ax.plot(H0_, dt._xrm_s(H0_, epsQuad, 2), "k--", label=r"$x^{*}_{RM}$ with $h^{*} \sim h^2$", linewidth=2)
	ax.set_xlabel(r"Front thickness: $h$ (m)")
	ax.set_ylabel(r"Moat position: $x_{RM}$ (m)")
	ax.legend()
	ax.set_xlim(90,300)
	ax.set_ylim(0, 750)
	fig.canvas.manager.full_screen_toggle()
	title = f'{figTitle}_h0_xrm_wrm'
	fig.savefig(f"{figPath_}{title}.png", dpi=300)
	plt.close(fig=fig)


	fig = plt.figure(figsize=(16,9), constrained_layout=True)
	gs = fig.add_gridspec(1, 2, wspace=0.005)
	ax1 = fig.add_subplot(gs[0,0])
	ax2 = fig.add_subplot(gs[0,1])
	bins1 = np.linspace(0, 35, 36)
	bins2 = np.linspace(30, 162, 45)
	ax1.hist(Lf_s2, bins=bins1, color="tab:red", alpha=0.5, label=r"$l_f$ with $h^{*} \sim h^2$")
	ax1.hist(Lf_s1, bins=bins1, color="gray", alpha=0.5, label=r"$l_f$ with $h^{*}\sim h$")
	ax2.hist(Lw_s2, bins=bins2, color="tab:red", alpha=0.5, label=r"$l_w$ with $h^{*} \sim h^2$")
	ax2.hist(Lw_s1, bins=bins2, color="gray", alpha=0.5, label=r"$l_w$ with $h^{*}\sim h$")
	ax1.set_xlabel(r"Foot length: $l_f$ (m)")
	ax2.set_xlabel(r"Buoyancy wavelength: $l_w$ (m)")
	ax1.set_ylabel("Count")
	ax1.set_ylim(0, 120)
	ax2.set_ylim(0, 120)
	ax2.set_yticks(ax1.get_yticks())
	ax2.set_yticklabels("")
	ax1.legend()
	ax2.legend(loc="upper right")
	ax1.annotate("a)", xy=(ax1.get_xlim()[0], ax1.get_ylim()[1]), xytext=(3, -4),textcoords='offset points', ha="left", va="top")
	ax2.annotate("b)", xy=(ax2.get_xlim()[0], ax2.get_ylim()[1]), xytext=(3, -4),textcoords='offset points', ha="left", va="top")
	fig.canvas.manager.full_screen_toggle()
	title = f'{figTitle}_hist_lf+lw'
	fig.savefig(f"{figPath_}{title}.png", dpi=300)
	plt.close(fig=fig)


	fig = plt.figure(figsize=(16,9), constrained_layout=True)
	ax = fig.add_subplot()
	ax.plot(Lw_o, Lw_s1, "o", label=r"$h^{*}=a_1 h$", alpha=0.3)
	ax.plot(Lw_o, Lw_s2, "o", label=r"$h^{*}=a_2 h^2$", alpha=0.3)
	ax.plot(Lw_o, Lw_t, "o", label=r"$h^{*}=h$", alpha=0.3)
	ax.set_xlabel(r"Observed buoyancy wavelength: $l_w^{obs}$ (m)")
	ax.set_ylabel(r"Estimated buoyancy wavelength: $l_w^{*}$ (m)")
	ax.plot([0, 1e4], [0, 1e4],"k:")
	ax.legend()
	ax.set_xlim(0, 250)
	ax.set_ylim(0, 700)
	fig.canvas.manager.full_screen_toggle()
	title = f'{figTitle}_lws'
	fig.savefig(f"{figPath_}{title}.png", dpi=300)
	plt.close(fig=fig)

	fig = plt.figure(figsize=(16,9), constrained_layout=True)
	ax = fig.add_subplot()
	ax.plot(H0[mask_], Wrm_o[mask_], "k.", alpha=0.4)
	ax.plot(H0[mask], Wrm_o[mask], "g.", label="Data", alpha=0.4)
	fct = lambda h, a, b: a*h**b
	parm = sp.optimize.curve_fit(f=fct, xdata=H0[mask], ydata=Wrm_o[mask])[0]
	x = np.linspace(H0.min(), H0.max(), 1000)
	y = fct(x, parm[0], parm[1])
	ax.plot(x,y,"k", label=f"Fit: {parm[0]:.1e}*h^{parm[1]:.1f}")
	ax.set_xlabel(r"Front thickness: $h$ (m)")
	ax.set_ylabel(r"Rampart height: $w_{RM}$ (m)")
	ax.legend()
	fig.canvas.manager.full_screen_toggle()
	title = f'{figTitle}_h0_hrm'
	fig.savefig(f"{figPath_}{title}.png", dpi=300)
	plt.close(fig=fig)

def FrontTypeMap(): #FIGURE S1: Elevation map of ICESat2 transects along RIS, with dots where the front is detected colored by the height of the rampart.
	figTitle = "ICESAT2_"+it.stack()[0][3]
	df = dt.getData()

	fig = plt.figure(figsize=(20,6), constrained_layout=True)
	ax = fig.add_subplot(projection=crs.SouthPolarStereo(central_longitude=180))
	Lat, Lon, F = [],[],[]

	for i, ind in enumerate(df.index):
		dfi = df.loc[i]
		if dfi.flag in [0, 1, 2]:
			lat = dfi.lat
			lon = dfi.lon
			f = dfi.f
			mask = np.logical_not(np.logical_or(np.logical_or(np.isnan(f), np.isnan(lat)), np.isnan(lon)))
			Lat += lat[mask].tolist()
			Lon += lon[mask].tolist()
			F += f[mask].tolist()
	cs = ax.scatter(Lon, Lat, c=F, s=0.1, alpha=1, transform=crs.PlateCarree(), zorder=3, cmap="terrain", vmin=0, vmax=80)
	cb = fig.colorbar(cs, pad=0.03)
	cb.set_label("Elevation (m)")

	cc = {
		0:"k",
		1:"k",
		2:"#f41b02",
		3:"#f46c02",
		4:"#f4cc02",
	}

	for i, ind in enumerate(df.index)	:
		dfi = df.loc[i]
		if dfi.flag in [1,2]:
			lat, lon = dfi.frontLat, dfi.frontLon
			if dfi.flag==1:
				flag = 0
			if dfi.flag==2:
				wrm = dfi.wrm_o
				if wrm<1:
					flag = 1
				elif wrm<2:
					flag = 2
				elif wrm<5:
					flag = 3
				else:
					flag = 4
			ax.scatter(lon, lat, color=cc[flag], alpha=0.4, transform=crs.PlateCarree(), zorder=4)#facecolors='none' if flag in [0,1] else None
			flag = None

	ax.coastlines(zorder=6)
	ax.add_feature(crf.LAND, facecolor=[74/255,47/255,12/255], zorder=2)
	gl = ax.gridlines( draw_labels=True, dms=True, x_inline=False, y_inline=False, rotate_labels=False, zorder=7)
	gl.xlocator = mp.ticker.FixedLocator([160,165,170,175,180,-160,-165,-170,-175,-155])
	ax.stock_img()
	title = f"{figTitle}.png"
	fig.savefig(f"{figPath}{title}.png", dpi=300)
	plt.close(fig=fig)
