import os
import inspect as it
import numpy as np
import scipy as sp
import pandas as pd
import multiprocessing as mg
import colorcet as cc
import cartopy.feature as crf
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.pylab as pll
import cartopy.crs as crs

import dataICESAT2ATL06 as dt
import physics as ph

from params import figPath, rhotio,g ,rhow
os.makedirs(figPath, exist_ok=True)

plt.rc('font', size=24)         # controls default text sizes
plt.rc('axes', titlesize=24)    # fontsize of the axes title
plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=21)   # fontsize of the tick labels
plt.rc('ytick', labelsize=21)   # fontsize of the tick labels
plt.rc('legend', fontsize=24)   # legend fontsize
plt.rc('figure', titlesize=24)  # fontsize of the figure title
figSupName = "ICESAT2_"

def RMProfile(): #Overlap ICESat2 transect ID = 215 with the elastic beam theory
	figTitle = figSupName + it.stack()[0][3]
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
	figTitle = "Figure_5"
	df = dt.getData()

	fig = plt.figure(figsize=(24,9), constrained_layout=True)
	gs = fig.add_gridspec(2, 3, width_ratios=[1.4, 1.4, 1], height_ratios=[1,1] )
	ax = fig.add_subplot(gs[:,0:2])
	ax1 = fig.add_subplot(gs[0, 2])
	ax2 = fig.add_subplot(gs[1, 2])
	cmapName = pll.cm.Reds
	cMin, cMax = 0, 12
	cNbPoints = int(1e8)
	cmap = cmapName(np.linspace(0, 1, cNbPoints))
	Index = df[df.flag==2].sort_values("wrm_o").index
	nbRemovedTransect = 0
	Xrm = []
	Wrm = []
	for i in Index:
		dfi = df.loc[i]
		r, f = dt.getNormalizedRF(dfi)
		r /= dfi.lw_o
		f /= dfi.wrm_o
		f -= f[0]-1 
		if f[r<1].max()>1.20:
			nbRemovedTransect+=1
			continue
		color = cmap[min(int(cNbPoints*(dfi.wrm_o-cMin)/(cMax-cMin)), cNbPoints-1)]
		ax.plot(r, f, color=color, alpha=0.25, zorder=4)
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

	m = ph.modelSimpleNoDim(h0=0)
	y = m.getProfile(Lf=1, H=1)(x)
	wrm = m.getRM()[1]
	y /= wrm
	y -= y[0] - 1
	ax.plot(x, y, "-", zorder=6, color="k")

	m = ph.modelLinNoTenNoDim(h0=1)
	y2 = m.getProfile3(Lf=1, a=-1e-1, H=1)(x)
	wrm = m.getRM()[1]
	y2 /= wrm
	y2 -= y2[0] - 1
	y2 -= y2[-1]-y[-1]
	ax.plot(x, y2, "--", zorder=6, color="k")
	y2 = m.getProfile3(Lf=1, a=2e-1, H=1)(x)
	wrm = m.getRM()[1]
	y2 /= wrm
	y2 -= y2[0] - 1
	y2 -= y2[-1]-y[-1]
	ax.plot(x, y2, "--", zorder=6, color="k")

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

	ax1b = ax1.secondary_xaxis('top')
	print(ax1.get_xticks())
	ax1b.set_xticks([0, 300, 600])
	ax1b.set_xticklabels((ax1b.get_xticks()/3).astype(int))
	ax1b.set_xlabel(r"Calving length: $L_{calve}$ (m)", labelpad=8)
	ax1b.tick_params(axis='x', which='major', pad=-3)

	fig.canvas.manager.full_screen_toggle()
	title = f'{figTitle}'
	fig.savefig(f"{figPath}{title}.png", dpi=300)
	plt.close(fig=fig)
	
	fig = plt.figure(figsize=(22,9), constrained_layout=True)
	ax = fig.add_subplot()
	cmapName = pll.cm.Reds
	cMin, cMax = 0, 12
	cNbPoints = int(1e8)
	cmap = cmapName(np.linspace(0, 1, cNbPoints))
	Index = df[df.flag==2].sort_values("wrm_o").index
	nbRemovedTransect = 0
	for i in Index:
		dfi = df.loc[i]
		r, f = dt.getNormalizedRF(dfi)
		if f[r<dfi.lw_o].max()-f[0]>1.2:
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

def Overimpose_def(): #FIGURE 4: overlap of all normalized ICESat2 transects with the beam theory. Additional distribution plots of the observed moat positions and the observed rampart heights.
	figTitle = "ICESAT2_def_"+it.stack()[0][3]
	df = dt.getData()

	fig = plt.figure(figsize=(24,9), constrained_layout=True)
	gs = fig.add_gridspec(2, 3, width_ratios=[1.4, 1.4, 1], height_ratios=[1,1] )
	ax = fig.add_subplot(gs[:,0:2])
	ax1 = fig.add_subplot(gs[0, 2])
	ax2 = fig.add_subplot(gs[1, 2])
	cmapName = pll.cm.Blues
	cMin, cMax = 0, 12
	cNbPoints = int(1e8)
	cmap = cmapName(np.linspace(0, 1, cNbPoints))
	Index = df[df.flag==2].sort_values("wrm_o").index
	nbRemovedTransect = 0
	Xrm = []
	Wrm = []
	for i in Index:
		dfi = df.loc[i]
		r, f = dt.getNormalizedRF(dfi)
		r /= dfi.lw_o
		f /= dfi.wrm_o
		f -= f[0]-1 
		if f[r<1].max()>1.20:
			nbRemovedTransect+=1
			continue
		color = cmap[min(int(cNbPoints*(dfi.wrm_o-cMin)/(cMax-cMin)), cNbPoints-1)]
		ax.plot(r, f, color=color, alpha=0.25, zorder=4)
		Wrm.append(dfi.wrm_o)
		Xrm.append(dfi.xrm_o)
	print(f"Removed {nbRemovedTransect} / {len(df[df.flag==2])}")
	ax1.hist(Xrm, bins=30, color="#E5F6F6")
	ax1.yaxis.tick_right()
	ax1.yaxis.set_label_position("right")
	ax1.set_ylabel("Count", color="#E5F6F6")
	ax1.set_xlabel(r"Moat position: $x_{RM}$ (m)", color="#E5F6F6")
	ax2.hist(Wrm, bins=30, color="#E5F6F6")
	ax2.yaxis.tick_right()
	ax2.yaxis.set_label_position("right")
	ax2.set_ylabel("Count", color="#E5F6F6")
	ax2.set_xlabel(r"Rampart height: $w_{RM}$ (m)", color="#E5F6F6")
	sm = mp.cm.ScalarMappable(cmap=cmapName, norm=mp.colors.Normalize(vmin=cMin, vmax=cMax)) 
	cb = fig.colorbar(sm, ax=ax)
	x = np.linspace(0,20,10000)

	m = ph.modelSimpleNoDim(h0=0)
	y = m.getProfile(Lf=1, H=1)(x)
	wrm = m.getRM()[1]
	y /= wrm
	y -= y[0] - 1
	ax.plot(x, y, "-", zorder=6, color="k")

	m = ph.modelLinNoTenNoDim(h0=1)
	y2 = m.getProfile3(Lf=1, a=-1e-1, H=1)(x)
	wrm = m.getRM()[1]
	y2 /= wrm
	y2 -= y2[0] - 1
	y2 -= y2[-1]-y[-1]
	ax.plot(x, y2, "--", zorder=6, color="k")
	y2 = m.getProfile3(Lf=1, a=2e-1, H=1)(x)
	wrm = m.getRM()[1]
	y2 /= wrm
	y2 -= y2[0] - 1
	y2 -= y2[-1]-y[-1]
	ax.plot(x, y2, "--", zorder=6, color="k")

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
	ax.set_xlabel(r"Normalized distance from front: $x/l_w$", color="#E5F6F6")
	ax.set_ylabel(r"Normalized deflection: $w/w\left(0\right)$", color="#E5F6F6")
	cb.set_label(r"Rampart height: $w_{RM}$ (m)"+"\n", color="#E5F6F6")

	ax1b = ax1.secondary_xaxis('top')
	print(ax1.get_xticks())
	ax1b.set_xticks([0, 300, 600])
	ax1b.set_xticklabels((ax1b.get_xticks()/3).astype(int))
	ax1b.set_xlabel(r"Calving length: $L_{calve}$ (m)", labelpad=8)
	ax1b.tick_params(axis='x', which='major', pad=-3)
	ax.spines['bottom'].set_color('#E5F6F6')
	ax.spines['top'].set_color('#E5F6F6')
	ax.spines['left'].set_color('#E5F6F6')
	ax.spines['right'].set_color('#E5F6F6')
	ax.xaxis.label.set_color('#E5F6F6')
	ax.tick_params(axis='x', colors='#E5F6F6')
	ax.tick_params(axis='y', colors='#E5F6F6')
	fig.canvas.manager.full_screen_toggle()
	title = f'{figTitle}'
	fig.savefig(f"{figPath}{title}.png", dpi=300, transparent=True)
	plt.close(fig=fig)
	quit()
	




	fig = plt.figure(figsize=(22,9), constrained_layout=True)
	ax = fig.add_subplot()
	cmapName = pll.cm.Reds
	cMin, cMax = 0, 12
	cNbPoints = int(1e8)
	cmap = cmapName(np.linspace(0, 1, cNbPoints))
	Index = df[df.flag==2].sort_values("wrm_o").index
	nbRemovedTransect = 0
	for i in Index:
		dfi = df.loc[i]
		r, f = dt.getNormalizedRF(dfi)
		if f[r<dfi.lw_o].max()-f[0]>1.2:
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
	figTitle = figSupName + it.stack()[0][3]
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
	ax.plot(x,y,"k", label=f"Fit: {parm[0]:.1e}*h^{parm[1]:.1f} m")
	ax.set_xlabel(r"Front thickness: $h$ (m)")
	ax.set_ylabel(r"Rampart height: $w_{RM}$ (m)")
	ax.legend()
	fig.canvas.manager.full_screen_toggle()
	title = f'{figTitle}_h0_hrm'
	fig.savefig(f"{figPath_}{title}.png", dpi=300)
	plt.close(fig=fig)

def FrontTypeMap(): #FIGURE S1: Elevation map of ICESat2 transects along RIS, with dots where the front is detected colored by the height of the rampart.
	figTitle = "Figure_A1"
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
	title = f"{figTitle}"
	fig.savefig(f"{figPath}{title}.png", dpi=300)
	plt.close(fig=fig)

def FrontPositionTimeMap(): #FIGURE S1: Elevation map of ICESat2 transects along RIS, with dots where the front is detected colored by the height of the rampart.
	figName = figSupName + it.stack()[0][3]
	# df = dt.getData()
	# pd.to_pickle(df, "tmp.pkl")
	df = pd.read_pickle("tmp.pkl")
	fig = plt.figure(figsize=(20,6), constrained_layout=True)
	ax = fig.add_subplot(projection=crs.SouthPolarStereo(central_longitude=180))
	buoy = {"name":"DR03", "lat":-78.263, "lon":-175.117%360, "V":993.02, "U":222.30}

	dff = df.loc[(df.flag == 1) | (df.flag == 2), ["frontLat", "frontLon", "mtime"]]
	print(dff["mtime"].values.min())
	print(dff["mtime"].values.max())
	cs = ax.scatter(dff["frontLon"].values, dff["frontLat"].values, c=dff["mtime"].values, s=10,  alpha=1, transform=crs.PlateCarree(), zorder=4)
	cs = ax.scatter(buoy["lon"], buoy["lat"], color="r", s=10,  alpha=1, transform=crs.PlateCarree(), zorder=4)
	fig.colorbar(cs)
	ax.coastlines(zorder=6)
	# ax.add_feature(crf.LAND, facecolor=[74/255,47/255,12/255], zorder=2)
	# gl = ax.gridlines( draw_labels=True, dms=True, x_inline=False, y_inline=False, rotate_labels=False, zorder=7)
	# gl.xlocator = mp.ticker.FixedLocator([160,165,170,175,180,-160,-165,-170,-175,-155])
	# ax.stock_img()
	fig.savefig(f"{figPath}{figName}.png", dpi=300)
	plt.close(fig=fig)

def CompareLfForNewModels():
	df = dt.getData()
	df = df.loc[df.flag==2]
	fig = plt.figure(figsize=(16,9), constrained_layout=True)
	ax = fig.add_subplot()
	ax.hist(df.lf_s2, label="Linear-simple", color="b", alpha=0.7, bins=np.arange(0, 120, 1))
	# ax.hist(df.loc[:, "lf_linear-noTen-approxXrm"], label="linear-noTen-approxXrm", color="r", alpha=0.7, bins=np.arange(0,120,1))
	# ax.hist(df.loc[:, "lf_linear-noTen"], label="linear-noTen", color="m", alpha=0.7, bins=np.arange(0,120,1))
	ax.hist(df.loc[:, "lf_linear-noTen-RMFit"], label="linear-noTen-RMFit", color="g", alpha=0.7, bins=np.arange(0,120,1))
	ax.legend()
	ax.set_xlabel("Foot length (m)")
	ax.set_ylabel("Count")
	ax.set_xlim(0,80)
	figTitle = figSupName + it.stack()[0][3]
	title = f"{figTitle}"
	fig.savefig(f"{figPath}{title}.png", dpi=300)
	plt.close(fig=fig)

def model_TillProfileMathematica(): #Plot modelFullDim profiles compared to mathematica
	fig = plt.figure(figsize=(7, 15))
	ax = fig.subplots(4, 3, sharex=True, sharey=True)

	initParams = [
		[1, 0],
		[2, 0],
		[3, 0],

		[1, 0.1],
		[2, 0.1],
		[3, 0.1],

		[1, 0.3],
		[2, 0.3],
		[3, 0.3],

		[1, 0.7],
		[2, 0.7],
		[3, 0.7],
		]

	x = np.linspace(0,10,10000)
	modelFull = ph.modelFullNoDim(h0=1, verbose=0)
	modelFull.rhotio = 0.9
	modelLinNoTen = ph.modelLinNoTenNoDim(h0=1, verbose=0)
	modelLinNoTen.rhotio = 0.9

	for i, param in enumerate(initParams):
		ix = i%3
		iy = i//3
		y_modelFull = modelFull.getProfile(H=param[0], Lf=param[1])(x)
		y_modelLinNoTen = modelLinNoTen.getProfile(H=param[0], Lf=param[1])(x)
		ax[iy,ix].plot(x, y_modelFull, color="k")
		ax[iy,ix].plot(x, y_modelLinNoTen, color="b")
	ax[0,0].set_xlim(-0.1, 6)
	ax[0,0].set_ylim(-0.3, 0.3)
	plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.25)
	plt.show()

def modelFullNoDim_RMspace(nbCore=12): #Plot modelFullDim xrm-wrm space compared to mathematica
	import multiprocessing as mg
	import pandas as pd
	global _figRMspace
	def _figRMspace(xrm, wrm):
		s = ph.modelFullNoDim(verbose=1)
		try:
			lwf = s.getLwf(xrm, wrm)
			H = lwf["H"]
			Lf = lwf["Lf"]
			print(f"RMspace SUCCESS: {xrm:.2e}, {wrm:.2e}")
		
		except ph.myError as eee:
			H = np.nan
			Lf = np.nan
			print(f"RMspace FAIL: {xrm:.2e}, {wrm:.2e} --> {eee}")
		
		return {"xrm":xrm, "wrm":wrm, "H":H, "Lf":Lf}


	size = 50
	size_ = 50
	Xrm = np.linspace(3.40, 4.2, size_)
	Wrm = np.linspace(0.01, 0.3, size)
	Xrm, Wrm = np.meshgrid(Xrm, Wrm)
	Xrm = Xrm.reshape(-1, 1).T[0]
	Wrm = Wrm.reshape(-1, 1).T[0] 
	
	pool = mg.Pool(processes=nbCore)
	res = pool.starmap(_figRMspace, zip(Xrm, Wrm))

	res = pd.DataFrame(res)
	fig = plt.figure(figsize=(15, 15), constrained_layout=True)
	gs = fig.add_gridspec(1, 2)
	ax1 = fig.add_subplot(gs[0])
	ax2 = fig.add_subplot(gs[1])
	X = res.xrm.values.reshape(-1, size_)
	Y = res.wrm.values.reshape(-1, size_)
	Z1 = res.H.values.reshape(-1, size_)
	Z2 = res.Lf.values.reshape(-1, size_)
	cs1 = ax1.pcolormesh(X, Y, Z1, vmin=0, vmax=5, cmap="magma")
	cs2 = ax2.pcolormesh(X, Y, Z2, vmin=0, vmax=2.5, cmap="magma")

	ax1.set_xlabel("xrm")
	ax2.set_xlabel("xrm")
	ax1.set_ylabel("wrm")
	fig.colorbar(cs1, ax=ax1, label="H")
	fig.colorbar(cs2, ax=ax2, label="Lf")
	plt.show()

def modelFullNoDim_RMspaceTrasect():#Plot modelFullDim xrm-wrm space transect compared to mathematica
	fig = plt.figure(figsize=(15, 15), constrained_layout=True)
	H_ = [1,2,3]
	ax = fig.subplots(1, len(H_), sharex=True, sharey=True)

	Lf = np.linspace(0,1,1000)
	for i, H in enumerate(H_):
		s = ph.modelFullNoDim(h0=1, verbose=0)
		Xrm = []
		Wrm = []
		for lf in Lf:
			try:
				s.getProfile(H=H, Lf=lf)
				xrm, wrm = s.getRM()
				Xrm.append(xrm)
				Wrm.append(wrm)
			except ph.myError as eee:
				print(eee)
				Xrm.append(np.nan)
				Wrm.append(np.nan)

		ax[i].plot(Lf, Xrm, 'r')
		ax[i].set_title(f"H = {H}")
		ax[i].set_xlabel("Lf")
		ax[i].set_ylabel("Xrm", color="r")

		axb = ax[i].twinx()
		axb.plot(Lf, Wrm, 'g')
		axb.set_ylabel("Wrm", color="g")

	plt.show()

def modelFullDim_RMspaceInverted(nbCore=12, xrm=None, wrm=None): #Plot modelFullDim lw-lf space
	global __modelFullDim_RMspaceInverted

	def __modelFullDim_RMspaceInverted(h0, lw, lf):
		s = ph.modelFullDim(h0=h0, verbose=0)
		try:
			s.getProfile(lw=lw, lf=lf)
			xrm, wrm = s.getRM()
			print(f"RMspace SUCCESS: h0 = {h0}, lw = {lw:.2e}, lf = {lf:.2e}")
		
		except ph.myError as eee:
			xrm = np.nan
			wrm = np.nan
			print(f"RMspace FAIL: h0 = {h0}, lw = {lw:.2e}, lf = {lf:.2e} --> {eee}")
		
		except IndexError as eee:
			xrm = np.nan
			wrm = np.nan
			print(f"RMspace FAIL: h0 = {h0}, lw = {lw:.2e}, lf = {lf:.2e} --> {eee}")

		return {"xrm":xrm, "wrm":wrm, "lw":lw, "lf":lf, "h0":h0}

	size = 150
	size_ = 151
	h0Range = [50,100,250,500]
	lwRange = np.logspace(0, 3, size_)
	lfRange = np.logspace(0, 3, size)
	lwRange, lfRange = np.meshgrid(lwRange, lfRange)
	lwRange = lwRange.reshape(-1, 1).T[0]
	lfRange = lfRange.reshape(-1, 1).T[0] 
	

	fig = plt.figure(figsize=(16,9), constrained_layout=True)
	gs = fig.add_gridspec(2, len(h0Range))
	for i, h0 in enumerate(h0Range):
		pool = mg.Pool(processes=nbCore)
		res = pd.DataFrame(pool.starmap(__modelFullDim_RMspaceInverted, zip([h0]*len(lwRange), lwRange, lfRange)))
		ax1 = fig.add_subplot(gs[0, i])
		ax2 = fig.add_subplot(gs[1, i])
		
		X = res.lw.values.reshape(-1, size_)
		Y = res.lf.values.reshape(-1, size_)
		Z1 = res.xrm.values.reshape(-1, size_)
		Z2 = res.wrm.values.reshape(-1, size_)
		cs1 = ax1.pcolormesh(X, Y, Z1, norm=mp.colors.LogNorm())
		cs2 = ax2.pcolormesh(X, Y, Z2)

		ax1.set_title(h0)
		ax1.set_ylabel("lf")
		ax2.set_xlabel("lw")
		ax2.set_ylabel("lf")
		ax1.set_xscale("log")
		ax1.set_yscale("log")
		ax2.set_xscale("log")
		ax2.set_yscale("log")
		fig.colorbar(cs1, ax=ax1, label="xrm")
		fig.colorbar(cs2, ax=ax2, label="wrm")

		if xrm is not None:
			mask1 = (Z1>=xrm*0.9) & (Z1<=xrm*1.1)
			ax1.contourf(X, Y, mask1, levels=[0.5, 1], colors='red', hatches=['x', 'x'], alpha=0.5)
			ax2.contourf(X, Y, mask1, levels=[0.5, 1], colors='red', hatches=['x', 'x'], alpha=0.5)

		if wrm is not None:
			mask2 = (Z2>=wrm*0.9) & (Z2<=wrm*1.1)
			ax1.contourf(X, Y, mask2, levels=[0.5, 1], colors='green', hatches=['x', 'x'], alpha=0.5)
			ax2.contourf(X, Y, mask2, levels=[0.5, 1], colors='green', hatches=['x', 'x'], alpha=0.5)

	plt.show()

def modelLinNoTenDim_RMspaceInverted(nbCore=12, h0Range=None, xrm=None, wrm=None): #Plot modelFullDim lw-lf space
	global __modelLinNoTenDim_RMspaceInverted
	def __modelLinNoTenDim_RMspaceInverted(h0, lw, lf):
		s = ph.modelLinNoTenDim(h0=h0, verbose=0)
		print(h0, lw, lf)
		try:
			s.getProfile(lw=lw, lf=lf)
			xrm, wrm = s.getRM()
			print(f"RMspace SUCCESS: h0 = {h0}, lw = {lw:.2e}, lf = {lf:.2e}")
		
		except ph.myError as eee:
			xrm = np.nan
			wrm = np.nan
			print(f"RMspace FAIL: h0 = {h0}, lw = {lw:.2e}, lf = {lf:.2e} --> {eee}")
		
		except IndexError as eee:
			xrm = np.nan
			wrm = np.nan
			print(f"RMspace FAIL: h0 = {h0}, lw = {lw:.2e}, lf = {lf:.2e} --> {eee}")

		return {"xrm":xrm, "wrm":wrm, "lw":lw, "lf":lf, "h0":h0}

	size = 150
	size_ = 151
	if h0Range is None:
		h0Range = [50,100,250,500]
	lwRange = np.logspace(0, 3, size_)
	lfRange = np.logspace(0, 3, size)
	lwRange, lfRange = np.meshgrid(lwRange, lfRange)
	lwRange = lwRange.reshape(-1, 1).T[0]
	lfRange = lfRange.reshape(-1, 1).T[0] 
	

	fig = plt.figure(figsize=(16,9), constrained_layout=True)
	gs = fig.add_gridspec(2, len(h0Range))
	for i, h0 in enumerate(h0Range):
		pool = mg.Pool(processes=nbCore)
		res = pd.DataFrame(pool.starmap(__modelLinNoTenDim_RMspaceInverted, zip([h0]*len(lwRange), lwRange, lfRange)))
		ax1 = fig.add_subplot(gs[0, i])
		ax2 = fig.add_subplot(gs[1, i])
		
		X = res.lw.values.reshape(-1, size_)
		Y = res.lf.values.reshape(-1, size_)
		Z1 = res.xrm.values.reshape(-1, size_)
		Z2 = res.wrm.values.reshape(-1, size_)
		cs1 = ax1.pcolormesh(X, Y, Z1, norm=mp.colors.LogNorm())
		cs2 = ax2.pcolormesh(X, Y, Z2)

		ax1.set_title(h0)
		ax1.set_ylabel("lf")
		ax2.set_xlabel("lw")
		ax2.set_ylabel("lf")
		ax1.set_xscale("log")
		ax1.set_yscale("log")
		ax2.set_xscale("log")
		ax2.set_yscale("log")
		fig.colorbar(cs1, ax=ax1, label="xrm")
		fig.colorbar(cs2, ax=ax2, label="wrm")

		if xrm is not None:
			mask1 = (Z1>=xrm*0.9) & (Z1<=xrm*1.1)
			ax1.contourf(X, Y, mask1, levels=[0.5, 1], colors='red', hatches=['x', 'x'], alpha=0.5)
			ax2.contourf(X, Y, mask1, levels=[0.5, 1], colors='red', hatches=['x', 'x'], alpha=0.5)

		if wrm is not None:
			mask2 = (Z2>=wrm*0.9) & (Z2<=wrm*1.1)
			ax1.contourf(X, Y, mask2, levels=[0.5, 1], colors='green', hatches=['x', 'x'], alpha=0.5)
			ax2.contourf(X, Y, mask2, levels=[0.5, 1], colors='green', hatches=['x', 'x'], alpha=0.5)

	plt.show()

def modelDim_slidder():
		x = np.linspace(0, 1500, 3000)
		hi = 500
		lfi = 10
		lwi = 10

		modelFullDim = ph.modelFullDim(h0=hi, verbose=0)
		modelLinNoTenDim = ph.modelLinNoTenDim(h0=hi, verbose=0)
		modelSimpleDim = ph.modelSimpleDim(h0=hi, verbose=0)

		y_modelFullDim = modelFullDim.getProfile(lf=lfi, lw=lwi)(x)
		y_modelLinNoTenDim = modelLinNoTenDim.getProfile(lf=lfi, lw=lwi)(x)
		y_modelSimpleDim = modelSimpleDim.getProfile(lf=lfi, lw=lwi)(x)

		fig, ax = plt.subplots()
		fig.subplots_adjust(left=0.1, bottom=0.35)
		line_modelFullDim, = ax.plot(x, y_modelFullDim, "k", lw=2)
		line_modelLinNoTenDim, = ax.plot(x, y_modelLinNoTenDim, '-.b', lw=2)
		line_modelSimpleDim, = ax.plot(x, y_modelSimpleDim, ':g', lw=2)

		axcolor = 'lightgoldenrodyellow'
		ax_h = plt.axes([0.1, 0.2, 0.65, 0.03], facecolor=axcolor)
		ax_lf = plt.axes([0.1, 0.15, 0.65, 0.03], facecolor=axcolor)
		ax_lw = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor=axcolor)

		h0_slide = mp.widgets.Slider(ax_h, 'h', 10, 1000, valinit=hi)
		lf_slide = mp.widgets.Slider(ax_lf, 'lf', 0, 1000, valinit=lfi)
		lw_slide = mp.widgets.Slider(ax_lw, 'lw', 0, 200, valinit=lwi)

		def update(val):
			h0 = h0_slide.val
			lf = lf_slide.val
			lw = lw_slide.val

			modelFullDim.h0 = h0
			try:
				y_modelFullDim = modelFullDim.getProfile(lf=lf, lw=lw)(x)
			except ph.myError as eee:
				print(eee)
				y_modelFullDim = np.zeros(x.shape)
			line_modelFullDim.set_ydata(y_modelFullDim)

			modelLinNoTenDim.h0 = h0
			y_modelLinNoTenDim = modelLinNoTenDim.getProfile(lf=lf, lw=lw)(x)
			line_modelLinNoTenDim.set_ydata(y_modelLinNoTenDim)

			modelSimpleDim.h0 = h0
			y_modelSimpleDim = modelSimpleDim.getProfile(lf=lf, lw=lw)(x)
			line_modelSimpleDim.set_ydata(y_modelSimpleDim)

			minY = np.nanmin(np.concatenate([y_modelSimpleDim, y_modelFullDim, y_modelLinNoTenDim]))
			maxY = np.nanmax(np.concatenate([y_modelSimpleDim, y_modelFullDim, y_modelLinNoTenDim]))
			ax.set_ylim([minY - 0.1, maxY + 0.1])
			fig.canvas.draw_idle()

		h0_slide.on_changed(update)
		lf_slide.on_changed(update)
		lw_slide.on_changed(update)
		plt.show()

def modelNoDim_slidder():
		x = np.linspace(0, 15, 3000)
		Lfi = 0
		Hi = 1

		modelFullNoDim = ph.modelFullNoDim(h0=1, verbose=0)
		modelLinNoTenNoDim = ph.modelLinNoTenNoDim(h0=1, verbose=0)
		modelSimpleNoDim = ph.modelSimpleNoDim(h0=1, verbose=0)

		y_modelFullNoDim = modelFullNoDim.getProfile(Lf=Lfi, H=Hi)(x)
		y_modelLinNoTenNoDim = modelLinNoTenNoDim.getProfile(Lf=Lfi,H=Hi)(x)
		y_modelSimpleNoDim = modelSimpleNoDim.getProfile(Lf=Lfi, H=Hi)(x)

		fig, ax = plt.subplots()
		fig.subplots_adjust(left=0.1, bottom=0.35)
		line_modelFullNoDim, = ax.plot(x, y_modelFullNoDim, "k", lw=2)
		line_modelLinNoTenNoDim, = ax.plot(x, y_modelLinNoTenNoDim, '-.b', lw=2)
		line_modelSimpleNoDim, = ax.plot(x, y_modelSimpleNoDim, ':g', lw=2)

		axcolor = 'lightgoldenrodyellow'
		ax_Lf = plt.axes([0.1, 0.15, 0.65, 0.03], facecolor=axcolor)
		ax_H = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor=axcolor)

		Lf_slide = mp.widgets.Slider(ax_Lf, 'Lf', 0, 1, valinit=Lfi)
		H_slide = mp.widgets.Slider(ax_H, 'H', 0, 5, valinit=Hi)

		def update(val):
			Lf = Lf_slide.val
			H = H_slide.val

			try:
				y_modelFullNoDim = modelFullNoDim.getProfile(Lf=Lf, H=H)(x)
			except ph.myError as eee:
				print(eee)
				y_modelFullNoDim = np.zeros(x.shape)
			line_modelFullNoDim.set_ydata(y_modelFullNoDim)

			y_modelLinNoTenNoDim = modelLinNoTenNoDim.getProfile(Lf=Lf, H=H)(x)
			line_modelLinNoTenNoDim.set_ydata(y_modelLinNoTenNoDim)

			y_modelSimpleNoDim = modelSimpleNoDim.getProfile(Lf=Lf, H=H)(x)
			line_modelSimpleNoDim.set_ydata(y_modelSimpleNoDim)

			minY = np.nanmin(np.concatenate([y_modelSimpleNoDim, y_modelFullNoDim, y_modelLinNoTenNoDim]))
			maxY = np.nanmax(np.concatenate([y_modelSimpleNoDim, y_modelFullNoDim, y_modelLinNoTenNoDim]))
			ax.set_ylim([minY - 0.1, maxY + 0.1])
			fig.canvas.draw_idle()

		Lf_slide.on_changed(update)
		H_slide.on_changed(update)
		plt.show()

def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if exponent is None:
        exponent = int(np.floor(np.log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits

    return r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision)

def quadProfileComparison():
	df = dt.getData()
	fig = plt.figure(figsize=(16, 10), constrained_layout=True)
	gs = fig.add_gridspec(2, 2)
	ax1 = fig.add_subplot(gs[0,0])
	ax1i = ax1.inset_axes([0.2, 0.65, 0.78, 0.33])
	ax2 = fig.add_subplot(gs[0,1])
	ax3 = fig.add_subplot(gs[1,0])
	ax4 = fig.add_subplot(gs[1,1])

	ID = [37, 438, 206, 1060]

	Id = ID[0]
	print()
	x = np.linspace(0, 2500, 100000)
	r, f = dt.getNormalizedRF(df.loc[Id])
	m = ph.modelLinNoTenDim(h0=df.loc[Id, "h0"])
	idx = m.getLwfa3(r, f, df.loc[Id, "xrm_o"])["idx"]
	ax1.plot(r[:idx], f[:idx], "ko", markersize=9)
	ax1.fill_between(r, [0]*len(r),f, color="black", alpha=0.1)
	ax1.plot([0,0], [0, f[0]], color="black", alpha=0.6)
	ax1i.plot(r[:idx], f[:idx], "ko", markersize=9)
	ax1i.fill_between(r, [0]*len(r),f, color="black", alpha=0.1)
	ax1i.plot([0,0], [0, f[0]], color="black", alpha=0.6)
	ax1.plot(r[idx:], f[idx:], "o",color="gray", markersize=8)
	ax1i.plot(r[idx:], f[idx:], "o",color="gray", markersize=8)
	idx = (np.abs(r - df.loc[Id, "xwrm_o"])).argmin()
	mSimple = lambda x : df.loc[Id, "wrm_o"]/(1 + np.exp(-3*np.pi/4)/2**0.5)*np.cos(x/(2**0.5)/df.loc[Id, "lw_o"])*np.exp(-x/(2**0.5)/df.loc[Id, "lw_o"])
	y_simple = mSimple(x)
	y_simple = y_simple - y_simple[0] + f[idx]
	ax1.plot(x, y_simple, "r", label=r"$l_f$ = " + f"{df.loc[Id, 'lf_s0']:.1f} m, M = 0", linewidth=4)
	ax1i.plot(x, y_simple, "r", label=r"$l_f$ = " + f"{df.loc[Id, 'lf_s0']:.1f} m, M = 0", linewidth=4)
	mLinearNoTen = ph.modelLinNoTenDim(h0=df.loc[Id, 'h0'], verbose=0)
	y_linRMFit = mLinearNoTen.getProfile3(lw=df.loc[Id, 'lw_linear-noTen-RMFit'], lf=df.loc[Id, 'lf_linear-noTen-RMFit'], a=df.loc[Id, 'a_linear-noTen-RMFit'])(x)
	y_linRMFit = y_linRMFit - y_linRMFit[0] + f[0]
	Mdim = df.loc[Id, 'lw_linear-noTen-RMFit']**3*g*rhow*df.loc[Id, 'a_linear-noTen-RMFit']
	ax1.plot(x, y_linRMFit, "g", label=r"$l_f$ = " + f"{df.loc[Id, 'lf_linear-noTen-RMFit']:.1f} m, M = "+ sci_notation(Mdim,1) + r" N$\,$m".replace("e+07",r"x$10^7$").replace("e+08",r"x$10^8$").replace("e+09",r"x$10^9$"), linewidth=4)
	ax1i.plot(x, y_linRMFit, "g", label=r"$l_f$ = " + f"{df.loc[Id, 'lf_linear-noTen-RMFit']:.1f} m, M = "+ sci_notation(Mdim,1) + r" N$\,$m".replace("e+07",r"x$10^7$").replace("e+08",r"x$10^8$").replace("e+09",r"x$10^9$"), linewidth=4)
	print(f"{df.loc[Id, 'frontLat']:.2f}° N, {((df.loc[Id, 'frontLon'] - 180) % 360) - 180:.2f}° W")

	Id = ID[1]
	x = np.linspace(0, 2500, 100000)
	r, f = dt.getNormalizedRF(df.loc[Id])
	m = ph.modelLinNoTenDim(h0=df.loc[Id, "h0"])
	idx = m.getLwfa3(r, f, df.loc[Id, "xrm_o"])["idx"]
	ax2.plot(r[:idx], f[:idx], "ko", markersize=9)
	ax2.fill_between(r, [0]*len(r),f, color="black", alpha=0.1)
	ax2.plot([0,0], [0, f[0]], color="black", alpha=0.6)
	ax2.plot(r[idx:], f[idx:], "o",color="gray", markersize=8)

	idx = (np.abs(r - df.loc[Id, "xwrm_o"])).argmin()
	mSimple = lambda x : df.loc[Id, "wrm_o"]/(1 + np.exp(-3*np.pi/4)/2**0.5)*np.cos(x/(2**0.5)/df.loc[Id, "lw_o"])*np.exp(-x/(2**0.5)/df.loc[Id, "lw_o"])
	y_simple = mSimple(x)
	y_simple = y_simple - y_simple[0] + f[idx]
	ax2.plot(x, y_simple, "r", label=r"$l_f$ = " + f"{df.loc[Id, 'lf_s0']:.1f} m, M = 0", linewidth=4)
	
	mLinearNoTen = ph.modelLinNoTenDim(h0=df.loc[Id, 'h0'], verbose=0)
	y_linRMFit = mLinearNoTen.getProfile3(lw=df.loc[Id, 'lw_linear-noTen-RMFit'], lf=df.loc[Id, 'lf_linear-noTen-RMFit'], a=df.loc[Id, 'a_linear-noTen-RMFit'])(x)
	y_linRMFit = y_linRMFit - y_linRMFit[0] + f[0]
	Mdim = df.loc[Id, 'lw_linear-noTen-RMFit']**3*g*rhow*df.loc[Id, 'a_linear-noTen-RMFit']
	ax2.plot(x, y_linRMFit, "g", label=r"$l_f$ = " + f"{df.loc[Id, 'lf_linear-noTen-RMFit']:.1f} m, M = "+ sci_notation(Mdim,1) + r" N$\,$m".replace("e+07",r"x$10^7$").replace("e+08",r"x$10^8$").replace("e+09",r"x$10^9$"), linewidth=4)
	print(f"{df.loc[Id, 'frontLat']:.2f}° N, {((df.loc[Id, 'frontLon'] - 180) % 360) - 180:.2f}° W")


	Id = ID[2]
	x = np.linspace(0, 2500, 100000)
	r, f = dt.getNormalizedRF(df.loc[Id])
	m = ph.modelLinNoTenDim(h0=df.loc[Id, "h0"])
	idx = m.getLwfa3(r, f, df.loc[Id, "xrm_o"])["idx"]
	ax3.plot(r[:idx], f[:idx], "ko", markersize=9)
	ax3.fill_between(r, [0]*len(r),f, color="black", alpha=0.1)
	ax3.plot([0,0], [0, f[0]], color="black", alpha=0.6)
	ax3.plot(r[idx:], f[idx:], "o",color="gray", markersize=8)

	idx = (np.abs(r - df.loc[Id, "xwrm_o"])).argmin()
	mSimple = lambda x : df.loc[Id, "wrm_o"]/(1 + np.exp(-3*np.pi/4)/2**0.5)*np.cos(x/(2**0.5)/df.loc[Id, "lw_o"])*np.exp(-x/(2**0.5)/df.loc[Id, "lw_o"])
	y_simple = mSimple(x)
	y_simple = y_simple - y_simple[0] + f[idx]
	ax3.plot(x, y_simple, "r", label=r"$l_f$ = " + f"{df.loc[Id, 'lf_s0']:.1f} m, M = 0", linewidth=4)
	
	mLinearNoTen = ph.modelLinNoTenDim(h0=df.loc[Id, 'h0'], verbose=0)
	y_linRMFit = mLinearNoTen.getProfile3(lw=df.loc[Id, 'lw_linear-noTen-RMFit'], lf=df.loc[Id, 'lf_linear-noTen-RMFit'], a=df.loc[Id, 'a_linear-noTen-RMFit'])(x)
	y_linRMFit = y_linRMFit - y_linRMFit[0] + f[0]
	Mdim = df.loc[Id, 'lw_linear-noTen-RMFit']**3*g*rhow*df.loc[Id, 'a_linear-noTen-RMFit']
	ax3.plot(x, y_linRMFit, "g", label=r"$l_f$ = " + f"{df.loc[Id, 'lf_linear-noTen-RMFit']:.1f} m, M = "+ sci_notation(Mdim,1) + r" N$\,$m".replace("e+07",r"x$10^7$").replace("e+08",r"x$10^8$").replace("e+09",r"x$10^9$"), linewidth=4)
	print(f"{df.loc[Id, 'frontLat']:.2f}° N, {((df.loc[Id, 'frontLon'] - 180) % 360) - 180:.2f}° W")


	Id = ID[3]
	x = np.linspace(0, 2500, 100000)
	r, f = dt.getNormalizedRF(df.loc[Id])
	m = ph.modelLinNoTenDim(h0=df.loc[Id, "h0"])
	idx = m.getLwfa3(r, f, df.loc[Id, "xrm_o"])["idx"]
	ax4.plot(r[:idx], f[:idx], "ko", markersize=9)
	ax4.fill_between(r, [0]*len(r),f, color="black", alpha=0.1)
	ax4.plot([0,0], [0, f[0]], color="black", alpha=0.6)
	ax4.plot(r[idx:], f[idx:], "o",color="gray", markersize=8)

	idx = (np.abs(r - df.loc[Id, "xwrm_o"])).argmin()
	mSimple = lambda x : df.loc[Id, "wrm_o"]/(1 + np.exp(-3*np.pi/4)/2**0.5)*np.cos(x/(2**0.5)/df.loc[Id, "lw_o"])*np.exp(-x/(2**0.5)/df.loc[Id, "lw_o"])
	y_simple = mSimple(x)
	y_simple = y_simple - y_simple[0] + f[idx]
	ax4.plot(x, y_simple, "r", label=r"$l_f$ = " + f"{df.loc[Id, 'lf_s0']:.1f} m, M = 0", linewidth=4)
	
	mLinearNoTen = ph.modelLinNoTenDim(h0=df.loc[Id, 'h0'], verbose=0)
	y_linRMFit = mLinearNoTen.getProfile3(lw=df.loc[Id, 'lw_linear-noTen-RMFit'], lf=df.loc[Id, 'lf_linear-noTen-RMFit'], a=df.loc[Id, 'a_linear-noTen-RMFit'])(x)
	y_linRMFit = y_linRMFit - y_linRMFit[0] + f[0]
	Mdim = df.loc[Id, 'lw_linear-noTen-RMFit']**3*g*rhow*df.loc[Id, 'a_linear-noTen-RMFit']
	ax4.plot(x, y_linRMFit, "g", label=r"$l_f$ = " + f"{df.loc[Id, 'lf_linear-noTen-RMFit']:.1f} m, M = "+ sci_notation(Mdim,1) + r" N$\,$m".replace("e+07",r"x$10^7$").replace("e+08",r"x$10^8$").replace("e+09",r"x$10^9$"), linewidth=4)
	print(f"{df.loc[Id, 'frontLat']:.2f}° N, {((df.loc[Id, 'frontLon'] - 180) % 360) - 180:.2f}° W")

	ax1.set_xlim(850, -100)
	ax2.set_xlim(850, -100)
	ax3.set_xlim(850, -100)
	ax4.set_xlim(850, -100)
	ax1.set_ylim(34, 50)
	ax2.set_ylim(34, 50)
	ax3.set_ylim(34, 50)
	ax4.set_ylim(34, 50)
	ax1i.set_xlim(650, -50)
	ax1i.set_ylim(39.3, 40.3)
	ax1.set_ylabel("Freeboard (m)")
	ax3.set_ylabel("Freeboard (m)")
	ax4.set_xlabel("Distance from front (m)")
	ax3.set_xlabel("Distance from front (m)")
	leg1 = ax1.legend(loc="lower left", frameon=False, fontsize=18)
	leg1.get_frame().set_facecolor('none')
	leg2 = ax2.legend(loc="upper right", frameon=False, fontsize=18)
	leg2.get_frame().set_facecolor('none')
	leg3 = ax3.legend(loc="upper right", frameon=False, fontsize=18)
	leg3.get_frame().set_facecolor('none')
	leg4 = ax4.legend(loc="upper right", frameon=False, fontsize=18)
	leg4.get_frame().set_facecolor('none')
	ax1i.set_xticks([0,200,400,600])
	ax1i.set_yticks([39.4, 39.8, 40.2])
	# ax1.indicate_inset_zoom(ax1i, edgecolor="black")
	ax1.plot([650, 650], [39.3, 40.3], color="k", linewidth=1)
	ax1.plot([-50, -50], [39.3, 40.3], color="k", linewidth=1)
	ax1.plot([650, -50], [39.3, 39.3], color="k", linewidth=1)
	ax1.plot([650, -50], [40.3, 40.3], color="k", linewidth=1)
	ax1.plot([650, 659.55], [39.3, 44.4], color="k", linewidth=1)
	ax1.plot([-50, -81], [39.3, 44.4], color="k", linewidth=1)

	ax1.set_xticks([0,200,400,600,800])
	ax2.set_xticks([0,200,400,600,800])
	ax2.set_yticks(np.arange(34, 52, 2))
	ax4.set_yticks(np.arange(34, 52, 2))

	ax1.set_xticklabels(["","","","",""])
	ax2.set_xticklabels(["","","","",""])
	ax2.set_yticklabels(["","","","","", "","","",""])
	ax4.set_yticklabels(["","","","","", "","","",""])

	ax1.annotate("a)", xy=(ax1.get_xlim()[0], ax1.get_ylim()[1]), xytext=(3, -4), textcoords='offset points', ha="left", va="top", fontsize=20)
	ax2.annotate("b)", xy=(ax2.get_xlim()[0], ax2.get_ylim()[1]), xytext=(3, -4), textcoords='offset points', ha="left", va="top", fontsize=20)
	ax3.annotate("c)", xy=(ax3.get_xlim()[0], ax3.get_ylim()[1]), xytext=(3, -4), textcoords='offset points', ha="left", va="top", fontsize=20)
	ax4.annotate("d)", xy=(ax4.get_xlim()[0], ax4.get_ylim()[1]), xytext=(3, -4), textcoords='offset points', ha="left", va="top", fontsize=20)

	fig.canvas.manager.full_screen_toggle()
	figTitle = "Figure_6"
	title = f"{figTitle}"
	fig.savefig(f"{figPath}{title}.png", dpi=300)
	plt.close(fig=fig)

def LFASpace():
	fig = plt.figure(figsize=(16,9), constrained_layout=True)
	gs = fig.add_gridspec(2, 2)
	ax1 = fig.add_subplot(gs[1,0])
	ax2 = fig.add_subplot(gs[0,1])
	ax3 = fig.add_subplot(gs[1,1], sharex=ax2, sharey=ax1)

	LfCrit1 = 1
	LfCrit2 = 0.1
	df = dt.getData()

	mask = df["lf_linear-noTen-RMFit"] > LfCrit1
	ax1.scatter(df.loc[mask, "wrm_o"], df.loc[mask, "h0"], c=df.loc[mask, "lf_linear-noTen-RMFit"], marker="o", alpha=0.6, vmin=0, vmax=20)
	ax2.scatter(df.loc[mask, "xrm_o"], df.loc[mask, "wrm_o"], c=df.loc[mask, "lf_linear-noTen-RMFit"], marker="o", alpha=0.6, vmin=0, vmax=20)
	ax3.scatter(df.loc[mask, "xrm_o"], df.loc[mask, "h0"], c=df.loc[mask, "lf_linear-noTen-RMFit"], marker="o", alpha=0.6, vmin=0, vmax=20)

	mask = np.logical_and(df["lf_linear-noTen-RMFit"] <= LfCrit1, df["lf_linear-noTen-RMFit"] > LfCrit2) 
	ax1.scatter(df.loc[mask, "wrm_o"], df.loc[mask, "h0"], c=df.loc[mask, "lf_linear-noTen-RMFit"], marker="s", alpha=0.6, vmin=0, vmax=20)
	ax2.scatter(df.loc[mask, "xrm_o"], df.loc[mask, "wrm_o"], c=df.loc[mask, "lf_linear-noTen-RMFit"], marker="s", alpha=0.6, vmin=0, vmax=20)
	ax3.scatter(df.loc[mask, "xrm_o"], df.loc[mask, "h0"], c=df.loc[mask, "lf_linear-noTen-RMFit"], marker="s", alpha=0.6, vmin=0, vmax=20)

	mask = df["lf_linear-noTen-RMFit"] <= LfCrit2
	ax1.scatter(df.loc[mask, "wrm_o"], df.loc[mask, "h0"], c=df.loc[mask, "lf_linear-noTen-RMFit"], marker="^", alpha=0.6, vmin=0, vmax=20)
	ax2.scatter(df.loc[mask, "xrm_o"], df.loc[mask, "wrm_o"], c=df.loc[mask, "lf_linear-noTen-RMFit"], marker="^", alpha=0.6, vmin=0, vmax=20)
	ax3.scatter(df.loc[mask, "xrm_o"], df.loc[mask, "h0"], c=df.loc[mask, "lf_linear-noTen-RMFit"], marker="^", alpha=0.6, vmin=0, vmax=20)

	ax1.set_xlabel("wrm")
	ax1.set_ylabel("h0")

	ax2.set_xlabel("xrm")
	ax2.set_ylabel("wrm")

	ax3.set_xlabel("xrm")
	ax3.set_ylabel("h0")

	fig.canvas.manager.full_screen_toggle()
	figTitle = figSupName + it.stack()[0][3]
	title = f"{figTitle}"
	fig.savefig(f"{figPath}{title}.png", dpi=300)
	plt.close(fig=fig)

def LFAHistogram():
	figName = figSupName + it.stack()[0][3]
	
	d = dt.getData()
	d = d.loc[d.flag==2].reset_index()
	fig = plt.figure(figsize=(16,9), constrained_layout=True)
	ax = fig.add_subplot()
	ax.plot(d.h0, d.lw_o, "o", label="Model: Foot only", alpha=0.3)
	ax.plot(d.h0, d["lw_linear-noTen-RMFit"], "o", label="Model: Foot + Moment", alpha=0.3)
	ax.plot(d.h0, d.lw/6, "o", label=r"$\frac{1}{6}$Theory", alpha=0.3)
	ax.set_xlabel(r"Front thickness: $h_0$ (m)")
	ax.set_ylabel(r"Buoyancy wavelength: $l_w$ (m)")
	ax.legend()
	fig.canvas.manager.full_screen_toggle()
	fig.savefig(f"{figPath}{figName}_lws.png", dpi=300)
	plt.close(fig=fig)

	d = dt.getData()
	fig = plt.figure(figsize=(8,9), constrained_layout=True)
	ax = fig.add_subplot()
	d = d.loc[d.flag==2].reset_index()
	nb = 6
	lfs = np.linspace(0, 50, nb)
	Xcrit = [[] for i in range(nb)]
	bins = np.arange(0,400,5)
	for i in d.index:
		m = ph.modelLinNoTenDim(h0=d.loc[i, "h0"])
		lf = d.loc[i, "lf_linear-noTen-RMFit"]
		xcrit = m.getXCrit(lw=d.loc[i, "lw_linear-noTen-RMFit"], lf=d.loc[i, "lf_linear-noTen-RMFit"], a=d.loc[i, "a_linear-noTen-RMFit"])
		for j in range(nb-1):
			if lf>=lfs[j] and lf<=lfs[j+1]:
				break
		Xcrit[j].append(xcrit)

	ax.hist(Xcrit, density=False, histtype='bar', stacked=True, bins=bins,alpha=0.3, label=[f"Range lf = {lfs[i]} - {lfs[i+1]}" for i in range(nb-1)])
	ax.hist(d["xrm_o"]/3, color="r", label="Simple model", bins=bins, histtype='step', fill=False)
	ax.set_xlabel("Calving length: L (m)")
	ax.legend(fontsize=14)
	fig.savefig(f"{figPath}{figName}_xcrit.png", dpi=300)
	plt.close(fig=fig)



	d = dt.getData()
	fig = plt.figure(figsize=(10,9), constrained_layout=True)
	ax = fig.add_subplot()
	d = d.loc[d.flag==2].reset_index()
	bins = np.arange(0,320,10)
	mask_neg_m = d["a_linear-noTen-RMFit"]<=0
	mask_pos_m = d["a_linear-noTen-RMFit"]>0
	ax.hist([d.loc[mask_neg_m, "lw_linear-noTen-RMFit"].values, d.loc[mask_pos_m, "lw_linear-noTen-RMFit"].values], density=False, histtype='bar', stacked=True, label=["Full Model: Negative moment","Full Model: Positive moment"], alpha=0.3, bins=bins, color=["g","b"])
	ax.hist(d["lw_o"], color="r", label="Simple model", bins=bins, histtype='step', fill=False)
	ax.set_xlabel(r"Buoyancy wavelength: $l_w$ (m)")
	ax.set_ylabel("Count")
	fig.savefig(f"{figPath}{figName}_lw.png", dpi=300)
	plt.close(fig=fig)



	d = dt.getData()
	fig = plt.figure(figsize=(16,9), constrained_layout=True)
	gs = fig.add_gridspec(1, 2)
	ax1 = fig.add_subplot(gs[0])
	d = d.loc[d.flag==2].reset_index()
	bins = np.arange(0,320,10)
	mask_neg_m = d["a_linear-noTen-RMFit"]<=0
	mask_pos_m = d["a_linear-noTen-RMFit"]>0
	ax1.hist([d.loc[mask_neg_m, "lw_linear-noTen-RMFit"].values, d.loc[mask_pos_m, "lw_linear-noTen-RMFit"].values], density=False, histtype='bar', stacked=True, label=["Full Model: Negative moment","Full Model: Positive moment"], alpha=0.3, bins=bins, color=["g","b"])
	ax1.hist(d["lw_o"], color="r", label="Simple model", bins=bins, histtype='step', fill=False)
	ax1.set_xlabel(r"Buoyancy wavelength: $l_w$ (m)")
	ax1.set_ylabel("Count")

	ax2 = fig.add_subplot(gs[1], sharey=ax1)
	bins = np.arange(0,400,10)
	xcritF = []
	for i in d[mask_neg_m].index:
		m = ph.modelLinNoTenDim(h0=d.loc[i, "h0"])
		if 1e-3*m.getSigMax(lw=d.loc[i, "lw_linear-noTen-RMFit"], lf=d.loc[i, "lf_linear-noTen-RMFit"], a=d.loc[i, "a_linear-noTen-RMFit"])<0:
			continue

		xcritF.append(m.getXCrit(lw=d.loc[i, "lw_linear-noTen-RMFit"], lf=d.loc[i, "lf_linear-noTen-RMFit"], a=d.loc[i, "a_linear-noTen-RMFit"]))
	xcritM = []
	for i in d[mask_pos_m].index:
		m = ph.modelLinNoTenDim(h0=d.loc[i, "h0"])
		if 1e-3*m.getSigMax(lw=d.loc[i, "lw_linear-noTen-RMFit"], lf=d.loc[i, "lf_linear-noTen-RMFit"], a=d.loc[i, "a_linear-noTen-RMFit"])<0:
			continue
		xcritM.append(m.getXCrit(lw=d.loc[i, "lw_linear-noTen-RMFit"], lf=d.loc[i, "lf_linear-noTen-RMFit"], a=d.loc[i, "a_linear-noTen-RMFit"]))
	xcritZ = []
	cpt = 0
	for i in d.index:
		m = ph.modelLinNoTenDim(h0=d.loc[i, "h0"])
		if m.getXCrit(lw=d.loc[i, "lw_linear-noTen-RMFit"], lf=d.loc[i, "lf_linear-noTen-RMFit"], a=d.loc[i, "a_linear-noTen-RMFit"])<10:
			cpt+=1
			continue
		xcritZ.append(d.loc[i, "xrm_o"]/3)

	print(cpt)
	print(np.mean(xcritZ), np.std(xcritZ))

	ax2.hist([xcritF, xcritM], density=False, histtype='bar', stacked=True, label=["Full Model: Negative moment","Full Model: Positive moment"], alpha=0.3, bins=bins, color=["g","b"])
	ax2.hist(d["xrm_o"]/3, color="r", label="Simple model", bins=bins, histtype='step', fill=False)
	ax2.hist(xcritZ, color="k", label="Simple model", bins=bins, histtype='step', fill=False)
	ax2.set_xlabel("Calving length: L (m)")
	ax1.annotate("a)", xy=(ax1.get_xlim()[0], ax1.get_ylim()[1]), xytext=(3, -4), textcoords='offset points', ha="left", va="top", fontsize=18)
	ax2.annotate("b)", xy=(ax2.get_xlim()[0], ax2.get_ylim()[1]), xytext=(3, -4), textcoords='offset points', ha="left", va="top", fontsize=18)
	ax2.legend()
	fig.savefig(f"{figPath}{figName}_lwL.png", dpi=300)
	plt.close(fig=fig)



	d = dt.getData()
	fig = plt.figure(figsize=(16,9), constrained_layout=True)
	gs = fig.add_gridspec(1, 2)
	d = d.loc[d.flag==2].reset_index()

	ax1 = fig.add_subplot(gs[0])
	bins = np.arange(0,65,1)
	mask_FF = d["a_linear-noTen-RMFit"]<=0
	mask_MM = d["a_linear-noTen-RMFit"]>0
	ax1.hist([d.loc[mask_FF, "lf_linear-noTen-RMFit"].values, d.loc[mask_MM, "lf_linear-noTen-RMFit"].values], density=False, histtype='bar', stacked=True, label=["Full Model: Negative moment","Full Model: Positive moment"], alpha=0.3, bins=bins, color=["g","b"])
	ax1.hist(d["lf_s0"], color="r", label="Simple model", bins=bins, histtype='step', fill=False)
	ax1.set_xlabel(r"Foot length: $l_f$ (m)")

	ax2 = fig.add_subplot(gs[1], sharey=ax1)
	bins = np.arange(0,300,5)
	sigMaxF = []
	for i in d.index:
		if d.loc[i, "a_linear-noTen-RMFit"]<=0:
			m = ph.modelLinNoTenDim(h0=d.loc[i, "h0"])
			sigMaxF.append(1e-3*m.getSigMax(lw=d.loc[i, "lw_linear-noTen-RMFit"], lf=d.loc[i, "lf_linear-noTen-RMFit"], a=d.loc[i, "a_linear-noTen-RMFit"]))
	sigMaxM = []
	for i in d.index:
		if d.loc[i, "a_linear-noTen-RMFit"]>0:
			m = ph.modelLinNoTenDim(h0=d.loc[i, "h0"])
			xcrit = m.getXCrit(lw=d.loc[i, "lw_linear-noTen-RMFit"], lf=d.loc[i, "lf_linear-noTen-RMFit"], a=d.loc[i, "a_linear-noTen-RMFit"])
			if xcrit<=1:
				continue
			sigMaxM.append(1e-3*m.getSigMax(lw=d.loc[i, "lw_linear-noTen-RMFit"], lf=d.loc[i, "lf_linear-noTen-RMFit"], a=d.loc[i, "a_linear-noTen-RMFit"]))

	sigMaxS = []
	for i in d.index:
		m = ph.modelLinNoTenDim(h0=d.loc[i, "h0"])
		sigMaxS.append(1e-3*m.getSigMax(lw=d.loc[i, "lw_o"], lf=d.loc[i, "lf_s0"], a=0))
	
	ax2.hist([sigMaxF, sigMaxM], density=False, histtype='bar', stacked=True, label=["Full Model: Negative moment","Full Model: Positive moment"], alpha=0.3, bins=bins, color=["g","b"])
	ax2.hist(sigMaxS, color="r", label="Simple model", bins=bins, histtype='step', fill=False)
	ax2.set_xlabel(r"Maximum stress: $\sigma_{max}$ (kPa)")
	ax1.annotate("a)", xy=(ax1.get_xlim()[0], ax1.get_ylim()[1]), xytext=(3, -4), textcoords='offset points', ha="left", va="top", fontsize=18)
	ax2.annotate("b)", xy=(ax2.get_xlim()[0], ax2.get_ylim()[1]), xytext=(3, -4), textcoords='offset points', ha="left", va="top", fontsize=18)
	fig.savefig(f"{figPath}{figName}_lfxcrit.png", dpi=300)
	plt.close(fig=fig)



	d = dt.getData()
	fig = plt.figure(figsize=(16,9), constrained_layout=True)
	gs = fig.add_gridspec(1, 3)
	d = d.loc[d.flag==2].reset_index()

	ax1 = fig.add_subplot(gs[0])
	bins = np.arange(0,320,10)
	mask_neg_m = d["a_linear-noTen-RMFit"]<=0
	mask_pos_m = d["a_linear-noTen-RMFit"]>0
	ax1.hist([d.loc[mask_neg_m, "lw_linear-noTen-RMFit"].values, d.loc[mask_pos_m, "lw_linear-noTen-RMFit"].values], density=False, histtype='bar', stacked=True, label=["Full Model: Negative moment","Full Model: Positive moment"], alpha=0.3, bins=bins, color=["g","b"])
	ax1.hist(d["lw_o"], color="r", label="Foot-only Model", bins=bins, histtype='step', fill=False)
	ax1.set_xlabel(r"Buoyancy wavelength: $l_w$ (m)")
	ax1.set_ylabel("Count")
	fig.savefig(f"{figPath}{figName}_lw.png", dpi=300)

	ax2 = fig.add_subplot(gs[1], sharey=ax1)
	bins = np.arange(0,65,1)
	mask_FF = d["a_linear-noTen-RMFit"]<=0
	mask_MM = d["a_linear-noTen-RMFit"]>0
	ax2.hist([d.loc[mask_FF, "lf_linear-noTen-RMFit"].values, d.loc[mask_MM, "lf_linear-noTen-RMFit"].values], density=False, histtype='bar', stacked=True, label=["Full Model: Negative moment","Full Model: Positive moment"], alpha=0.3, bins=bins, color=["g","b"])
	ax2.hist(d["lf_s0"], color="r", label="Simple model", bins=bins, histtype='step', fill=False)
	ax2.set_xlabel(r"Foot length: $l_f$ (m)")

	ax3 = fig.add_subplot(gs[2], sharey=ax2)
	bins = np.arange(0,300,5)
	sigMaxF = []
	for i in d.index:
		if d.loc[i, "a_linear-noTen-RMFit"]<=0:
			m = ph.modelLinNoTenDim(h0=d.loc[i, "h0"])
			sigMaxF.append(1e-3*m.getSigMax(lw=d.loc[i, "lw_linear-noTen-RMFit"], lf=d.loc[i, "lf_linear-noTen-RMFit"], a=d.loc[i, "a_linear-noTen-RMFit"]))
	sigMaxM = []
	for i in d.index:
		if d.loc[i, "a_linear-noTen-RMFit"]>0:
			m = ph.modelLinNoTenDim(h0=d.loc[i, "h0"])
			xcrit = m.getXCrit(lw=d.loc[i, "lw_linear-noTen-RMFit"], lf=d.loc[i, "lf_linear-noTen-RMFit"], a=d.loc[i, "a_linear-noTen-RMFit"])
			if xcrit<=1:
				continue
			sigMaxM.append(1e-3*m.getSigMax(lw=d.loc[i, "lw_linear-noTen-RMFit"], lf=d.loc[i, "lf_linear-noTen-RMFit"], a=d.loc[i, "a_linear-noTen-RMFit"]))

	sigMaxS = []
	for i in d.index:
		m = ph.modelLinNoTenDim(h0=d.loc[i, "h0"])
		sigMaxS.append(1e-3*m.getSigMax(lw=d.loc[i, "lw_o"], lf=d.loc[i, "lf_s0"], a=0))
	
	ax3.hist([sigMaxF, sigMaxM], density=False, histtype='bar', stacked=True, label=["Full Model: Negative moment","Full Model: Positive moment"], alpha=0.3, bins=bins, color=["g","b"])
	ax3.hist(sigMaxS, color="r", label="Foot-only Model", bins=bins, histtype='step', fill=False)
	ax3.set_xlabel(r"Maximum stress: $\sigma_{max}$ (kPa)")
	ax3.set_xlabel(r"Maximum stress: $\sigma_{max}$ (kPa)")
	ax1.legend(fontsize=15)
	ax1.annotate("a)", xy=(ax1.get_xlim()[0], ax1.get_ylim()[1]), xytext=(3, -4), textcoords='offset points', ha="left", va="top", fontsize=18)
	ax2.annotate("b)", xy=(ax2.get_xlim()[0], ax2.get_ylim()[1]), xytext=(3, -4), textcoords='offset points', ha="left", va="top", fontsize=18)
	ax3.annotate("c)", xy=(ax3.get_xlim()[0], ax3.get_ylim()[1]), xytext=(3, -4), textcoords='offset points', ha="left", va="top", fontsize=18)
	fig.savefig(f"{figPath}{figName}_lwlfxcrit.png", dpi=300)
	plt.close(fig=fig)

def tripleHistogram():
	figName = "Figure_7"
	d = dt.getData()
	fig = plt.figure(figsize=(16,9), constrained_layout=True)
	gs = fig.add_gridspec(1, 3)
	d = d.loc[d.flag==2].reset_index()

	ax1 = fig.add_subplot(gs[0])
	bins = np.arange(0,320,10)
	mask_neg_m = d["a_linear-noTen-RMFit"]<=0
	mask_pos_m = d["a_linear-noTen-RMFit"]>0
	ax1.hist([d.loc[mask_neg_m, "lw_linear-noTen-RMFit"].values, d.loc[mask_pos_m, "lw_linear-noTen-RMFit"].values], density=False, histtype='bar', stacked=True, label=["Full Model: Negative moment","Full Model: Positive moment"], alpha=0.3, bins=bins, color=["g","b"])
	ax1.hist(d["lw_o"], color="r", label="Foot-only Model", bins=bins, histtype='step', fill=False)
	ax1.set_xlabel(r"Buoyancy wavelength: $l_w$ (m)")
	ax1.set_ylabel("Count")

	ax2 = fig.add_subplot(gs[1], sharey=ax1)
	bins = np.arange(0,65,1)
	mask_FF = d["a_linear-noTen-RMFit"]<=0
	mask_MM = d["a_linear-noTen-RMFit"]>0
	ax2.hist([d.loc[mask_FF, "lf_linear-noTen-RMFit"].values, d.loc[mask_MM, "lf_linear-noTen-RMFit"].values], density=False, histtype='bar', stacked=True, label=["Full Model: Negative moment","Full Model: Positive moment"], alpha=0.3, bins=bins, color=["g","b"])
	ax2.hist(d["lf_s0"], color="r", label="Simple model", bins=bins, histtype='step', fill=False)
	ax2.set_xlabel(r"Foot length: $l_f$ (m)")

	ax3 = fig.add_subplot(gs[2], sharey=ax2)
	bins = np.arange(0,300,5)
	sigMaxF = []
	for i in d.index:
		if d.loc[i, "a_linear-noTen-RMFit"]<=0:
			m = ph.modelLinNoTenDim(h0=d.loc[i, "h0"])
			sigMaxF.append(1e-3*m.getSigMax(lw=d.loc[i, "lw_linear-noTen-RMFit"], lf=d.loc[i, "lf_linear-noTen-RMFit"], a=d.loc[i, "a_linear-noTen-RMFit"]))
	sigMaxM = []
	for i in d.index:
		if d.loc[i, "a_linear-noTen-RMFit"]>0:
			m = ph.modelLinNoTenDim(h0=d.loc[i, "h0"])
			xcrit = m.getXCrit(lw=d.loc[i, "lw_linear-noTen-RMFit"], lf=d.loc[i, "lf_linear-noTen-RMFit"], a=d.loc[i, "a_linear-noTen-RMFit"])
			if xcrit<=1:
				continue
			sigMaxM.append(1e-3*m.getSigMax(lw=d.loc[i, "lw_linear-noTen-RMFit"], lf=d.loc[i, "lf_linear-noTen-RMFit"], a=d.loc[i, "a_linear-noTen-RMFit"]))

	sigMaxS = []
	for i in d.index:
		m = ph.modelLinNoTenDim(h0=d.loc[i, "h0"])
		sigMaxS.append(1e-3*m.getSigMax(lw=d.loc[i, "lw_o"], lf=d.loc[i, "lf_s0"], a=0))
	
	ax3.hist([sigMaxF, sigMaxM], density=False, histtype='bar', stacked=True, label=["Full Model: Negative moment","Full Model: Positive moment"], alpha=0.3, bins=bins, color=["g","b"])
	ax3.hist(sigMaxS, color="r", label="Foot-only Model", bins=bins, histtype='step', fill=False)
	ax3.set_xlabel(r"Maximum stress: $\sigma_{max}$ (kPa)")
	ax3.set_xlabel(r"Maximum stress: $\sigma_{max}$ (kPa)")
	ax1.legend(fontsize=15)
	ax1.annotate("a)", xy=(ax1.get_xlim()[0], ax1.get_ylim()[1]), xytext=(3, -4), textcoords='offset points', ha="left", va="top", fontsize=18)
	ax2.annotate("b)", xy=(ax2.get_xlim()[0], ax2.get_ylim()[1]), xytext=(3, -4), textcoords='offset points', ha="left", va="top", fontsize=18)
	ax3.annotate("c)", xy=(ax3.get_xlim()[0], ax3.get_ylim()[1]), xytext=(3, -4), textcoords='offset points', ha="left", va="top", fontsize=18)
	fig.savefig(f"{figPath}{figName}.png", dpi=300)
	plt.close(fig=fig)
	
def getStatRange():
	d = dt.getData()
	d = d.loc[d.flag==2].reset_index()

	sOnFt = []
	for i in d.index:
		m = ph.modelLinNoTenDim(h0=d.loc[i, "h0"])
		if m.getXCrit(lw=d.loc[i, "lw_linear-noTen-RMFit"], lf=d.loc[i, "lf_linear-noTen-RMFit"], a=d.loc[i, "a_linear-noTen-RMFit"]) >=10:
			sOnFt.append(d.loc[i, "h0"])

	print("-----h0-----")
	print(f"{d['h0'].mean():.2f} +- {d['h0'].std():.2f}")
	print(f"From foot-only selected: {np.mean(sOnFt):.2f} +- {np.std(sOnFt):.2f}")
	print()

	sOnFt = []
	for i in d.index:
		m = ph.modelLinNoTenDim(h0=d.loc[i, "h0"])
		if m.getXCrit(lw=d.loc[i, "lw_linear-noTen-RMFit"], lf=d.loc[i, "lf_linear-noTen-RMFit"], a=d.loc[i, "a_linear-noTen-RMFit"]) >=10:
			sOnFt.append(d.loc[i, "lf_linear-noTen-RMFit"])

	print("-----LF-----")
	print(f"Foot only : {d['lf_s0'].mean():.2f} +- {d['lf_s0'].std():.2f}")
	print(f"Full model : {d['lf_linear-noTen-RMFit'].mean():.2f} +- {d['lf_linear-noTen-RMFit'].std():.2f}")
	print(f"Full model (neg mm) : {d.loc[d['a_linear-noTen-RMFit']<=0, 'lf_linear-noTen-RMFit'].mean():.2f} +- {d.loc[d['a_linear-noTen-RMFit']<=0, 'lf_linear-noTen-RMFit'].std():.2f}")
	print(f"Full model (pos mm) : {d.loc[d['a_linear-noTen-RMFit']>0, 'lf_linear-noTen-RMFit'].mean():.2f} +- {d.loc[d['a_linear-noTen-RMFit']>0, 'lf_linear-noTen-RMFit'].std():.2f}")
	print(f"From foot-only selected: {np.mean(sOnFt):.2f} +- {np.std(sOnFt):.2f}")
	print()

	sOnFt = []
	for i in d.index:
		m = ph.modelLinNoTenDim(h0=d.loc[i, "h0"])
		if m.getXCrit(lw=d.loc[i, "lw_linear-noTen-RMFit"], lf=d.loc[i, "lf_linear-noTen-RMFit"], a=d.loc[i, "a_linear-noTen-RMFit"]) >=10:
			sOnFt.append(d.loc[i, "lw_linear-noTen-RMFit"])


	print("-----LW-----")
	print(f"From Th: {d['lw'].mean():.2f} +- {d['lw'].std():.2f}")
	print(f"Foot only : {d['lw_o'].mean():.2f} +- {d['lw_o'].std():.2f}")
	print(f"Full model : {d['lw_linear-noTen-RMFit'].mean():.2f} +- {d['lw_linear-noTen-RMFit'].std():.2f}")
	print(f"Full model (neg mm) : {d.loc[d['a_linear-noTen-RMFit']<=0, 'lw_linear-noTen-RMFit'].mean():.2f} +- {d.loc[d['a_linear-noTen-RMFit']<=0, 'lw_linear-noTen-RMFit'].std():.2f}")
	print(f"Full model (pos mm) : {d.loc[d['a_linear-noTen-RMFit']>0, 'lw_linear-noTen-RMFit'].mean():.2f} +- {d.loc[d['a_linear-noTen-RMFit']>0, 'lw_linear-noTen-RMFit'].std():.2f}")
	print(f"From foot-only selected: {np.mean(sOnFt):.2f} +- {np.std(sOnFt):.2f}")
	print()

	sAll = []
	sOnFt = []
	for i in d.index:
		m = ph.modelLinNoTenDim(h0=d.loc[i, "h0"])
		xcrit = m.getXCrit(lw=d.loc[i, "lw_o"], lf=d.loc[i, "lf_s0"], a=0)
		sAll.append(xcrit)
		if m.getXCrit(lw=d.loc[i, "lw_linear-noTen-RMFit"], lf=d.loc[i, "lf_linear-noTen-RMFit"], a=d.loc[i, "a_linear-noTen-RMFit"]) >=10:
			sOnFt.append(xcrit)

	print("-----Lcalve-----")
	print(f"From foot-only: {np.mean(sAll):.2f} +- {np.std(sAll):.2f}")
	print(f"From foot-only selected: {np.mean(sOnFt):.2f} +- {np.std(sOnFt):.2f}")
	print("")

def plotBermExample():
	figName = "Figure_A2"
	i = 51
	df = dt.getData()
	dfi = df.loc[i]
	r, f = dt.getNormalizedRF(dfi)
	fig = plt.figure(figsize=(16,9), constrained_layout=True)
	ax = fig.add_subplot()
	ax.plot(r[:350]/1e3, f[:350], "ko", zorder=5, markersize=4)
	ax.set_xlabel("Distance from front (km)")
	ax.set_ylabel("Freeboard (m)")
	ax.set_xlim(7.2, -0.5)
	print(dfi.frontLat, ((dfi.frontLon - 180) % 360) - 180 )
	fig.savefig(f"{figPath}{figName}.png", dpi=300)
	plt.close(fig=fig)

def findBestFit():
	df = dt.getData()
	for i in df.loc[df.flag==2].index:
		print(i)
		dfi = df.loc[i]
		xrm = dfi["xrm_o"]
		wrm = dfi["wrm_o"]
		h0 = dfi["h0"]
		lw_o = dfi["lw_o"]
		lw_s2 = dfi["lw_s2"]
		lf_s2 = dfi["lf_s2"]
		lw_lin = dfi["lw_linear-noTen"]
		lf_lin = dfi["lf_linear-noTen"]
		lw_RMFit = dfi["lw_linear-noTen-RMFit"]
		lf_RMFit = dfi["lf_linear-noTen-RMFit"]
		a_RMFit = dfi["a_linear-noTen-RMFit"]

		if not (lf_RMFit<=1 and a_RMFit<0):
			continue

		r, f = dt.getNormalizedRF(dfi)

		dire = "test14-verif"
		os.makedirs(dire, exist_ok=True)
		filePath = f"{dire}/{i:03d}.png"

		if os.path.isfile(filePath.replace("/", '/nm_')) or os.path.isfile(filePath.replace("/", '/ff_')) or os.path.isfile(filePath.replace("/", '/mm_')):
			continue
		open(filePath.replace("/", '/nm_'), "w").close()
		open(filePath.replace("/", '/mm_'), "w").close()
		open(filePath.replace("/", '/ff_'), "w").close()

		fig = plt.figure(figsize=(16,9), constrained_layout=True)
		ax = fig.add_subplot()

		idxMin = (np.abs(r - xrm)).argmin()
		idxMax = int(idxMin*1.7) + 4
		x = np.linspace(0, r[idxMax+10], 10000)

		ax.plot(r[:idxMax+10], f[:idxMax+10], "ks", label="ICESat 2 raw", zorder=5)
		xSpline = np.linspace(0,dfi["xrm_o"],1000000)
		ySpline = sp.interpolate.make_interp_spline(x=r[:idxMax+10], y=f[:idxMax+10], k=3)(xSpline)
		wp = np.gradient(ySpline, xSpline)
		wpp_np_xrm4 = np.mean(np.gradient(wp, xSpline)[xSpline<=dfi["xrm_o"]/4])
		wpp_np_xrm8 = np.mean(np.gradient(wp, xSpline)[xSpline<=dfi["xrm_o"]/8])
		wpp_np_0 = np.gradient(wp, xSpline)[0]
		wpp_raw = (f[2] - 2*f[1] + f[0]) / (( r[2] - r[0])/2)**2
		ax.plot(xSpline[::25], ySpline[::25], "k", label=f"Spline: {wpp_raw:.1e} {wpp_np_0:.1e} {wpp_np_xrm8:.1e} {wpp_np_xrm4:.1e} {wp[xSpline<=xrm/8].mean():.1e}", zorder=4)

		idx = (np.abs(r - dfi["xwrm_o"])).argmin()

		ax.plot([dfi["xrm_o"]]*2, [f[:idxMax+10].min(), f[:idxMax+10].max()],"k", zorder=10)
		mSimple = lambda x : wrm/(1 + np.exp(-3*np.pi/4)/2**0.5)*np.cos(x/(2**0.5)/dfi.lw_o)*np.exp(-x/(2**0.5)/dfi.lw_o)
		y_simple = mSimple(x)
		y_simple = y_simple - y_simple[0] + f[idx]
		ax.plot(x, y_simple, "r-.", label=f"Simple model: lw = {lw_o:.1f}, lf = {0:.1f}, a = {0:.1f}", zorder=8)
		mSimple2 = lambda x : np.sqrt(2)*(1-rhotio)*rhotio*h0*lf_s2/lw_s2*np.exp(-x/np.sqrt(2)/lw_s2)*np.cos(x/np.sqrt(2)/lw_s2)
		y_simple2 = mSimple2(x)
		y_simple2 = y_simple2 - y_simple2[0] + f[idx]
		ax.plot(x, y_simple2, "m-.", label=f"Simple model : lw = {lw_s2:.1f}, lf = {lf_s2:.1f}, a = {0:.1f}", zorder=6)

		mLinearNoTen = ph.modelLinNoTenDim(h0=h0, verbose=0)
		if not np.isnan(lw_RMFit):
			y_linRMFit = mLinearNoTen.getProfile3(lw=lw_RMFit, lf=lf_RMFit, a=a_RMFit)(x)
			y_linRMFit = y_linRMFit - y_linRMFit[0] + f[0]
			xrm = mLinearNoTen.getRM()[0]

			ax.plot(x, y_linRMFit, "-g", label=f"Linear model RMFit: lw = {lw_RMFit:.1f}, lf = {lf_RMFit:.1f}, a = {a_RMFit:.1e}", zorder=7)
			ax.plot([xrm]*2, [f[:idxMax+10].min(), f[:idxMax+10].max()], "-.", color=plt.gca().lines[-1].get_color(), alpha=0.75, zorder=7)
		else:
			ax.plot([0,r[idxMax]], [f[0], f[0]],"-g", label="No sol")

		def fit(x, lw, lf, a):
				profile = s.getProfile3(lw=lw, lf=lf, a=a)
				return profile(x) - profile(0) 
		s = ph.modelLinNoTenDim(h0=h0, verbose=0)
		for idx in range(idxMin, idxMax):
			r_, f_ = r[:idx], f[:idx]
			f_ = f_ - f_[0]
			try:
				var = sp.optimize.curve_fit(f=fit, xdata=r_, ydata=f_, p0=(100, 0, 0.01), bounds=([0,0,-10], [2000,2000,10]) )[0]
			except RuntimeError:
				continue
			var = {"a":var[2], "lf":var[1], "lw":var[0]}
			y  = s.getProfile3(**var)(x)
			y = y - y[0] + f[0]
			xrm = s.getRM()[0]
			ax.plot(x,y, ":"if var["lf"]<0.1 else "-.", label=f"idx = {idx}, lf ={var['lf']:.1f} m, a = {var['a']:.1e}, e={ph.rootSquareError(lambda x: fit(x, **var), r_[r_<=dfi['xrm_o']*1.2], f_[r_<=dfi['xrm_o']*1.2]):.1e}", alpha=0.75, zorder=6)
			ax.plot([xrm]*2, [f[:idxMax+10].min(), f[:idxMax+10].max()], ":" if var["lf"]<0.1 else "-.", color=plt.gca().lines[-1].get_color(), alpha=0.75, zorder=6)


		idx = int(idxMin*1.1) + 2
		ax.plot([r[:idx][-1]]*2, [f[:idxMax+10].min(), f[:idxMax+10].max()],"k" )

		r_, f_ = r[:idx], f[:idx]
		f_ = f_ - f_[0]

		cpt1 = 0
		for i in range(10):
			f__ = f_ + np.random.normal(loc=0, scale=0.1, size=f_.shape)
			
			try:
				var = sp.optimize.curve_fit(f=fit, xdata=r_, ydata=f__, p0=(100, 0, 0.01), bounds=([0,0,-10], [2000,2000,10]) )[0]
			except RuntimeError:
				continue
			var = {"a":var[2], "lf":var[1], "lw":var[0]}
			y  = s.getProfile3(**var)(x)
			y = y - y[0] + f[0]
			xrm = s.getRM()[0]
			ax.plot(x,y,color="red", alpha=0.01, zorder=4)
			ax.plot([xrm]*2, [f[:idxMax+10].min(), f[:idxMax+10].max()],color="red", alpha=0.01, zorder=4)
			if var["lf"]<1:
				cpt1+=1

		cpt2 = 0
		for i in range(10):
			f__ = f_ + np.random.normal(loc=0, scale=0.1, size=f_.shape)
			
			try:
				var = sp.optimize.curve_fit(f=fit, xdata=r_, ydata=f__, p0=(100, 10, -0.01), bounds=([0,0,-10], [2000,2000,10]) )[0]
			except RuntimeError:
				continue
			var = {"a":var[2], "lf":var[1], "lw":var[0]}
			y  = s.getProfile3(**var)(x)
			y = y - y[0] + f[0]
			xrm = s.getRM()[0]
			ax.plot(x,y,color="green", alpha=0.01, zorder=4)
			ax.plot([xrm]*2, [f[:idxMax+10].min(), f[:idxMax+10].max()],color="green", alpha=0.01, zorder=4)
			if var["lf"]<1:
				cpt2+=1

		ax.set_title(f"{round(cpt1/10)}% || {round(cpt2/10)}%")
		ax.legend(fontsize=10, loc="upper right")
		ax.set_xlim(-1,r[idxMax+10])
		ax.set_ylim(f[:idxMax+10].min(), f[:idxMax+10].max())

		if wp[xSpline<dfi["xrm_o"]/8].mean()>=0:
			os.remove(filePath.replace("/", '/ff_'))
			os.remove(filePath.replace("/", '/mm_'))
			fig.savefig(filePath.replace("/", '/nm_'), dpi=300)
			plt.close(fig=fig)
			continue
		
		elif lf_RMFit>=1:
			os.remove(filePath.replace("/", '/nm_'))
			os.remove(filePath.replace("/", '/mm_'))
			fig.savefig(filePath.replace("/", '/ff_'), dpi=300)
			plt.close(fig=fig)
			continue

		else :
			os.remove(filePath.replace("/", '/ff_'))
			os.remove(filePath.replace("/", '/nm_'))
			fig.savefig(filePath.replace("/", '/mm_'), dpi=300)
			plt.close(fig=fig)
			continue
		
