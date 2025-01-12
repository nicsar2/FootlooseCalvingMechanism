import inspect as it
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as crs
import scipy as sp

import dataICESAT as dt

from params import figPath, Rt

plt.rc('font', size=24)         # controls default text sizes
plt.rc('axes', titlesize=24)    # fontsize of the axes title
plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=21)   # fontsize of the tick labels
plt.rc('ytick', labelsize=21)   # fontsize of the tick labels
plt.rc('legend', fontsize=24)   # legend fontsize
plt.rc('figure', titlesize=24)  # fontsize of the figure title

def plotTransectOnMap(): #Map plot of the RIS with the ICESat transect position overlaid
	df = dt.getData().sort_values("mTime")
	fig = plt.figure(figsize=(16,9))
	ax = fig.add_subplot(projection=crs.SouthPolarStereo(central_longitude=180))
	ext = [163,202,-82,-74]
	ax.set_extent(ext)
	for i in df.index:
		ax.plot(df.loc[i,"lon"], df.loc[i,"lat"],transform=crs.PlateCarree())
	ax.coastlines()
	ax.stock_img()
	plt.show()

def Transects3D(): #Plot of ICESat transect in 3D lat/lon/elevation
	figTitle = "ICESAT1_" + it.stack()[0][3]
	df = dt.getData()
	fig = plt.figure(figsize=(12,9), constrained_layout=True)
	ax = fig.add_subplot(projection='3d')
	for i in df.index:
		if not df.loc[i,"isRM"]:
			continue

		lat = df.loc[i, "lat"]
		lon = df.loc[i, "lon"]
		f = df.loc[i, "f"]

		ax.plot(lat, lon, f, alpha=0.7)

	title = f"{figTitle}"
	fig.savefig(f"{figPath}{title}.png", dpi=300)
	plt.show()
	plt.close(fig=fig)

def RMInfoTimeSeries(): #Time series of the ICESat transect elevations with the rampart-moat features (height, position, foot-length)
	figTitle = "ICESAT1_" + it.stack()[0][3]
	df = dt.getData()

	fig = plt.figure(figsize=(15, 15), constrained_layout=True)
	gs = fig.add_gridspec(6, 1)
	ax1 = fig.add_subplot(gs[0:3, 0])
	ax2 = fig.add_subplot(gs[3, 0])
	ax3 = fig.add_subplot(gs[4, 0])
	ax4 = fig.add_subplot(gs[5, 0])

	color = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:brown","tab:purple"]
	k = 0
	for i in df.index:
		if not df.loc[i,"isRM"]:
			continue
	   
		t = df.loc[i, "mTime"]
		dfi = df.loc[i]
		r, f = dt.getRF(dfi)
		r, f = dt.normalizeProfile(r, f)
		Info = dt.getRMInfo(dfi)

		x = np.linspace(r.min(), 1500, 10000)
		y = sp.interpolate.make_interp_spline(r, f, 3)(x)
		ax1.plot(x/1000,y,"-.", color=color[k], alpha=0.6)
		ax1.plot(r/1000, f, "-o",label=df.loc[i,"mTime"], color=color[k], alpha=0.6)
		ax1.plot(Info["xrm"]/1000, Info["fxrm"], "s", color=color[k], alpha=0.6)
		ax1.plot([Info["xrm"]/1000]*100, np.linspace(Info["fxrm"],Info["fxrm"]+Info["hrm"],100), ":", color=color[k], alpha=0.6)
		ax1.plot([Info["xrm"]/1000]*100, np.linspace(Info["fxrm"],Info["fxrm"]+Info["hrm"],100), ":", color=color[k], alpha=0.6)
		ax1.plot(np.linspace(0,Info["xrm"]/1000,100), [Info["fxrm"]+Info["hrm"]]*100, ":", color=color[k], alpha=0.6)
		ax2.plot(t, Info["xrm"], "o", color=color[k])
		ax3.plot(t, Info["hrm"], "o", color=color[k])
		ax4.plot(t, Info["lf"], "o", color=color[k])
		k+=1

		ax1.set_xlim(-0.05, 2)
		ax1.set_ylim(35.5, 40)
		ax1.legend()
		ax1.set_ylabel("Profiles")
		ax2.set_ylabel("Xrm (m)")
		ax3.set_ylabel("Hrm (m)")
		ax4.set_ylabel("Lf (m)")
		ax1.set_xlabel("Projected atd (km)")
		ax4.set_xlabel("Time (y)")
	fig.canvas.manager.full_screen_toggle()
	title = f'{figTitle}.png'
	fig.savefig(f"{figPath}{title}.png", dpi=300)
	plt.close(fig=fig)

def TimeSerieSplit(): #Time series of the ICESat transect elevations split at the calving event
	figTitle = "ICESAT1_" + it.stack()[0][3]
	nvals = 1000000
	cmap = plt.cm.viridis(np.linspace(0,1,nvals))

	fig = plt.figure(figsize=(12,9), constrained_layout=True)
	gs = fig.add_gridspec(6, 1)

	ax1a = fig.add_subplot(gs[0:2,0])
	ax1b = fig.add_subplot(gs[2,0])
	ax2a = fig.add_subplot(gs[3:5,0])
	ax2b = fig.add_subplot(gs[5,0])

	df = dt.getData()
	for i in df.index:
		if df.loc[i, "mTime"]<2007: 
			ax1a.plot(df.loc[i,"atd"]-15, df.loc[i,"f"], color=cmap[int((df.loc[i, "mTime"]-2003)/(2010-2003)*nvals)])
			ax1b.plot(df.loc[i,"atd"]-15, df.loc[i,"f"], color=cmap[int((df.loc[i, "mTime"]-2003)/(2010-2003)*nvals)])
		else:
			ax2a.plot(df.loc[i,"atd"]-15, df.loc[i,"f"], color=cmap[int((df.loc[i, "mTime"]-2003)/(2010-2003)*nvals)])
			ax2b.plot(df.loc[i,"atd"]-15, df.loc[i,"f"], color=cmap[int((df.loc[i, "mTime"]-2003)/(2010-2003)*nvals)])

	ax1a.set_ylim(32, 45) 
	ax1b.set_ylim(-3, 4)
	ax1a.set_xlim([0,15])
	ax1b.set_xlim([0,15])
	ax2a.set_ylim(32, 45) 
	ax2b.set_ylim(-3, 4)
	ax2a.set_xlim([0,15])
	ax2b.set_xlim([0,15])

	ax1a.spines.bottom.set_visible(False)
	ax1b.spines.top.set_visible(False)
	ax1a.xaxis.tick_top()
	ax1a.tick_params(labeltop=False)  
	ax1b.xaxis.tick_bottom()
	ax2a.spines.bottom.set_visible(False)
	ax2b.spines.top.set_visible(False)
	ax2a.xaxis.tick_top()
	ax2a.tick_params(labeltop=False)  
	ax2b.xaxis.tick_bottom()

	d = .4  
	kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
				  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
	ax1a.plot([0, 1], [0, 0], transform=ax1a.transAxes, **kwargs)
	ax1b.plot([0, 1], [1, 1], transform=ax1b.transAxes, **kwargs)
	ax2a.plot([0, 1], [0, 0], transform=ax2a.transAxes, **kwargs)
	ax2b.plot([0, 1], [1, 1], transform=ax2b.transAxes, **kwargs)

	ax1a.xaxis.grid()
	ax1b.xaxis.grid()
	ax1b.set_yticks([-2.5, 0 , 2.5])
	ax1b.set_yticklabels([-2.5, 0 , 2.5])
	ax1a.set_yticks([35,40,45])
	ax1a.set_yticklabels([35,40,45])

	ax2a.xaxis.grid()
	ax2a.set_yticks([35,40,45])
	ax2a.set_yticklabels([35,40,45])

	ax2b.xaxis.grid()
	ax2b.set_xlabel('Along-track distance [km]', fontsize=18)
	ax2b.set_yticks([-2.5, 0 , 2.5])
	ax2b.set_yticklabels([-2.5, 0 , 2.5])


	fig.supylabel('Elevation [m]', fontsize=18)
	norm = mpl.colors.Normalize(vmin=2003,vmax=2010)
	sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('viridis'), norm=norm)
	cb = fig.colorbar(sm, label = 'year', ax = [ax1a,ax1b,ax2a,ax2b], orientation='vertical')
	cb.set_label("Year", fontsize=18)
	cb.ax.tick_params(labelsize=18)

	ax1a.set_title('ICESat track 0068', fontsize=18)
	ax1b.text(0.5,0,'(a) pre-calving', fontsize=18)
	ax2b.text(0.5,0,'(b) post-calving', fontsize=18)
	title = f"{figTitle}.png"
	fig.savefig(f"{figPath}{title}", dpi=300)
	plt.show()
	plt.close(fig=fig)

def TimeSerieRaw(plotVelocity=True): #Time series of the ICESat transect front positions
	figTitle = "ICESAT1_" + it.stack()[0][3]
	fig = plt.figure(figsize=(16,9), constrained_layout=True)
	ax = fig.add_subplot()

	df = dt.getData()
	X,Y = [], []
	for i in df.index:
		r, f = df.loc[i,"lat"]*Rt*np.pi/180/1000, df.loc[i,"f"]
		if "r0" not in locals():
			r0 = r[np.argmin(f>7)]
		r = r-r0
		Y.append(r[np.argmin(f>7)])
		X.append(df.loc[i,"mTime"])
	
	ax.plot(X, Y, "s-")

	velBC = sp.optimize.curve_fit(lambda x, a: a*x, X[:6]-np.min(X[:6]), Y[:6])[0][0]
	velAC = sp.optimize.curve_fit(lambda x, a: a*x, X[6:]-np.min(X[6:]), Y[6:]-Y[6])[0][0]
	print(f"Speed BC = {velBC:.3f} km/yr" )
	print(f"Speed AC = {velAC:.3f} km/yr" )
	
	if plotVelocity:
		fitPos = lambda x: velBC*x 
		x_ = np.linspace(X[0], X[6], 100)
		y_ = fitPos(x_-x_[0])
		ax.plot(x_, y_,"k:")

		fitPos = lambda x: velAC*x 
		x_ = np.linspace(X[6], X[-1], 100)
		y_ = fitPos(x_-x_[0])+Y[6]
		ax.plot(x_, y_,"k:")

	title = f"{figTitle}.png"
	fig.savefig(f"{figPath}{title}", dpi=300)
	plt.show()
	plt.close(fig=fig)
	
def TimeSeriesRM(plotVelocity=True): #FIGURE 2: Time series of the ICESat transect front positions and elevations colored
	figTitle = "Figure_2"
	fig = plt.figure(figsize=(16,16), constrained_layout=True)
	gs = fig.add_gridspec(3, 1)
	ax1 = fig.add_subplot(gs[0,0])
	ax2 = fig.add_subplot(gs[1,0])
	ax3 = fig.add_subplot(gs[2,0])
	# cmapName = plt.cm.gist_rainbow
	# cmap = cmapName(np.linspace(0,1,10001))
	col = ['tab:gray', 'tab:brown','tab:blue','tab:cyan','tab:green','tab:olive', 'tab:orange','tab:red','tab:pink','tab:purple' ,]

	df = dt.getData().sort_values("mTime", ignore_index=True)
	X, Y = [], []
	for i in df.index:
		r, f = df.loc[i,"lat"]*Rt*np.pi/180/1000, df.loc[i,"f"]
		if "r0" not in locals():
			r0 = r[np.argmin(f>7)]
		r = r-r0
		Y.append(r[np.argmin(f>7)])
		X.append(df.loc[i,"mTime"])

	for i in df.index:
		r, f = df.loc[i,"lat"]*Rt*np.pi/180/1000, df.loc[i,"f"]
		if "r0" not in locals():
			r0 = r[np.argmin(f>7)]
		r = r-r0
		# color = cmap[int(10000*(X[i]-X[0])/(X[-1]-X[0]))]
		color = col[i]
		if i<=5:
			ax1.plot(r,f, color=color, zorder=80, linewidth=3, label=f"{round(12*(X[i] - int(X[i]))):02d}/{int(X[i])}")
			ax1.fill_between(r,[0]*len(f),f, color=color, alpha=0.1, zorder=12-i)
		else:
			ax2.plot(r,f, color=color, zorder=80, linewidth=3, label=f"{round(12*(X[i] - int(X[i]))):02d}/{int(X[i])}")
			ax2.fill_between(r,[0]*len(f),f, color=color, alpha=0.1, zorder=12-i)

	ax1.plot([2.73]*10, np.linspace(30,50,10),":", color=col[6], linewidth=3.5, zorder=250)
	ax2.plot([2.73]*10, np.linspace(30,50,10),":", color=col[6], linewidth=3.5, zorder=250)
	ax1.plot([3.151]*10, np.linspace(30,50,10),":", color=col[5], linewidth=3.5, zorder=250)
	ax2.plot([3.151]*10, np.linspace(30,50,10),":", color=col[5], linewidth=3.5, zorder=250)

	ax3.plot(X, Y, "-k", zorder=1)
	# ax3.scatter(X, Y, c=[X[i]-X[0] for i in range(len(X))], s=250, cmap=cmapName, zorder=2)
	ax3.scatter(X, Y, color=col, s=250, zorder=2)
	ax1.add_patch(mpl.patches.FancyArrowPatch((3.151+0.01, 41), (2.73-0.01, 41), mutation_scale=50, color="g"))
	# ax1.text(2.15, 41.7, "Calving retreat: 950 m", zorder=252, color="g")

	velBC = sp.optimize.curve_fit(lambda x, a: a*x, X[:6]-np.min(X[:6]), Y[:6])[0][0]
	velAC = sp.optimize.curve_fit(lambda x, a: a*x, X[6:]-np.min(X[6:]), Y[6:]-Y[6])[0][0]
	print(f"Speed BC = {velBC:.3f} km/yr" )
	print(f"Speed AC = {velAC:.3f} km/yr" )
	
	if plotVelocity:
		fitPos = lambda x: velBC*x 
		x_ = np.linspace(X[0], X[6], 100)
		y_ = fitPos(x_-x_[0])
		ax3.plot(x_, y_,"k:")

		fitPos = lambda x: velAC*x 
		x__ = np.linspace(X[6], X[-1], 100)
		y__ = fitPos(x__-x__[0])+Y[6]
		ax3.plot(x__, y__,"k:")

	ax3.add_patch(mpl.patches.FancyArrowPatch((X[6], y_[-1]+0.03), (X[6], y__[0]-0.03), mutation_scale=30, color="g",zorder=250))
	ax3.text(X[6]-0.17, y_[-1]+0.1, "950 m", zorder=252, color="g")
	ax1.annotate("406 m", xy=(3.5, 40.93), color="green", ha="center", va="center")

	ax1.set_xlim(-2, 6)
	ax2.set_xlim(-2, 6)
	ax1.set_ylim(32.5, 42.5)
	ax2.set_ylim(32.5, 42.5)
	ax1.set_xticks([])
	ax1.set_ylabel("Freeboard (m)")
	ax2.set_xlabel("Advance (km)")
	ax2.set_ylabel("Freeboard (m)")
	ax3.set_xlabel("Time")
	ax3.set_ylabel("Advance (km)")
	ax1.text(-1.95, 41.8, "a)")
	ax2.text(-1.95, 41.8, "b)")
	ax3.text(2003.56, 5.47, "c)")
	ax1.legend()
	ax2.legend()

	title = f"{figTitle}"
	fig.savefig(f"{figPath}{title}.png", dpi=300)
	plt.close(fig=fig)

def TimeSerieColors(): #Time series of the ICESat transect front positions and elevations colored
	import dataMEaSUREs2 as dVEL
	figTitle = "ICESAT1_" + it.stack()[0][3]
	df = dt.getData().sort_values("mTime")
	print(df.lon[0].min())
	X,Y = [], []
	for i in df.index:
		r, f = df.loc[i,"lat"]*Rt*np.pi/180/1000, df.loc[i,"f"]
		if "r0" not in locals():
			r0 = r[np.argmin(f>7)]
		r = r-r0
		Y.append(r[np.argmin(f>7)])
		X.append(df.loc[i,"mTime"])

	cmap = plt.cm.jet(np.linspace(0,1,10001))

	fig = plt.figure(figsize=(16,9), constrained_layout=True)
	ax = fig.add_subplot()
	ax.plot(X, Y, "-k", zorder=1)
	ax.scatter(X, Y, c=[X[i]-X[0] for i in range(len(X))], s=250, cmap="jet", zorder=2)
	ax.set_xlabel("Time")
	ax.set_ylabel("Advance (km)")
	title = f"{figTitle}.png"
	fig.savefig(f"{figPath}{title}_0.png", dpi=300)

	iSplit=5
	for i in range(len(X)):	
		fig = plt.figure(figsize=(16,9), constrained_layout=True)
		ax = fig.add_subplot()
		ax2 = fig.add_axes([0.07, 0.64, 0.4, 0.35])
		if i>iSplit:
			ax3 = fig.add_axes([0.575, 0.115, 0.4, 0.35])

		for k in range(i+1):
			j = df.index[k]
			r, f = df.loc[j,"lat"]*Rt*np.pi/180/1000, df.loc[j,"f"]
			if "r0" not in locals():
				r0 = r[np.argmin(f>7)]
			r = r-r0
			if k>iSplit:
				ax3.plot(r,f, color=cmap[int(10000*(X[k]-X[0])/(X[-1]-X[0]))], zorder=80)
				if k==i:
					ax3.fill_between(r,[0]*len(f),f, color=cmap[int(10000*(X[k]-X[0])/(X[-1]-X[0]))], alpha=0.3, zorder=12-k)
				else:
					ax3.fill_between(r,[0]*len(f),f, color=cmap[int(10000*(X[k]-X[0])/(X[-1]-X[0]))], alpha=0.1, zorder=12-k)

			else:
				ax2.plot(r,f, color=cmap[int(10000*(X[k]-X[0])/(X[-1]-X[0]))], zorder=80)
				if k==i:
					ax2.fill_between(r,[0]*len(f),f, color=cmap[int(10000*(X[k]-X[0])/(X[-1]-X[0]))], alpha=0.3, zorder=12-k)
				else:
					ax2.fill_between(r,[0]*len(f),f, color=cmap[int(10000*(X[k]-X[0])/(X[-1]-X[0]))], alpha=0.1, zorder=12-k)


		ax2.set_xlim(-2, 6)
		ax2.set_ylim(32.5, 42.5)
		if i>iSplit:
			ax3.set_xlim(-2, 6)
			ax3.set_ylim(32.5, 42.5)
		ax.plot(X, Y, "-k", zorder=1)
		ax.scatter(X, Y, c=[X[i]-X[0] for i in range(len(X))], s=250, cmap="jet", zorder=2)
		ax.scatter(X[i], Y[i], color=cmap[int(10000*(X[i]-X[0])/(X[-1]-X[0]))], s=550, marker="o", zorder=3)
		ax.set_xlabel("Time")
		ax.set_ylabel("Advance (km)")
		title = f"{figTitle}.png"
		fig.savefig(f"{figPath}{title}_{i+1}.png", dpi=300)
	plt.close(fig=fig)

	fig = plt.figure(figsize=(16,9), constrained_layout=True)
	ax = fig.add_subplot()
	ax.plot(X, Y, "-k", zorder=1)
	
	vel = sp.optimize.curve_fit(lambda x,a: a*x, X[:6]-np.min(X[:6]),Y[:6])[0][0]
	print("icesat",vel)
	Xs = np.linspace(np.min(X[:6]), np.max(X[:6])+0.7, 10)
	ax.plot(Xs, vel*(Xs-np.min(Xs)), linestyle=(5, (10, 3)), linewidth=3, color="b")

	vel += 80/1e3
	ax.plot(Xs, vel*(Xs-np.min(Xs)), linestyle=(5, (10, 3)), linewidth=3, color="y")
	ax.text(2007, 4.2, "V", color="y", fontsize=30)
	ax.text(2007.5, 4.3, "V-M", color="b", fontsize=30)
	ax.plot([2007]*15, np.linspace(2.6, 3.6 ,15), "r", linestyle=":", linewidth=3)
	ax.text(2007.05,3.1,"C",color="r", fontsize=30)
	d = dVEL.getDataBand()
	lon = 179
	vel = d.sel(lon=lon, method="nearest").values
	print("measure",vel)
	ax.plot(np.linspace(np.min(X), np.max(X), 10), vel/1e3*np.linspace(0, np.max(X)-np.min(X), 10),"g", linestyle=(0, (5, 5)), linewidth=2)
	ax.text(2009, 4.5, "Buoy", color="g", fontsize=30)
	vel = 1100
	ax.plot(np.linspace(np.min(X), np.max(X), 10), vel/1e3*np.linspace(0, np.max(X)-np.min(X), 10),"k", linestyle=(0, (5, 5)), linewidth=2)
	ax.text(2008.1, 6, "MEaSUREs", color="k", fontsize=30)

	vel = sp.optimize.curve_fit(lambda x,a: a*x, X[6:]-np.min(X[6:]),Y[6:]-np.min(Y[6:]))[0][0]
	print("icesat",vel)
	Xs = np.linspace(np.min(X[6:]), np.max(X[6:]), 10)
	ax.plot(Xs, vel*(Xs-np.min(Xs))+np.min(Y[6:]), linestyle=(5, (10, 3)), linewidth=3, color="cyan")

	Xs = np.linspace(np.min(X[6:])-1, np.max(X[6:]), 10)
	ax.plot(Xs, vel*(Xs-np.min(Xs)-1)+np.min(Y[6:]), linestyle=(5, (10, 3)), linewidth=3, color="brown")

	ax.scatter(X, Y, c=[X[i]-X[0] for i in range(len(X))], s=250, cmap="jet", zorder=2)
	ax.set_xlabel("Time")
	ax.set_ylabel("Advance (km)")
	title = f"{figTitle}.png"
	fig.savefig(f"{figPath}{title}_V.png", dpi=300)
	plt.close(fig=fig)
