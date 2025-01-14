import os
import subprocess as ss
import pandas as pd
import numpy as np
import scipy as sp
import h5py as hp
import matplotlib.pyplot as plt
import physics as ph

from params import dataPath, Rt, rhotio, rhotiob, nu, E

dataDir = "ICESat2ATL06/"

def convertRawTransectUsingBeker():
	savePath = dataPath + dataDir + 'Preprocess/'
	os.makedirs(savePath, exist_ok=True)

	# ss.run("""matlab -nodisplay -nosplash -nodesktop -r "run('/home/sartore/Documents/Projects/Ross/Code/Becker/Step_02.m');exit;" """, shell=True)
	# ss.run("""matlab -nodisplay -nosplash -nodesktop -r "run('/home/sartore/Documents/Projects/Ross/Code/Becker/Step_03.m');exit;" """, shell=True)

	with hp.File(savePath + "step3.mat", 'r') as file:
		df = []
		Lon = file["ross_front_crossing_data/lon"]
		Lat = file["ross_front_crossing_data/lat"]
		Hss = file["ross_front_crossing_data/h_ss"]
		Dt = file["ross_front_crossing_data/delta_time"]
		# Name = file["ross_front_crossing_data/file"]
		for i in range(Lon.size):
			lon = np.array(file[Lon[i][0]])[0]
			lat = np.array(file[Lat[i][0]])[0]
			hss = np.array(file[Hss[i][0]])[0]
			dt = np.array(file[Dt[i][0]])[0]
			# name = [np.array(file[Name[i][0]])
			# print(name)
			print(hss[np.logical_not(np.isnan(hss))])
			input()
			for j in range(len(lon)):
				df.append({"ID":i, "lon":lon[j], "lat":lat[j], "h":hss[j], "delta_time":dt[j]})
		df = pd.DataFrame.from_records(df)
		df.to_csv(savePath + "dataAfterBeckerPreprocess.csv")

	# os.remove(savePath+"step2.mat")
	# os.remove(savePath+"step3.mat")

def getData(wrmMin=1):
	savePathPkl = dataPath + dataDir + 'Pickle/'
	savePathPreprocess = dataPath + dataDir + 'Preprocess/'
	os.makedirs(savePathPkl, exist_ok=True)
	os.makedirs(savePathPreprocess, exist_ok=True)

	filePathFin = savePathPkl + "finalizedData.pkl"
	if os.path.isfile(filePathFin):
		return pd.read_pickle(filePathFin)
	else:
		data = pd.read_csv(savePathPreprocess + "dataAfterBeckerPreprocess.csv")
		df = []
		for ID in data.ID.unique():
			row = {}
			datai = data.loc[data.ID==ID]
			# row["name"] = datai["name"]
			row["lon"] = datai["lon"].values % 360 # longitude
			row["lat"] = datai["lat"].values # Latitude
			row["mlat"] = row["lat"].mean() # Mean latitude
			row["mlon"] = row["lon"].mean() # Mean longitude
			row["f"] = datai["h"].values # Freeboard
			row["dTime"] = datai["delta_time"].values  # Delta time
			row["time"] = pd.to_datetime("2018-01-01") + pd.to_timedelta(row["dTime"], unit="s")
			row["mdTime"] = row["dTime"].mean() # Mean delta time
			row["mtime"] = pd.to_datetime("2018-01-01") + pd.to_timedelta(row["mdTime"], unit="s")

			row["frontLat"] = np.nan
			row["frontLon"] = np.nan
			row["flag"] = 3 # Flag for the profile type, default is unknown

			row["h0"] = np.nan 
			row["xrm_o"] = np.nan 
			row["fxrm_o"] = np.nan 
			row["wrm_o"] = np.nan 
			row["xwrm_o"] = np.nan 

			row["lw_o"] = np.nan
			row["B"] = np.nan
			row["lw"] = np.nan
			row["lf_s0"] = np.nan
			row["h_s1"] = np.nan
			row["lw_s1"] = np.nan
			row["B_s1"] = np.nan
			row["H_s1"] = np.nan
			row["lf_s1"] = np.nan
			row["h_s2"] = np.nan
			row["lw_s2"] = np.nan
			row["B_s2"] = np.nan
			row["H_s2"] = np.nan
			row["lf_s2"] = np.nan

			row["lf_linear-noTen-approxXrm"] = np.nan
			row["lw_linear-noTen-approxXrm"] = np.nan
			row["a_linear-noTen-approxXrm"] = np.nan
			row["lf_linear-noTen"] = np.nan
			row["lw_linear-noTen"] = np.nan
			row["a_linear-noTen"] = np.nan
			row["lf_linear-noTen-RMFit"] = np.nan
			row["lw_linear-noTen-RMFit"] = np.nan
			row["a_linear-noTen-RMFit"] = np.nan

			df.append(row)
		df = pd.DataFrame.from_records(df)
		
		flag = pd.read_pickle(savePathPkl + "flag.pkl") # Open pickle with transect flags
		df["flag"] = pd.Series(flag, dtype=int)

		# Remove some sea points that prevent good front detection
		df.loc[136,"f"][3922:3936] = np.nan
		df.loc[166,"f"][[8395, 8396]] = np.nan
		df.loc[188,"f"][[1570, 1571, 1572]] = np.nan
		df.loc[213,"f"][1590] = np.nan
		df.loc[228,"f"][[6693, 6694]] = np.nan
		df.loc[229,"f"][[6690, 6692]] = np.nan
		df.loc[279,"f"][8702] = np.nan
		df.loc[436,"f"][[3358, 3359]] = np.nan
		df.loc[439,"f"][[3387, 3389]] = np.nan
		df.loc[441,"f"][3408] = np.nan
		df.loc[455,"f"][[8223, 8673]] = np.nan
		df.loc[457,"f"][8542] = np.nan
		df.loc[489,"f"][4682] = np.nan
		df.loc[802,"f"][10247] = np.nan
		df.loc[810,"f"][7101] = np.nan
		df.loc[831,"f"][[3309, 3310]] = np.nan
		df.loc[886,"f"][[2178, 2179]] = np.nan
		df.loc[888,"f"][3270] = np.nan
		df.loc[980,"f"][6334] = np.nan
		df.loc[982,"f"][6587] = np.nan
		df.loc[1099,"f"][8657:8662] = np.nan
		df.loc[1100,"f"][7716] = np.nan
		df.loc[1130,"f"][[2461,2462,2465]] = np.nan
		df.loc[1213,"f"][3324:3345] = np.nan
		df.loc[1274,"f"][9465] = np.nan
		df.loc[1322,"f"][7827] = np.nan
		df.loc[1679,"f"][[2021,2022]] = np.nan
		df.loc[1687,"f"][[1720,1721,1722]] = np.nan
		df.loc[2238,"f"][274] = np.nan
		df.loc[2521,"f"][[1667,1668]] = np.nan
		df.loc[2526,"f"][804] = np.nan
		df.loc[2579,"f"][[4002,4003]] = np.nan
		df.loc[3016,"f"][[3411,3412]] = np.nan
		df.loc[3064,"f"][[2175,2176]] = np.nan
		df.loc[3440,"f"][1625] = np.nan

		# for idx in [228,229,279,436,439,441,886,1130,1679,1684,1685,1686,1687,1782,2238,2521,2526,2577,2579,3016,3064,3092,3440]:
		# 	if idx <2000:
		# 		continue
		# 	plt.plot(df.loc[idx,"f"],"-s")
		# 	plt.title(idx)
		# 	plt.show()


		# Remove transects when unable to find front
		for i in [191, 549, 1540, 1541, 1722, 1723, 1724, 1725, 1806, 2232, 2355, 2439, 2551, 2711]:
			df.loc[i, "flag"] = 0
		# Remove other transects because bad quality
		for i in [28,29,185,211,224,225,619,639,641,642,643,690,1108,1109,1171,1172,1173,1174,1175,1178,1179,1274,1275,1384,1385,1392,1393,1665,1688,1824,1825,1915,2052,2053,2435,2760,2924,3108,3130,3352,3353]:
			df.loc[i, "flag"] = 0
		# Remove other transects because bad quality
		for i in [18,19,20,21,1050,1051,1067,1094,1095,1096,1097,1171,1172,1173,1178,1179,1372,1373,1374,1375, \
					28,29,22,23,297,299,301,347,349,500,619,641,642,804,807,1169,1174,1175,1330,1331,1332,1333, \
					1334,1346,1347,1366,1367,1408,1409,1410,1411]:
			df.loc[i, "flag"] = 0

		for i in df[df.flag==2].index:
			Info = getRMInfo(df.loc[i])
			for key in Info.keys():
				df.loc[i, key] = Info.get(key)

		for i in df[np.logical_or(df.flag==1, df.flag==2)].index:
			df.loc[i, "frontLat"], df.loc[i, "frontLon"] = getFrontPosition(df.loc[i])

		df.attrs['wrmMin'] = wrmMin
		df.attrs['epsLin'] = getFitLw(df, 1)
		df.attrs['epsQuad'] = getFitLw(df, 2)
		
		for i in df[df.flag==2].index:
			Info = getBeamInfo(df.loc[i])
			for key in Info.keys():
				df.loc[i, key] = Info.get(key)

		for model in ["linear-noTen-approxXrm", "linear-noTen-RMFit"]:
			for i in df[(df.flag==2)].index:
				print(i)
				modelVar = getLfFromNewModel(df.loc[i], model)
				for key in modelVar.keys():
					df.loc[i, key] = modelVar.get(key)

		df.to_pickle(filePathFin)
		return df

def makeFlag(flagToReview="all"):
	"""
	Function to manually assign flag to transects:
		0 --> Unusable because not clear, too small, too noisy
		1 --> Berm
		2 --> Rampart moat
		3 --> Unknown

	Just click on the button with the flag to assign. Data is saved after each click on a button.
	"""

	from matplotlib.widgets import Button
	savePath = dataPath + dataDir + 'Pickle/'
	fileName = "flag.pkl"
	df = getData()

	if os.path.isfile(savePath+fileName): # Verify if flag pickle file already exists
		data = pd.read_pickle(savePath+fileName)
	else:
		data = {}

	def fButtonRemove(event):
		data[i] = 0
		plt.close(fig=fig)

	def fButtonFront(event):
		data[i] = 1
		plt.close(fig=fig)

	def fButtonRM(event):
		data[i] = 2
		plt.close(fig=fig)

	def fButtonUnknown(event):
		data[i] = 3
		plt.close(fig=fig)

	if flagToReview=="all":
		flagToReview = [0,1,2,3]

	for key in flagToReview:
		for i in data.keys():
			if data[i] != key:
				continue

			fig = plt.figure()
			ax = fig.subplots()
			ax.plot(df.loc[i, "x"], df.loc[i, "h"], color="g" if df.loc[i, "isRM"] else "r")
			plt.title(f"{i:04d} / {len(df)} -- {key}")

			axbRemove = plt.axes([0.14, 0.005, 0.15, 0.075])
			axbFront = plt.axes([0.14+1*(0.15+0.05), 0.005, 0.15, 0.075])
			axbRM = plt.axes([0.14+2*(0.15+0.05), 0.005, 0.15, 0.075])
			axbUnknown = plt.axes([0.14+3*(0.15+0.05), 0.005, 0.15, 0.075])

			bRemove = Button(axbRemove, 'Remove', color="red")
			bFront = Button(axbFront, 'Front', color="blue")
			bRM = Button(axbRM, 'RM', color="green")
			bUnknown = Button(axbUnknown, 'Unknown', color="grey")
			bRemove.on_clicked(fButtonRemove)
			bFront.on_clicked(fButtonFront)
			bRM.on_clicked(fButtonRM)
			bUnknown.on_clicked(fButtonUnknown)
			
			plt.get_current_fig_manager().full_screen_toggle()
			plt.show()
			pd.to_pickle(data, savePath+fileName)

def _xrm_s(h, eps, exp):
	h_s = ph.h_s(h0=h, eps=eps, exp=exp)
	B_s = ph._B(h0=h_s, E=E, nu=nu)
	lw_s = ph._lw(B=B_s)
	xrm_s = ph._xrm(lw=lw_s)
	return xrm_s

def getFitLw(df, exp):
	wrmMin = df.attrs.get("wrmMin")
	mask = (df.flag==2) & (df.wrm_o > wrmMin)
	Xrm = df.loc[mask, "xrm_o"]
	H0 = df.loc[mask, "h0"]
		
	fct = lambda h, eps: _xrm_s(h, eps, exp)
	eps = sp.optimize.curve_fit(f=fct, xdata=H0, ydata=Xrm)[0][0]
	return eps

def getRF(dfi):
	r = Rt*dfi.lat*np.pi/180 
	f = dfi.f

	mask = np.logical_not(np.logical_or(np.isnan(f), np.isnan(r)))
	r = r[mask]
	f = f[mask]
	f = f[(-r).argsort()]
	r = r[(-r).argsort()]
	r = r[0] - r
	
	return r, f

def getNormalizedRF(dfi):
	r, f = getRF(dfi)
	i = np.argmax(f>7)
	k = 0
	while np.abs(f[i]-f[i+1])>3:
		i += 1
		if k>=5:
			raise ValueError("Failed")

	rFront = r[i]
	r, f = r[r>rFront], f[r>rFront]
	r -= r[0]
	return r, f

def getFrontPosition(dfi):
	r = Rt*dfi.lat*np.pi/180 
	f = dfi.f

	mask = np.logical_not(np.logical_or(np.isnan(f), np.isnan(r)))
	r = r[mask]
	f = f[mask]
	lat = dfi.lat[mask]
	lon = dfi.lon[mask]
	f = f[(-r).argsort()]
	lat = lat[(-r).argsort()]
	lon = lon[(-r).argsort()]
	r = r[(-r).argsort()]
	r = r[0] - r

	i = np.argmax(f>7)
	k=0
	while np.abs(f[i]-f[i+1])>3:
		i+=1
		if k>=5:
			raise ValueError("Failed")
	return lat[i], lon[i]

def smoothXY(x, y, method):
	mask = np.logical_not(np.isnan(y))
	if method=="moat":
		b, a = sp.signal.butter(3, 0.18)
		yf = sp.signal.filtfilt(b, a, y[mask])
		yf = np.concatenate([yf, [np.nan]*(len(x)-len(yf))])

	elif method=="rampart":
		b, a = sp.signal.butter(2, 0.1)
		yf = sp.signal.filtfilt(b, a, y[mask])
		yf = np.concatenate([yf, [np.nan]*(len(x)-len(yf))])

	elif method=="light":
		b, a = sp.signal.butter(1, 0.008)
		yf = sp.signal.filtfilt(b, a, y[mask])
		yf = np.concatenate([yf, [np.nan]*(len(x)-len(yf))])

	elif method=="light+":
		b, a = sp.signal.butter(1, 0.002)
		yf = sp.signal.filtfilt(b, a, y[mask])
		yf = np.concatenate([yf, [np.nan]*(len(x)-len(yf))])

	elif method=="medium":
		b, a = sp.signal.butter(1, 0.001)
		yf = sp.signal.filtfilt(b, a, y[mask])
		yf = np.concatenate([yf, [np.nan]*(len(x)-len(yf))])

	else:
		raise ValueError("Wrong keyword for smoothing")

	return yf

def getRMInfo(dfi):
	if dfi.flag!=2:
		raise ValueError("Transect not a rampart-moat")
	r, f = getNormalizedRF(dfi)
	x = np.linspace(0, 5000, 50000)
	f = np.interp(x, r, f, right=np.nan)
	y = smoothXY(x, f, "medium")
	yxrm = y[x<1000].min()
	xrm = x[y==yxrm][0]
	y2 = smoothXY(x, f, "light+" if xrm>150 else "light")
	fxrm = y2[x==xrm]
	# r, f = getNormalizedRF(dfi)
	# x = np.linspace(0, 5000, 500)
	# f = np.interp(x, r, f, right=np.nan)
	# wrm = (f[x<=xrm].max() - fxrm)
	# xwrm = x[f==f[x<=xrm].max()][0]
	wrm = (f[0] - fxrm)
	xwrm = 0

	h0 = (fxrm+wrm/2)/rhotiob
	# if dfi.name>3000:
	# 	r, f = getNormalizedRF(dfi)
	# 	r,f = r[r<2*xrm], f[r<2*xrm]
	# 	plt.plot(r,f,"ko")
	# 	plt.plot([xrm]*2, [f.min(), f.max()],"k")
	# 	plt.plot([r[0], r[-1]], [wrm+fxrm]*2,"k")
	# 	plt.title(dfi.name)
	# 	plt.get_current_fig_manager().window.showMaximized()
	# 	plt.show()

	return {"h0":h0, "xrm_o":xrm, "fxrm_o":fxrm, "wrm_o":wrm, "xwrm_o":xwrm,}

def getRMInfoTest(dfi):
	if dfi.flag!=2:
		raise ValueError("Transect not a rampart-moat")

	r, f = getNormalizedRF(dfi)
	x1 = np.linspace(0, 1000, 25000)

	x2 = np.linspace(0, 1000, 100)
	y2 = np.interp(x2, r, f, right=np.nan)

	y2s = np.interp(x1, x2, smoothXY(x2, y2, "moat"))
	y3s = np.interp(x1, x2, smoothXY(x2, y2, "rampart"))

	xrm = x1[y2s==y2s.min()][0]
	fxrm = y2s[y2s==y2s.min()][0]

	xwrm = x1[x1<xrm][y3s[x1<xrm]==y3s[x1<xrm].max()][0]
	wrm = y3s[x1<xrm][y3s[x1<xrm]==y3s[x1<xrm].max()][0]-fxrm

	h0 = (fxrm+wrm/2)/rhotiob

	r, f = getNormalizedRF(dfi)
	r, f = r[r<1500], f[r<1500]

	fig = plt.figure(figsize=(16,9), constrained_layout=True)
	ax = fig.add_subplot()
	ax.set_title(dfi.flag)
	ax.plot(r, f ,"-ks")
	ax.plot([r.min(), r.max()], [h0*rhotiob]*2, color="g")
	ax.plot([xrm]*2, [fxrm, fxrm+wrm], color="g")
	ax.plot(xrm, fxrm, "xg", markersize=15)
	ax.plot([xwrm]*2, [fxrm, fxrm+wrm], color="g")

	del wrm, xwrm, xrm, fxrm, h0

	r, f = getNormalizedRF(dfi)
	x = np.linspace(0, 5000, 50000)
	f = np.interp(x, r, f, right=np.nan)
	y = smoothXY(x, f, "medium")
	yxrm = y[x<1000].min()
	xrm = x[y==yxrm][0]
	y2 = smoothXY(x, f, "light+" if xrm>150 else "light")
	fxrm = y2[x==xrm]
	r, f = getNormalizedRF(dfi)
	x = np.linspace(0, 5000, 500)
	f = np.interp(x, r, f, right=np.nan)
	wrm = (f[x<=xrm].max() - fxrm)
	xwrm = x[f==f[x<=xrm].max()][0]
	h0 = (fxrm+wrm/2)/rhotiob

	ax.plot([r.min(), r.max()], [h0*rhotiob]*2, "--r")
	ax.plot([xrm]*2, [fxrm, fxrm+wrm], "--r")
	ax.plot(xrm, fxrm, "xr", markersize=15)
	ax.plot([xwrm]*2, [fxrm, fxrm+wrm], "--r")
	ax.set_xlim(-5,1550)
	fig.canvas.manager.full_screen_toggle()
	plt.show()

	return {"h0":h0, "xrm":xrm, "fxrm":fxrm, "wrm":wrm, "xwrm":xwrm,}

def getBeamInfo(dfi):
	wrm = dfi.wrm_o
	h0 = dfi.h0
	epsLin = dfi.attrs.get("epsLin")
	epsQuad = dfi.attrs.get("epsQuad")
	
	lw_o = ph._lw_o(dfi.xrm_o)
	H_s0 = ph._H(h0=h0, lw=lw_o)
	lf_s0 = ph._lf(wrm=wrm, H=H_s0)

	h_s1 = ph.h_s(h0=h0, eps=epsLin, exp=1)
	B_s1 = ph._B(h0=h_s1, E=E, nu=nu)
	lw_s1 = ph._lw(B_s1)
	H_s1 = ph._H(h0=h0, lw=lw_s1)
	lf_s1 = ph._lf(wrm=wrm, H=H_s1)

	h_s2 = ph.h_s(h0=h0, eps=epsQuad, exp=2)
	B_s2 = ph._B(h0=h_s2, E=E, nu=nu)
	lw_s2 = ph._lw(B_s2)
	H_s2 = ph._H(h0=h0, lw=lw_s2)
	lf_s2 = ph._lf(wrm=wrm, H=H_s2)
	
	B = ph._B(h0=h0, E=E, nu=nu)
	lw = ph._lw(B)
	
	Info = {"lw_o":lw_o, "B":B, "lw":lw,\
		"h_s1":h_s1, "lw_s1":lw_s1, "B_s1":B_s1, "H_s1":H_s1, "lf_s1":lf_s1,\
		"h_s2":h_s2, "lw_s2":lw_s2, "B_s2":B_s2, "H_s2":H_s2, "lf_s2":lf_s2, "lf_s0":lf_s0}
	return Info

def getMeanH(dfi):
	r, f = getNormalizedRF(dfi)
	h = f/rhotiob
	x = np.linspace(500, 14000, 11000)
	y = np.interp(x, r, h, right=np.nan, left=np.nan)
	h1k  = np.nanmean(y[(x>1000*0.98)  & (x<1000*1.02)])
	h3k  = np.nanmean(y[(x>3000*0.99)  & (x<3000*1.01)])
	h5k  = np.nanmean(y[(x>5000*0.99)  & (x<5000*1.01)])
	h10k = np.nanmean(y[(x>10000*0.99) & (x<10000*1.01)])

	return {"h1k":h1k, "h3k":h3k, "h5k":h5k, "h10k":h10k,}

def getLfFromNewModel(dfi, model):
	xrm = dfi["xrm_o"]
	wrm = dfi["wrm_o"]
	h0 = dfi["h0"]

	if model == "linear-noTen-approxXrm":
		a1 = rhotio*(1 - 3*rhotio + 2*rhotio**2)/12
		b1 = rhotio*(1 - rhotio)/2
		lf = wrm*xrm/3/np.pi/b1/h0 - 3*np.pi*a1*h0**2/8/b1/xrm
		lw = np.nan
		a = np.nan

	elif model == "linear-noTen":
		s = ph.modelLinNoTenDim(h0=h0, verbose=0)
		a = np.nan
		try:
			var = s.getLwf(xrm=xrm, wrm=wrm)
			lw = var["lw"]
			lf = var["lf"]
		except ph.myError as eee:
			print(dfi.index, eee)
			lw = np.nan
			lf = np.nan

	elif model == "linear-noTen-RMFit":
		s = ph.modelLinNoTenDim(h0=h0, verbose=0)
		r, f = getNormalizedRF(dfi)
		var = s.getLwfa3(r=r, f=f, xrm=xrm)
		lw = var["lw"]
		lf = var["lf"]
		a = var["a"]

	else:
		raise ValueError("Model type unknown.")

	return {f"lw_{model}":lw, f"lf_{model}":lf, f"a_{model}":a}

if __name__ == '__main__':
	# d = getData()
	convertRawTransectUsingBeker()
