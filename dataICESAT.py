import pandas as pd
import numpy as np
import scipy as sp
import glob as gl

from params import dataPath, rhoi, rhow, Rt, rhotio

dataDir = "ICESat1/"

def getData():
	File = sorted(gl.glob(dataPath+dataDir+'*.txt'))
	data = []
	rmLabel = { "L2a":True,
				"L2b":True,
				"L2f":False,
				"L3b":True,
				"L3e":True,
				"L3f":True,
				"L3g":True,
				"L3h":False,
				"L3i":False,
				"L3k":False,
				}
	for i, file in enumerate(File):
		fileName = file.split("/")[-1].replace(".txt","").replace("icesat_t0068_","")
		track = pd.read_csv(file, names=['lon','lat','elev','time','atd'])
		lon = track.lon.values
		lat = track.lat.values
		f = track.elev.values
		atd = track.atd.values
		time = track.time.values
		mTime = track.time.values.mean()
		isFront = True
		isRM = rmLabel[fileName]

		data.append({"lon":lon, "lat":lat, "f":f, "atd":atd, "time":time, "mTime":mTime, "isFront":isFront, "isRM":isRM, "name":fileName})

	data = pd.DataFrame(data).sort_values("mTime", ignore_index=True)
	return data

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

def getFront(r, f):
	i = np.argmax(f>7)
	k=0
	while np.abs(f[i]-f[i+1])>3:
		i+=1
		if k>=5:
			raise ValueError("Failed")
	return r[i]

def normalizeProfile(r, f):
	rFront = getFront(r,f)
	r, f = r[r>rFront], f[r>rFront]
	r -= r[0]
	return r, f

def getXrm(dfi):
	r, f = getRF(dfi)
	r, f = normalizeProfile(r, f)
	x = np.linspace(r.min(), 1500, 10000)
	y = smoothXY(r, f, "spline")(x)

	yxrm = y.min()
	xrm = x[y==yxrm][0]

	fxrm = y[x==xrm]

	return xrm, fxrm

def getHrm(dfi):
	xrm, fxrm = getXrm(dfi)
	r, f = getRF(dfi)
	r, f = normalizeProfile(r, f)
	x = np.linspace(r.min(), 1500, 10000)
	y = smoothXY(r, f, "spline")(x)
	hrm = (y[x<=xrm].max() - fxrm)
	return hrm

def smoothXY(x, y, method):
	mask = np.logical_not(np.isnan(y))
	if method=="light":
		b, a = sp.signal.butter(1, 0.008)
		yf = sp.signal.filtfilt(b, a, y[mask])
		yf = np.concatenate([yf, [np.nan]*(len(x)-len(yf))])
	elif method=="spline":
		yf = sp.interpolate.make_interp_spline(x, y, 3)
	else:
		raise ValueError("Wrong keyword for smoothing")

	return yf

def getRMInfo(dfi):
	r, f = getRF(dfi)
	r, f = normalizeProfile(r, f)
	xrm, fxrm = getXrm(dfi)
	hrm = getHrm(dfi)

	h = f/(1-rhotio)
	x = np.linspace(0, 14000, 14000)
	h = np.interp(x, r, h, right=np.nan)
	
	y = smoothXY(x, h, "light")
	yp = np.gradient(y, x)
	ypp = np.gradient(yp, x)
	xmax = x[ypp==ypp[x<xrm].max()][0]

	h0 = ((fxrm + hrm/2)/(1-rhotio))[0]
	h1k  = (h[(x>1000*0.98)  & (x<1000*1.02)].mean())
	h2k  = (h[(x>2000*0.99)  & (x<2000*1.01)].mean())
	h3k  = (h[(x>3000*0.99)  & (x<3000*1.01)].mean())
	h5k  = (h[(x>5000*0.99)  & (x<5000*1.01)].mean())
	h10k = (h[(x>10000*0.99) & (x<10000*1.01)].mean())

	lw = 4*xrm/3/np.sqrt(2)/np.pi
	deltaRho = rhow - rhoi
	d = h0*deltaRho/rhow
	H = d*rhoi/rhow/lw
	lf = (H*(np.sqrt(2)+np.exp(-3*np.pi/4))/hrm)**-1

	Info = {
		"h0": h0,
		"hrm": hrm,
		"xrm": xrm,
		"xmax": xmax,
		"lw": lw,
		"lf": lf,
		"h1k": h1k,
		"h2k": h2k,
		"h3k": h3k,
		"h5k": h5k,
		"h10k": h10k,
		"fxrm": fxrm,
		"H":H,
	}
	return Info