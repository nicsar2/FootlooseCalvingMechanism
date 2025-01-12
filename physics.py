import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
import scipy as sp

from params import a1, a2, b1, b2, Rt, rhoi, rhow, g, rhotio, rhotiob, nu

def Me(ds, a1=a1, a2=a2, b1=b1, b2=b2): #Wave-induced melt equation
	return (b1 + b2*ds.T) * (a1*ds.w**0.5 + a2*ds.w) * (1 + np.cos(np.pi*ds.C**3))/2

def MeRed(ds, a1=a1, a2=a2, b1=b1, b2=b2): #Wave-induced melt equation
	return (b1 + b2*ds.T) * (a1*ds.w**0.5 + a2*ds.w) * (1 + np.cos(np.pi*ds.C))/2

def sphericalDst(lat1, lon1, lat2, lon2): #Distance between two points on a sphere
	dlat = np.pi/180*(lat2-lat1)/2
	dlon = np.pi/180*(lon2-lon1)/2
	mlat = np.pi/180*(lat1+lat2)/2
	dst = 2*Rt*np.arcsin( (np.sin(dlat)**2 + (1 - np.sin(dlat)**2 - np.sin(mlat)**2)*np.sin(dlon)**2)**0.5 )
	return dst
 
def gaussian(x, mu, sig):
	return (
		1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
	)

def rootSquareError(fct, x, y):
	if not callable(fct):
		raise TypeError("Input variable fxt should be a function.")

	err = ((fct(x) - y)**2).sum()
	return err  

class ode(): # ODE system to solve non-uniform ice shelf --- DO NOT WORK ---
	def __init__(self, xmax=100, dx=0.1, ode="Simple"):
		self.dx = dx
		self.xmax = xmax
		self.ode = ode

	def _Ydot(self, x, Y):
		a, b, c, d = Y

		if self.ode == "Simple":
			return [b, c, d, -a]
		elif self.ode=="SimpleI":
			return [b, c, d, -a*x**-8 -b*24*x**-3 -c*36*x**-2 -d*12/x] 
		elif self.ode=="Full":
			return [b, c, d, -6*d/x -6*c/x**2 -1*a/x**3] 
		else:
			raise ValueError("ode unknown")

	def solve(self, method="BDF", y0=(1, 1, 0, 1)):
		x = np.arange(self.dx, self.xmax+self.dx, self.dx)
		res = sp.integrate.solve_ivp(fun=self._Ydot, y0=y0, t_eval=x, t_span=(x.min(), x.max()),\
			method=method)

		# if res.status!=1:
		#     print(res.message)

		return res

	def _root(self, x):
		# res = self.solve(y0=(x[0], x[1], wpp0, wppp0))
		# return np.sqrt( res.y[0,-1]**2 + res.y[1,-1]**2 )

		res = self.solve(y0=(x[0], x[1], 0, 1))
		return np.sqrt( (res.y[0,-1]-0)**2 + (res.y[1,-1]-0)**2 )

		# res = self.solve(y0=(x[0], x[1], x[2], x[3]))
		# return np.array( [res.y[0,-1], res.y[1,-1], res.y[2,0], res.y[3,0]-1,] )

	def _jac(self, x, wpp0, wppp0):
		if self.ode=="Simple":
			return np.array( [  [ 0, 1, 0, 0],
								[ 0, 0, 1, 0],
								[ 0, 0, 0, 1],
								[-1, 0, 0, 0]  ] )
		elif self.ode=="Full":
			return np.array( [  [      0,   1,       0,    0],
								[      0,   0,       1,    0],
								[      0,   0,       0,    1],
								[-1/x**3,   0, -6/x**2, -6/x]  ] )
		else:
			raise ValueError("ode unknown")

	def shooting(self,x0=(1, 1)):
		# res = sp.optimize.root(fun=self._root, x0=x0, jac=self._jac, args=(wpp0, wppp0) )
		res = sp.optimize.minimize(fun=self._root, x0=x0, method="Nelder-Mead" )
		return res.x

####################    ELASTIC BEAM MODEL    ####################

def h_s(h0, eps, exp): #Effective ice thickness
	return eps*h0**exp

def _B(h0, E, nu): #Bending stiffness 
	return E*h0**3/12/(1-nu**2)

def _lw(B): #Theoretical Buoyancy wavelength
	return (B/rhow/g)**0.25

def _lw_o(xrm_o): #Observed Buoyancy wavelength
	return xrm_o*2*2**0.5/3/np.pi

def _xrm(lw): #Moat position
	return lw*3*np.pi/2/2**0.5

def _H(h0, lw): #Reduced front thickness
	return h0*rhotiob*rhoi/rhow/lw

def _lf(wrm, H): #Foot length
	return wrm/H/(2**0.5 + np.exp(-3*np.pi/4))

class myError(Exception):
	pass

class modelSimpleDim():
	def __init__(self, h0, verbose=0):
		self.verbose = verbose
		self.rhotio = rhotio
		self.h0 = h0
		self.reset()

	def verbosity(self, verboseLvl, text):
		if self.verbose >= verboseLvl:
				print("--"*verboseLvl+">", text)

	def reset(self):
		self.lf = None
		self.lw = None

		self.H = None
		self.Lf = None

		self.profile = None
		self.Hbb = None

	def getProfile(self, lf, lw):
		if not isinstance(lf, (float, int, np.float64, np.int64)) or lf is np.nan:
			raise TypeError("Input variable lf should be a scalar.")
		if not isinstance(lw, (float, int, np.float64, np.int64)) or lw is np.nan:
			raise TypeError("Input variable lw should be a scalar.")
		self.reset()
		h = self.h0
		self.lf = lf
		self.lw = lw

		rhotio = self.rhotio
		H = h/lw
		Lf = lf/lw
		D = rhotio*H
		Hbb = (1 - D/H)*D

		w = lambda x: lw*np.sqrt(2)*Hbb*Lf*np.exp(-x/np.sqrt(2)/lw)*np.cos(x/np.sqrt(2)/lw)

		self.verbosity(2, "")
		self.verbosity(2, f"H = {H:.1e}, Lf = {Lf:.1e}")
		self.verbosity(2, f"w(0) = {w(0):.2e}")
		self.verbosity(2, "")
		self.verbosity(2, "")

		self.profile = w
		return w

class modelSimpleNoDim():
	def __init__(self, h0, verbose=0):
		self.verbose = verbose
		self.rhotio = rhotio
		self.h0 = h0
		self.reset()

	def verbosity(self, verboseLvl, text):
		if self.verbose >= verboseLvl:
				print("--"*verboseLvl+">", text)

	def reset(self):
		self.lf = None
		self.lw = None

		self.H = None
		self.Lf = None

		self.profile = None
		self.Hbb = None

	def getMinMax(self):
		bmin = 0; bmax = 25
		for i in range(8):
			x = np.linspace(bmin, bmax, 10000)
			y = self.profile(x)
			idx = (np.abs(y - y.max())).argmin()
			bmin = x[max(idx-3,0)]
			bmax = x[min(idx+3, len(x)-1)]
		xmax = (bmin + bmax)/2
		if xmax<0:
			xmax = 0

		bmin = xmax; bmax = 50
		for i in range(8):
			x = np.linspace(bmin, bmax, 10000)
			y = self.profile(x)
			idx = (np.abs(y - y.min())).argmin()
			bmin = x[max(idx-3,0)]
			bmax = x[min(idx+3, len(x)-1)]
		xmin = (bmin + bmax)/2
		return xmin, xmax

	def getRM(self):
		minPosition, maxPosition = self.getMinMax()
		xrm = minPosition
		wrm = self.profile(0) - self.profile(xrm)
		return np.array([xrm, wrm])

	def getProfile(self, Lf, H):
		if not isinstance(Lf, (float, int, np.float64, np.int64)) or Lf is np.nan:
			raise TypeError("Input variable Lf should be a scalar.")
		if not isinstance(H, (float, int, np.float64, np.int64)) or H is np.nan:
			raise TypeError("Input variable H should be a scalar.")
		self.reset()
		self.Lf = Lf
		self.H = H

		rhotio = self.rhotio
		D = rhotio*H
		Hbb = (1 - D/H)*D

		W = lambda X: np.sqrt(2)*Hbb*Lf*np.exp(-X/np.sqrt(2))*np.cos(X/np.sqrt(2))

		self.verbosity(2, "")
		self.verbosity(2, f"H = {H:.1e}, Lf = {Lf:.1e}")
		self.verbosity(2, f"w(0) = {W(0):.2e}")
		self.verbosity(2, "")
		self.verbosity(2, "")

		self.profile = W
		return W
		
class modelFullNoDim():
	def __init__(self, h0, verbose=0):
		self.verbose = verbose
		self.h0 = h0
		self.rhotio = rhotio
		self.reset()

	def verbosity(self, verboseLvl, text):
		if self.verbose >= verboseLvl:
				print("--"*verboseLvl+">", text)

	def reset(self):
		self.lf = None
		self.lw = None

		self.H = None
		self.Lf = None

		self.profile = None
		self.tau = None
		self.Hbb = None

	def getProfile(self, H, Lf):
		if not isinstance(H, (float, int, np.float64, np.int64)) or H is np.nan:
			raise TypeError("Input variable h should be a scalar.")
		if not isinstance(Lf, (float, int, np.float64, np.int64)) or Lf is np.nan:
			raise TypeError("Input variable lf should be a scalar.")
		self.reset()
		self.H = H
		self.Lf = Lf

		rhotio = self.rhotio
		eps = 1e-12
		D = rhotio*H
		Hbb = (1 - rhotio)*D
		a1 = rhotio*(1 - 3*rhotio + 2*rhotio**2)/12
		a = a1*H**3
		b1 = rhotio*(1 - rhotio)/2
		b = b1*H**2

		self.Hbb = Hbb
		self.lw = (1-rhotio)*rhotio*self.h0/self.Hbb
		self.lf = self.Lf*self.lw

		taup = lambda tau_: np.sqrt(2 + tau_)/2
		taum = lambda tau_: np.sqrt(2 - tau_)/2
		taus = lambda tau_: (np.sqrt(2)*(1 - tau_)*taup(tau_))**-1
		eta = lambda tau_: (1 + tau_) * np.sqrt(4 - tau_**2) / (tau_**2 + tau_ - 2)
		xi = lambda tau_: (b + tau_)/D
		root = lambda tau_: xi(tau_) + 2*taum(tau_)*Hbb*Lf - (a - b*xi(tau_))*(tau_ - 1)

		for x0 in np.linspace(0, 2, 20)[::-1]:
			rootFinding = sp.optimize.root(root , x0=x0, method="lm", tol=1e-15)
			tau = rootFinding.x[0]
			if np.abs(root(tau)) <= eps:
				break
		else:
			self.verbosity(2, f"tau root finding failed: accuracy not reached --> err = {np.abs(root(tau))}")
			if self.verbose >= 3:
				x = np.linspace(-10,10,500000)
				y = root(x)
				plt.plot(x,y)
				plt.show()
			raise myError("tau root finding failed: accuracy not reached.")

		self.tau = tau
		W = lambda X: np.exp(-taum(tau)*X)*( (taus(tau)*np.sqrt(2)*Hbb*Lf - eta(tau)*xi(tau))*np.sin(taup(tau)*X) - xi(tau)*np.cos(taup(tau)*X) )

		if np.isnan(W(np.linspace(0,10,100))).any():
			raise myError("Nan in solution")

		self.verbosity(2, "")
		self.verbosity(2, f"a1 = {a1:.2e}, b1 = {b1:.2e}")
		self.verbosity(2, f"H = {H:.1e}, Lf = {Lf:.1e}")
		self.verbosity(2, f"tau = {tau:.2e}, res = {root(tau):.1e}, cond = {np.abs(root(tau)) > 1e-13}")
		self.verbosity(2, f"w(0) = {W(0):.2e}")
		self.verbosity(2, f"{rootFinding.status} {rootFinding.message}")
		self.verbosity(2, "")
		self.verbosity(2, "")

		self.profile = W
		return W

	def getMinMax(self):
		bmin = 0; bmax = 25
		for i in range(8):
			x = np.linspace(bmin, bmax, 10000)
			y = self.profile(x)
			idx = (np.abs(y - y.max())).argmin()
			bmin = x[max(idx-3,0)]
			bmax = x[min(idx+3, len(x)-1)]
		xmax = (bmin + bmax)/2
		if xmax<0:
			xmax = 0

		bmin = xmax; bmax = 50
		for i in range(8):
			x = np.linspace(bmin, bmax, 10000)
			y = self.profile(x)
			idx = (np.abs(y - y.min())).argmin()
			bmin = x[max(idx-3,0)]
			bmax = x[min(idx+3, len(x)-1)]
		xmin = (bmin + bmax)/2
		return xmin, xmax

	def getRM(self):
		minPosition, maxPosition = self.getMinMax()
		xrm = minPosition
		wrm = self.profile(0) - self.profile(xrm)
		return np.array([xrm, wrm])

	def plotSlidder(self):
		x = np.linspace(0, 15, 1000)
		Hi = 1
		Lfi = 0
		y = self.getProfile(H=Hi, Lf=Lfi)(x)
		fig, ax = plt.subplots()
		plt.subplots_adjust(left=0.1, bottom=0.35)
		l, = plt.plot(x, y, lw=2)

		minPosition, maxPosition = self.getMinMax()
		m, = plt.plot(maxPosition, self.profile(maxPosition), 'ro')
		p, = plt.plot(minPosition, self.profile(minPosition), 'bo')

		axcolor = 'lightgoldenrodyellow'
		ax_H = plt.axes([0.1, 0.2, 0.65, 0.03], facecolor=axcolor)
		ax_Lf = plt.axes([0.1, 0.15, 0.65, 0.03], facecolor=axcolor)

		s_H = mp.widgets.Slider(ax_H, 'H', 0, 5, valinit=Hi)
		s_Lf = mp.widgets.Slider(ax_Lf, 'Lf', 0, 1, valinit=Lfi)

		def update(val):
			H = s_H.val
			Lf = s_Lf.val
			try:
				self.getProfile(H=H, Lf=Lf)
				y = self.profile(x)
				ax.set_ylim([y.min() - np.abs(y.min())*0.5, y.max() + np.abs(y.max())*0.5])
			except:
				y = np.zeros(x.shape)
			l.set_ydata(y)

			minPosition, maxPosition = self.getMinMax()
			m.set_data(maxPosition, self.profile(maxPosition))
			p.set_data(minPosition, self.profile(minPosition))
			
			fig.canvas.draw_idle()

		s_H.on_changed(update)
		s_Lf.on_changed(update)
		plt.show()
	
	def getLwf(self, Xrm, Wrm):
		def fff(x):
			self.getProfile(H=x[0], Lf=x[1])
			return self.getRM()
		self.verbosity(1, f"Lwf RF start --> Xrm = {Xrm:.2e}, Wrm = {Wrm:.2e}")

		eps = 1e-11
		for H0 in np.concatenate([[1],np.linspace(0,5,50)]):
			for Lf0 in np.concatenate([[0.5],np.linspace(0,1,50)]):
				try:
					res = sp.optimize.root(lambda x: fff(x) - (Xrm, Wrm) , x0=(H0, Lf0), tol=1e-14, method="lm", options={"factor":0.1})
					H = res.x[0]
					Lf = res.x[1]
					if H < 0 or Lf < 0:
						raise myError("lwf RF step failed: H or Lf is negative")

					self.getProfile(H=H, Lf=Lf)
					XrmR, WrmR = self.getRM()
					error = np.sqrt((Xrm-XrmR)**2 + (Wrm-WrmR)**2)
					print(error)

				except myError as eee:
					continue
					self.verbosity(2, f"Lwf RF step failed --> {eee}")

				if error <= eps:
					self.verbosity(2, f"Lwf RF step succed --> H = {H:.2e}, Lf = {Lf:.2e}")
					break
			else:
				continue
			break
		else:
			raise myError("Lwf RF failed: no solution found")

		self.getProfile(H=H, Lf=Lf)
		XrmR, WrmR = self.getRM()
		error = np.sqrt((Xrm-XrmR)**2 + (Wrm-WrmR)**2)
		if not error <= eps:
			raise myError("Lwf RF failed: no solution found")

		return {"H":H, "Lf":Lf}
	
	def plot(self, save=False, title=None):
		x = np.linspace(0, 15, 100000)
		y = self.profile(x)
		minPosition, maxPosition = self.getMinMax()

		fig = plt.figure(figsize=(15, 15), constrained_layout=True)
		ax = fig.add_subplot()
		ax.plot(x,y)
		ax.plot(maxPosition, self.profile(maxPosition), 'ro')
		ax.plot(minPosition, self.profile(minPosition), 'bo')
		ax.set_title(f"H = {self.H:.2e}, Lf = {self.Lf:.2e}")
		if save:
			if title is None:
				title = f"./{np.random.uniform(0,10000)}.png"
			fig.savefig(title)
		else:
			plt.show()
		plt.close(fig=fig)

class modelFullDim():
	def __init__(self, h0, verbose=0):
		self.verbose = verbose
		self.rhotio = rhotio
		self.h0 = h0
		self.reset()

	def verbosity(self, verboseLvl, text):
		if self.verbose >= verboseLvl:
				print("--"*verboseLvl+">", text)

	def reset(self):
		self.lf = None
		self.lw = None

		self.H = None
		self.Lf = None

		self.profile = None
		self.tau = None
		self.Hbb = None

	def getProfile(self, lf, lw):
		if not isinstance(lf, (float, int, np.float64, np.int64)) or lf is np.nan:
			raise TypeError("Input variable lf should be a scalar.")
		if not isinstance(lw, (float, int, np.float64, np.int64)) or lw is np.nan:
			raise TypeError("Input variable lw should be a scalar.")
		self.reset()
		h = self.h0
		self.lf = lf
		self.lw = lw

		eps = 1e-10
		rhotio = self.rhotio
		H = h/lw
		Lf = lf/lw
		D = rhotio*H
		Hbb = (1 - D/H)*D
		a1 = rhotio*(1 - 3*rhotio + 2*rhotio**2)/12
		a = a1*H**3
		b1 = rhotio*(1 - rhotio)/2
		b = b1*H**2

		taup = lambda tau_: np.sqrt(2 + tau_)/2
		taum = lambda tau_: np.sqrt(2 - tau_)/2
		taus = lambda tau_: (np.sqrt(2)*(1 - tau_)*taup(tau_))**-1
		eta = lambda tau_: (1 + tau_) * np.sqrt(4 - tau_**2) / (tau_**2 + tau_ - 2)
		xi = lambda tau_: (b + tau_)/D
		root = lambda tau_: xi(tau_) + 2*taum(tau_)*Hbb*Lf - (a - b*xi(tau_))*(tau_ - 1)

		for x0 in np.linspace(0, 2, 5):
			rootFinding = sp.optimize.root(root , x0=x0, method="lm", tol=1e-15)
			tau = rootFinding.x[0]
			if np.abs(root(tau)) <= eps:
				break
		else:
			self.verbosity(2, f"tau root finding failed: accuracy not reached --> err = {np.abs(root(tau))}")
			if self.verbose >= 3:
				x = np.linspace(-25,25,500000)
				y = root(x)
				plt.plot(x,y,".-")
				plt.show()
			raise myError("tau root finding failed: accuracy not reached.")

		W = lambda X: lw*np.exp(-taum(tau)*X/lw)*( (taus(tau)*np.sqrt(2)*Hbb*Lf - eta(tau)*xi(tau))*np.sin(taup(tau)*X/lw) - xi(tau)*np.cos(taup(tau)*X/lw) )

		self.verbosity(2, "")
		self.verbosity(2, f"a1 = {a1:.2e}, b1 = {b1:.2e}")
		self.verbosity(2, f"H = {H:.1e}, Lf = {Lf:.1e}")
		self.verbosity(2, f"tau = {tau:.2e}, res = {root(tau):.1e}, cond = {np.abs(root(tau)) > 1e-13}")
		self.verbosity(2, f"w(0) = {W(0):.2e}")
		self.verbosity(2, f"{rootFinding.status} {rootFinding.message}")
		self.verbosity(2, "")
		self.verbosity(2, "")

		self.profile = W
		return W

	def getMinMax(self):
		bmin = 0; bmax = 25*self.lw
		for i in range(8):
			x = np.linspace(bmin, bmax, 10000)
			y = self.profile(x)
			idx = (np.abs(y - y.max())).argmin()
			bmin = x[max(idx-3,0)]
			bmax = x[min(idx+3, len(x)-1)]
		xmax = (bmin + bmax)/2
		if xmax<0:
			xmax = 0

		bmin = xmax; bmax = 50*self.lw
		for i in range(8):
			x = np.linspace(bmin, bmax, 10000)
			y = self.profile(x)
			idx = (np.abs(y - y.min())).argmin()
			bmin = x[max(idx-3,0)]
			bmax = x[min(idx+3, len(x)-1)]
		xmin = (bmin + bmax)/2
		return xmin, xmax

	def getRM(self):
		minPosition, maxPosition = self.getMinMax()
		xrm = minPosition
		wrm = self.profile(0) - self.profile(xrm)
		return np.array([xrm, wrm])

	def plotSlidder(self):
		x = np.linspace(0, 500, 3000)
		hi = 500
		lfi = 10
		lwi = 10
		self.h0 = hi
		y = self.getProfile(lf=lfi, lw=lwi)(x)
		fig, ax = plt.subplots()
		fig.subplots_adjust(left=0.1, bottom=0.35)
		l, = ax.plot(x, y, lw=2)

		minPosition, maxPosition = self.getMinMax()
		m, = plt.plot(maxPosition, self.profile(maxPosition), 'ro')
		p, = plt.plot(minPosition, self.profile(minPosition), 'bo')

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
			x = np.linspace(0, 10*lw, 3000)
			l.set_xdata(x)
			ax.set_xlim(x.min(), x.max())
			self.h0 = h0
			try:
				y = self.getProfile(lf=lf, lw=lw)(x)
			except myError as eee:
				print(eee)
				y = np.zeros(x.shape)
			ax.set_ylim([np.nanmin(y) - 0.1, np.nanmax(y) + 0.1])
			l.set_ydata(y)
			if not y.sum()==0:
				minPosition, maxPosition = self.getMinMax()
				m.set_data(maxPosition, self.profile(maxPosition))
				p.set_data(minPosition, self.profile(minPosition))
			fig.canvas.draw_idle()

		h0_slide.on_changed(update)
		lf_slide.on_changed(update)
		lw_slide.on_changed(update)
		plt.show()

	def getLwf(self, xrm, wrm):
		def fff(x):
			try:
				self.getProfile(lw=x[0], lf=x[1])
				a = self.getRM()
				return a
			except myError:
				return np.array([np.inf, np.inf])
		self.verbosity(1, f"Lwf RF start --> xrm = {xrm:.2e}, wrm = {wrm:.2e}")

		eps = 1e-7
		for lw0 in np.logspace(1, 3, 10):
			for lf0 in np.logspace(0, 3, 10):
				print(lw0, lf0)
				try:
					res = sp.optimize.root(lambda x: fff(x) - (xrm, wrm) , x0=(lw0, lf0), method="lm")
					lw = res.x[0]
					lf = res.x[1]
					if lw < 0 or lf < 0:
						raise myError("lwf RF step failed: lw or lf is negative")

					self.getProfile(lw=lw, lf=lf)
					xrmR, wrmR = self.getRM()
					error = np.sqrt((xrm-xrmR)**2 + (wrm-wrmR)**2)

				except myError as eee:
					self.verbosity(2, f"Lwf RF step failed --> {eee}")
					continue

				print("Error =", error)
				if error <= eps:
					self.verbosity(2, f"Lwf RF step succed --> lw = {lw:.2e}, lf = {lf:.2e}")
					break
			else:
				continue
			break
		else:
			raise myError("Lwf RF failed: no solution found")

		self.getProfile(lw=lw, lf=lf)
		xrmR, wrmR = self.getRM()
		error = np.sqrt((xrm-xrmR)**2 + (wrm-wrmR)**2)
		if not error <= eps:
			raise myError("Lwf RF failed: no solution found")

		return {"lw":lw, "lf":lf}

	def plot(self, save=False, title=None):
		x = np.linspace(0, self.lw*15, 10000000)
		y = self.profile(x)
		minPosition, maxPosition = self.getMinMax()

		fig = plt.figure(figsize=(15, 15), constrained_layout=True)
		ax = fig.add_subplot()
		ax.plot(x,y)
		ax.plot(maxPosition, self.profile(maxPosition), 'ro')
		ax.plot(minPosition, self.profile(minPosition), 'bo')
		ax.set_title(f"lf = {self.lf:.2e}, lw = {self.lw:.2e}")
		ax.set_xlim(0,15*self.lw)
		if save:
			if title is None:
				title = f"./{np.random.uniform(0,10000)}.png"
			fig.savefig(title)
		else:
			plt.show()
		plt.close(fig=fig)

class modelLinNoTenNoDim():
	def __init__(self, h0, verbose=0):
		self.verbose = verbose
		self.h0 = h0
		self.rhotio = rhotio
		self.reset()

	def verbosity(self, verboseLvl, text):
		if self.verbose >= verboseLvl:
				print("--"*verboseLvl+">", text)

	def reset(self):
		self.lf = None
		self.lw = None

		self.H = None
		self.Lf = None

		self.profile = None
		self.tau = None
		self.Hbb = None

	def getProfile(self, Lf, H):
		if not isinstance(Lf, (float, int, np.float64, np.int64)) or Lf is np.nan:
			raise TypeError("Input variable Lf should be a scalar.")
		if not isinstance(H, (float, int, np.float64, np.int64)) or H is np.nan:
			raise TypeError("Input variable H should be a scalar.")
		self.reset()
		h = self.h0
		self.Lf = Lf
		self.H = H

		rhotio = self.rhotio
		D = rhotio*H
		Hbb = (1 - D/H)*D
		a1 = rhotio*(1 - 3*rhotio + 2*rhotio**2)/12
		a = a1*H**3

		w = lambda x: np.exp(-x/np.sqrt(2)) * ( (a + np.sqrt(2)*Hbb*Lf)*np.cos(x/np.sqrt(2)) - a*np.sin(x/np.sqrt(2)) )

		if np.isnan(w(np.linspace(0,10,100))).any():
			raise myError("Nan in solution")

		self.verbosity(3, "")
		self.verbosity(3, f"a1 = {a1:.2e}")
		self.verbosity(3, f"H = {H:.1e}, Lf = {Lf:.1e}")
		self.verbosity(3, f"w(0) = {w(0):.2e}")
		self.verbosity(3, "")
		self.verbosity(3, "")

		self.profile = w
		return w

	def getProfile3(self, Lf, H, a):
		if not isinstance(Lf, (float, int, np.float64, np.int64)) or Lf is np.nan:
			raise TypeError("Input variable Lf should be a scalar.")
		if not isinstance(H, (float, int, np.float64, np.int64)) or H is np.nan:
			raise TypeError("Input variable H should be a scalar.")
		self.reset()
		h = self.h0
		self.Lf = Lf
		self.H = H

		rhotio = self.rhotio
		D = rhotio*H
		Hbb = (1 - D/H)*D

		w = lambda x: np.exp(-x/np.sqrt(2)) * ( (a + np.sqrt(2)*Hbb*Lf)*np.cos(x/np.sqrt(2)) - a*np.sin(x/np.sqrt(2)) )

		if np.isnan(w(np.linspace(0,10,100))).any():
			raise myError("Nan in solution")

		self.verbosity(3, "")
		self.verbosity(3, f"a1 = {a1:.2e}")
		self.verbosity(3, f"H = {H:.1e}, Lf = {Lf:.1e}")
		self.verbosity(3, f"w(0) = {w(0):.2e}")
		self.verbosity(3, "")
		self.verbosity(3, "")

		self.profile = w
		return w

	def getMinMax(self):
		bmin = 0; bmax = 25
		for i in range(8):
			x = np.linspace(bmin, bmax, 10000)
			y = self.profile(x)
			idx = (np.abs(y - y.max())).argmin()
			bmin = x[max(idx-3,0)]
			bmax = x[min(idx+3, len(x)-1)]
		xmax = (bmin + bmax)/2
		if xmax<0:
			xmax = 0

		bmin = xmax; bmax = 50
		for i in range(8):
			x = np.linspace(bmin, bmax, 10000)
			y = self.profile(x)
			idx = (np.abs(y - y.min())).argmin()
			bmin = x[max(idx-3,0)]
			bmax = x[min(idx+3, len(x)-1)]
		xmin = (bmin + bmax)/2
		return xmin, xmax

	def getRM(self):
		minPosition, maxPosition = self.getMinMax()
		xrm = minPosition
		wrm = self.profile(0) - self.profile(xrm)
		return np.array([xrm, wrm])

	def plotSlidder(self):
		x = np.linspace(0, 15, 1000)
		Hi = 1
		Lfi = 0
		y = self.getProfile(H=Hi, Lf=Lfi)(x)
		fig, ax = plt.subplots()
		plt.subplots_adjust(left=0.1, bottom=0.35)
		l, = plt.plot(x, y, lw=2)

		minPosition, maxPosition = self.getMinMax()
		m, = plt.plot(maxPosition, self.profile(maxPosition), 'ro')
		p, = plt.plot(minPosition, self.profile(minPosition), 'bo')

		axcolor = 'lightgoldenrodyellow'
		ax_H = plt.axes([0.1, 0.2, 0.65, 0.03], facecolor=axcolor)
		ax_Lf = plt.axes([0.1, 0.15, 0.65, 0.03], facecolor=axcolor)

		s_H = mp.widgets.Slider(ax_H, 'H', 0, 5, valinit=Hi)
		s_Lf = mp.widgets.Slider(ax_Lf, 'Lf', 0, 1, valinit=Lfi)

		def update(val):
			H = s_H.val
			Lf = s_Lf.val
			try:
				self.getProfile(H=H, Lf=Lf)
				y = self.profile(x)
				ax.set_ylim([y.min() - np.abs(y.min())*0.5, y.max() + np.abs(y.max())*0.5])
			except:
				y = np.zeros(x.shape)
			l.set_ydata(y)

			minPosition, maxPosition = self.getMinMax()
			m.set_data(maxPosition, self.profile(maxPosition))
			p.set_data(minPosition, self.profile(minPosition))
			
			fig.canvas.draw_idle()

		s_H.on_changed(update)
		s_Lf.on_changed(update)
		plt.show()

	def plotSlidder3(self):
		x = np.linspace(0, 15, 1000)
		Hi = 1
		Lfi = 0
		ai = 0
		y = self.getProfile3(H=Hi, Lf=Lfi, a=ai)(x)
		fig, ax = plt.subplots()
		plt.subplots_adjust(left=0.1, bottom=0.35)
		l, = plt.plot(x, y, lw=2)

		minPosition, maxPosition = self.getMinMax()
		m, = plt.plot(maxPosition, self.profile(maxPosition), 'ro')
		p, = plt.plot(minPosition, self.profile(minPosition), 'bo')

		axcolor = 'lightgoldenrodyellow'
		ax_H = plt.axes([0.1, 0.2, 0.65, 0.03], facecolor=axcolor)
		ax_Lf = plt.axes([0.1, 0.15, 0.65, 0.03], facecolor=axcolor)
		ax_a = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor=axcolor)

		s_H = mp.widgets.Slider(ax_H, 'H', 0, 5, valinit=Hi)
		s_Lf = mp.widgets.Slider(ax_Lf, 'Lf', 0, 2, valinit=Lfi)
		s_a = mp.widgets.Slider(ax_a, 'a', -1, 1, valinit=ai)

		def update(val):
			H = s_H.val
			Lf = s_Lf.val
			a = s_a.val
			try:
				y = self.getProfile3(H=H, Lf=Lf, a=a)(x)
				ax.set_ylim([y.min() - np.abs(y.min())*0.5, y.max() + np.abs(y.max())*0.5])
			except:
				y = np.zeros(x.shape)
			l.set_ydata(y)

			minPosition, maxPosition = self.getMinMax()
			m.set_data(maxPosition, self.profile(maxPosition))
			p.set_data(minPosition, self.profile(minPosition))
			
			fig.canvas.draw_idle()

		s_H.on_changed(update)
		s_Lf.on_changed(update)
		s_a.on_changed(update)
		plt.show()

	def getLwf(self, Xrm, Wrm):
		def fff(x):
			self.getProfile(H=x[0], Lf=x[1])
			return self.getRM()
		self.verbosity(1, f"Lwf RF start --> Xrm = {Xrm:.2e}, Wrm = {Wrm:.2e}")

		eps = 1e-11
		for H0 in np.concatenate([[1],np.linspace(0,5,50)]):
			for Lf0 in np.concatenate([[0.5],np.linspace(0,1,50)]):
				try:
					res = sp.optimize.root(lambda x: fff(x) - (Xrm, Wrm) , x0=(H0, Lf0), tol=1e-14, method="lm", options={"factor":0.1})
					H = res.x[0]
					Lf = res.x[1]
					if H < 0 or Lf < 0:
						raise myError("lwf RF step failed: H or Lf is negative")

					self.getProfile(H=H, Lf=Lf)
					XrmR, WrmR = self.getRM()
					error = np.sqrt((Xrm-XrmR)**2 + (Wrm-WrmR)**2)
					print(error)

				except myError as eee:
					continue
					self.verbosity(2, f"Lwf RF step failed --> {eee}")

				if error <= eps:
					self.verbosity(2, f"Lwf RF step succed --> H = {H:.2e}, Lf = {Lf:.2e}")
					break
			else:
				continue
			break
		else:
			raise myError("Lwf RF failed: no solution found")

		self.getProfile(H=H, Lf=Lf)
		XrmR, WrmR = self.getRM()
		error = np.sqrt((Xrm-XrmR)**2 + (Wrm-WrmR)**2)
		if not error <= eps:
			raise myError("Lwf RF failed: no solution found")

		return {"H":H, "Lf":Lf}

	def plot(self, save=False, title=None):
		x = np.linspace(0, 15, 100000)
		y = self.profile(x)
		minPosition, maxPosition = self.getMinMax()

		fig = plt.figure(figsize=(15, 15), constrained_layout=True)
		ax = fig.add_subplot()
		ax.plot(x,y)
		ax.plot(maxPosition, self.profile(maxPosition), 'ro')
		ax.plot(minPosition, self.profile(minPosition), 'bo')
		ax.set_title(f"H = {self.H:.2e}, Lf = {self.Lf:.2e}")
		if save:
			if title is None:
				title = f"./{np.random.uniform(0,10000)}.png"
			fig.savefig(title)
		else:
			plt.show()
		plt.close(fig=fig)

class modelLinNoTenDim():
	def __init__(self, h0, verbose=0):
		self.verbose = verbose
		self.h0 = h0
		self.rhotio = rhotio
		self.E = 10e6
		self.g = g
		self.rhow = rhow
		self.nu = nu
		self.reset()

	def verbosity(self, verboseLvl, text):
		if self.verbose >= verboseLvl:
				print("--"*verboseLvl+">", text)

	def reset(self):
		self.lf = None
		self.lw = None

		self.H = None
		self.Lf = None

		self.profile = None
		self.tau = None
		self.Hbb = None
		self.a1 = None
		self.b1 = None
		self.a = None
		self.b = None

	def getProfile(self, lf, lw):
		if not isinstance(lf, (float, int, np.float64, np.int64)) or lf is np.nan:
			raise TypeError("Input variable lf should be a scalar.")
		if not isinstance(lw, (float, int, np.float64, np.int64)) or lw is np.nan:
			raise TypeError("Input variable lw should be a scalar.")
		
		self.reset()
		h = self.h0
		rhotio = self.rhotio

		H = h/lw
		Lf = lf/lw
		D = rhotio*H
		Hbb = (1 - D/H)*D
		a1 = rhotio*(1 - 3*rhotio + 2*rhotio**2)/12
		a = a1*H**3
		b1 = rhotio*(1 - rhotio)/2
		b = b1*H**2

		w = lambda x: lw*np.exp(-x/np.sqrt(2)/lw) * ( (a + np.sqrt(2)*Hbb*Lf)*np.cos(x/np.sqrt(2)/lw) - a*np.sin(x/np.sqrt(2)/lw) )

		if np.isnan(w(np.linspace(0,10,100))).any():
			raise myError("Nan in solution")

		self.verbosity(3, "")
		self.verbosity(3, f"a1 = {a1:.2e}")
		self.verbosity(3, f"H = {H:.1e}, Lf = {Lf:.1e}")
		self.verbosity(3, f"w(0) = {w(0):.2e}")
		self.verbosity(3, "")
		self.verbosity(3, "")

		self.profile = w
		self.lf = lf
		self.Lf = Lf
		self.lw = lw
		self.a1 = a1
		self.b1 = b1
		self.a = a
		self.b = b

		return w

	def getProfile2(self, lf, a):
		if not isinstance(lf, (float, int, np.float64, np.int64)) or lf is np.nan:
			raise TypeError("Input variable lf should be a scalar.")
		if not isinstance(a, (float, int, np.float64, np.int64)) or a is np.nan:
			raise TypeError("Input variable a should be a scalar.")
		self.reset()
		h = self.h0

		lw = (self.E*h**3/12/self.rhow/self.g/(1 - self.nu**2))**0.25

		self.lf = lf
		self.lw = lw

		rhotio = self.rhotio
		H = h/lw
		Lf = lf/lw
		D = rhotio*H
		Hbb = (1 - D/H)*D

		w = lambda x: lw*np.exp(-x/np.sqrt(2)/lw) * ( (a + np.sqrt(2)*Hbb*Lf)*np.cos(x/np.sqrt(2)/lw) - a*np.sin(x/np.sqrt(2)/lw) )

		if np.isnan(w(np.linspace(0,10,100))).any():
			raise myError("Nan in solution")

		self.verbosity(3, "")
		self.verbosity(3, f"a = {a:.1e}, Lf = {Lf:.1e}")
		self.verbosity(3, f"w(0) = {w(0):.2e}")
		self.verbosity(3, "")
		self.verbosity(3, "")

		self.profile = w
		return w

	def getProfile3(self, lf, lw, a):
		if not isinstance(lf, (float, int, np.float64, np.int64)) or np.isnan(lf):
			raise TypeError("Input variable lf should be a scalar.")
		if not isinstance(lw, (float, int, np.float64, np.int64)) or np.isnan(lw):
			raise TypeError("Input variable lw should be a scalar.")
		if not isinstance(a, (float, int, np.float64, np.int64)) or np.isnan(a):
			raise TypeError("Input variable a should be a scalar.")
		self.reset()
		h = self.h0

		self.lf = lf
		self.lw = lw

		rhotio = self.rhotio
		H = h/lw
		Lf = lf/lw
		D = rhotio*H
		Hbb = (1 - D/H)*D

		w = lambda x: lw*np.exp(-x/np.sqrt(2)/lw) * ( (a + np.sqrt(2)*Hbb*Lf)*np.cos(x/np.sqrt(2)/lw) - a*np.sin(x/np.sqrt(2)/lw) )

		if np.isnan(w(np.linspace(0,10,100))).any():
			raise myError("Nan in solution")

		self.verbosity(3, "")
		self.verbosity(3, f"H = {H:.1e}, Lf = {Lf:.1e}")
		self.verbosity(3, f"w(0) = {w(0):.2e}")
		self.verbosity(3, "")
		self.verbosity(3, "")

		self.profile = w
		return w

	def getMinMax(self):
		bmin = 0; bmax = 25*self.lw
		for i in range(8):
			x = np.linspace(bmin, bmax, 10000)
			y = self.profile(x)
			idx = (np.abs(y - y.max())).argmin()
			bmin = x[max(idx-3,0)]
			bmax = x[min(idx+3, len(x)-1)]
		xmax = (bmin + bmax)/2
		if xmax<0:
			xmax = 0

		bmin = xmax; bmax = 50*self.lw
		for i in range(8):
			x = np.linspace(bmin, bmax, 10000)
			y = self.profile(x)
			idx = (np.abs(y - y.min())).argmin()
			bmin = x[max(idx-3,0)]
			bmax = x[min(idx+3, len(x)-1)]
		xmin = (bmin + bmax)/2
		return xmin, xmax

	def getRM(self):
		minPosition, maxPosition = self.getMinMax()
		xrm = minPosition
		wrm = self.profile(0) - self.profile(xrm)
		return np.array([xrm, wrm])

	def plotSlidder(self):
		x = np.linspace(0, 500, 3000)
		hi = 500
		lfi = 10
		lwi = 10
		self.h0 = hi
		y = self.getProfile(lf=lfi, lw=lwi)(x)
		fig, ax = plt.subplots()
		plt.subplots_adjust(left=0.1, bottom=0.35)
		l, = plt.plot(x, y, lw=2)

		minPosition, maxPosition = self.getMinMax()
		m, = plt.plot(maxPosition, self.profile(maxPosition), 'ro')
		p, = plt.plot(minPosition, self.profile(minPosition), 'bo')

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
			x = np.linspace(0, 10*lw, 3000)
			l.set_xdata(x)
			ax.set_xlim(x.min(), x.max())
			try:
				self.h0 = h0
				y = self.getProfile(lf=lf, lw=lw)(x)
				ax.set_ylim([y.min() - 0.1, y.max() + 0.1])
			except myError as eee:
				print(eee)
				y = np.zeros(x.shape)
			l.set_ydata(y)

			minPosition, maxPosition = self.getMinMax()
			m.set_data(maxPosition, self.profile(maxPosition))
			p.set_data(minPosition, self.profile(minPosition))

			fig.canvas.draw_idle()

		h0_slide.on_changed(update)
		lf_slide.on_changed(update)
		lw_slide.on_changed(update)
		plt.show()

	def plotSlidder3(self):
		x = np.linspace(0, 500, 3000)
		hi = 500
		lfi = 10
		lwi = 10
		ai = -0.006
		self.h0 = hi
		y = self.getProfile3(lf=lfi, lw=lwi, a=ai)(x)
		fig, ax = plt.subplots()
		plt.subplots_adjust(left=0.1, bottom=0.35)
		l, = plt.plot(x, y, lw=2)

		minPosition, maxPosition = self.getMinMax()
		m, = plt.plot(maxPosition, self.profile(maxPosition), 'ro')
		p, = plt.plot(minPosition, self.profile(minPosition), 'bo')

		axcolor = 'lightgoldenrodyellow'
		ax_h = plt.axes([0.1, 0.25, 0.65, 0.03], facecolor=axcolor)
		ax_lf = plt.axes([0.1, 0.2, 0.65, 0.03], facecolor=axcolor)
		ax_lw = plt.axes([0.1, 0.15, 0.65, 0.03], facecolor=axcolor)
		ax_a = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor=axcolor)

		h0_slide = mp.widgets.Slider(ax_h, 'h', 10, 600, valinit=hi)
		lf_slide = mp.widgets.Slider(ax_lf, 'lf', 0, 100, valinit=lfi)
		lw_slide = mp.widgets.Slider(ax_lw, 'lw', 0, 200, valinit=lwi)
		a_slide = mp.widgets.Slider(ax_a, 'a', -0.1, 0.1, valinit=ai)

		def update(val):
			h0 = h0_slide.val
			lf = lf_slide.val
			lw = lw_slide.val
			a = a_slide.val
			x = np.linspace(0, 10*lw, 3000)
			l.set_xdata(x)
			ax.set_xlim(x.min(), x.max())
			try:
				self.h0 = h0
				y = self.getProfile3(lf=lf, lw=lw, a=a)(x)
				ax.set_ylim([y.min() - 0.1, y.max() + 0.1])
			except myError as eee:
				print(eee)
				y = np.zeros(x.shape)
			l.set_ydata(y)

			minPosition, maxPosition = self.getMinMax()
			m.set_data(maxPosition, self.profile(maxPosition))
			p.set_data(minPosition, self.profile(minPosition))

			fig.canvas.draw_idle()

		h0_slide.on_changed(update)
		lf_slide.on_changed(update)
		lw_slide.on_changed(update)
		a_slide.on_changed(update)
		plt.show()

	def getLwf(self, xrm, wrm, lw0=None, lf0=None):
		def fff(x):
			try:
				self.getProfile(lw=x[0], lf=x[1])
				a = self.getRM()
				return a
			except myError:
				return np.array([np.inf, np.inf])
		self.verbosity(1, f"Lwf RF start --> xrm = {xrm:.2e}, wrm = {wrm:.2e}")

		eps = 5e-8
		lw0Range = np.logspace(1, 3, 10)
		lf0Range = np.logspace(0, 3, 10)
		if lw0 is not None:
			lw0Range = np.concatenate([[lw0], lw0Range])
		if lf0 is not None:
			lf0Range = np.concatenate([[lf0], lf0Range])

		for lw0 in lw0Range:
			for lf0 in lf0Range:
				self.verbosity(2, f"Lwf RF step --> lf0 = {lf0:.2e}, lw0 = {lw0:.2e}")
				try:
					res = sp.optimize.root(lambda x: fff(x) - (xrm, wrm) , x0=(lw0, lf0), method="lm")
					lw = res.x[0]
					lf = res.x[1]
					if lw < 0 or lf < 0:
						raise myError("lwf RF step failed: lw or lf is negative")

					self.getProfile(lw=lw, lf=lf)
					xrmR, wrmR = self.getRM()
					error = np.sqrt((xrm-xrmR)**2 + (wrm-wrmR)**2)

				except myError as eee:
					self.verbosity(2, f"Lwf RF step failed --> {eee}")
					continue

				self.verbosity(2, f"Lwf RF step --> Error = {error:.2e}")
				if error <= eps:
					self.verbosity(2, f"Lwf RF step succed --> lw = {lw:.2e}, lf = {lf:.2e}")
					break
			else:
				continue
			break
		else:
			raise myError("Lwf RF failed: no solution found")

		self.getProfile(lw=lw, lf=lf)
		xrmR, wrmR = self.getRM()
		error = np.sqrt((xrm-xrmR)**2 + (wrm-wrmR)**2)
		if not error <= eps:
			raise myError("Lwf RF failed: no solution found")

		return {"lw":lw, "lf":lf}

	def getLaf(self, xrm, wrm, a0=None, lf0=None):
		def rootFindingFct(x):
			try:
				self.getProfile2(a=x[0], lf=x[1])
				xrmR, wrmR = self.getRM()
				return [(xrm-xrmR)/xrm, (wrm-wrmR)/wrm]
			except myError:
				return [np.inf, np.inf]
		
		self.verbosity(1, f"Lwf RF start --> xrm = {xrm:.2e}, wrm = {wrm:.2e}")
		eps = 5e-8
		a0Range = np.linspace(-2e-3, -1.5e-3, 30)
		lf0Range = np.logspace(0, 3, 5)
		if a0 is not None:
			a0Range = np.concatenate([[a0], a0Range])
		if lf0 is not None:
			lf0Range = np.concatenate([[lf0], lf0Range])

		for a0 in a0Range:
			for lf0 in lf0Range:
				self.verbosity(2, f"Laf RF step --> lf0 = {lf0:.2e}, a0 = {a0:.2e}")
				try:
					res = sp.optimize.root(rootFindingFct, x0=(a0, lf0), method="lm")
					a = res.x[0]
					lf = res.x[1]
					if lf < 0:
						raise myError("laf RF step failed: a or lf is negative")

					self.getProfile2(a=a, lf=lf)
					xrmR, wrmR = self.getRM()
					error = np.sqrt((xrm-xrmR)**2 + (wrm-wrmR)**2)

				except myError as eee:
					self.verbosity(2, f"Laf RF step failed --> {eee}")
					continue

				self.verbosity(2, f"Laf RF step --> Error = {error:.2e}")
				if error <= eps:
					self.verbosity(2, f"Laf RF step succed --> a = {a:.2e}, lf = {lf:.2e}")
					break
			else:
				continue
			break
		else:
			raise myError("Laf RF failed: no solution found")

		return {"a":a, "lf":lf}

	def getLwfa(self, xrm, wrm, a0=None, lf0=None, lw0=None):
		def rootFindingFct(x):
			try:
				self.getProfile3(a=x[0], lf=x[1], lw=x[2])
				xrmR, wrmR = self.getRM()
				return [np.sqrt(((xrm-xrmR)/xrm)**2 + ((wrm-wrmR)/wrm)**2),0,0]
			except myError:
				return [np.inf,0,0]
		
		self.verbosity(1, f"Lwf RF start --> xrm = {xrm:.2e}, wrm = {wrm:.2e}")
		eps = 1e-8
		a0Range = np.logspace(-4, 0, 5)
		a0Range = np.concatenate([-a0Range, a0Range])
		lf0Range = np.logspace(0, 3, 5)
		lw0Range = np.logspace(0, 3, 5)
		if a0 is not None:
			a0Range = np.concatenate([[a0], a0Range])
		if lf0 is not None:
			lf0Range = np.concatenate([[lf0], lf0Range])
		if lw0 is not None:
			lw0Range = np.concatenate([[lw0], lw0Range])

		for a0 in a0Range:
			for lf0 in lf0Range:
				for lw0 in lw0Range:
					self.verbosity(2, f"Laf RF step --> lf0 = {lf0:.2e}, a0 = {a0:.2e}, lw0 = {lw0:.2e}")
					try:
						res = sp.optimize.root(rootFindingFct, x0=(a0, lf0, lw0), method="lm")
						a = res.x[0]
						lf = res.x[1]
						lw = res.x[2]
						if lw < 0 or lf < 0:
							raise myError("laf RF step failed: a or lf is negative")

						self.getProfile3(a=a, lf=lf, lw=lw)
						xrmR, wrmR = self.getRM()
						error = np.sqrt(((xrm-xrmR)/xrm)**2 + ((wrm-wrmR)/wrm)**2)

						print(f"{a0:.2e}, {lf0:.2e}, {error:.2e}, {xrm:.2f} {xrmR:.2f}")
					except myError as eee:
						self.verbosity(2, f"Laf RF step failed --> {eee}")
						continue

					self.verbosity(2, f"Laf RF step --> Error = {error:.2e}")
					if error <= eps:
						print("BREAAAAAAA")
						self.verbosity(2, f"Laf RF step succed --> a = {a:.2e}, lf = {lf:.2e}, lw = {lw:.2e}")
						break
				else:
					continue
				break
			else:
				continue
			break
		else:
			raise myError("Laf RF failed: no solution found")

		return {"a":a, "lf":lf, "lw":lw}

	def getLwfa2(self, xrm, wrm, a0=None, lf0=None, lw0=None):
		def rootFindingFct(x):
			try:
				self.getProfile3(a=x[0], lf=x[1], lw=x[2])
				xrmR, wrmR = self.getRM()
				return [np.sqrt(((xrm-xrmR)/xrm)**2 + ((wrm-wrmR)/wrm)**2),0,0]
			except myError:
				return [np.inf,0,0]
		
		self.verbosity(1, f"Lwf RF start --> xrm = {xrm:.2e}, wrm = {wrm:.2e}")
		eps = 1e-8
		a0Range = np.logspace(-4, 0, 8)
		a0Range = np.concatenate([-a0Range, a0Range])
		lf0Range = np.logspace(0, 3, 8)
		lw0Range = np.logspace(0, 3, 8)
		if a0 is not None:
			a0Range = np.concatenate([[a0], a0Range])
		if lf0 is not None:
			lf0Range = np.concatenate([[lf0], lf0Range])
		if lw0 is not None:
			lw0Range = np.concatenate([[lw0], lw0Range])

		R = []
		for a0 in a0Range:
			for lf0 in lf0Range:
				for lw0 in lw0Range:
					self.verbosity(2, f"Laf RF step --> lf0 = {lf0:.2e}, a0 = {a0:.2e}, lw0 = {lw0:.2e}")
					try:
						res = sp.optimize.root(rootFindingFct, x0=(a0, lf0, lw0), method="lm")
						a = res.x[0]
						lf = res.x[1]
						lw = res.x[2]
						if lw < 0 or lf < 0:
							raise myError("laf RF step failed: a or lf is negative")

						self.getProfile3(a=a, lf=lf, lw=lw)
						xrmR, wrmR = self.getRM()
						error = np.sqrt(((xrm-xrmR)/xrm)**2 + ((wrm-wrmR)/wrm)**2)

						print(f"{a0:.2e}, {lf0:.2e}, {error:.2e}, {xrm:.2f} {xrmR:.2f}")
					except myError as eee:
						self.verbosity(2, f"Laf RF step failed --> {eee}")
						continue

					self.verbosity(2, f"Laf RF step --> Error = {error:.2e}")
					if error <= eps:
						print("BREAAAAAAA")
						self.verbosity(2, f"Laf RF step succed --> a = {a:.2e}, lf = {lf:.2e}, lw = {lw:.2e}")
						R.append({"a":a, "lf":lf, "lw":lw})
		import pandas as pd
		return pd.DataFrame.from_dict(R).drop_duplicates()

	def getLwfa3(self, r, f, xrm):
		def fit(x, lw, lf, a):
			profile = self.getProfile3(lw=lw, lf=lf, a=a)
			return profile(x) - profile(0) 

		idxMin = (np.abs(r - xrm)).argmin()
		idxMax = int(idxMin*1.5) + 4
		Var = []
		for idx in range(idxMin, idxMax):
			r_, f_ = r[:idx], f[:idx]
			f_ = f_ - f_[0]
			try:
				var = sp.optimize.curve_fit(f=fit, xdata=r_, ydata=f_, p0=(100, 0, 0.01), bounds=([0,0,-10], [2000,2000,10]) )[0]
			except RuntimeError:
				continue
			errFit = rootSquareError(lambda x: fit(x, a=var[2], lf=var[1], lw=var[0]), r_[:idxMin+2], f_[:idxMin+2])
			errXrm = np.abs(self.getRM()[0] - xrm)
			Var.append({"a":var[2], "lf":var[1], "lw":var[0], "errFit":errFit, "errXrm":errXrm, "idx":idx})
		Var = pd.DataFrame.from_dict(Var)

		# xSpline = np.linspace(0, xrm/8, 124000)
		# ySpline = sp.interpolate.make_interp_spline(x=r[:idxMax+10], y=f[:idxMax+10], k=3)(xSpline)
		# wp = np.gradient(ySpline, xSpline)
		# wpp_raw = (f[2] - 2*f[1] + f[0]) / (( r[2] - r[0])/2)**2

		# if wpp_raw<=7.5e-5 or wp.mean()>=0:
		# 	if len(Var[Var.lf>0.1])!=0:
		# 		Var = Var[Var.lf>0.1]
		# else:
		# 	if len(Var[Var.lf<0.1])!=0:
		# 		Var = Var[Var.lf<0.1]

		Var = Var.sort_values(by='errFit').head(int(len(Var)*0.30+1))
		Var = Var[Var.errXrm == Var.errXrm.min()]
		Var = {"lf":Var.lf.values[0], "lw":Var.lw.values[0], "a":Var.a.values[0], "idx":Var.idx.values[0]}
		return Var

	def plot(self, save=False, title=None):
		x = np.linspace(0, self.lw*15, 10000000)
		y = self.profile(x)
		minPosition, maxPosition = self.getMinMax()

		fig = plt.figure(figsize=(15, 15), constrained_layout=True)
		ax = fig.add_subplot()
		ax.plot(x,y)
		ax.plot(maxPosition, self.profile(maxPosition), 'ro')
		ax.plot(minPosition, self.profile(minPosition), 'bo')
		ax.set_title(f"lf = {self.lf:.2e}, lw = {self.lw:.2e}")
		ax.set_xlim(0,15*self.lw)
		if save:
			if title is None:
				title = f"./{np.random.uniform(0,10000)}.png"
			fig.savefig(title)
		else:
			plt.show()
		plt.close(fig=fig)

	def getXCrit(self, lf, lw, a):
		h0 = self.h0
		r = self.rhotio*(1 - self.rhotio)
		inCosUp = 2*a**2*lw**4 + 2**1.5*a*h0*lf*lw**2*r + h0**2*lf**2*r**2
		inCosDown = 2*a**2*lw**4 + 2**1.5*a*h0*lf*lw**2*r + 2*h0**2*lf**2*r**2
		inCos = (inCosUp)**0.5/(inCosDown)**0.5
		inSide = 2**0.5*a*lw**2/h0/r + lf
		return 2**0.5*lw*( \
			np.arccos(-inCos)*np.heaviside(-inSide,0) + \
			np.arccos(inCos)*np.heaviside(inSide,1) \
		  	)

	def getWppMax(self, lf, lw, a):
		h0 = self.h0
		r = self.rhotio*(1 - self.rhotio)
		X = self.getXCrit(lf=lf, lw=lw, a=a)/2**0.5/lw

		return lw**-3*np.exp(-X)*(\
			a*lw**2*np.cos(X) + \
			(a*lw**2 + 2**0.5*h0*lf*r)*np.sin(X)
			)

	def getSigMax(self, lf, lw, a):
		h0 = self.h0

		Y = 6*self.rhow*g*lw**4/h0**2
		return Y*self.getWppMax(lf=lf, lw=lw, a=a)
