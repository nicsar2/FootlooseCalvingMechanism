import numpy as np
import scipy as sp

from params import a1, a2, b1, b2, Rt, rhoi, rhow, rhotiob, g

def Me(ds, a1=a1, a2=a2, b1=b1, b2=b2): #Wave-induced melt equation
	return (b1 + b2*ds.T) * (a1*ds.w**0.5 + a2*ds.w) * (1 + np.cos(np.pi*ds.C**3))/2

def sphericalDst(lat1, lon1, lat2, lon2): #Distance between two points on a sphere
	dlat = np.pi/180*(lat2-lat1)/2
	dlon = np.pi/180*(lon2-lon1)/2
	mlat = np.pi/180*(lat1+lat2)/2
	dst = 2*Rt*np.arcsin( (np.sin(dlat)**2 + (1 - np.sin(dlat)**2 - np.sin(mlat)**2)*np.sin(dlon)**2)**0.5 )
	return dst
 
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

# def _lcrit(h0, E, nu, sigY):
# 	lw = _lw(h0, E, nu)
# 	return np.exp(np.pi/4)*rhow*h0*sigY/6/rhoi/lw/g/(rhow-rhoi)