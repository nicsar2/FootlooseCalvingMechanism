dataPath = './Data/' #Path to data
figPath = './Fig/' #Path to figure
minLon = 150 #Minimum longitude to shrink data when downloading
maxLon = 210 #Maximum longitude to shrink data when downloading
minLat = -89 #Minimum latitude to shrink data when downloading
maxLat = -63 #Maximum latitude to shrink data when downloading
a1 = 8.7e-6 #Coef for wave-induced melting equation
a2 = 5.8e-7 #Coef for wave-induced melting equation
b1 = 0.67 #Coef for wave-induced melting equation
b2 = 0.33 #Coef for wave-induced melting equation
Rt = 6371e3 #Earth radius
g = 9.81 #Gravitational constant
E = 1e9 #Pure ice Young's modulus
rhoi = 850 #Density of ice shelf (ice+firn)
rhow = 1030 #Density of Ross Sea water
rhotio = rhoi/rhow #Ratio of density
rhotiob = 1-rhotio #1 minus ratio of density
nu = 0.3 #Poisson ration
sigmaY = 1e5 #Yield strength of ice
cmap = {
    'T':   'inferno',
    'C':   'PuBu_r',
    'u':   'seismic',
    'v':   'seismic',
    'w':   'cividis',
    'Me':  'viridis', 
}
lim = {
    'T':  [-2, 2],
    'C':  [0, 1],
    'u':  [-10, 10],
    'v':  [-10, 10],
    'w':  [0, 10],  
    'Me': [0, 1000],       
}
