import numpy as np
import warnings

def  cosgrowRhoDE(z = 1, w0 = -1, wprime = 0, rhoDE = 1):
	"""Function to calculate the reference energy density for dark energy

	"""

	return rhoDE * ((1/(1 + z))**(-(3 + 3 * w0 + 6 * wprime))) * np.exp(-6 * wprime * (1 - 1/(1 + z)))


def cosgrowH(z = 1, H0 = 100, OmegaM = 0.3, OmegaR = 0, w0 = -1, wprime = 0): 
	""" Function to grow H0 to the required redshift

	"""

	OmegaL = 1 - OmegaM - OmegaR
	OmegaK = 1 - OmegaM - OmegaL - OmegaR
	return H0 * np.sqrt(OmegaR * (1 + z)**4 + OmegaM * (1 + z)**3 + OmegaK * (1 + z)**2 + OmegaL * cosgrowRhoDE(z = z, w0 = w0, wprime = wprime, rhoDE = 1))



def cosgrowRhoCrit(z = 1, H0 = 100, OmegaM = 0.3, OmegaR = 0, w0 = -1, wprime = 0, Dist = "Ang", Mass = "Msun"):
	"""Function to grow &rho<SUB>crit</SUB> to the required redshift

	""" 

	OmegaL = 1 - OmegaM - OmegaR
	G = 6.67384e-11
	km2m = 1000
	Mpc2m = 3.08567758e+22
	Msol2kg = 1.9891e+30
	Msun10=10e10
	Hub2 = cosgrowH(z = z, H0 = H0, OmegaM = OmegaM,OmegaR = OmegaR, w0 = w0, wprime = wprime)**2
	RhoCrit = (3 * Hub2)/(8 * np.pi * G) * (km2m**2)/Mpc2m**2
	if (Mass == "Msun"): 
	    RhoCrit = RhoCrit/Msol2kg
	if (Dist == "Ang"): 
	    RhoCrit = RhoCrit * Mpc2m**3
	if (Dist == "Co"): 
	    RhoCrit = RhoCrit * Mpc2m**3/(1 + z)**3
	return RhoCrit

def cosgrowOmegaM(z = 1, OmegaM = 0.3, OmegaR = 0, w0 = -1, wprime = 0):
	"""Function to grow &Omega<SUB>M</SUB> 

	"""

	OmegaL = 1 - OmegaM - OmegaR

	OmegaK = 1 - OmegaM - OmegaL - OmegaR
	return (OmegaM * (1 + z)**3)/(OmegaR * (1 + z)**4 + OmegaM * (1 + z)**3 + OmegaK * (1 + z)**2 + OmegaL * cosgrowRhoDE(z = z, w0 = w0, wprime = wprime, rhoDE = 1))

def cosgrowRhoMean(z = 1, H0 = 100, OmegaM = 0.3, OmegaR = 0, w0 = -1, wprime = 0, Dist = "Ang", Mass = "Msun"):
	
	"""Function to grow &rho<SUB>mean</SUB> to the required redshift

	"""



	OmegaL = 1 - OmegaM - OmegaR
	G = 6.67384e-11
	km2m = 1000
	Mpc2m = 3.08567758e+22
	Msol2kg = 1.9891e+30
	Msun10=10e10
	Hub2 = cosgrowH(z = z, H0 = H0, OmegaM = OmegaM,OmegaR = OmegaR, w0 = w0, wprime = wprime)**2
	RhoCrit = (3 * Hub2)/(8 * np.pi * G) * (km2m**2)/Mpc2m**2
	if (Mass == "Msun"): 
	    RhoCrit = RhoCrit/Msol2kg

	if (Dist == "Ang"):
	    RhoCrit = RhoCrit * Mpc2m**3

	if (Dist == "Co"): 
	    RhoCrit = RhoCrit * Mpc2m**3/(1 + z)**3

	OmegaMatz = cosgrowOmegaM(z = z, OmegaM = OmegaM, OmegaR = OmegaR, w0 = w0, wprime = wprime)
	RhoMean = RhoCrit * OmegaMatz
	return(RhoMean)




def coshaloRvirToMvir(Rvir,Rho,DeltaVir,Munit=1e10): 

	"""Functions for convert R<SUB>vir</SUB> to M<SUB>vir</SUB> 


	"""

	return ((4 *np.pi/3) * DeltaVir * Rho * Rvir**3)/Munit

def coshaloMvirToRvir(Mvir = 1e+12, z = 0, H0 = 67.51, OmegaM = 0.3, OmegaL=0.7, OmegaR = 0, Rho = "crit", Dist = "Co", DeltaVir = 200, Munit = 1e10, Lunit = 1e+06, Vunit = 1000, Dim = 3): 

	"""Function to M<SUB>vir</SUB> to R<SUB>vir</SUB> 

	"""

	OmegaL = 1 - OmegaM - OmegaR

	G = 6.67384e-11
	msol_to_kg = 1.98892e+30
	pc_to_m = 3.08568e+16
	g = G * msol_to_kg/(pc_to_m)
	g = g * Munit/(Lunit * Vunit**2)
	if (Rho == "crit"):
	    RhoVal = (cosgrowRhoCrit(z = z, H0 = H0, OmegaM = OmegaM, OmegaR = OmegaR, Dist = Dist)/1e+09) * ((Lunit/1000)**3)/Munit
	   
	Rvir = ((3 * Mvir)/(4 * np.pi * DeltaVir * RhoVal))**(1/3)
	if (Dim == 2):
	    Rvir = Rvir/1.37
	return Rvir



def cosNFWmass_Rmax(Rho0=2.412e15, Rs=0.03253, Rmax=0.16265):
  return 4*np.pi*Rho0*(Rs**3)*(np.log((Rs+Rmax)/Rs)-Rmax/(Rs+Rmax))

def cosFindR200(r,rhocrit,DeltaVir,particlemass):

	r = np.sort(r)

	rhodif= np.abs(DeltaVir*rhocrit - (np.arange(len(r)) + 1) * particlemass / ((3/4 * np.pi * r**3)))

	return r[rhodif.argmin()] 


def cosNFWvesc(Rad=0.16264, Mvir=1e12, c=5, f=float("inf"), z = 0, H0 = 100.00, OmegaM = 0.3, OmegaL = 0.7, OmegaR = 0, Rho = "crit", Dist = "Co", DeltaVir = 200, Munit = 1e10, Lunit = 1e+06, Vunit = 1e3):
	G = 6.67384e-11
	msol_to_kg = 1.98892e+30
	pc_to_m = 3.08568e+16
	g = G * msol_to_kg/(pc_to_m)
	g = g * Munit/(Lunit * Vunit**2)

	R200=coshaloMvirToRvir(Mvir=Mvir, z=z, H0=H0, OmegaM=OmegaM, OmegaL=OmegaL, OmegaR=OmegaR, Rho=Rho, Dist=Dist, DeltaVir=200, Munit = Munit, Lunit=Lunit, Vunit=Vunit)
	vcircR200=np.sqrt(g*Mvir/R200)

	x=Rad/R200
	g=np.log(1+c)-c/(1+c)
	vesc=np.zeros(len(x))
	try:
		vesc[x<=f]=np.sqrt((vcircR200**2/g) * (np.log(1+c*x[x<=f])/x[x<=f]-c/(1+f*c)))
	except:
		print(x[x<=f][((vcircR200**2/g) * (np.log(1+c*x[x<=f])/x[x<=f]-c/(1+f*c)))<0])
	vesc[x==0]=np.sqrt((vcircR200**2/g) * (c-c/(1+f*c)))
	vesc[x>f]=vcircR200/np.sqrt(x[x>f])
	return vesc*np.sqrt(2) 



def cosNFWvcirc(Rad=0.16264, Mvir=1e12, c=5, f=float("inf"), z = 0, H0 = 100.00, OmegaM = 0.3, OmegaL = 0.7, OmegaR = 0, Rho = "crit", Dist = "Co", DeltaVir = 200, Munit = 1e10, Lunit = 1e+06, Vunit = 1e3):
	G = 6.67384e-11
	msol_to_kg = 1.98892e+30
	pc_to_m = 3.08568e+16
	g = G * msol_to_kg/(pc_to_m)
	g = g * Munit/(Lunit * Vunit**2)
	  
	R200=coshaloMvirToRvir(Mvir=Mvir, z=z, H0=H0, OmegaM=OmegaM, OmegaL=OmegaL, OmegaR=OmegaR, Rho=Rho, Dist=Dist, DeltaVir=200, Munit = Munit, Lunit=Lunit, Vunit=Vunit)
	Rmax=coshaloMvirToRvir(Mvir=Mvir, z=z, H0=H0, OmegaM=OmegaM, OmegaL=OmegaL, OmegaR=OmegaR, Rho=Rho, Dist=Dist, DeltaVir=DeltaVir, Munit = Munit, Lunit=Lunit, Vunit=Vunit)
	Rho0=Mvir/cosNFWmass_Rmax(Rho0=1, Rs=R200/c, Rmax=Rmax)
	MassCont=cosNFWmass_Rmax(Rho0=Rho0, Rs=R200/c, Rmax=Rad)
	MassContf=cosNFWmass_Rmax(Rho0=Rho0, Rs=R200/c, Rmax=f*Rmax)
	MassCont[Rad/R200>f]=MassContf

	vcirc=np.sqrt(g*MassCont/Rad)
	vcirc[Rad==0]=0
	return vcirc
