from astropy.table import Table, vstack
import numpy as np
from plot import *
import matplotlib.pyplot as p
from astropy.modeling.models import Voigt1D
import scipy.optimize as op
p.rcParams['font.size']= 20.0
#p.rcParams.update({'font.sans-serif':'Arial'})
p.rcParams['pdf.fonttype'] = 42
p.rcParams['savefig.dpi']=300
p.rcParams['savefig.format']='pdf'

#p.rc('text', usetex=True)
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import astropy.units as u
import glob
import emcee
import corner
import random
import scipy.stats as st
from scipy.stats import mode
import vaex
from scipy import ndimage
import pickle



def likelihood(x, mu, icov):
    x[2] = x[2]
    diff = x-mu
    return -np.dot(diff,np.dot(icov,diff))/2.0

def posterior(guess,means,covs,hlp,dist,cts):
    #print(guess[2])
    if guess[2] >0 and guess[2] < 50:
        return likelihood(guess,means,covs) + prior_d_mock(1.0/guess[2],hlp,dist,cts)
    else:
        return -np.inf

def prior_d_mock(d,hlp,dists,cts):
    dist = np.round(d*1000,decimals=0)
    #print(inds)
    unp = np.interp(dist,dists,cts)
    return np.log(unp)

def dmode(L,p,sp):
    #def func(d,L,p,sp):
        #return (d**3/L)-(2*d**2)+(p/sp**2*d)-(1/sp**2)
    #return op.fsolve(func,1/p,args=(L,p,sp))
    coeff = [1/L,-2,p/sp**2,-1/sp**2]
    roots = np.roots(coeff)
    roots = roots[np.where(np.isreal(roots))[0]]
    roots= roots.real
    if len(roots) ==1:
        return roots
    if len(roots) >1 and p>0:
        return min(roots)
    if len(roots)>1 and p<0:
        return roots[np.where(roots>0)[0]]


def MC(meanv,icov,L,hlp,dists,cts,ndim=3,nwalkers=100,burnsteps = 100,nsteps=1000):
    ndim, nwalkers = ndim, nwalkers
    sampler = emcee.EnsembleSampler(nwalkers, ndim, posterior, args=[meanv,icov,hlp,dists,cts])
    print(sampler)
    covm = np.linalg.inv(icov)
    p0 = np.zeros((3,100))
    print(dmode(L,meanv[2],np.sqrt(covm[2,2])))
    for i in range(nwalkers):
        p0[2,i] = 1/np.random.normal(dmode(L,meanv[2],np.sqrt(covm[2,2])),np.sqrt(covm[2,2]),1)
        p0[0,i] = np.random.normal(meanv[0],np.sqrt(covm[0,0]),1)
        p0[1,i] = np.random.normal(meanv[1],np.sqrt(covm[1,1]),1)
    #print(p0,np.shape(p0))
    p0 = p0.T
    #print p0
    pos, prob, state = sampler.run_mcmc(p0, burnsteps)
    sampler.reset()
    sampler.run_mcmc(pos,nsteps, rstate0=random.getstate())
    return sampler, dmode(L,meanv[2],np.sqrt(covm[2,2]))

def MC_G1(ind=0,offset=-0.029):
    import corner
    table = Table.read('/home/mrl2968/Desktop/giraffe/XBJ_cov.fits')
    #bjtable = Table.read('/home/mrl2968/Desktop/bacchus_r63_pub/Bulge/BulgexGaiadist.fits')
    mean = np.zeros(3)
    #ind = int(404)
    #ind = int(0)
    mean[2] = table['parallax_1'][ind]-offset
    mean[0] = table['pmra_1'][ind]
    mean[1] = table['pmdec_1'][ind]
    ra = table['RA_2'][ind]
    dec = table['DEC_2'][ind]
    hp = HEALPix(nside=32,frame="icrs",order='nested')
    coord = SkyCoord(ra*units.deg,dec*units.deg)
    hlp = hp.skycoord_to_healpix(coord)
    ps = Table.read('/home/mrl2968/Desktop/giraffe/prior.fits')
    inds = np.where(ps['healpix']== hlp)[0]
    #print(hlp)
    dists = ps['dist'][inds] ; cts = ps['ct'][inds]
    L = table['r_len'][ind]/1000.0
    cov = np.zeros((3,3))
    items = ['pmra','pmdec','parallax']
    for i in range(0,3):
        for j in range(0,3):
            if i == j:
                cov[i,j] = float(table[items[i]+'_error_1'][ind])**2
            if j <i:
                if i == 2:
                    cov[i,j] = float(table[items[i]+'_error_1'][ind])*float(table[items[j]+'_error_1'][ind])*float(table[items[i]+'_'+items[j]+'_corr'][ind])
                    cov[j,i] = cov[i,j]
                else:
                    cov[i,j] = float(table[items[i]+'_error_1'][ind])*float(table[items[j]+'_error_1'][ind])*float(table[items[j]+'_'+items[i]+'_corr'][ind])
                    cov[j,i] = cov[i,j]
    #print(mean)
    #print(cov)
    icov = np.linalg.inv(cov)
    sampler, rest = MC(mean,icov,L,hlp,dists,cts)
    a = sampler.flatchain
    a[:,2] = 1./a[:,2]
    #corner.corner(a,labels=['pmra','pmdec','d (kpc)'],quantiles=[.16,.5,.84],show_titles=True)
    return a, rest


def baye_uvw(sampler,rest,ra,sra,dec,sdec,rv,srv):
    import corner
    a = sampler
    u=[]
    v= []
    w = []
    x= []; y= []; z =[]
    #for i in range(len(a)):
    RV = np.random.normal(rv,srv,len(a))
    RA = np.random.normal(ra,sra,len(a))
    DEC= np.random.normal(dec,sdec,len(a))
    print(RA,DEC)
    X,Y,Z,U,V,W = gal_uvw(distance=a[:,2]*1000, ra=RA,dec=DEC,pmra=a[:,0], pmdec = a[:,1],vrad = RV,lsr='',galxyz=True)
    R = np.sqrt((X)**2+Y**2)
    #corner.corner(np.array([U,V,W,R,Z]).T,labels=['U', 'V','W','R','Z'],quantiles=[.16,.5,.84],show_titles=True)
    return U,V,W,X,Y,Z

def gal_uvw(distance=None, lsr=None, ra=None, dec=None, pmra=None, pmdec=None, vrad=None, plx=None, galxyz=False, R0=8.3,l=None,b=None,lsrvel=[-14.0, 12.24, 7.25]):
   """
    NAME:
        GAL_UVW
    PURPOSE:
        Calculate the Galactic space velocity (U,V,W) of star
    EXPLANATION:
        Calculates the Galactic space velocity U, V, W of star given its
        (1) coordinates, (2) proper motion, (3) distance (or parallax), and
        (4) radial velocity.
    CALLING SEQUENCE:
        GAL_UVW [/LSR, RA=, DEC=, PMRA= ,PMDEC=, VRAD= , DISTANCE=
                 PLX= ]
    OUTPUT PARAMETERS:
         U - Velocity (km/s) positive toward the Galactic *anti*center
         V - Velocity (km/s) positive in the direction of Galactic rotation
         W - Velocity (km/s) positive toward the North Galactic Pole
    REQUIRED INPUT KEYWORDS:
         User must supply a position, proper motion,radial velocity and distance
         (or parallax).    Either scalars or vectors can be supplied.
        (1) Position:
         RA - Right Ascension in *Degrees*
         Dec - Declination in *Degrees*
        (2) Proper Motion
         PMRA = Proper motion in RA in arc units (typically milli-arcseconds/yr)
         PMDEC = Proper motion in Declination (typically mas/yr)
        (3) Radial Velocity
         VRAD = radial velocity in km/s
        (4) Distance or Parallax
         DISTANCE - distance in parsecs
                    or
         PLX - parallax with same distance units as proper motion measurements
               typically milliarcseconds (mas)

    OPTIONAL INPUT KEYWORD:
         /LSR - If this keyword is set, then the output velocities will be
                corrected for the solar motion (U,V,W)_Sun = -8.5, +13.38, +6.49)
                Coskunoglu et al. 2011 to the local standard of rest
     EXAMPLE:
         (1) Compute the U,V,W coordinates for the halo star HD 6755.
             Use values from Hipparcos catalog, and correct to the LSR
         ra = ten(1,9,42.3)*15.    & dec = ten(61,32,49.5)
         pmra = 627.89  &  pmdec = 77.84         ;mas/yr
         dis = 144    &  vrad = -321.4
         gal_uvw,u,v,w,ra=ra,dec=dec,pmra=pmra,pmdec=pmdec,vrad=vrad,dis=dis,/lsr
             ===>  u=154  v = -493  w = 97        ;km/s

         (2) Use the Hipparcos Input and Output Catalog IDL databases (see
         http://idlastro.gsfc.nasa.gov/ftp/zdbase/) to obtain space velocities
         for all stars within 10 pc with radial velocities > 10 km/s

         dbopen,'hipparcos,hic'      ;Need Hipparcos output and input catalogs
         list = dbfind('plx>100,vrad>10')      ;Plx > 100 mas, Vrad > 10 km/s
         dbext,list,'pmra,pmdec,vrad,ra,dec,plx',pmra,pmdec,vrad,ra,dec,plx
         ra = ra*15.                 ;Need right ascension in degrees
         GAL_UVW,u,v,w,ra=ra,dec=dec,pmra=pmra,pmdec=pmdec,vrad=vrad,plx = plx
         forprint,u,v,w              ;Display results
    METHOD:
         Follows the general outline of Johnson & Soderblom (1987, AJ, 93,864)
         except that U is positive outward toward the Galactic *anti*center, and
         the J2000 transformation matrix to Galactic coordinates is taken from
         the introduction to the Hipparcos catalog.
    REVISION HISTORY:
         Written, W. Landsman                       December   2000
         fix the bug occuring if the input arrays are longer than 32767
           and update the Sun velocity           Sergey Koposov June 2008
   	   vectorization of the loop -- performance on large arrays
           is now 10 times higher                Sergey Koposov December 2008
   """
   import numpy


   n_params = 3

   if n_params == 0:
      print 'Syntax - GAL_UVW, U, V, W, [/LSR, RA=, DEC=, PMRA= ,PMDEC=, VRAD='
      print '                  Distance=, PLX='
      print '         U, V, W - output Galactic space velocities (km/s)'
      return None

   if ra is None or dec is None:
      raise Exception('ERROR - The RA, Dec (J2000) position keywords must be supplied (degrees)')
   if plx is None and distance is None:
      raise Exception('ERROR - Either a parallax or distance must be specified')
   if distance is not None:
      if numpy.any(distance==0):
         raise Exception('ERROR - All distances must be > 0')
      plx = 1e3 / distance          #Parallax in milli-arcseconds
   if plx is not None and numpy.any(plx==0):
      raise Exception('ERROR - Parallaxes must be > 0')

   cosd = numpy.cos(numpy.deg2rad(dec))
   sind = numpy.sin(numpy.deg2rad(dec))
   cosa = numpy.cos(numpy.deg2rad(ra))
   sina = numpy.sin(numpy.deg2rad(ra))

   k = 4.74047     #Equivalent of 1 A.U/yr in km/s
   a_g = numpy.array([[0.0548755604, +0.4941094279, -0.8676661490],
                [0.8734370902, -0.4448296300, -0.1980763734],
                [0.4838350155, 0.7469822445, +0.4559837762]])

   vec1 = vrad
   vec2 = k * pmra / plx  # --- Added cos(dec) to this as it is implicit
   vec3 = k * pmdec / plx

   u = (a_g[0,0] * cosa * cosd + a_g[1,0] * sina * cosd + a_g[2,0] * sind) * vec1 + (-a_g[0,0] * sina + a_g[1,0] * cosa) * vec2 + (-a_g[0,0] * cosa * sind - a_g[1,0] * sina * sind + a_g[2,0] * cosd) * vec3
   v = (a_g[0,1] * cosa * cosd + a_g[1,1] * sina * cosd + a_g[2,1] * sind) * vec1 + (-a_g[0,1] * sina + a_g[1,1] * cosa) * vec2 + (-a_g[0,1] * cosa * sind - a_g[1,1] * sina * sind + a_g[2,1] * cosd) * vec3
   w = (a_g[0,2] * cosa * cosd + a_g[1,2] * sina * cosd + a_g[2,2] * sind) * vec1 + (-a_g[0,2] * sina + a_g[1,2] * cosa) * vec2 + (-a_g[0,2] * cosa * sind - a_g[1,2] * sina * sind + a_g[2,2] * cosd) * vec3

   lsr_vel = numpy.array(lsrvel)
   if (lsr is not None):
      u = u + lsr_vel[0]
      v = v + lsr_vel[1]
      w = w + lsr_vel[2]

   # --- This part rturns the X,Y,Z (in kpc) and UVW of the star
   if galxyz == True:
      if l is not None and b is not None:
      	l = l*(pi/180.0)
      	b = b*(pi/180.0)
      else:
        coord = SkyCoord(ra*units.deg,dec*units.deg,frame='icrs')
        gal =coord.galactic
        l,b = (gal.l).radian, (gal.b).radian

      #print distance
      #print l*(180/pi),b*(180/pi)
      distance = 1000*1/plx
      X = -R0 + (distance/1000.00)*np.cos(l)*np.cos(b) #sun at (8.3,0,0)
      Y = (distance/1000.00)*np.cos(b)*np.sin(l)
      Z = (distance/1000.00)*np.sin(b)
      #print sqrt((X+R0)**2+Y**2+Z**2) #should be distance to star!

   if galxyz is False:
      return [-u,v,w] # -u is becuase U in code is defined as +if directed toward anti-galactic center
   elif galxyz is True:
      return [X,Y,Z,-u,v,w]  # -u is becuase U in code is defined as +if directed toward anti-galactic center
