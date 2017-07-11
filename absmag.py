import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

#Setting type distributions here
std_1a = 0.50
mean_1a = -19.25
std_1b = 1.12
mean_1b = -17.45
std_1c = 1.18
mean_1c = -17.66
std_2b = 0.92
mean_2b = -16.99
std_2l = 0.86
mean_2l = -17.98
std_2n = 1.36
mean_2n = -18.53
std_2p = 0.98
mean_2p = -18.53

#This sets the grid spacing
M=np.linspace(-14,-23,1000)


#Defines the distribution, for illustrative plotting plotting
def f(M,std,mean):
    f=(1./(2.*np.pi*std**2.)**(1./2.))*np.exp(-(M-mean)**2./(2.*std**2))
    return f


y_1a=f(M,std_1a,mean_1a)
y_1b=f(M,std_1b,mean_1b)
y_1c=f(M,std_1c,mean_1c)
y_2b=f(M,std_2b,mean_2b)
y_2l=f(M,std_2l,mean_2l)
y_2n=f(M,std_2n,mean_2n)
y_2p=f(M,std_2p,mean_2p)
#Uncomment this if you want to see the distribution functions
#plt.plot(M,y_1a,M,y_1b,M,y_1c,M,y_2b,M,y_2l,M,y_2n,M,y_2p)


#This produces random numbers within that same normal distrubtion,
#using the 'normal' function
M_1a = np.random.normal(mean_1a,std_1a,600)
M_1b = np.random.normal(mean_1b,std_1b,600)
M_1c = np.random.normal(mean_1c,std_1c,600)
M_2b = np.random.normal(mean_2b,std_2b,600)
M_2l = np.random.normal(mean_2l,std_2l,600)
M_2n = np.random.normal(mean_2n,std_2n,600)
M_2p = np.random.normal(mean_2p,std_2p,600)

#Uncomment this if you want to see all the distributions + functions
"""
count, bins, ignored = plt.hist(M_1a, 30, normed=True,histtype='step')
count, bins, ignored = plt.hist(M_1b, 30, normed=True,histtype='step')
count, bins, ignored = plt.hist(M_1c, 30, normed=True,histtype='step')
count, bins, ignored = plt.hist(M_2b, 30, normed=True,histtype='step')
count, bins, ignored = plt.hist(M_2l, 30, normed=True,histtype='step')
count, bins, ignored = plt.hist(M_2n,30, normed=True,histtype='step')
count, bins, ignored = plt.hist(M_2p,30,normed=True,histtype='step')
plt.plot(M,y_1a,M,y_1b,M,y_1c,M,y_2b,M,y_2l,M,y_2n,M,y_2p)
"""

#For calculating the luminosity distance, formula in notes. Taking Plank values for O_m, O_l, and H_0
def integrand(x):
    return 1.0/(0.308*(1+x)**(3.0)+0.692*(1+x)**(6.0))**(0.5)

#z=1.0
c=3e5
H_0=67.8



###############
#This section is for plotting just the distribution with a 'shift' of all the supernovae.

z=1.0
I = quad(integrand, 0, z)
d_L = ((1.0+z)*c)/H_0*I[0]


sn_mean_test_1a = M_1a + 5.0*np.log10(d_L)+25.0
sn_mean_test_1b = M_1b + 5.0*np.log10(d_L)+25.0
sn_mean_test_1c = M_1c + 5.0*np.log10(d_L)+25.0
sn_mean_test_2b = M_2b + 5.0*np.log10(d_L)+25.0
sn_mean_test_2l = M_2l + 5.0*np.log10(d_L)+25.0
sn_mean_test_2n = M_2n + 5.0*np.log10(d_L)+25.0
sn_mean_test_2p = M_2p + 5.0*np.log10(d_L)+25.0
"""
#Plotting all
count, bins, ignored = plt.hist(sn_mean_test_1a,10,normed=True,histtype='step')
count, bins, ignored = plt.hist(sn_mean_test_1b,10,normed=True,histtype='step')
count, bins, ignored = plt.hist(sn_mean_test_1c,10,normed=True,histtype='step')
count, bins, ignored = plt.hist(sn_mean_test_2b,10,normed=True,histtype='step')
count, bins, ignored = plt.hist(sn_mean_test_2l,10,normed=True,histtype='step')
count, bins, ignored = plt.hist(sn_mean_test_2n,10,normed=True,histtype='step')
count, bins, ignored = plt.hist(sn_mean_test_2p,10,normed=True,histtype='step')
count, bins, ignored = plt.hist(M_1a,10,normed=True,histtype='step')
count, bins, ignored = plt.hist(M_1b,10,normed=True,histtype='step')
count, bins, ignored = plt.hist(M_1c,10,normed=True,histtype='step')
count, bins, ignored = plt.hist(M_2b,10,normed=True,histtype='step')
count, bins, ignored = plt.hist(M_2l,10,normed=True,histtype='step')
count, bins, ignored = plt.hist(M_2n,10,normed=True,histtype='step')
count, bins, ignored = plt.hist(M_2p,10,normed=True,histtype='step')
plt.show()
"""
#Proposal plot
count, bins, ignored = plt.hist(sn_mean_test_1a,20,normed=True,range=(20,24.5),linewidth=1.5,linestyle='dashdot',histtype='step',label='SNe Ia')
count, bins, ignored = plt.hist(sn_mean_test_2n,20,normed=True,range=(20,24.5),linewidth=1.35,linestyle='solid',histtype='step',label='SNe IIn')
count, bins, ignored = plt.hist(sn_mean_test_2p,20,normed=True,range=(20,24.5),linewidth=1.5,linestyle='dashed',histtype='step',label='SNe IIP')
plt.axvline(24.5, color='k', linestyle='dashed', linewidth=2)
#plt.axvline(27.5, color='r', linestyle='dashed', linewidth=2)
plt.xlabel('Apparent magnitude (mag)')
plt.ylabel('Frequency (normalized)')
plt.text(20.0, 0.5, '-- LSST single')
plt.text(20.0, 0.45, 'visit limit')
#plt.text(25.45,0.6,'LSST co-added',color='r')
#plt.text(25.45,0.55,'depth limit',color='r')
plt.text(20.0,0.575,'z=1')
plt.title('Magnitude cut for z=1, LSST single visit')
plt.legend()
plt.show()

###############


#I'm using the random exponential distribution for the redshifts
expzgrid=np.random.exponential(0.35,600)

#This is a pairing with M and z, uniform distribution
i=0
Mz_1a=np.zeros(shape=(600,2))
for i in range(len(M_1a)):
    Mz_1a[i]=[M_1a[i],expzgrid[i]]

#Calling d_L as an array
d_L=np.zeros(shape=(600,))
for i in range(len(Mz_1a)):
    d_L[i] = ((1.0+Mz_1a[i,1])*c)/H_0*(quad(integrand, 0, Mz_1a[i,1])[0])

#Apparent magnitude distribution (with ALL SN, all z)
mdist_1a = M_1a + 5.0*np.log10(d_L)+25.0
#count, bins, ignored = plt.hist(mdist_1a,10,normed=True,histtype='step')
#plt.title('Exponential z')
#plt.show()

#Creating pairing of m and z
mdistz_1a=np.zeros(shape=(600,2))
for i in range(len(M_1a)):
    mdistz_1a[i]=[mdist_1a[i],expzgrid[i]]

#Distance modulus
mu=mdist_1a-M_1a

#Plotting the Absolute Magnitude vs distance modulus, from Richardson paper. Includes reversed axis
plt.plot(mu,M_1a,'k.')
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])
plt.xlabel(r'$\mu$')
plt.ylabel('Absolute Magnitude')
plt.title('Exponential z distribution')
plt.show()

#Plotting z vs mu
plt.plot(expzgrid,mu,'k.')
plt.xlabel('z')
plt.ylabel(r'$\mu$')
plt.title('Exponential z distribution')
plt.show()

#Plotting z vs m
plt.plot(expzgrid,mdist_1a,'k.')
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])
plt.ylabel('Apparent magnitude m')
plt.xlabel('z')
plt.show()

#Plotting mu vs m
plt.plot(mu,mdist_1a,'k.')
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])
plt.xlabel('mu')
plt.ylabel('Apparent magnitude m')
plt.title('Exponential z distribution')
plt.show()

#####
#This is using a different distribution for z, to try and have more smaller redshifts
"""
#I'm using the random non-centered F distribution for the redshifts
zgrid=np.random.noncentral_f(2.0,1000.0,0.01,600)

#This is a pairing with M and z, uniform distribution
i=0
Mz_1a=np.zeros(shape=(600,2))
for i in range(len(M_1a)):
    Mz_1a[i]=[M_1a[i],zgrid[i]]

#Calling d_L as an array
d_L=np.zeros(shape=(600,))
for i in range(len(Mz_1a)):
    d_L[i] = ((1.0+Mz_1a[i,1])*c)/H_0*(quad(integrand, 0, Mz_1a[i,1])[0])

#Apparent magnitude distribution (with ALL SN, all z)
mdist_1a = M_1a + 5.0*np.log10(d_L)+25.0
#count, bins, ignored = plt.hist(mdist_1a,10,normed=True,histtype='step')
#plt.title('Non-center F z')
#plt.show()

#Creating pairing of m and z
mdistz_1a=np.zeros(shape=(600,2))
for i in range(len(M_1a)):
    mdistz_1a[i]=[mdist_1a[i],zgrid[i]]

#Distance modulus
mu=mdist_1a-M_1a

#Plotting the Absolute Magnitude vs distance modulus, from Richardson paper. Includes reversed axis
plt.plot(mu,M_1a,'k.')
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])
plt.xlabel(r'$\mu$')
plt.ylabel('Absolute Magnitude')
plt.title('Non-center F z distribution')
plt.show()

#Plotting z vs mu
plt.plot(zgrid,mu,'k.')
plt.xlabel('z')
plt.ylabel(r'$\mu$')
plt.title('Non-center F z distribution')
plt.show()


#Plotting mu vs m
plt.plot(mu,mdist_1a,'k.')
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])
plt.xlabel('mu')
plt.ylabel('m')
plt.title('Non-center F z distribution')
plt.show()
"""
