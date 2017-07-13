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
"""
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
"""
###############


#I'm using the random noncentral f distribution for the redshifts
zgrid1a=np.random.noncentral_f(5.0,1e100,0.001,600)
zgrid1b=np.random.noncentral_f(5.0,1e100,0.001,600)
zgrid1c=np.random.noncentral_f(5.0,1e100,0.001,600)
zgrid2b=np.random.noncentral_f(5.0,1e100,0.001,600)
zgrid2l=np.random.noncentral_f(5.0,1e100,0.001,600)
zgrid2n=np.random.noncentral_f(5.0,1e100,0.001,600)
zgrid2p=np.random.noncentral_f(5.0,1e100,0.001,600)

#This is a pairing with M and z, uniform distribution
i=0
Mz_1a=np.zeros(shape=(600,2))
for i in range(len(M_1a)):
    Mz_1a[i]=[M_1a[i],zgrid1a[i]]

Mz_1b=np.zeros(shape=(600,2))
for i in range(len(M_1b)):
    Mz_1b[i]=[M_1b[i],zgrid1b[i]]
    
Mz_1c=np.zeros(shape=(600,2))
for i in range(len(M_1c)):
    Mz_1c[i]=[M_1c[i],zgrid1c[i]]
    
Mz_2b=np.zeros(shape=(600,2))
for i in range(len(M_2b)):
    Mz_2b[i]=[M_2b[i],zgrid2b[i]]
    
Mz_2l=np.zeros(shape=(600,2))
for i in range(len(M_2l)):
    Mz_2l[i]=[M_2l[i],zgrid2l[i]]
    
Mz_2n=np.zeros(shape=(600,2))
for i in range(len(M_2n)):
    Mz_2n[i]=[M_2n[i],zgrid2n[i]]
    
Mz_2p=np.zeros(shape=(600,2))
for i in range(len(M_2p)):
    Mz_2p[i]=[M_2p[i],zgrid2p[i]]

#Calling d_L as an array
d_L=np.zeros(shape=(600,))
for i in range(len(Mz_1a)):
    d_L[i] = ((1.0+Mz_1a[i,1])*c)/H_0*(quad(integrand, 0, Mz_1a[i,1])[0])

d_L1b=np.zeros(shape=(600,))
for i in range(len(Mz_1b)):
    d_L1b[i] = ((1.0+Mz_1b[i,1])*c)/H_0*(quad(integrand, 0, Mz_1b[i,1])[0])

d_L1c=np.zeros(shape=(600,))
for i in range(len(Mz_1c)):
    d_L1c[i] = ((1.0+Mz_1c[i,1])*c)/H_0*(quad(integrand, 0, Mz_1c[i,1])[0])
    
d_L2b=np.zeros(shape=(600,))
for i in range(len(Mz_2b)):
    d_L2b[i] = ((1.0+Mz_2b[i,1])*c)/H_0*(quad(integrand, 0, Mz_2b[i,1])[0])
    
d_L2l=np.zeros(shape=(600,))
for i in range(len(Mz_2l)):
    d_L2l[i] = ((1.0+Mz_2l[i,1])*c)/H_0*(quad(integrand, 0, Mz_2l[i,1])[0])
    
d_L2n=np.zeros(shape=(600,))
for i in range(len(Mz_2n)):
    d_L2n[i] = ((1.0+Mz_2n[i,1])*c)/H_0*(quad(integrand, 0, Mz_2n[i,1])[0])
    
d_L2p=np.zeros(shape=(600,))
for i in range(len(Mz_2p)):
    d_L2p[i] = ((1.0+Mz_2p[i,1])*c)/H_0*(quad(integrand, 0, Mz_2p[i,1])[0])

#Apparent magnitude distribution (with ALL SN, all z)
mdist_1a = M_1a + 5.0*np.log10(d_L)+25.0
mdist_1b = M_1b + 5.0*np.log10(d_L1b)+25.0
mdist_1c = M_1c + 5.0*np.log10(d_L1c)+25.0
mdist_2b = M_2b + 5.0*np.log10(d_L2b)+25.0
mdist_2l = M_2l + 5.0*np.log10(d_L2l)+25.0
mdist_2n = M_2n + 5.0*np.log10(d_L2n)+25.0
mdist_2p = M_2p + 5.0*np.log10(d_L2p)+25.0

#Apparent magnitude distributions with LSST 'single visit' cuts and 'co-added visit' cuts
mdist_1a_cut = mdist_1a[(0.0 < mdist_1a) & (mdist_1a < 24.5)]
mdist_1b_cut = mdist_1b[(0.0 < mdist_1b) & (mdist_1b < 24.5)]
mdist_1c_cut = mdist_1c[(0.0 < mdist_1c) & (mdist_1c < 24.5)]
mdist_2b_cut = mdist_2b[(0.0 < mdist_2b) & (mdist_2b < 24.5)]
mdist_2l_cut = mdist_2l[(0.0 < mdist_2l) & (mdist_2l < 24.5)]
mdist_2n_cut = mdist_2n[(0.0 < mdist_2n) & (mdist_2n < 24.5)]
mdist_2p_cut = mdist_2p[(0.0 < mdist_2p) & (mdist_2p < 24.5)]

mdist_1a_cut_add = mdist_1a[(0.0 < mdist_1a) & (mdist_1a < 27.5)]
mdist_1b_cut_add = mdist_1b[(0.0 < mdist_1b) & (mdist_1b < 27.5)]
mdist_1c_cut_add = mdist_1c[(0.0 < mdist_1c) & (mdist_1c < 27.5)]
mdist_2b_cut_add = mdist_2b[(0.0 < mdist_2b) & (mdist_2b < 27.5)]
mdist_2l_cut_add = mdist_2l[(0.0 < mdist_2l) & (mdist_2l < 27.5)]
mdist_2n_cut_add = mdist_2n[(0.0 < mdist_2n) & (mdist_2n < 27.5)]
mdist_2p_cut_add = mdist_2p[(0.0 < mdist_2p) & (mdist_2p < 27.5)]

#Finding the ordering of the cut apparent magnitudes, so that the proper redshift and absolute magnitudes can be paired
w1a=[np.where((mdist_1a==i))[0][0] for i in mdist_1a_cut]
w1b=[np.where((mdist_1b==i))[0][0] for i in mdist_1b_cut]
w1c=[np.where((mdist_1c==i))[0][0] for i in mdist_1c_cut]
w2b=[np.where((mdist_2b==i))[0][0] for i in mdist_2b_cut]
w2l=[np.where((mdist_2l==i))[0][0] for i in mdist_2l_cut]
w2n=[np.where((mdist_2n==i))[0][0] for i in mdist_2n_cut]
w2p=[np.where((mdist_2p==i))[0][0] for i in mdist_2p_cut]

w1a_add=[np.where((mdist_1a==i))[0][0] for i in mdist_1a_cut_add]
w1b_add=[np.where((mdist_1b==i))[0][0] for i in mdist_1b_cut_add]
w1c_add=[np.where((mdist_1c==i))[0][0] for i in mdist_1c_cut_add]
w2b_add=[np.where((mdist_2b==i))[0][0] for i in mdist_2b_cut_add]
w2l_add=[np.where((mdist_2l==i))[0][0] for i in mdist_2l_cut_add]
w2n_add=[np.where((mdist_2n==i))[0][0] for i in mdist_2n_cut_add]
w2p_add=[np.where((mdist_2p==i))[0][0] for i in mdist_2p_cut_add]


#Distance modulus
mu1a=mdist_1a_cut-M_1a[w1a]
mu1b=mdist_1b_cut-M_1b[w1b]
mu1c=mdist_1c_cut-M_1c[w1c]
mu2b=mdist_2b_cut-M_2b[w2b]
mu2l=mdist_2l_cut-M_2l[w2l]
mu2n=mdist_2n_cut-M_2n[w2n]
mu2p=mdist_2p_cut-M_2p[w2p]

mu1a_add=mdist_1a_cut_add-M_1a[w1a_add]
mu1b_add=mdist_1b_cut_add-M_1b[w1b_add]
mu1c_add=mdist_1c_cut_add-M_1c[w1c_add]
mu2b_add=mdist_2b_cut_add-M_2b[w2b_add]
mu2l_add=mdist_2l_cut_add-M_2l[w2l_add]
mu2n_add=mdist_2n_cut_add-M_2n[w2n_add]
mu2p_add=mdist_2p_cut_add-M_2p[w2p_add]

mu1a_og=mdist_1a-M_1a
mu1b_og=mdist_1b-M_1b
mu1c_og=mdist_1c-M_1c
mu2b_og=mdist_2b-M_2b
mu2l_og=mdist_2l-M_2l
mu2n_og=mdist_2n-M_2n
mu2p_og=mdist_2p-M_2p

#Plotting the Absolute Magnitude vs distance modulus, from Richardson paper. Includes reversed axis
plt.plot(mu1a_og,M_1a,'.',label='Ia all')
plt.plot(mu1a_add,M_1a[w1a_add],'.',label='Ia co-added visit')
plt.plot(mu1a,M_1a[w1a],'.',label='Ia single visit')
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])
plt.xlabel(r'$\mu$')
plt.ylabel('Absolute magnitude M')
plt.legend()
plt.show()

#A panel plot of all the other types of SNe
fig=plt.figure(figsize=(20,10))
ax1=fig.add_subplot(2,3,1)
ax1.plot(mu1b_og,M_1b,'.',label='Ib all')
ax1.plot(mu1b_add,M_1b[w1b_add],'.',label='Ib co-added')
ax1.plot(mu1b,M_1b[w1b],'.',label='Ib single')
ax1.set_ylim(ax1.get_ylim()[::-1])
ax1.set_ylabel('Absolute magnitude M')
plt.legend(loc=2)

ax2=fig.add_subplot(2,3,2)
ax2.plot(mu1c_og,M_1c,'.',label='Ic all')
ax2.plot(mu1c_add,M_1c[w1c_add],'.',label='Ic co-added')
ax2.plot(mu1c,M_1c[w1c],'.',label='Ib single')
ax2.set_ylim(ax2.get_ylim()[::-1])
plt.legend(loc=2)

ax3=fig.add_subplot(2,3,3)
ax3.plot(mu2b_og,M_2b,'.',label='IIb all')
ax3.plot(mu2b_add,M_2b[w2b_add],'.',label='IIb co-added')
ax3.plot(mu2b,M_2b[w2b],'.',label='IIb single')
ax3.set_ylim(ax3.get_ylim()[::-1])
plt.legend(loc=2)

ax4=fig.add_subplot(2,3,4)
ax4.plot(mu2l_og,M_2l,'.',label='IIl all')
ax4.plot(mu2l_add,M_2l[w2l_add],'.',label='IIl co-added')
ax4.plot(mu2l,M_2l[w2l],'.',label='IIl single')
ax4.set_ylim(ax4.get_ylim()[::-1])
ax4.set_xlabel(r'$\mu$')
ax4.set_ylabel('Absolute magnitude M')
plt.legend(loc=2)

ax5=fig.add_subplot(2,3,5)
ax5.plot(mu2n_og,M_2n,'.',label='IIn all')
ax5.plot(mu2n_add,M_2n[w2n_add],'.',label='IIn co-added')
ax5.plot(mu2n,M_2n[w2n],'.',label='IIn single')
ax5.set_ylim(ax5.get_ylim()[::-1])
ax5.set_xlabel(r'$\mu$')
plt.legend(loc=2)

ax6=fig.add_subplot(2,3,6)
ax6.plot(mu2p_og,M_2p,'.',label='IIP all')
ax6.plot(mu2p_add,M_2p[w2p_add],'.',label='IIP co-added')
ax6.plot(mu2p,M_2p[w2p],'.',label='IIP single')
ax6.set_ylim(ax6.get_ylim()[::-1])
ax6.set_xlabel(r'$\mu$')
plt.legend(loc=2)

#plt.tight_layout()
plt.show()










#Different types of plots
"""
#Plotting z vs mu
plt.plot(zgrid1a,mu,'k.')
plt.xlabel('z')
plt.ylabel(r'$\mu$')
plt.title('Exponential z distribution')
plt.show()

#Plotting z vs m
plt.plot(zgrid1a,mdist_1a,'k.')
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
"""

