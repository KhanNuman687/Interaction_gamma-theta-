import numpy as np
from numba import jit
import numba
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import root
from scipy import interpolate
from scipy.linalg import eig
import matplotlib.pyplot as plt
import h5py

@jit(nopython=True)
def nf(x,T):
    if np.real(x)/T > 500:
        return 0.0
    elif np.real(x)/T < -500:
        return 1.0
    else:
        return 1./(np.exp(x/T)+1.)

@jit(nopython=True)
def ek(kx, ky, t, tp):
    return -2.*t*(np.cos(kx)+np.cos(ky))-4*tp*np.cos(kx)*np.cos(ky)

def vf_an(kx,ky,t,tp):
    return (2*t*np.sin(kx) + 4*tp*np.cos(ky)*np.sin(kx), 2*t*np.sin(ky) + 4*tp*np.cos(kx)*np.sin(ky))

@jit(nopython=True)
def compute_n(eks,mu,T):
    n = 0.0
    for i in range(len(eks)):
        n = n + nf( eks[i]-mu ,T)
    return 2*n/len(eks)

@jit(nopython=True)
def find_mu(x, *args):
    # find chemical potential
    eks, nfill, T = args
    n = compute_n(eks,x[0],T)
    return n-nfill

def compute_Vopp(kx, ky, kpx, kpy, U, chispq_interp, chichq_interp):
    kpkpx = kx + kpx
    kpkpy = ky + kpy
    kmkpx = kx - kpx
    kmkpy = ky - kpy

    # Ensure values stay within 0 and 2*pi
    kpkpx = np.mod(kpkpx, 2*np.pi)
    kpkpy = np.mod(kpkpy, 2*np.pi)
    kmkpx = np.mod(kmkpx, 2*np.pi)
    kmkpy = np.mod(kmkpy, 2*np.pi)

    # Interpolate using RegularGridInterpolator, pass the coordinates as tuples
    return U + 0.5 * U**2 * chispq_interp((kmkpx, kmkpy)) - 0.5 * U**2 * chichq_interp((kmkpx, kmkpy)) \
           + U**2 * chispq_interp((kpkpx, kpkpy))

def compute_VRPA(kx, ky, kpx, kpy, U, chispq_interp, chichq_interp):
    return compute_Vopp(kx, ky, kpx, kpy,U,chispq_interp,chichq_interp) \
         + compute_Vopp(-kx, -ky, kpx, kpy,U,chispq_interp,chichq_interp)
    #return compute_Vopp(kx, ky, kpx, kpy,U,chispq_interp,chichq_interp) \
    #     - compute_Vopp(-kx, -ky, kpx, kpy,U,chispq_interp,chichq_interp)

def g_se(kx,ky):
    return np.cos(kx) + np.cos(ky)

def g_s(kx,ky):
    return 1.0

def g_dx2_y2(kx,ky):
    return np.cos(kx) - np.cos(ky)

def g_dxy(kx,ky):
    return np.sin(kx)*np.sin(ky)

def g_g(kx,ky):
    return (np.cos(kx) - np.cos(ky))*np.sin(kx)*np.sin(ky)

def V_FS_avg(kx_fs, ky_fs, gk_func, U, t, tp, chispq_interp, chichq_interp):
    avg1 = 0
    avg2 = 0
    for i in range(len(kx_fs)):
        #print 'i=', i
        for j in range(len(kx_fs)):
            #print 'i=', i, 'j=', j
            kx = kx_fs[i]
            ky = ky_fs[i]
            kpx = kx_fs[j]
            kpy = ky_fs[j]
            tmp = compute_VRPA(kx, ky, kpx, kpy, U, chispq_interp, chichq_interp)
            dk = np.sqrt((kx - kx_fs[i-1])**2+(ky-ky_fs[i-1])**2)
            dkp = np.sqrt((kpx - kx_fs[j-1])**2+(kpy-ky_fs[j-1])**2)
            vfk = vf_an(kx,ky,t,tp)
            vfkp = vf_an(kpx,kpy,t,tp)
            vfka = np.sqrt(vfk[0]**2+vfk[1]**2)
            vfkpa = np.sqrt(vfkp[0]**2+vfkp[1]**2)
            gk = gk_func(kx,ky)
            gkp = gk_func(kpx,kpy)
            #print tmp, gk, gkp, dk, dkp, vfka, vfkpa

            avg2 += gk*tmp*gkp*dk*dkp/(vfka*vfkpa)
            avg1 += dk* gk * gk / vfka
    
    #print avg2, avg1
   # return -avg2/avg1
    return -avg2/(avg1**2)*0.5


t = 1.0
tp = -0.35
eta = 0.000000001

nfill = 0.3
T = 0.015

Nkx = 128#256
Nky = 128#256
kxs = np.arange(-np.pi,np.pi,2*np.pi/Nkx)
kys = np.arange(-np.pi,np.pi,2*np.pi/Nky)

eks = []
ekpqs = []
for kx in kxs:
    for ky in kys:
        eks.append(ek(kx,ky,t,tp))
# find mu
sols = root( find_mu, 0.1, args=(eks, nfill, T))
mu = sols.x[0]
print ('nfill=', compute_n(eks, mu, T))

# find Fermi surface
Nkx = 256#89
Nky = 256#89
kxs = np.arange(-np.pi,np.pi+2*np.pi/Nkx,2*np.pi/Nkx)
kys = np.arange(-np.pi,np.pi+2*np.pi/Nky,2*np.pi/Nky)
#kxs = np.arange(0.0,2*np.pi,2*np.pi/Nkx)
#kys = np.arange(0.0,2*np.pi,2*np.pi/Nky)

# find Fermi surface and perform average
kx_fs = []
ky_fs = []
kx_fs_1bz = []
ky_fs_1bz = []

for ikx,kx in enumerate(kxs):
    for iky,ky in enumerate(kys):
        if iky == 0:
            enew = np.abs(ek(kx,ky,t,tp) - mu)
            if np.abs(ek(kx,ky,t,tp) - mu) < 1e-10:
                kx_fs.append(kx)
                ky_fs.append(ky)
                #kx_fs.append()
                #ky_fs.append()
        else:
            enew = ek(kx,ky,t,tp) - mu
            if eold*enew <0:
                kx_fs.append(kx)
                ky_fs.append(ky)
                #kx_fs_1bz.append(kx)
                #ky_fs_1bz.append(ky)
        eold = enew

plt.figure(figsize=(6,6))
plt.plot(kx_fs,ky_fs,'o',ms =3)
#plt.xlim(-np.pi,np.pi)
#plt.ylim(-np.pi,np.pi)
plt.show()

#reorganize
kx_fs = kx_fs[::2] + kx_fs[-1::-2]
ky_fs = ky_fs[::2] + ky_fs[-1::-2]

print (len(kx_fs))

plt.figure(figsize=(6,6))
plt.plot(kx_fs,ky_fs,'-',ms =3)
#plt.xlim(-np.pi,np.pi)
#plt.ylim(-np.pi,np.pi)
plt.show()

#quit()

omRs = np.array([0.0])

Nqx = 128
Nqy = 128
qxs = np.arange(0,2*np.pi+2*np.pi/Nqx,2*np.pi/Nqx)
qys = np.arange(0,2*np.pi+2*np.pi/Nqx,2*np.pi/Nqy)

U_list = U_list = np.arange(0,10.1,0.1)


data = h5py.File('n%g/rpa_chis.h5'%(nfill),'r')
qxs = data['qxs'][...]
qys = data['qys'][...]
Nqx = len(qxs)
Nqy = len(qys) 
f = open('V_dx2y2_dxy_s_se_g_n%g.dat'%(nfill),'w')
for U in U_list:
    print ('---------------------- U=', U, '--------------------------')

    chi0q = data['U%.2f/chi0q'%(U)][...]
    chispq = data['U%.2f/chispq'%(U)][...]
    chichq = data['U%.2f/chichq'%(U)][...]

    chi0q_interp = RegularGridInterpolator((qxs, qys), chi0q, bounds_error=False, fill_value=None)
    chichq_interp = RegularGridInterpolator((qxs, qys), chichq, bounds_error=False, fill_value=None)
    chispq_interp = RegularGridInterpolator((qxs, qys), chispq, bounds_error=False, fill_value=None)

    print ('compute V_dx2y2')
    V_dx2_y2 = V_FS_avg(kx_fs, ky_fs, g_dx2_y2, U, t, tp, chispq_interp, chichq_interp)
    print ('compute V_dxy')
    V_dxy = V_FS_avg(kx_fs, ky_fs, g_dxy, U, t, tp, chispq_interp, chichq_interp)
    print ('compute V_s')
    V_s = V_FS_avg(kx_fs, ky_fs, g_s, U, t, tp, chispq_interp, chichq_interp)
    print ('compute V_se')
    V_se = V_FS_avg(kx_fs, ky_fs, g_se, U, t, tp, chispq_interp, chichq_interp)
    print ('compute V_g')
    V_g = V_FS_avg(kx_fs, ky_fs, g_g, U, t, tp, chispq_interp, chichq_interp)

    print ('U, V_dx2_y2, V_dxy, V_s, V_se, V_g=', U, V_dx2_y2, V_dxy, V_s, V_se, V_g)

    print(U, V_dx2_y2, V_dxy, V_s, V_se, V_g, file=f)


f.close()
data.close()


