import numpy as np
from numba import jit
import numba
from scipy.spatial import KDTree
from scipy.optimize import root
from scipy.interpolate import RegularGridInterpolator
from scipy import interpolate
from scipy.linalg import eig, inv
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
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
def compute_Vopp(kx, ky, kpx, kpy,U,chispq_interp,chichq_interp):
    kpkpx = kx + kpx
    kpkpy = ky + kpy
    kmkpx = kx - kpx
    kmkpy = ky - kpy
    #print 'kpkpx', kpkpx,'kpkpy', kpkpy, 'kmkpx', kmkpx, 'kmkpy', kmkpy
    if kpkpx >= 2*np.pi: kpkpx = kpkpx - 2*np.pi*(kpkpx//(2*np.pi))
    if kpkpx < 0.:       kpkpx = kpkpx + 2*np.pi*(-kpkpx//(2*np.pi)+1)
    if kpkpy >= 2*np.pi: kpkpy = kpkpy - 2*np.pi*(kpkpy//(2*np.pi))
    if kpkpy < 0.:       kpkpy = kpkpy + 2*np.pi*(-kpkpy//(2*np.pi)+1)
    if kmkpx >= 2*np.pi: kmkpx = kmkpx - 2*np.pi*(kmkpx//(2*np.pi))
    if kmkpx < 0.:       kmkpx = kmkpx + 2*np.pi*(-kmkpx//(2*np.pi)+1)
    if kmkpy >= 2*np.pi: kmkpy = kmkpy - 2*np.pi*(kmkpy//(2*np.pi))
    if kmkpy < 0.:       kmkpy = kmkpy + 2*np.pi*(-kmkpy//(2*np.pi)+1)
    #print 'kpkpx', kpkpx,'kpkpy', kpkpy, 'kmkpx', kmkpx, 'kmkpy', kmkpy
    return U+0.5*U**2*chispq_interp(kmkpx,kmkpy)-0.5*U**2*chichq_interp(kmkpx,kmkpy)+U**2*chispq_interp(kpkpx,kpkpy)

t = 1.0
tp = -0.35
eta = 0.000000001
U= 1.75
T = 0.015

Nkx = 256
Nky = 256
kxs = np.arange(-np.pi,np.pi,2*np.pi/Nkx)
kys = np.arange(-np.pi,np.pi,2*np.pi/Nky)

eks = []
ekpqs = []
for kx in kxs:
    for ky in kys:
        eks.append(ek(kx,ky,t,tp))

# find mu
nfill = 0.8
sols = root( find_mu, 0.1, args=(eks, nfill, T))
mu = sols.x[0]
print("Computed chemical potential (mu):", mu)
print ('nfill=', compute_n(eks, mu, T))

# find Fermi surface
Nkx = 128#256
Nky = 128#256
kxs = np.arange(-np.pi,np.pi+2*np.pi/Nkx,2*np.pi/Nkx)
kys = np.arange(-np.pi,np.pi+2*np.pi/Nky,2*np.pi/Nky)
#kxs = np.arange(0.0,2*np.pi,2*np.pi/Nkx)
#kys = np.arange(0.0,2*np.pi,2*np.pi/Nky)

# find Fermi surface and perform average

KX, KY = np.meshgrid(kxs, kys)
ek_values = ek(KX, KY, t, tp)

# Contour plot at the Fermi level (ek = mu)
plt.figure(figsize=(8, 8))
contour = plt.contour(KX, KY, ek_values, levels=[mu], colors='b')
plt.xlabel('$k_x$', fontsize=14)
plt.ylabel('$k_y$', fontsize=14)
plt.title("Fermi Surface at nfill=0.8")
plt.grid(True)
plt.show()
# Extract Fermi surface points from the contour
kx_fs = []
ky_fs = []

for path in contour.collections[0].get_paths():
    vertices = path.vertices
    kx_fs.extend(vertices[:, 0])
    ky_fs.extend(vertices[:, 1])

# Convert lists to arrays for further processing
kx_fs = np.array(kx_fs)
ky_fs = np.array(ky_fs)
plt.plot(kx_fs,ky_fs, '-o')
plt.show()
omRs = np.array([0.0])

Nqx = 128
Nqy = 129
qxs = np.arange(0,2*np.pi+2*np.pi/Nqx,2*np.pi/Nqx)
qys = np.arange(0,2*np.pi+2*np.pi/Nqx,2*np.pi/Nqy)

data = h5py.File('n0.8/rpa_chis.h5','r')
qxs = data['qxs'][...]
qys = data['qys'][...]
Nqx = len(qxs)
Nqy = len(qys)



chi0q = data['U%.2f/chi0q'%(U)][...]
chispq = data['U%.2f/chispq'%(U)][...]
chichq = data['U%.2f/chichq'%(U)][...]
chi0q_interp = interpolate.interp2d(qxs, qys, chi0q, kind='linear', bounds_error=True)
chichq_interp = interpolate.interp2d(qxs, qys, chichq, kind='linear', bounds_error=True)
chispq_interp = interpolate.interp2d(qxs, qys, chispq, kind='linear', bounds_error=True)

#k_radius_vals = np.sqrt(kx_fs**2 + ky_fs**2)
theta_vals = np.zeros(len(kx_fs))
for i in range(len(theta_vals)):
    tmp = np.arctan2(ky_fs[i], kx_fs[i]) 
    if tmp < 0:
        theta_vals[i] = tmp + 2 * np.pi
    else: 
        theta_vals[i] = tmp
print(theta_vals)
plt.plot(theta_vals,'o')
plt.show()
kpx = np.pi
kpy = 0.347
def compute_VRPA(theta_vals, kx_fs, ky_fs, kpx, kpy, U, chispq_interp, chichq_interp):
    gamma_thetas = []
    for i in range(len(theta_vals)):
        theta = theta_vals[i]
        kx = kx_fs[i]
        ky = ky_fs[i]

        print(theta,kx,ky)
        # If you need to use theta directly, you can do so here
        gamma_theta = (
            compute_Vopp(kx, ky, kpx, kpy, U, chispq_interp, chichq_interp)
            + compute_Vopp(-kx, -ky, kpx, kpy, U, chispq_interp, chichq_interp)
        )
        gamma_thetas.append(gamma_theta)
    return gamma_thetas

# Example usage:
# theta_vals, kx_fs, and ky_fs are arrays representing theta and points on the Fermi surface
gamma_theta = compute_VRPA(theta_vals, kx_fs, ky_fs, kpx, kpy, U, chispq_interp, chichq_interp)




plt.figure(figsize=(8, 6))

plt.plot(theta_vals, gamma_theta, marker='o', linestyle='--', markersize=1,color='red')
plt.xlabel("Theta (radians)")
plt.ylabel("Gamma (VRPA value)")
plt.ylim(-5, 55)
plt.title("Plot of Theta vs Gamma (VRPA values)")
plt.grid(True)
plt.show()
