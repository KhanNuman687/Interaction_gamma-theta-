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

#def vfr_an(kx,ky,R,t,tp):
    return ( R**2*(2*t*np.sin(kx) + 4*tp*np.cos(ky)*np.sin(kx)), R**2*(2*t*np.sin(ky) + 4*tp*np.cos(kx)*np.sin(ky)) )

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
    kpkpx = np.mod(kx + kpx, 2 * np.pi)
    kpkpy = np.mod(ky + kpy, 2 * np.pi)
    kmkpx = np.mod(kx - kpx, 2 * np.pi)
    kmkpy = np.mod(ky - kpy, 2 * np.pi)
    #print 'kpkpx', kpkpx,'kpkpy', kpkpy, 'kmkpx', kmkpx, 'kmkpy', kmkpy
    #print 'kpkpx', kpkpx,'kpkpy', kpkpy, 'kmkpx', kmkpx, 'kmkpy', kmkpy
    return U+0.5*U**2*chispq_interp((kmkpx,kmkpy))-0.5*U**2*chichq_interp((kmkpx,kmkpy))+U**2*chispq_interp((kpkpx,kpkpy))


#def compute_Vsame_sp(kx, ky, kpx, kpy, U, chispq_interp, chichq_interp):
    # Calculate the difference in momentum for susceptibility evaluation
    kmkpx = kx - kpx
    kmkpy = ky - kpy

    # Wrap kmkpx and kmkpy within the Brillouin zone (0 to 2π)
    if kmkpx >= 2 * np.pi:
        kmkpx -= 2 * np.pi * (kmkpx // (2 * np.pi))
    elif kmkpx < 0.:
        kmkpx += 2 * np.pi * (-kmkpx // (2 * np.pi) + 1)

    if kmkpy >= 2 * np.pi:
        kmkpy -= 2 * np.pi * (kmkpy // (2 * np.pi))
    elif kmkpy < 0.:
        kmkpy += 2 * np.pi * (-kmkpy // (2 * np.pi) + 1)

    # Compute the interaction potential for same-spin scattering
   # gamma_same_sp = -0.5 * U**2 * chispq_interp(kmkpx, kmkpy) - 0.5 * U**2 * chichq_interp(kmkpx, kmkpy)

    return gamma_same_sp


def compute_VRPA(kx, ky, kpx, kpy, U, chispq_interp, chichq_interp):
    return compute_Vopp(kx, ky, kpx, kpy,U,chispq_interp,chichq_interp) \
        + compute_Vopp(-kx, -ky, kpx, kpy,U,chispq_interp,chichq_interp)
    #return compute_Vopp(kx, ky, kpx, kpy,U,chispq_interp,chichq_interp) \
    #     - compute_Vopp(-kx, -ky, kpx, kpy,U,chispq_interp,chichq_interp)

def compute_Vch(kx, ky, kpx, kpy, R, t, tp, invMchq_interp):
    kpkpx = np.mod(kx + kpx, 2 * np.pi)
    kpkpy = np.mod(ky + kpy, 2 * np.pi)
    kmkpx = np.mod(kx - kpx, 2 * np.pi)
    kmkpy = np.mod(ky - kpy, 2 * np.pi)
    #print 'kpkpx', kpkpx,'kpkpy', kpkpy, 'kmkpx', kmkpx, 'kmkpy', kmkpy
    ekpekp = ek(kx,ky,t,tp) + ek(kpx,kpy,t,tp)
    return -0.5*((R*ekpekp)**2*invMchq_interp[0,0](kmkpx,kmkpy)+ 2*R*ekpekp*invMchq_interp[0,1](kmkpx,kmkpy) + invMchq_interp[1,1](kmkpx,kmkpy))

def compute_Vsp(kx, ky, kpx, kpy, R, t, tp, invMspq_interp):
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
    ekpekp = ek(kx,ky,t,tp) + ek(kpx,kpy,t,tp)
    return +1.5*((R*ekpekp)**2*invMspq_interp[0,0](kmkpx,kmkpy)+ 2*R*ekpekp*invMspq_interp[0,1](kmkpx,kmkpy) + invMspq_interp[1,1](kmkpx,kmkpy))

def compute_Mgap_RPA(kx_fs,ky_fs,chispq_interp,chichq_interp,U,t,tp):
    Mgap = np.zeros((len(kx_fs),len(ky_fs)))
    
    for i in range(len(kx_fs)):
        print ('i=', i)
        for j in range(len(kx_fs)):
            #print 'i=', i, 'j=', j
            kx = kx_fs[i]
            ky = ky_fs[i]
            kpx = kx_fs[j]
            kpy = ky_fs[j]
            tmp = compute_VRPA(kx, ky, kpx, kpy, U, chispq_interp, chichq_interp)
            vf = vf_an(kpx,kpy,t,tp)
            dk = np.sqrt((kpx - kx_fs[j-1])**2+(kpy-ky_fs[j-1])**2)
            Mgap[i,j] = -tmp*dk/(np.sqrt(vf[0]**2+vf[1]**2)*(2*(2*np.pi)**2))
    
    return Mgap

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
    return -avg2/(avg1**2)*0.5

   # return -avg2/avg1



t = 1.0
tp = -0.35
eta = 0.000000001

T = 0.015

Nkx = 512
Nky = 512
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
#reorganize

def reorder_fermi_surface(kx_fs, ky_fs):
    """
    Reorganize the Fermi surface points by following the nearest-neighbor path.
    """
    # Combine kx and ky into a list of 2D points
    points = np.array(list(zip(kx_fs, ky_fs)))

    # Start with the first point
    reordered_points = [points[0]]
    remaining_points = list(points[1:])

    # Greedily select the nearest neighbor to maintain continuity
    while remaining_points:
        last_point = reordered_points[-1]
        # Find the nearest neighbor to the last added point
        nearest_index = np.argmin(np.linalg.norm(remaining_points - last_point, axis=1))
        reordered_points.append(remaining_points.pop(nearest_index))

    # Separate the reordered points back into kx and ky arrays
    reordered_points = np.array(reordered_points)
    kx_reordered = reordered_points[:, 0]
    ky_reordered = reordered_points[:, 1]

    return kx_reordered, ky_reordered

# Example usage with your Fermi surface data
kx_fs_n0p8, ky_fs_n0p8 = reorder_fermi_surface(kx_fs, ky_fs)


# Plot to verify the reorganization
plt.figure(figsize=(6, 6))
plt.plot(kx_fs_n0p8, ky_fs_n0p8, '-', ms=3)
plt.show()


#quit()

omRs = np.array([0.0])

Nqx = 128
Nqy = 128
qxs = np.arange(0,2*np.pi+2*np.pi/Nqx,2*np.pi/Nqx)
qys = np.arange(0,2*np.pi+2*np.pi/Nqx,2*np.pi/Nqy)

data = h5py.File('n0.8/rpa_chis.h5','r')
qxs = data['qxs'][...]
qys = data['qys'][...]
Nqx = len(qxs)
Nqy = len(qys)

U = 1.8

chi0q = data['U%.2f/chi0q'%(U)][...]
chispq = data['U%.2f/chispq'%(U)][...]
chichq = data['U%.2f/chichq'%(U)][...]
chi0q_interp = RegularGridInterpolator((qxs, qys), chi0q, bounds_error=True, fill_value=None)
chispq_interp = RegularGridInterpolator((qxs, qys), chispq, bounds_error=True, fill_value=None)
chichq_interp = RegularGridInterpolator((qxs, qys), chichq, bounds_error=True, fill_value=None)


Mgap = compute_Mgap_RPA(kx_fs_n0p8,ky_fs_n0p8,chispq_interp,chichq_interp,U,t,tp)

#evals, evecs = eigh(Mgap)
w_n0p8, vr_n0p8 = eig(Mgap)
ind_n0p8 = range(len(w_n0p8))
ind_n0p8 = sorted(ind_n0p8, key=lambda i: w_n0p8[i])

#print w

#for i in range(len(ind_n0p8)):
#    print i, w_n0p8[ind_n0p8[i]]#, vr_n0p8[:,ind_n0p8[i]]
print (w_n0p8[ind_n0p8[-1]])
print (w_n0p8[ind_n0p8[-2]])
print (w_n0p8[ind_n0p8[-3]])
print (w_n0p8[ind_n0p8[-4]])

data.close()

###########################################################
# nfill=0.4
###########################################################

# find mu
nfill = 0.55
sols = root( find_mu, 0.1, args=(eks, nfill, T))
mu = sols.x[0]
print ('nfill=', compute_n(eks, mu, T))

# find Fermi surface
Nkx = 128#256
Nky = 128#256
kxs = np.arange(-np.pi,np.pi+2*np.pi/Nkx,2*np.pi/Nkx)
kys = np.arange(-np.pi,np.pi+2*np.pi/Nky,2*np.pi/Nky)
#kxs = np.arange(0.0,2*np.pi,2*np.pi/Nkx)
#kys = np.arange(0.0,2*np.pi,2*np.pi/Nky)

# find Fermi surface and perform average
Nkx = 252#256
Nky = 252#256
kxs = np.arange(-np.pi,np.pi+2*np.pi/Nkx,2*np.pi/Nkx)
kys = np.arange(-np.pi,np.pi+2*np.pi/Nky,2*np.pi/Nky)
#kxs = np.arange(0.0,2*np.pi,2*np.pi/Nkx)
#kys = np.arange(0.0,2*np.pi,2*np.pi/Nky)

# find Fermisurface
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
kx_fs_n0p3 = kx_fs[::2] + kx_fs[-1::-2]
ky_fs_n0p3 = ky_fs[::2] + ky_fs[-1::-2]

print( len(kx_fs))

plt.figure(figsize=(6,6))
plt.plot(kx_fs_n0p3,ky_fs_n0p3,'-',ms =3)
#plt.xlim(-np.pi,np.pi)
#plt.ylim(-np.pi,np.pi)
plt.show()

#quit()

omRs = np.array([0.0])

Nqx = 128
Nqy = 128
qxs = np.arange(0,2*np.pi+2*np.pi/Nqx,2*np.pi/Nqx)
qys = np.arange(0,2*np.pi+2*np.pi/Nqx,2*np.pi/Nqy)

data = h5py.File('n0.3/rpa_chis.h5','r')
qxs = data['qxs'][...]
qys = data['qys'][...]
Nqx = len(qxs)
Nqy = len(qys)

U = 1.5

chi0q = data['U%.2f/chi0q'%(U)][...]
chispq = data['U%.2f/chispq'%(U)][...]
chichq = data['U%.2f/chichq'%(U)][...]

chi0q_interp = RegularGridInterpolator((qxs, qys), chi0q, bounds_error=True, fill_value=None)
chispq_interp = RegularGridInterpolator((qxs, qys), chispq, bounds_error=True, fill_value=None)
chichq_interp = RegularGridInterpolator((qxs, qys), chichq, bounds_error=True, fill_value=None)

avg_interaction = V_FS_avg(kx_fs_n0p3, ky_fs_n0p3, g_g, U, t, tp, chispq_interp, chichq_interp)
print(f"Average interaction strength (g-wave): {avg_interaction}")
Mgap = compute_Mgap_RPA(kx_fs_n0p3,ky_fs_n0p3,chispq_interp,chichq_interp,U,t,tp)

#evals, evecs = eigh(Mgap)
w_n0p3, vr_n0p3 = eig(Mgap)
ind_n0p3 = range(len(w_n0p3))
ind_n0p3 = sorted(ind_n0p3, key=lambda i: w_n0p3[i])

#print w

#for i in range(len(ind_n0p3)):
#    print i, w_n0p3[ind_n0p3[i]]#, vr_n0p3[:,ind_n0p3[i]]
print (w_n0p3[ind_n0p3[-1]])
print (w_n0p3[ind_n0p3[-2]])
print (w_n0p3[ind_n0p3[-3]])
print (w_n0p3[ind_n0p3[-4]])



###########################################################
# plotting
###########################################################

fig, axs = plt.subplots(1, 2, figsize=(9, 4))

# Subplot 1
axs[0].set_title('(a) RPA $n=0.8$ $U=1.8$ $T=0.015$', size=15)

im1 = axs[0].scatter(kx_fs_n0p8, ky_fs_n0p8, 
                     c=vr_n0p8[:, ind_n0p8[-1]], cmap='jet')

axs[0].set_xlim(-np.pi, np.pi)
axs[0].set_ylim(-np.pi, np.pi)
axs[0].set_xlabel('$q_x$', size=15)
axs[0].set_ylabel('$q_y$', size=15)
axs[0].tick_params(axis='both', which='major', labelsize=15)

fig.colorbar(im1, ax=axs[0])

# Subplot 2
axs[1].set_title('(b) RPA $n=0.55$ $U=1.5$ $T=0.015$', size=15)

im2 = axs[1].scatter(kx_fs_n0p3, ky_fs_n0p3, 
                     c=vr_n0p3[:, ind_n0p3[-1]], cmap='jet')

axs[1].set_xlim(-np.pi, np.pi)
axs[1].set_ylim(-np.pi, np.pi)
axs[1].set_xlabel('$q_x$', size=15)
axs[1].set_ylabel('$q_y$', size=15)
axs[1].tick_params(axis='both', which='major', labelsize=15)

fig.colorbar(im2, ax=axs[1])

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('gap_func.png')
plt.show()

