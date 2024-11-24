import numpy as np
import matplotlib.pyplot as plt
import h5py

# Load RPA data from HDF5 file
data = h5py.File('n1.0/rpa_chis.h5', 'r')
qxs = data['qxs'][...]
qys = data['qys'][...]
Nqx = len(qxs)
Nqy = len(qys)


Nkx = 64
Nky = 64
k_points = []

# Part 1: (0, 0) to (pi, pi) (Diagonal path)
for i in range(Nkx):
    kx = (i / (Nkx - 1)) * np.pi
    ky = (i / (Nkx - 1)) * np.pi
    k_points.append((kx, ky))

# Part 2: (pi, pi) to (pi, 0)
for j in range(Nky):
    kx = np.pi
    ky = np.pi - (j / (Nky - 1)) * np.pi
    k_points.append((kx, ky))

# Part 3: (pi, 0) to (0, 0)
for k in range(Nkx):
    kx = np.pi - (k / (Nkx - 1)) * np.pi
    ky = 0
    k_points.append((kx, ky))

k_points = np.array(k_points)

# Output the result for verification
print("Generated k-points:")
print(k_points)


k_points = np.array(k_points)

U = 2.5 
chi0q = data[f'U{U:.2f}/chi0q'][...]
chispq = data[f'U{U:.2f}/chispq'][...]
chichq = data[f'U{U:.2f}/chichq'][...]

# Extract chi values along the k-path
chi_real = []
for kx, ky in k_points:
    # Find nearest qx and qy indices
    iqx = np.argmin(np.abs(qxs - kx))
    iqy = np.argmin(np.abs(qys - ky))
    # Extract real part of chi
    chi_real.append(np.real(chi0q[iqx, iqy]))

    # Plot chi_real along the k-path
    # Define x-axis positions for labels
# Define x-axis positions for symmetry points
x_ticks = [0, len(k_points) // 3, 2 * len(k_points) // 3, len(k_points) - 1]
x_labels = [r'$\Gamma$', r'$M$', r'$X$', r'$\Gamma$']

# Plot chi_real along the k-path
plt.figure(figsize=(8, 6))
plt.plot(np.array(chi_real)/2.0, label=r'Re[$\chi$]', color='b')
plt.xlabel('k-Path', fontsize=12)
plt.ylabel(r'Re[$\chi$]', fontsize=12)
plt.title(f'Real Part of Chi along k-Path (U={U:.2f})', fontsize=14)
plt.grid(True)

# Set custom x-axis ticks and labels
plt.xticks(x_ticks, x_labels, fontsize=12)

plt.legend()
plt.show()

# Close the data file
data.close()
