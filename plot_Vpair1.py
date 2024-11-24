import matplotlib.pyplot as plt
import numpy as np

# Load the RPA data
data = np.loadtxt('V_dx2y2_dxy_s_se_g_n0.8.dat').T

# Use genfromtxt with dtype=complex to handle complex numbers
data2 = np.genfromtxt('V_dx2y2_dxy_s_se_g_n0.3.dat', dtype=complex).T

plt.figure(figsize=(8,7))

# First subplot: RPA n = 0.8
plt.subplot(2, 1, 1)
plt.text(0.05, 0.32, '(a) RPA n=0.8 T=0.015', size=15)

plt.plot(data[0], data[1], 'b-', lw=2, label='$d_{x^2-y^2}$')
plt.plot(data[0], data[2], 'r-', lw=2, label='$d_{xy}$')
plt.plot(data[0], data[3], 'g-', lw=2, label='$s$')
# plt.plot(data[0], data[4], 'c-', lw=2, label='extended $s$')  # Commented out as per original code
plt.plot(data[0], data[5], 'm-', lw=2, label='$g$')

plt.xlim(0, 2.39)
plt.ylim(-0.3, 0.4)
plt.xticks(size=15)
plt.yticks(size=15)
plt.xlabel('$U$', size=15)
plt.ylabel('$-V_{pair}$', size=15)
plt.legend(loc='best', fontsize=12)

# Second subplot: RPA n = 0.3
plt.subplot(2, 1, 2)
plt.text(1.65, 0.0015, '(b) RPA n=0.3 T=0.015', size=15)

plt.plot(data2[0], data2[1].real, 'b-', lw=2, label='$d_{x^2-y^2}$')  # Use real part if necessary
plt.plot(data2[0], data2[2].real, 'r-', lw=2, label='$d_{xy}$')       # Use real part if necessary
plt.plot(data2[0], data2[3].real, 'g-', lw=2, label='$s$')             # Use real part if necessary
# plt.plot(data2[0], data2[4].real, 'c-', lw=2, label='extended $s$')  # Commented out as per original code
plt.plot(data2[0], data2[5].real, 'm-', lw=2, label='$g$')             # Use real part if necessary

plt.xlim(0, 6.8)
plt.ylim(-0.002, 0.002)
plt.xticks(size=15)
plt.yticks(size=15)
plt.xlabel('$U$', size=15)
plt.ylabel('$-V_{pair}$', size=15)
plt.legend(loc='best', fontsize=12)

plt.tight_layout()
plt.savefig('Vpair_RPA_only.png')
plt.show()

