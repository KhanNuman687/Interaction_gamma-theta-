import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

def tight_binding_dispersion(kx, ky, t, t_prime, mu):
    """
    Calculate the tight-binding dispersion relation for a 2D square lattice with next-nearest-neighbor hopping,
    including the chemical potential.
    
    Parameters:
    kx, ky : ndarray
        Arrays of kx and ky values.
    t : float
        Nearest-neighbor hopping parameter.
    t_prime : float
        Next-nearest-neighbor hopping parameter.
    mu : float
        Chemical potential.
    
    Returns:
    ndarray
        Dispersion relation xi_k.
    """
    return -2 * t * (np.cos(kx) + np.cos(ky)) - 4 * t_prime * np.cos(kx) * np.cos(ky) - mu

def compute_chemical_potential(t_prime, n_fill, Nk=400):
    """
    Compute the chemical potential for a given filling fraction in the 2D tight-binding model.
    
    Parameters:
    t_prime : float
        Next-nearest-neighbor hopping parameter.
    n_fill : float
        Desired filling fraction (electron density).
    Nk : int
        Number of grid points in each dimension.
    
    Returns:
    float
        Chemical potential mu.
    """
    kx = np.linspace(-np.pi, np.pi, Nk)
    ky = np.linspace(-np.pi, np.pi, Nk)
    KX, KY = np.meshgrid(kx, ky)
    xi_k = tight_binding_dispersion(KX, KY, 1.0, t_prime, 0)  # Start with mu=0 for initial guess
    
    def filling_fraction(mu):
        """
        Calculate the filling fraction for a given chemical potential mu.
        """
        xi_k = tight_binding_dispersion(KX, KY, 1.0, t_prime, mu)
        density = 2 * np.mean(xi_k < 0)
        return density - n_fill

    # Find the chemical potential mu for the given filling fraction
    mu = brentq(filling_fraction, -10, 10)
    
    return mu

def plot_fermi_surface(t, t_prime, n_fill, Nk=400):
    """
    Plot the Fermi surface for the 2D tight-binding model at a specified filling fraction.
    
    Parameters:
    t : float
        Nearest-neighbor hopping parameter.
    t_prime : float
        Next-nearest-neighbor hopping parameter.
    n_fill : float
        Desired filling fraction (electron density).
    Nk : int
        Number of grid points in each dimension.
    """
    kx = np.linspace(-np.pi, np.pi, Nk)
    ky = np.linspace(-np.pi, np.pi, Nk)
    KX, KY = np.meshgrid(kx, ky)
    mu = compute_chemical_potential(t_prime, n_fill, Nk)
    xi_k = tight_binding_dispersion(KX, KY, t, t_prime, mu)

    # Normalize kx and ky by pi for plotting
    KX_pi = KX / np.pi
    KY_pi = KY / np.pi

    plt.figure(figsize=(8, 8))
    plt.contour(KX_pi, KY_pi, xi_k, levels=[0], colors='r')
    plt.xlabel('$k_x / \pi$')
    plt.ylabel('$k_y / \pi$')
    plt.title(f'Fermi Surface for n_fill={n_fill} with t\'={t_prime}')
    plt.grid(True)
    plt.show()

# Parameters
t = 1.0        # Nearest-neighbor hopping parameter
t_prime = -0.35 # Next-nearest-neighbor hopping parameter
U = 1.75

# Fillings to visualize
fillings = [0.05, 0.35, 0.50, 0.65, 0.80, 1.05, 1.2, 1.35, 1.55, 1.90]

# Plot Fermi surfaces for different fillings
for n_fill in fillings:
    plot_fermi_surface(t, t_prime, n_fill)
