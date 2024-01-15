"""
Code D2EOptimization 2023/04/14
Author Mattia
"""
import Photonic_building_block as Pbb
import scipy.constants as const
import matplotlib.pyplot as plt
import numpy as np

# C B C U C B C U C B C ....
#  n0  n1  n2  n3  n4

# Polynomial are define as P(z) = p0 + p1*z^-1 + p2*z^-2....
def calculate_next_layer_balance(lattice_order, a, b, k, phi, n):
    """
    Uses the value of the previous An Bn polynomial to calculate An+1 Bn+1
    :param lattice_order: Order of the lattice. Uses to create vectors of correct size
    :param a: polynomial A of order n in matrix formula
    :param b: polynomial B of order n in matrix formula
    :param k: Power ratio of the n coupler
    :param phi: Extra phases of the n unbalance
    :param n: Index of the layer considered starting from 0
    :return: Polynomial A and B of order n+1 in matrix formula
    """
    coefficient_order = lattice_order + 1
    new_a = np.zeros(coefficient_order, dtype=np.complex128)
    new_b = np.zeros(coefficient_order, dtype=np.complex128)

    lattice_n = int((n+1)/2)
    # Modified Equation 50 and 51 Madsen
    if n % 2 == 0: # Balance
        for i in range(min(lattice_n, lattice_order) + 1):
            new_a[i] = np.sqrt(1 - k) * np.exp(-phi * 1j) * a[i] - np.sqrt(k) * b[i]
            new_b[i] = np.sqrt(k) * np.exp(-phi * 1j) * a[i] + np.sqrt(1 - k) * b[i]
    else:          # Unbalance
        for i in range(lattice_n + 1):
            new_a[i] = np.sqrt(1 - k) * np.exp(-phi * 1j) * a[i - 1] - np.sqrt(k) * b[i]
            new_b[i] = np.sqrt(k) * np.exp(-phi * 1j) * a[i - 1] + np.sqrt(1 - k) * b[i]
    return new_a, new_b

def calculate_transfer_function_balance(lattice_order, k, phis):
    """
    Calculate the current transfer function of the filter, given certain ks and phis
    :param lattice_order: Order of the lattice. Uses to create vectors of correct sizes and to iterate the right number of times
    :param k: Power ratio of all the couplers of the lattice.
    :param phis: Extra phases of the n unbalance of the lattice.
    :return: Polynomial A and B that describes the whole lattice.
    """

    coefficient_order = lattice_order+1
    stadium_order = lattice_order*2+2
    a = np.zeros((stadium_order, coefficient_order), dtype=np.complex128)  # Different vector A(z). Each A(z) = A[0] + A[1]*z^-1 + A[2]*z^-2 ...
    b = np.zeros((stadium_order, coefficient_order), dtype=np.complex128)  # Different vector B(z). Each B(z) = B[0] + B[1]*z^-1 + B[2]*z^-2 ...
    a[0][0], b[0][0] = np.sqrt(1 - k), np.sqrt(k)
    for i in range(stadium_order-1):
        a[i+1], b[i+1] = calculate_next_layer_balance(lattice_order, a[i], b[i], k, phis[i], i)
    return a, b

def calculate_prev_layer_balance(lattice_order, a, b, k, n):
    """
    Calculate the value of An-1 Bn-1 kn-1 and phi-1 to obtain the wanted An Bn polynomial
    :param lattice_order: Order of the lattice.
    :param a: An polynomial of order n+1 in matrix formula
    :param b: Bn polynomial of order n+1 in matrix formula
    :param k: Coupler of the lattice from design
    :param n: index of the layer considered
    :return: Power ratio, Phase and Polynomial A and B of order n-1 in matrix formula
    """
    # Bn-1 calculation: Equation 62 Madsen. Same for both balance and unbalance
    coefficient_order = lattice_order + 1
    prev_b = np.zeros(coefficient_order, dtype=np.complex128)
    for i in range(coefficient_order):
        prev_b[i] = -a[i] * np.sqrt(k) + b[i] * np.sqrt(1-k)

    # Phi calculation: Equation 65 Madsen. Same for both balance and unbalance
    lattice_n = int((n - 1) / 2)
    a_tilda = np.sqrt(1 - k) * a + np.sqrt(k) * b
    if n % 2 == 1: # Balance
        phi = -np.angle(a_tilda[lattice_n]) + np.angle(prev_b[lattice_n])
    else:          # Unbalance
        phi = -np.angle(a_tilda[lattice_n]) + np.angle(prev_b[lattice_n-1])-np.pi/2

    # An-1 calculation: Modified Equation 61 Madsen
    prev_a = np.zeros(coefficient_order, dtype=np.complex128)
    if n % 2 == 1: # Balance
        for i in range(coefficient_order):
            prev_a[i] = (a[i] * np.sqrt(1 - k) + b[i] * np.sqrt(k)) * np.exp(phi * 1j)
    if n % 2 == 0: # Unbalance
        for i in range(lattice_order):
            prev_a[i] = (a[i + 1] * np.sqrt(1 - k) + b[i + 1] * np.sqrt(k)) * np.exp(phi * 1j)

    return phi, prev_a, prev_b

def reverse_transfer_function_balance(a_n, b_n, k):
    """
    Calculate the list of all couplers and extra phases needed for obtaining An Bn polynomials at the end of the lattice
    :param a_n: Polynomial A of order n in matrix formula
    :param b_n: Polynomial B of order n in matrix formula
    :param k: Coupler of the lattice from design
    :return: Power ratio, Phase and Polynomial A and B that describes the whole lattice.
    """
    coefficient_order = len(a_n)
    stadium_order = coefficient_order * 2
    phis = np.zeros(stadium_order-1)
    a = np.zeros((stadium_order, coefficient_order), dtype=np.complex128)  # Different vector A(z). Each A(z) = A[0] + A[1]*z^-1 + A[2]*z^-2 ...
    b = np.zeros((stadium_order, coefficient_order), dtype=np.complex128)  # Different vector B(z). Each B(z) = B[0] + B[1]*z^-1 + B[2]*z^-2 ...
    a[-1] = a_n
    b[-1] = b_n
    for i in range(stadium_order):
        if i != stadium_order - 1:
            phis[-i - 1], a[-i - 2], b[-i - 2] = calculate_prev_layer_balance(filter_order, a[-i - 1], b[-i - 1], k, -i-1)
    return phis, a, b

def find_valid_b_balance(lattice_order, a, gamma=1, to_plot=False):
    """
    Return a valid cross output given a wanted bar output (a)
    :param lattice_order: Order of the lattice.
    :param a: Polynomial A of order n in matrix formula
    :param gamma: losses (not yet implemented)
    :param to_plot: Boolean to decide if you want to see the roots of the polynomial B
    :return: Return a valid cross output given a wanted bar output (a)
    """
    # Total Phase of last polynomial
    phi = -np.angle(a[-1])

    # B*Br Product: Equation 66 Madsen
    a_reverse = np.conjugate(a[::-1]) * np.exp(-1j * phi)
    product_b_br = -np.polymul(a, a_reverse)
    product_b_br[lattice_order] += np.exp(-1j * phi)

    # Find the B Br roots
    b_br_roots = np.roots(product_b_br[::-1]) # Remember I define the polynomial opposite to numpy

    # Couple the B Br roots (depending on the gamma circle)
    b_br_roots_coupled_index = []
    for root in b_br_roots:
        b_roots_close = np.isclose(np.abs(b_br_roots)*np.abs(root)*np.exp(1j*(np.angle(b_br_roots)-np.angle(root))), gamma)
        b_roots_close_index = np.argmax(b_roots_close)
        b_br_roots_coupled_index.append(b_roots_close_index)

    # Chose the B root with minimal magnitude
    b_roots_index = []
    for r, root in enumerate(b_br_roots):
        if (r not in b_roots_index) and (b_br_roots_coupled_index[r] not in b_roots_index):
            coupled_root = b_br_roots[b_br_roots_coupled_index[r]]

            # Decide which root has minimal magnitude and add it to the final roots
            if np.abs(root) > np.abs(coupled_root):
                b_roots_index.append(r)
            else:
                b_roots_index.append(b_br_roots_coupled_index[r])
    b_roots_index = np.array(b_roots_index, dtype=int)
    b_roots = b_br_roots[b_roots_index]

    #Take polynomial by roots
    b = np.poly(b_roots)
    b = b[::-1] # Remember I define the polynomial opposite to numpy

    # Normalize the output
    alpha = np.sqrt(-(a[0] * a[-1]) / (b[0] * b[-1]))
    b *= alpha

    # Plot roots if wanted
    if to_plot:
        plt.figure(1)
        # Total B B_reverse roots
        plot_roots(product_b_br, color='blue', width=3, unit_circle=True)

        # Only B roots
        plot_roots(b, color='red', width=2)

    return b * np.exp(1j*np.pi/2)

def plot_roots(a, color="red", width=2, unit_circle=False):
    """
    :param a: polynomial you want to plot.
    :param color: color of the points.
    :param width: radius of the points.
    :param unit_circle: boolean to decide if you want to plot also the unit circle
    """
    for root in np.roots(a[::-1]):
        plt.scatter(np.real(root), np.imag(root), color=color, linewidths=width)

    # Unit Circle
    if unit_circle:
        t_circle = np.linspace(0, np.pi * 2, 100)
        x_circle = np.cos(t_circle)
        y_circle = np.sin(t_circle)
        plt.plot(x_circle, y_circle, color='black')
    plt.grid()


# INPUTS
c = const.c/1000000
FSR = 5 # THz
n_points = 1001
frequencies = np.linspace(190, 190+FSR, n_points)
wavelengths = c/frequencies
input_field = [np.ones(n_points), np.zeros(n_points)]

waveguide_args_balance = {
    'neff0': 1.5,
    'ng': 1.5,
    'wavelength0': 1.55,    # um
    'wavelengths': wavelengths
}
waveguide_args_unbalance = waveguide_args_balance.copy()
waveguide_args_unbalance['dL'] = c/FSR/waveguide_args_balance['ng']

# Coupler arguments
coupler_value = np.pi/4
coupler_args = {
    'k0': coupler_value,
    'k1': 0,
    'k2': 0,
    'wavelength0': 1.55,
    'wavelengths': wavelengths}

coupling_loss_args = {
    'coupling_losses': 0,
    'wavelengths': wavelengths}

filter_order = 5

# BUILDING BLOCKS
structures = {0: Pbb.WaveguideFacet(**coupling_loss_args)}

for idx in range(filter_order):
    structures[4*idx + 1] = Pbb.Coupler(**coupler_args)
    structures[4*idx + 2] = Pbb.DoubleWaveguide(**waveguide_args_balance)
    structures[4*idx + 3] = Pbb.Coupler(**coupler_args)
    structures[4*idx + 4] = Pbb.DoubleWaveguide(**waveguide_args_unbalance)
structures[4*filter_order + 1] = Pbb.Coupler(**coupler_args)
structures[4*filter_order + 2] = Pbb.DoubleWaveguide(**waveguide_args_balance)
structures[4*filter_order + 3] = Pbb.Coupler(**coupler_args)
structures[4*filter_order + 4] = Pbb.WaveguideFacet(**coupling_loss_args)

# As, Bs calculation
Phis = np.random.uniform(0, np.pi/2, filter_order * 2 + 1)
As, Bs = calculate_transfer_function_balance(filter_order, np.sin(coupler_value)**2, Phis)

# Entering from top (Xin = 1, Yin = 0)
bar = np.zeros(n_points, dtype=np.complex128)
cross = np.zeros(n_points, dtype=np.complex128)
x = np.linspace(0, 1, n_points)
for fourier_coefficient in range(filter_order+1):
    bar = bar + np.exp(fourier_coefficient * 2j * np.pi * x) * As[-1][fourier_coefficient]
    cross = cross - np.exp(fourier_coefficient * 2j * np.pi * x) * Bs[-1][fourier_coefficient] * np.exp(1j)
plt.figure(10)
plt.plot(x, np.abs(bar)**2, label="MADSEN Lattice Bar")
# plt.plot(x, np.abs(cross)**2, label="MADSEN Lattice Cross")

Lattice = Pbb.ChipStructure(structures)
Lattice.calculate_internal_transfer_function()

#################################################
################# B-1 KNOWN #####################
#################################################

# # MADSEN ALGORITHM
# Target_As = As[-1]
# Target_Bs = Bs[-1]
# Phis_estimate, As_estimate, Bs_estimate = reverse_transfer_function_balance(Target_As, Target_Bs, np.sin(coupler_value)**2)
#
# phis_estimate_dict = {}
# for idp, phi_estimate in enumerate(Phis_estimate):
#     phis_estimate_dict[idp*2+2] = -phi_estimate
#
# # OUTPUT CALCULATION
# Lattice.set_heaters(phis_estimate_dict)
# Lattice.calculate_transfer_function()
# output_field = Lattice.calculate_output(input_field)
# output_power = np.abs(output_field[0])**2
#
# plt.figure(10)
# plt.plot(x, output_power, label="PBB Output Bar")
# plt.xlabel("2*pi*Frequencies/FSR")
# plt.ylabel("Filter Linear Transfer Function")
# plt.legend()
# plt.show()
# plt.grid()

#################################################
################ B-1 UNKNOWN ####################
#################################################

# MADSEN ALGORITHM
Target_As = As[-1]
Target_Bs = find_valid_b_balance(filter_order, Target_As, to_plot=True)

Phis_estimate, As_estimate, Bs_estimate = reverse_transfer_function_balance(Target_As, Target_Bs, np.sin(coupler_value)**2)

phis_estimate_dict = {}
for idp, phi_estimate in enumerate(Phis_estimate):
    phis_estimate_dict[idp*2+2] = -phi_estimate

# OUTPUT CALCULATION
Lattice.set_heaters(phis_estimate_dict)
Lattice.calculate_transfer_function()
output_field = Lattice.calculate_output(input_field)
output_power = np.abs(output_field[0])**2
plt.figure(10)
plt.plot(x, output_power, label="PBB Output Bar")
plt.xlabel("2*pi*Frequencies/FSR")
plt.ylabel("Filter Linear Transfer Function")
plt.grid()
plt.legend()
plt.show()