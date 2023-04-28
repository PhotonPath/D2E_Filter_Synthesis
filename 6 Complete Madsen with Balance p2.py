"""
Code Data 2023/04/14
Author Mattia
"""
import Photonic_building_block as Pbb
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

    # Phi calculation: Equation 65 Madsen. Modified for balance
    lattice_n = int((n - 1) / 2)

    if n % 2 == 1: # Balance
        print("B")
        numerator = b * np.sqrt(k) * np.sqrt(k) + a * np.sqrt(1 - k) * np.sqrt(k)
        denominator = b * np.sqrt(1 - k) * np.sqrt(1 - k) - a * np.sqrt(k) * np.sqrt(1 - k)
        r = numerator / denominator
        phi = -np.angle(r)
        print(phi)
    else:          # Unbalance
        print("U")
        a_tilda = np.sqrt(1 - k) * a + np.sqrt(k) * b
        phi = -np.angle(a_tilda[lattice_n]) + np.angle(prev_b[lattice_n - 1]) + np.arctan(0.2 / (k - 0.5) / 4)
        if k > 0.5:
            phi -= np.pi
        # phi = -np.angle(a_tilda[lattice_n]) + np.angle(prev_b[lattice_n - 1]) - np.arctan(4 / PHI * (k - 0.5)) - np.pi/2
        print(phi)

    phi = PHI

    # An-1 calculation: Equation 61 Madsen. Modified for balance
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
            phis[-i - 1], a[-i - 2], b[-i - 2] = calculate_prev_layer_balance(Lattice_Order, a[-i - 1], b[-i - 1], k, -i-1)
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


# Input data
Lattice_Order = 5
n_points = 500
c = 299.792458 # um * THz
dL = 30        # um
wavelength_neff = 1.55 # um
neff = 1.46    # @ wavelength_neff
ng = 1.52      # @ wavelength_neff
FSR = c/dL/ng # THz
frequencies = np.linspace(190, 190+FSR, n_points)
wavelengths = c/frequencies
coupler_value = 0.52
losses_parameter = {
    'A': 0,
    'B': 0,
    'C': 0,
    'D': 0,
    'wl1': 1.5,
    'wl2': 1.6
} # No losses
input_field = np.array([[1, 0], [0,  0]])
#Phis = np.random.uniform(0, np.pi/2, Lattice_Order * 2 + 1)
PHI = 0.5
Phis = np.ones(Lattice_Order * 2 + 1) * PHI
Phis[8] = 0.2
# As, Bs calculation
As, Bs = calculate_transfer_function_balance(Lattice_Order, coupler_value, Phis)

"""
# Entering from top (Xin = 1, Yin = 0)
bar = np.zeros(n_points, dtype=np.complex128)
cross = np.zeros(n_points, dtype=np.complex128)
x = np.linspace(0, 1, n_points)
for fourier_coefficient in range(Lattice_Order+1):
    bar = bar + np.exp(fourier_coefficient * 2j * np.pi * x) * As[-1][fourier_coefficient]
    cross = cross - np.exp(fourier_coefficient * 2j * np.pi * x) * Bs[-1][fourier_coefficient] * np.exp(1j)
plt.figure(10)
plt.plot(frequencies, np.abs(bar)**2, label="MADSEN Lattice Bar")
# plt.plot(x, np.abs(cross)**2, label="MADSEN Lattice Cross")
"""

# # LATTICE GENERATION
Couplers = []
Balance_traits = []
Unbalance_traits = []
for idc in range(2*Lattice_Order+2):
    Couplers += [Pbb.Coupler([1.5, 1.6], (coupler_value, coupler_value))]
for idb in range(Lattice_Order+1):
    Balance_traits += [Pbb.Balanced_propagation(neff, ng, wavelength_neff, losses_parameter, 0)]
for idu in range(Lattice_Order):
    Unbalance_traits += [Pbb.Unbalanced_propagation(neff, ng, wavelength_neff, losses_parameter, 0, dL)]
coupling_losses = 0
Lattice = Pbb.Chip_structure([Couplers, Balance_traits, Unbalance_traits], ['C', 'B', 'C', 'U'] * Lattice_Order + ['C', 'B', 'C'], coupling_losses)
heater_order = ['B', 'U'] * Lattice_Order + ['B']


# WORKING AREA ################
phi11, As11, Bs11 = calculate_prev_layer_balance(Lattice_Order, As[-1], Bs[-1], coupler_value, -1)
phi10, As10, Bs10 = calculate_prev_layer_balance(Lattice_Order, As11, Bs11, coupler_value, -2)
phi9, As9, Bs9 = calculate_prev_layer_balance(Lattice_Order, As10, Bs10, coupler_value, -3)
phi8, As8, Bs8 = calculate_prev_layer_balance(Lattice_Order, As9, Bs9, coupler_value, -4)
phi7, As7, Bs7 = calculate_prev_layer_balance(Lattice_Order, As8, Bs8, coupler_value, -5)

bar_original = np.zeros(n_points, dtype=np.complex128)
bar_reconstructed = np.zeros(n_points, dtype=np.complex128)
x = np.linspace(0, 1, n_points)
for fourier_coefficient in range(Lattice_Order+1):
    bar_original = bar_original + np.exp(fourier_coefficient * 2j * np.pi * x) * As[-3][fourier_coefficient]
    bar_reconstructed = bar_reconstructed + np.exp(fourier_coefficient * 2j * np.pi * x) * As10[fourier_coefficient]
# plt.figure(10)
# plt.plot(frequencies, bar_original, label="Original")
# plt.plot(frequencies, bar_reconstructed, label="Reconstructed")

# FINISH WORKING AREA #########
"""
# MADSEN ALGORITHM
Target_As = As[-1]
Target_Bs = find_valid_b_balance(Lattice_Order, Target_As, to_plot=False)

Phis_estimate, As_estimate, Bs_estimate = reverse_transfer_function_balance(Target_As, Target_Bs, coupler_value)

# NEFF COMPENSATION
Neff_shift_phis = np.zeros(len(Phis_estimate))
Neff_shift_phis[1::2] = np.ones(Lattice_Order) * 2 * np.pi * ((neff - ng) * dL / wavelength_neff + frequencies[0] / FSR)
# The unbalance shifts must compensate the selected band (frequencies[0] / FSR) because the filter consider as regular bands the
# multiples of the FSR, while we want the filter to be set in a given region of the spectrum
# The unbalance must also compensate the difference between neff and ng (neff - ng) * dL / wavelength_neff.

# OUTPUT CALCULATION
Lattice.set_heaters(-Phis_estimate-Neff_shift_phis, heater_order)
S = Lattice.calculate_S_matrix(wavelengths)
output_power = Pbb.calculate_outputs(input_field, S, dB=False)
plt.figure(10)
plt.plot(frequencies, output_power[:, 0], label="PBB Output Bar")
# plt.plot(x, output_power[:, 1], label="PBB Output Cross")
plt.xlabel("Frequencies")
plt.ylabel("Filter Linear Transfer Function")
plt.grid()
plt.legend()

"""
