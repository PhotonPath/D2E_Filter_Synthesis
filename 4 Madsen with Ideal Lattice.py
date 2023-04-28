"""
Code Data 2023/04/13
Author Mattia
"""

import Photonic_building_block as Pbb
import matplotlib.pyplot as plt
import numpy as np


def calculate_next_layer(lattice_order, a, b, k, phi, n):
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

    # Equation 50 and 51 Madsen
    if n == 0:
        new_a[0] = np.sqrt(1-k)*np.exp(-phi*1j)
        new_b[0] = np.sqrt(k)*np.exp(-phi*1j)
    else:
        for i in range(n+1):
            new_a[i] = np.sqrt(1-k)*np.exp(-phi*1j)*a[i-1] - np.sqrt(k)*b[i]
            new_b[i] = np.sqrt(k)*np.exp(-phi*1j)*a[i-1]   + np.sqrt(1-k)*b[i]
    return new_a, new_b

def calculate_transfer_function(lattice_order, ks, phis):
    """
    Calculate the current transfer function of the filter, given certain ks and phis
    :param lattice_order: Order of the lattice. Uses to create vectors of correct sizes and to iterate the right number of times
    :param ks: Power ratio of all the couplers of the lattice.
    :param phis: Extra phases of the n unbalance of the lattice.
    :return: Polynomial A and B that describes the whole lattice.
    """

    coefficient_order = lattice_order+1
    a = np.zeros((coefficient_order, coefficient_order), dtype=np.complex128)  # Different vector A(z). Each A(z) = A[0] + A[1]*z^-1 + A[2]*z^-2 ...
    b = np.zeros((coefficient_order, coefficient_order), dtype=np.complex128)  # Different vector B(z). Each B(z) = B[0] + B[1]*z^-1 + B[2]*z^-2 ...
    a[0], b[0] = calculate_next_layer(lattice_order, [], [], ks[0], phis[0], 0)
    for i in range(lattice_order):
        a[i+1], b[i+1] = calculate_next_layer(lattice_order, a[i], b[i], ks[i+1], phis[i+1], i+1)
    return a, b

def calculate_prev_layer(lattice_order, a, b, n):
    """
    Calculate the value of An-1 Bn-1 kn-1 and phi-1 to obtain the wanted An Bn polynomial
    :param lattice_order: Order of the lattice.
    :param a: An polynomial of order n+1 in matrix formula
    :param b: Bn polynomial of order n+1 in matrix formula
    :param n: index of the layer considered
    :return: Power ratio, Phase and Polynomial A and B of order n-1 in matrix formula
    """
    # K Calculation: Equation 60 Madsen
    if a[n] != 0:
        r = np.abs(b[n]/a[n])**2
        k = r/(1+r)
    else:
        k = 1

    # Bn-1 calculation: Equation 62 Madsen
    coefficient_order = lattice_order + 1
    prev_b = np.zeros(coefficient_order, dtype=np.complex128)
    for i in range(lattice_order):
        prev_b[i] = -a[i] * np.sqrt(k) + b[i] * np.sqrt(1-k)

    # Phi calculation: Equation 65 Madsen
    a_tilda = np.sqrt(1 - k) * a[n] + np.sqrt(k) * b[n]
    if -n < lattice_order + 1:
        phi = -np.angle(a_tilda) + np.angle(prev_b[n-1])
    else:
        phi = -np.angle(a_tilda)

    # An-1 calculation: Equation 61 Madsen
    prev_a = np.zeros(coefficient_order, dtype=np.complex128)
    for i in range(lattice_order):
        prev_a[i] = (a[i + 1] * np.sqrt(1 - k) + b[i + 1] * np.sqrt(k)) * np.exp(phi * 1j)

    return k, phi, prev_a, prev_b

def reverse_transfer_function(a_n, b_n):
    """
    Calculate the list of all couplers and extra phases needed for obtaining An Bn polynomials at the end of the lattice
    :param a_n: Polynomial A of order n in matrix formula
    :param b_n: Polynomial B of order n in matrix formula
    :return: Power ratio, Phase and Polynomial A and B that describes the whole lattice.
    """
    coefficient_order = len(a_n)
    ks = np.zeros(coefficient_order)
    phis = np.zeros(coefficient_order)
    a = np.zeros((coefficient_order, coefficient_order), dtype=np.complex128)  # Different vector A(z). Each A(z) = A[0] + A[1]*z^-1 + A[2]*z^-2 ...
    b = np.zeros((coefficient_order, coefficient_order), dtype=np.complex128)  # Different vector B(z). Each B(z) = B[0] + B[1]*z^-1 + B[2]*z^-2 ...
    a[-1] = a_n
    b[-1] = b_n
    for i in range(coefficient_order):
        if i != coefficient_order - 1:
            ks[-i - 1], phis[-i - 1], a[-i - 2], b[-i - 2] = calculate_prev_layer(coefficient_order - 1, a[-i - 1], b[-i - 1], -i - 1)
        else:
            ks[-i - 1], phis[-i - 1], _, _ = calculate_prev_layer(coefficient_order - 1, a[-i - 1], b[-i - 1], -i - 1)
    return ks, phis, a, b

def find_valid_b(lattice_order, a, gamma=1, to_plot=False):
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

    return b

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

def find_balance_phase(k1, k2, wanted_k):
    wanted_k = 1 - wanted_k
    c1 = np.sqrt(1 - k1)
    c2 = np.sqrt(1 - k2)
    s1 = np.sqrt(k1)
    s2 = np.sqrt(k2)
    argument = (wanted_k - c1*c1*c2*c2 - s1*s1*s2*s2) / (-2*c1*c2*s1*s2)
    return np.arccos(argument)

#################################################
######### First case - Variable Coupler #########
#################################################

# Input data
# Lattice_Order = 5
# c = 299.792458 #um THz
# FSR = 5 # THz
# n_points = 300
# frequencies = np.linspace(190, 190+FSR, n_points)
# wavelengths = c/frequencies
# coupler_value = 0.5
# neff = 1.5
# ng = 1.5
# losses_parameter = {
#     'A': 0,
#     'B': 0,
#     'C': 0,
#     'D': 0,
#     'wl1': 1.5,
#     'wl2': 1.6
# } # No losses
# dL = c/FSR/ng
# input_field = np.array([[1, 0], [0,  0]])
#
# # Wanted Output
# First_want_FFT_Output_ks = np.array([ 0.10970111-8.19491733e-02j, -0.03607001+7.88144078e-02j,
#         0.01581732+2.23046660e-01j,  0.12081527+1.88158632e-01j,
#        -0.55673725-3.04146946e-01j,  0.13693064+2.08166817e-17j], dtype=np.complex128)
# All_want_FFT_Output_ks = np.zeros(n_points, dtype=np.complex128)
# All_want_FFT_Output_ks[:Lattice_Order+1] = First_want_FFT_Output_ks
# t = np.linspace(0, 1, n_points)
# want_Output = np.zeros(n_points, dtype=np.complex128)
# for fks in range(0, Lattice_Order+1):
#     want_Output = want_Output + np.exp(fks * 2j * np.pi * t) * First_want_FFT_Output_ks[fks]
# plt.figure(2)
# plt.plot(frequencies, np.abs(want_Output)**2, label="Wanted Output")
#
# # MADSEN ALGORITHM
# Target_As = First_want_FFT_Output_ks
# Target_Bs = find_valid_b(Lattice_Order, Target_As, to_plot=False)
# Ks_estimate, Phis_estimate, As_estimate, Bs_estimate = reverse_transfer_function(Target_As, Target_Bs)
#
# # LATTICE GENERATION
# Couplers = []
# Unbalance_traits = []
# for idc in range(Lattice_Order+1):
#     Couplers += [Pbb.Coupler([1.5, 1.6], [Ks_estimate[idc], Ks_estimate[idc]])]
# for idu in range(Lattice_Order):
#     Unbalance_traits += [Pbb.Unbalanced_propagation(neff, ng, 1.55, losses_parameter, 0, dL)]
# coupling_losses = 0
# Lattice = Pbb.Chip_structure([Couplers, [], Unbalance_traits], ['C', 'U'] * Lattice_Order + ['C'], coupling_losses)
# heater_order = ['U'] * Lattice_Order
#
# # OUTPUT CALCULATION
# heaters = np.zeros(Lattice_Order)
# Lattice.set_heaters(-Phis_estimate[1:], heater_order)
# S = Lattice.calculate_S_matrix(wavelengths)
# output_power = Pbb.calculate_outputs(input_field, S, dB=False)[:, 0]
#
# # PLOTTING POWER OUTPUT
# plt.plot(frequencies, output_power, label="Reconstructed Output")
# plt.grid()
# plt.legend()

#################################################
########### Second case - Fix Coupler ###########
#################################################
# Input data
Lattice_Order = 5
c = 299.792458 #um THz
FSR = 5 # THz
n_points = 300
frequencies = np.linspace(190, 190+FSR, n_points)
wavelengths = c/frequencies
coupler_value = 0.5
neff = 1.5
ng = 1.5
losses_parameter = {
    'A': 0,
    'B': 0,
    'C': 0,
    'D': 0,
    'wl1': 1.5,
    'wl2': 1.6
} # No losses
dL = c/FSR/ng
input_field = np.array([[1, 0], [0,  0]])

# WANTED OUTPUT
First_want_FFT_Output_ks = np.array([ 0.10970111-8.19491733e-02j, -0.03607001+7.88144078e-02j,
        0.01581732+2.23046660e-01j,  0.12081527+1.88158632e-01j,
       -0.55673725-3.04146946e-01j,  0.13693064+2.08166817e-17j], dtype=np.complex128)
All_want_FFT_Output_ks = np.zeros(n_points, dtype=np.complex128)
All_want_FFT_Output_ks[:Lattice_Order+1] = First_want_FFT_Output_ks
t = np.linspace(0, 1, n_points)
want_Output = np.zeros(n_points, dtype=np.complex128)
for fks in range(0, Lattice_Order+1):
    want_Output = want_Output + np.exp(fks * 2j * np.pi * t) * First_want_FFT_Output_ks[fks]
plt.figure(2)
plt.plot(frequencies, np.abs(want_Output)**2, label="Wanted Output")

# MADSEN ALGORITHM
Target_As = First_want_FFT_Output_ks
Target_Bs = find_valid_b(Lattice_Order, Target_As, to_plot=False)
Ks_estimate, Phis_estimate, As_estimate, Bs_estimate = reverse_transfer_function(Target_As, Target_Bs)

# LATTICE GENERATION
Couplers = []
Balance_traits = []
Unbalance_traits = []
k_coupler = 0.5
for idc in range(2*Lattice_Order+2):
    Couplers += [Pbb.Coupler([1.5, 1.6], (k_coupler, k_coupler))]
for idb in range(Lattice_Order+1):
    Balance_traits += [Pbb.Balanced_propagation(neff, ng, 1.55, losses_parameter, 0)]
for idu in range(Lattice_Order):
    Unbalance_traits += [Pbb.Unbalanced_propagation(neff, ng, 1.55, losses_parameter, 0, dL)]
coupling_losses = 0
Lattice = Pbb.Chip_structure([Couplers, Balance_traits, Unbalance_traits], ['C', 'B', 'C', 'U'] * Lattice_Order + ['C', 'B', 'C'], coupling_losses)
heater_order = ['B', 'U'] * Lattice_Order + ['B']

# OUTPUT CALCULATION
heaters = np.zeros(Lattice_Order*2+1)
for idh in range(Lattice_Order*2+1):
    if idh % 2 == 0:
        heaters[idh] = find_balance_phase(k_coupler, k_coupler, Ks_estimate[int(idh/2)])
    else:
        heaters[idh] = -Phis_estimate[int((idh - 1) / 2)]
Lattice.set_heaters(heaters, heater_order)
S = Lattice.calculate_S_matrix(wavelengths)
output_power = Pbb.calculate_outputs(input_field, S, dB=False)[:, 0]

# PLOTTING POWER OUTPUT
plt.plot(frequencies, output_power, label="Reconstructed Output")
plt.grid()
plt.legend()

# NOT WORKING