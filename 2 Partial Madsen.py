"""
Code Improved 2024/01/14
Author Mattia
"""

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


# Lattice Parameters
Lattice_Order = 5
Ks = np.ones(Lattice_Order + 1) * 0.5
Phis = np.ones(Lattice_Order + 1) * 0
As, Bs = calculate_transfer_function(Lattice_Order, Ks, Phis)

# Entering from top (Xin = 1, Yin = 0)
n_points = 500
bar = np.zeros(n_points, dtype=np.complex128)
x = np.linspace(0, 1, n_points)
for fourier_coefficient in range(0, Lattice_Order+1):
    bar = bar + np.exp(fourier_coefficient * 2j * np.pi * x) * As[-1][fourier_coefficient]
plt.plot(np.abs(bar)**2, label="Correct Lattice")

# Reverse Layer
Ks_estimate, Phis_estimate, As_estimate, Bs_estimate = reverse_transfer_function(As[-1], Bs[-1])
As_recalculated, Bs_recalculated = calculate_transfer_function(Lattice_Order, Ks_estimate, Phis_estimate)
time_domain_estimate = np.zeros(n_points, dtype=np.complex128)
for fourier_coefficient in range(0, Lattice_Order+1):
    time_domain_estimate = time_domain_estimate + np.exp(fourier_coefficient * 2j * np.pi * x) * As_recalculated[-1][fourier_coefficient]
plt.plot(np.abs(time_domain_estimate)**2, label="Reconstructed Lattice")
plt.grid()
plt.legend()
plt.show()