"""
Code Data 2023/04/12
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
        t = np.linspace(0, np.pi * 2, 100)
        x_circle = np.cos(t)
        y_circle = np.sin(t)
        plt.plot(x_circle, y_circle, color='black')
    plt.grid()

# Lattice Parameters
Lattice_Order = 5
Ks = [0.5, 0.5, 0.4, 0.5, 0.5, 0.5]      # np.ones(Lattice_Order + 1) * 0.5
Phis = [0, 0.5, 0.5, 0.5, 0.5, 0.5]                # np.ones(Lattice_Order + 1) * 0.5
Phis[0] -= np.sum(Phis)
As, Bs = calculate_transfer_function(Lattice_Order, Ks, Phis)

# Entering from top (Xin = 1, Yin = 0)
n_points = 500
bar = np.zeros(n_points, dtype=np.complex128)
cross = np.zeros(n_points, dtype=np.complex128)
x = np.linspace(0, 1, n_points)
for fourier_coefficient in range(0, Lattice_Order+1):
    bar = bar + np.exp(fourier_coefficient * 2j * np.pi * x) * As[-1][fourier_coefficient]
    cross = cross - np.exp(fourier_coefficient * 2j * np.pi * x) * Bs[-1][fourier_coefficient] * np.exp(1j)
plt.figure(10)
plt.plot(x, np.abs(bar)**2, label="Correct Lattice Bar")
# plt.plot(x, np.abs(cross)**2, label="Correct Lattice Cross")
plt.xlabel("2*pi*Frequencies/FSR")
plt.ylabel("Filter Linear Transfer Function")
plt.grid()
plt.legend()

# Reverse Layer. Bs now is unknown!
Target_As = As[-1]
Target_Bs = find_valid_b(Lattice_Order, Target_As, to_plot=True)
Ks_estimate, Phis_estimate, As_estimate, Bs_estimate = reverse_transfer_function(Target_As, Target_Bs)
As_recalculated, Bs_recalculated = calculate_transfer_function(Lattice_Order, Ks_estimate, Phis_estimate)
bar_estimate = np.zeros(n_points, dtype=np.complex128)
for fourier_coefficient in range(0, Lattice_Order+1):
    bar_estimate = bar_estimate + np.exp(fourier_coefficient * 2j * np.pi * x) * As_recalculated[-1][fourier_coefficient]
plt.figure(10)
plt.plot(x, np.abs(bar_estimate)**2, label="Reconstructed Lattice Bar")
plt.xlabel("2*pi*Frequencies/FSR")
plt.ylabel("Filter Linear Transfer Function")
plt.grid()
plt.legend()

