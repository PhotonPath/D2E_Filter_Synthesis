"""
Code Improved 2024/01/15
Author Mattia
"""

import Photonic_building_block as Pbb
import scipy.constants as const
import matplotlib.pyplot as plt
import scipy.signal as ss
import numpy as np

# Polynomials in this script are expressed as A(z) = A[0] + A[1]*z^-1 + A[2]*z^-2 ...

def calculate_next_layer_balance(lattice_order, a_n, b_n, k, phi, n):
    """
    Uses the value of the previous An Bn polynomial to calculate An+1 Bn+1
    :param lattice_order: Order of the lattice. Uses to create vectors of correct size
    :param a_n: polynomial A of order n in matrix formula
    :param b_n: polynomial B of order n in matrix formula
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
            new_a[i] = np.sqrt(1 - k) * np.exp(-phi * 1j) * a_n[i] - np.sqrt(k) * b_n[i]
            new_b[i] = np.sqrt(k) * np.exp(-phi * 1j) * a_n[i] + np.sqrt(1 - k) * b_n[i]
    else:          # Unbalance
        for i in range(lattice_n + 1):
            new_a[i] = np.sqrt(1 - k) * np.exp(-phi * 1j) * a_n[i - 1] - np.sqrt(k) * b_n[i]
            new_b[i] = np.sqrt(k) * np.exp(-phi * 1j) * a_n[i - 1] + np.sqrt(1 - k) * b_n[i]
    return new_a, new_b

def calculate_prev_layer(lattice_order, a_n, b_n, n):
    """
    Calculate the value of An-1 Bn-1 kn-1 and phi-1 to obtain the wanted An Bn polynomial
    :param lattice_order: Order of the lattice.
    :param a_n: An polynomial of order n+1 in matrix formula
    :param b_n: Bn polynomial of order n+1 in matrix formula
    :param n: index of the layer considered
    :return: Power ratio, Phase and Polynomial A and B of order n-1 in matrix formula
    """

    # K Calculation: Equation 60 Madsen
    if a_n[n] != 0:
        r = np.abs(b_n[n]/a_n[n])**2
        k = r/(1+r)
    else:
        k = 1

    # Bn-1 calculation: Equation 62 Madsen
    coefficient_order = lattice_order + 1
    prev_b = np.zeros(coefficient_order, dtype=np.complex128)
    for i in range(lattice_order+1):
        prev_b[i] = -a_n[i] * np.sqrt(k) + b_n[i] * np.sqrt(1-k)

    # Phi calculation: Equation 65 Madsen
    a_tilda = np.sqrt(1 - k) * a_n[n] + np.sqrt(k) * b_n[n]
    if -n < lattice_order + 1:
        phi = -np.angle(a_tilda) + np.angle(prev_b[n-1])
    else:
        phi = -np.angle(a_tilda)

    # An-1 calculation: Equation 61 Madsen
    prev_a = np.zeros(coefficient_order, dtype=np.complex128)
    for i in range(lattice_order):
        prev_a[i] = (a_n[i + 1] * np.sqrt(1 - k) + b_n[i + 1] * np.sqrt(k)) * np.exp(phi * 1j)
    return k, phi, prev_a, prev_b

def find_balance_phase(lattice_k, wanted_k):
    """
    :param lattice_k: Power ratio of the n coupler (assuming that both couplers of the balance have the same value)
    :param wanted_k: Power ratio we want for the CBC block
    :return: the phase to the balance of the CBC block to obtain the wanted coupler
    """
    wanted_k = 1 - wanted_k
    cos_k = np.sqrt(1 - lattice_k)
    sin_k = np.sqrt(lattice_k)
    t_k = (wanted_k - cos_k**4 - sin_k**4) / (-2*cos_k**2*sin_k**2)
    return np.arccos(t_k)

def find_correction_phase(lattice_k, coupler_estimate, phis_estimate):
    """
    This is the Beyond-Madsen algorithm. This function takes the phase and coupler estimates from the Madsen algorithm (Lattice Order + 1 parameters)
    and returns all the phases to our lattice model (2 * Lattice Order + 1 parameters)
    :param lattice_k: Power ratio of the real couplers (assuming all couplers of the balance have the same value)
    :param coupler_estimate: Power ratio founded by Madsen
    :param phis_estimate: Phases found by Madsen
    :return: correct phase to be applied to the lattice to obtain the wanted target
    """
    lattice_order = len(coupler_estimate) - 1

    as_n, bs_n = [], []
    # INPUT FIELD
    a, b = np.zeros(len(coupler_estimate), dtype=np.complex128), np.zeros(len(coupler_estimate), dtype=np.complex128)
    a[0] = 1 + 0j

    as_n.append(a)
    bs_n.append(b)
    # INITIAL COUPLER
    coupler_phase = find_balance_phase(lattice_k, coupler_estimate[0])
    a, b = calculate_next_layer_balance(lattice_order, a, b, lattice_k, phis_estimate[0], 0)
    as_n.append(a)
    bs_n.append(b)

    a, b = calculate_next_layer_balance(lattice_order, a, b, lattice_k, coupler_phase, 0)
    as_n.append(a)
    bs_n.append(b)

    # REMAINING LATTICE
    a_next, b_next = a, b
    lattice_phases = [coupler_phase]
    for i in range(lattice_order):
        # FIRST CALCULATION
       # phi_tot = 0 if i == 0 else -np.sum(phis_estimate[1:i + 1])
        phi_tot = -np.sum(phis_estimate[1:i + 1])
        a, b = calculate_next_layer_balance(lattice_order, a_next, b_next, lattice_k, phi_tot, 2 * i + 1)
        coupler_phase = find_balance_phase(lattice_k, coupler_estimate[i + 1])
        a_start, b_start = calculate_next_layer_balance(lattice_order, a, b, lattice_k, coupler_phase, 2 * i + 2)

        # PHASE ADJUSTMENT
        correction_phase = np.angle(b_start[i + 1]) - np.angle(b_start[0])
        a, b = calculate_next_layer_balance(lattice_order, a_next, b_next, lattice_k, correction_phase + phis_estimate[i + 1], 2 * i + 1)
        a_next, b_next = calculate_next_layer_balance(lattice_order, a, b, lattice_k, coupler_phase, 2 * i + 2)

        # ADD PHASES TO THE LATTICE PHASES LIST
        lattice_phases.append(correction_phase + phis_estimate[i + 1])
        lattice_phases.append(coupler_phase)
        as_n.append(a)
        bs_n.append(b)
        as_n.append(a_next)
        bs_n.append(b_next)
    return as_n, bs_n, np.array(lattice_phases)

def field_from_z_coefficients(a_n, num_points):
    """
    Generate the function f from the z-coefficients a_n
    :param a_n: polynomial A of order n in matrix formula
    :param num_points: number of points of the function f
    :return: the function f
    """
    f = np.zeros(num_points, dtype=np.complex128)
    x = np.linspace(0, 1, num_points, endpoint=False)
    for fourier_coefficient in range(len(a_n)):
        f = f + np.exp(-fourier_coefficient * 2j * np.pi * x) * a_n[fourier_coefficient]
    return f

def find_valid_b(lattice_order, a_n, gamma=1, to_plot=False):
    """
    Return a valid cross output given a wanted bar output (a)
    :param lattice_order: Order of the lattice.
    :param a_n: Polynomial A of order n in matrix formula
    :param gamma: losses
    :param to_plot: Boolean to decide if you want to see the roots of the polynomial B
    :return: Return a valid cross output given a wanted bar output (a)
    """
    # Total Phase of last polynomial
    phi_tot = -np.angle(a_n[-1])
    if np.abs(phi_tot) > np.pi / 2:
        a_n = np.array(a_n)
        a_n *= -1

    # B*Br Product: Equation 66 Madsen
    a_reverse = np.conjugate(a_n[::-1]) * np.exp(-1j * phi_tot)
    product_b_br = -np.polymul(a_n, a_reverse)
    product_b_br[lattice_order] += np.exp(-1j * phi_tot)

    # Find the B Br roots
    b_br_roots = np.roots(product_b_br[::-1]) # Remember I define the polynomial opposite to numpy, so I have to reverse the order

    # Couple the B Br roots (depending on the gamma circle)
    b_br_roots_coupled_index = []
    for root in b_br_roots:
        b_roots_close = np.isclose(np.abs(b_br_roots)*np.abs(root)*np.exp(1j*(np.angle(b_br_roots)-np.angle(root))), gamma)
        b_roots_close_index = np.argmax(b_roots_close)
        b_br_roots_coupled_index.append(b_roots_close_index)

    # Chose the B root with minimal magnitude (inside the circle)
    b_roots_index = []
    for r, root in enumerate(b_br_roots):
        if (r not in b_roots_index) and (b_br_roots_coupled_index[r] not in b_roots_index) and len(b_roots_index) < lattice_order:
            coupled_root = b_br_roots[b_br_roots_coupled_index[r]]

            # Decide which root has minimal magnitude and add it to the final roots
            if np.abs(root) < np.abs(coupled_root):
                b_roots_index.append(r)
            else:
                b_roots_index.append(b_br_roots_coupled_index[r])
    b_roots_index = np.array(b_roots_index, dtype=int)
    b_roots = b_br_roots[b_roots_index]

    # Take polynomial by roots
    b_n = np.poly(b_roots)
    b_n = b_n[::-1] # Remember I define the polynomial opposite to numpy, so I have to reverse the order

    # Normalize the output
    alpha = np.sqrt(-(a_n[0] * a_n[-1]) / (b_n[0] * b_n[-1]))
    b_n *= alpha

    # Plot roots if wanted
    if to_plot:
        plt.figure(1)
        # Total B B_reverse roots
        plot_roots(product_b_br, color='blue', width=3, unit_circle=True)

        # Only B roots
        plot_roots(b_n, color='red', width=2)

    return a_n, b_n

def plot_roots(a_n, color="black", width=2, unit_circle=False):
    """
    :param a_n: polynomial you want to plot.
    :param color: color of the points.
    :param width: radius of the points.
    :param unit_circle: boolean to decide if you want to plot also the unit circle
    """
    for root in np.roots(a_n[::-1]):
        plt.scatter(np.real(root), np.imag(root), color=color, linewidths=width)

    # Unit Circle
    if unit_circle:
        t_circle = np.linspace(0, np.pi * 2, 100)
        x_circle = np.cos(t_circle)
        y_circle = np.sin(t_circle)
        plt.plot(x_circle, y_circle, color='black')
    plt.grid()

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

def z_cut(f, f_index):
    """
    Cut the function up to the f_index harmonic
    :param f: function or target
    :param f_index: number of frequencies to be considered
    :return: function f without the high order harmonics, coefficient of the low order harmonics
    """
    # Take the first harmonics coefficients
    fft_f_coefficients = []
    num_points = len(f)
    x = np.linspace(0, 1, num_points, endpoint=False)
    for fourier_coefficient in range(f_index + 1):
        fft_f_coefficients.append(np.sum(np.exp(fourier_coefficient * 2j * np.pi * x) * f) / num_points)

    # Take the field build with those coefficients
    cut_field = field_from_z_coefficients(fft_f_coefficients, num_points)
    return cut_field, np.array(fft_f_coefficients)


# INPUTS
c = const.c/1000000
wavelength0 = 1.55
neff = 1.46
ng = 1.52
dL = 30
FSR = c/dL/ng # THz
n_points = 500
frequencies = np.linspace(192, 192+FSR, n_points, endpoint=False)
normalized_frequencies = np.linspace(0, 1, n_points, endpoint=False)
wavelengths = c/frequencies
input_field = [np.ones(n_points), np.zeros(n_points)]

waveguide_args_balance = {
    'neff0': neff,
    'ng': ng,
    'wavelength0': wavelength0,    # um
    'wavelengths': wavelengths
}
waveguide_args_unbalance = waveguide_args_balance.copy()
waveguide_args_unbalance['dL'] = dL

# Coupler arguments
coupler_value = np.pi/4
coupler_args = {
    'k0': coupler_value,
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


# LINEAR TARGET POWER (mW) AND FIELD USING HILBERT OPERATOR
# Target = 3.8 * (normalized_frequencies - 0.5) ** 2 + 0.1
# Target = (0.7 - normalized_frequencies*0.6)
Target = np.abs(np.sin(5.2 * normalized_frequencies * np.pi)) * 0.8 + 0.1

Target_Field_amplitude = np.sqrt(Target)
Target_Field_phase = -ss.hilbert(np.log(Target_Field_amplitude))
Target_Field = Target_Field_amplitude * np.exp(1j*np.imag(Target_Field_phase))  # NEEDED FOR HAVING A FEASIBLE FIELD

# LINEAR CUT TARGET FIELD (removed high order frequencies)
Target_Field_cut, Target_As = z_cut(Target_Field, filter_order)

# MADSEN ALGORITHM B PART -> find the correct target B
Target_As, Target_Bs = find_valid_b(filter_order, Target_As, to_plot=False)

# MADSEN ALGORITHM for a LATTICE formed by only couplers and unbalance (BACK)
K_estimate, Phis_estimate, As_estimate, Bs_estimate = reverse_transfer_function(Target_As, Target_Bs)

# BEYOND MADSEN -> Re-adjust the unbalance phases to compensate the coupler and CBC difference
afs, bfs, phases = find_correction_phase(np.sin(coupler_value)**2, K_estimate, Phis_estimate)

# NEFF COMPENSATION
Neff_shift_phis = np.zeros(len(phases))
Neff_shift_phis[1::2] = np.ones(filter_order) * 2 * np.pi * ((neff - ng) * dL / wavelength0 + frequencies[0] / FSR)
# The unbalance shifts must compensate the selected band (frequencies[0] / FSR) because the filter consider as regular bands the
# multiples of the FSR, while we want the filter to be set in a given region of the spectrum
# The unbalance must also compensate the difference between neff and ng (neff - ng) * dL / wavelength_neff.

# LATTICE GENERATION
Lattice = Pbb.ChipStructure(structures)
Lattice.calculate_internal_transfer_function()

# LATTICE OUTPUT CALCULATION
phis_estimate_dict = {}
for idp, phi_estimate in enumerate(phases):
    phis_estimate_dict[idp * 2 + 2] = phi_estimate-Neff_shift_phis[idp]

Lattice.set_heaters(phis_estimate_dict)
Lattice.calculate_transfer_function()
output_field = Lattice.calculate_output(input_field)

# PLOTTING
plt.figure(2)
plt.plot(frequencies, Target, label="Power Target")
plt.plot(frequencies, np.abs(Target_Field_cut)**2, label="Power Target Cut")
# plt.plot(frequencies, np.abs(field_from_z_coefficients(Target_As, n_points))**2)
plt.plot(frequencies, np.abs(output_field[0])**2, label="PBB Output Bar")
plt.xlabel("Frequencies")
plt.ylabel("Filter Linear Transfer Function")
plt.legend()
plt.grid()
plt.show()