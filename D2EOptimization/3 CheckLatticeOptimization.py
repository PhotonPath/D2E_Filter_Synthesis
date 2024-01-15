"""
Code Improved 2024/01/14
Author Mattia
"""

import Photonic_building_block as Pbb
import matplotlib.pyplot as plt
import scipy.constants as const
import scipy.optimize as opt
import numpy as np


def calculate_output_bar(phases, lattice):
    """
    Calculate the output of the lattice in input for all the wavelengths of interest, given the correspondent heater phases.
    :param phases: Phases in radiant to apply to the heater to extract the output power
    :param lattice: Pbb lattice used to perform the calculation
    :return: The output power in dB, of the lattice
    """
    heater_phases = {}
    for idp, phase in enumerate(phases):
        heater_phases[2*idp+2] = phase

    # Apply heater phases to the lattice
    lattice.set_heaters(heater_phases)
    lattice.calculate_transfer_function()

    # Calculate the transfer matrix of the lattice
    output_field = lattice.calculate_output(input_field)
    output_power_bar = 20*np.log10(np.abs(output_field[0]))
    return output_power_bar


def calculate_output_error(phases, lattice, target):
    """
    Calculate the absolute squared error between the bar output of the lattice and the target.
    :param phases: Phases in radiant to apply to the heater to extract the output power
    :param lattice: Pbb lattice used to determine the output
    :param target: Target in dB that the lattice should assume
    :return: Absolute squared error between the bar output of the lattice and the target
    """
    # Calculate the bar of the lattice
    bar_dB = calculate_output_bar(phases, lattice)

    # Calculate the absolute squared error
    error = (bar_dB - target)**2
    error = np.sum(error) + np.max(error)
    return error


# Band selector: Write C or L
Band = 'C'

# Lattice order selector: Write an integer number representing the number of stage of the D2E filter
filter_order = 4
heater_number = filter_order * 2 + 1

# Scan input
gain = 10
dLs = [30, 30, 30, 30]
precision_digits = 3

# Load data
c = const.c / 1000000
file_path = f"P:/Drive condivisi/4 - Technology Office/4 - Design and Simulations/P03 - Amplifier/07 - VPI with Python/DualStageC+L - seprated {Band} - Copropagating - FIXGFF.vtmu_pack/Inputs/"
targets = {}
frequencies = []
wavelengths = []
file_name = f"GEFprofile_GAIN{gain}.txt"
data = np.loadtxt(file_path + file_name, skiprows=2)
frequencies = data[:, 0] / 1e12
wavelengths = c / frequencies
targets[gain] = data[:, 1]
n_points = len(wavelengths)
input_field = [np.ones(n_points), np.zeros(n_points)]
target_index_internal = 1

waveguide_args_balance = {
    'neff0': 1.489 if Band == 'C' else 1.486,
    'ng':  1.54 if Band == 'L' else 1.54,
    'wavelength0':  1.55 if Band == 'C' else 1.59,    # um
    'L': 0.3,                                         # cm
    'A': 0.4,                                         # dB/cm
    'B': 4655.664,                                    # um ** -2
    'C': 3.614e-04,                                   # um ** 2
    'D': 0.05,                                        # dB/cm
    'wl1': 1.502293,                                  # um
    'wl2': 1.511449,                                  # um
    'wavelengths': wavelengths
}

# Coupler arguments
coupler_args = {
    'k0': np.pi/4,
    'k1': -2.63,
    'k2': 0,
    'wavelength0': 1.55 if Band == 'C' else 1.59 ,
    'wavelengths': wavelengths}

coupling_loss_args = {
    'coupling_losses': 0,
    'wavelengths': wavelengths}

# BUILDING BLOCKS
structures = {0: Pbb.WaveguideFacet(**coupling_loss_args)}
waveguide_args_unbalance = waveguide_args_balance.copy()

unbalance_idx = 0
for idx in range(filter_order):
    structures[4 * idx + 1] = Pbb.Coupler(**coupler_args)
    structures[4 * idx + 2] = Pbb.DoubleWaveguide(**waveguide_args_balance)
    structures[4 * idx + 3] = Pbb.Coupler(**coupler_args)
    waveguide_args_unbalance['dL'] = dLs[unbalance_idx]
    structures[4 * idx + 4] = Pbb.DoubleWaveguide(**waveguide_args_unbalance) # Changed to unbalance later in the code
    unbalance_idx += 1
structures[4*filter_order + 1] = Pbb.Coupler(**coupler_args)
structures[4*filter_order + 2] = Pbb.DoubleWaveguide(**waveguide_args_balance)
structures[4*filter_order + 3] = Pbb.Coupler(**coupler_args)
structures[4*filter_order + 4] = Pbb.WaveguideFacet(**coupling_loss_args)

# Generate Lattice
Lattice = Pbb.ChipStructure(structures)
Lattice.calculate_internal_transfer_function()
Lattice.calculate_transfer_function()

# Lattice components generation (Balance and Coupler)
max_ripples = {}
max_ripples_sum = {}

minimum_error = np.inf
optimal_phase = None
for i in range(10):
    initial_heater_phase = np.random.uniform(0, np.pi * 2, heater_number)
    optimization = opt.minimize(calculate_output_error, x0=initial_heater_phase, args=(Lattice, targets[gain]))
    if optimization.fun < minimum_error:
        minimum_error = optimization.fun
        optimal_phase = optimization.x

# Extract the optimal profile
output_power_bar_dB = calculate_output_bar(optimal_phase, Lattice)
plt.figure(figsize=(13, 8))
plt.title("Profile", fontsize=23)
plt.plot(frequencies, output_power_bar_dB, label="Simulation Output")
plt.plot(frequencies, targets[gain], label="Target")
plt.xlabel("Wavelength [um]", fontsize=23)
plt.ylabel("Power [dB]", fontsize=23)
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)
plt.legend(fontsize=23)
plt.grid()
plt.show()

# Extract useful information
max_ripple = np.round(max(abs(output_power_bar_dB - targets[gain])), precision_digits)
max_ripple_internal = np.round(max(abs(output_power_bar_dB[target_index_internal:-target_index_internal] - targets[gain][target_index_internal:-target_index_internal])), precision_digits)
square_error = np.round(np.sqrt(np.sum((output_power_bar_dB - targets[gain])**2)), precision_digits)
print(f"dLs: {dLs} Max ripple: {max_ripple} dB, Max ripple internal: {max_ripple_internal} dB, Square Error: {square_error} dB")
print(f"Power Consumption: {np.sum(np.abs(optimal_phase%(2*np.pi)-np.pi)*220/np.pi)} mW")