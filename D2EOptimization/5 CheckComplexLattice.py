"""
Code D2EOptimization 2023/12/29
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
        heater_phases[heater_codes[idp]] = phase

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
    :param lattice: Pbb lattice used to determine the output
    :param phases: Phases in radiant to apply to the heater to extract the output power
    :param target: Target in dB that the lattice should assume
    :return: Absolute squared error between the bar output of the lattice and the target
    """
    # Calculate the bar of the lattice
    bar_dB = calculate_output_bar(phases, lattice)

    # Calculate the absolute squared error
    error = (bar_dB - target) ** 2
    error = np.sum(error) + np.max(error)
    return error

def final_plot(vars1, vars2, var_name1, var_name2, gains, max_ripples_dict, max_ripples_sum_dict, plot_type):
    # Plot data: Preparation
    color_start = np.array([1, 0, 0])
    color_end = np.array([0, 0, 1])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    ax1.set_title(f"Band {Band}", fontsize=23)

    # Plot
    n = len(gains)
    colors = [color_start * p / n + color_end * (1 - p / n) for p in range(n)]
    for idx1, var1 in enumerate(vars1):
        for idx2, var2 in enumerate(vars2):
            codename_max = f"_{var_name1}{var1}_{var_name2}{var2}"
            ax2.scatter(var2 if plot_type == 1 else var1, float(max_ripples_sum_dict[codename_max]), color=colors[0])

            for idg, gain in enumerate(gains):
                codename = f"gain{gain}_{var_name1}{var1}_{var_name2}{var2}"
                if (idx2 == 0 and plot_type == 1) or (idx1 == 0 and plot_type == 2):
                    ax1.scatter(var2 if plot_type == 1 else var1, float(max_ripples_dict[codename]), color=colors[idg], label=f"gain {gain}")
                else:
                    ax1.scatter(var2 if plot_type == 1 else var1, float(max_ripples_dict[codename]), color=colors[idg])

    for axes_idx in range(2):
        if axes_idx == 0:
            ax1.set_ylabel("Max Ripple [dB]", fontsize=12)
            ax1.set_xlabel(var_name2 if plot_type == 1 else var_name1, fontsize=12)
            ax1.grid()
            ax1.legend()
        elif axes_idx == 1:
            ax2.set_ylabel("Max Ripple Sum [dB]", fontsize=12)
            ax2.set_xlabel("dL0 um", fontsize=12)
            ax2.grid()
    plt.show()

# Band selector: Write C or L
Band = 'C'

# Scan input
gain = 10
x = 31
y = 101
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
    'ng': 1.54 if Band == 'L' else 1.54,
    'wavelength0': 1.55 if Band == 'C' else 1.59,  # um
    'L': 0.3,  # cm
    'A': 0.4,  # dB/cm
    'B': 4655.664,  # um ** -2
    'C': 3.614e-04,  # um ** 2
    'D': 0.05,  # dB/cm
    'wl1': 1.502293,  # um
    'wl2': 1.511449,  # um
    'wavelengths': wavelengths
}

# Coupler arguments
coupler_args = {
    'k0': np.pi / 4,
    'k1': -2.63,
    'k2': 0,
    'wavelength0': 1.55 if Band == 'C' else 1.59,
    'wavelengths': wavelengths}

waveguide_choice_args = {
    'input_choice': 1,
    'output_choice':1,
    'wavelengths': wavelengths}

coupling_loss_args = {
    'coupling_losses': 0,
    'wavelengths': wavelengths}

# BUILDING BLOCKS
n_unbalances = 4
unbalances_codes = [4, 8, 16, 20]
heater_codes = [2, 4, 6, 8, 10, 14, 16, 18, 20, 22]
# B U B U B - B U B U B
structures = {0: Pbb.WaveguideFacet(**coupling_loss_args)}
for idx in range(2):
    structures[4 * idx + 1] = Pbb.Coupler(**coupler_args)
    structures[4 * idx + 2] = Pbb.DoubleWaveguide(**waveguide_args_balance)
    structures[4 * idx + 3] = Pbb.Coupler(**coupler_args)
    structures[4 * idx + 4] = Pbb.DoubleWaveguide(**waveguide_args_balance)  # Changed to unbalance later in the code
structures[9] = Pbb.Coupler(**coupler_args)
structures[10] = Pbb.DoubleWaveguide(**waveguide_args_balance)
structures[11] = Pbb.Coupler(**coupler_args)
structures[12] = Pbb.WaveguideChoice(**waveguide_choice_args)
for idx in range(2):
    structures[4 * idx + 13] = Pbb.Coupler(**coupler_args)
    structures[4 * idx + 14] = Pbb.DoubleWaveguide(**waveguide_args_balance)
    structures[4 * idx + 15] = Pbb.Coupler(**coupler_args)
    structures[4 * idx + 16] = Pbb.DoubleWaveguide(**waveguide_args_balance)
structures[21] = Pbb.Coupler(**coupler_args)
structures[22] = Pbb.DoubleWaveguide(**waveguide_args_balance)
structures[23] = Pbb.Coupler(**coupler_args)
structures[24] = Pbb.WaveguideFacet(**coupling_loss_args)

# Lattice components generation (Unbalance)
for idx in range(n_unbalances):
    waveguide_args_unbalance = waveguide_args_balance.copy()
    waveguide_args_unbalance['dL'] = x if idx < 3 else y
    structures[unbalances_codes[idx]] = Pbb.DoubleWaveguide(**waveguide_args_unbalance)

# Lattice component finalization
Lattice = Pbb.ChipStructure(structures)
Lattice.calculate_internal_transfer_function()

# Optimize the current architecture
max_ripples = {}
max_ripples_sum = {}

minimum_error = np.inf
optimal_phase = None
for i in range(10):
    initial_heater_phase = np.random.uniform(0, np.pi*2, len(heater_codes))
    optimization = opt.minimize(calculate_output_error, x0=initial_heater_phase, args=(Lattice, targets[gain]))
    if optimization.fun < minimum_error:
        minimum_error = optimization.fun
        optimal_phase = optimization.x

# Extract the optimal profile
output_power_bar_dB = calculate_output_bar(optimal_phase, Lattice)

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
print(f"Max ripple: {max_ripple} dB, Max ripple internal: {max_ripple_internal} dB, Square Error: {square_error} dB")
print(f"Power Consumption: {np.sum(np.abs(optimal_phase%(2*np.pi)-np.pi)*220/np.pi)} mW")