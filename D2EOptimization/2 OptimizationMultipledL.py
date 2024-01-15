"""
Code Improved 2024/01/15
Author Mattia
"""

import Photonic_building_block as Pbb
import matplotlib.pyplot as plt
import scipy.constants as const
import scipy.optimize as opt
import numpy as np
import yaml

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
    ax1.set_title(f"Band {Band} - Order {Lattice_Order}", fontsize=23)

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

# Lattice order selector: Write an integer number representing the number of stage of the D2E filter
Lattice_Order = 5

# Scan input
Gains = np.linspace(10, 30, 5).astype(int)
dLs = np.linspace(25, 35, 1)
dL0s = np.linspace(25, 65, 3)
plot_kind = 1 if len(dL0s) == 1 else 2  # 1 = dL constant, 2 = dL variable

precision_digits = 3

# Saving information
file_name_save = f"Band {Band} - Order {Lattice_Order} - High Range ({dLs[0]} - {dLs[-1]})"
file_path_save = "Data C-Band/"

# Load data
c = const.c / 1000000
file_path = f"P:/Drive condivisi/4 - Technology Office/4 - Design and Simulations/P03 - Amplifier/07 - VPI with Python/DualStageC+L - seprated {Band} - Copropagating - FIXGFF.vtmu_pack/Inputs/"
targets = {}
frequencies = []
wavelengths = []
for Gain in Gains:
    file_name = f"GEFprofile_GAIN{Gain}.txt"
    data = np.loadtxt(file_path + file_name, skiprows=2)
    frequencies = data[:, 0] / 1e12
    wavelengths = c / frequencies
    targets[Gain] = data[:, 1]
n_points = len(wavelengths)
input_field = [np.ones(n_points), np.zeros(n_points)]

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

heater_number = Lattice_Order * 2 + 1

# BUILDING BLOCKS
structures = {0: Pbb.WaveguideFacet(**coupling_loss_args)}

for idx in range(Lattice_Order):
    structures[4 * idx + 1] = Pbb.Coupler(**coupler_args)
    structures[4 * idx + 2] = Pbb.DoubleWaveguide(**waveguide_args_balance)
    structures[4 * idx + 3] = Pbb.Coupler(**coupler_args)
    structures[4 * idx + 4] = Pbb.DoubleWaveguide(**waveguide_args_balance) # Changed to unbalance later in the code
structures[4*Lattice_Order + 1] = Pbb.Coupler(**coupler_args)
structures[4*Lattice_Order + 2] = Pbb.DoubleWaveguide(**waveguide_args_balance)
structures[4*Lattice_Order + 3] = Pbb.Coupler(**coupler_args)
structures[4*Lattice_Order + 4] = Pbb.WaveguideFacet(**coupling_loss_args)

max_ripples = {}
max_ripples_sum = {}

# Test constant dL combination against all gain profiles
for Gain in Gains:
    for dL0 in dL0s:
        for dL in dLs:
            code = f"gain{Gain}_dL0{dL0}_dL{dL}"
            print(f"Simulation started: gain {Gain} dB, dL0 {dL0} um, dL: {dL}")
            # Lattice components generation (Unbalance)
            for idx in range(Lattice_Order):
                waveguide_args_unbalance = waveguide_args_balance.copy()
                waveguide_args_unbalance['dL'] = dL0 if idx == 0 else dL
                structures[4 * idx + 4] = Pbb.DoubleWaveguide(**waveguide_args_unbalance)

            # Lattice component finalization
            Lattice = Pbb.ChipStructure(structures)
            Lattice.calculate_internal_transfer_function()

            # Optimize the current architecture
            minimum_error = np.inf
            optimal_phase = None
            for i in range(10):
                initial_heater_phase = np.random.uniform(0, np.pi*2, heater_number)
                optimization = opt.minimize(calculate_output_error, x0=initial_heater_phase, args=(Lattice, targets[Gain]))
                if optimization.fun < minimum_error:
                    minimum_error = optimization.fun
                    optimal_phase = optimization.x

            # Extract the optimal profile
            output_power_bar_dB = calculate_output_bar(optimal_phase, Lattice)

            # Extract useful information
            max_ripple = np.round(max(abs(output_power_bar_dB - targets[Gain])), precision_digits)
            print(f"dL0: {dL0}, dL: {dL}, Max ripple: {max_ripple} dB")

            max_ripples[code] = str(max_ripple)

for dL0 in dL0s:
    for dL in dLs:
        code_max = f"_dL0{dL0}_dL{dL}"
        max_ripples_sum[code_max] = 0
        for Gain in Gains:
            code = f"gain{Gain}_dL0{dL0}_dL{dL}"
            max_ripples_sum[code_max] += float(max_ripples[code])


final_plot(dL0s, dLs, 'dL0', 'dL', Gains, max_ripples, max_ripples_sum, plot_kind)

# Plot Data: Saving
# plt.savefig(file_path_save + file_name_save + ".png")
# plt.show()

# Save data
# save_data = {'max_ripples':max_ripples,
#              'max_ripples_sum': max_ripples_sum}

# with open(file_path_save + file_name_save + ".yaml", 'w') as file:
#     yaml.dump(save_data, file)