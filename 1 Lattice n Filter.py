"""
Code Improved 2024/01/14
Author Mattia
"""

import Photonic_building_block as Pbb
import matplotlib.pyplot as plt
import scipy.constants as const
import numpy as np

# INPUTS
c = const.c
FSR = 5 # THz
n_points = 1001
frequencies = np.linspace(190, 190+FSR, n_points)
wavelengths = c/frequencies
input_field = [np.ones(n_points), np.zeros(n_points)]

waveguide_args_balance = {
    'neff0': 1.5,
    'ng': 1.5,
    'wavelength0': 1.55,    # um
    'dL': c/FSR/1.5,
    'wavelengths': wavelengths
}
waveguide_args_unbalance = waveguide_args_balance.copy()
waveguide_args_unbalance['dL'] = 37

# Coupler arguments
coupler_args = {
    'k0': np.pi/4,
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

# Output calculation
heaters = {2: np.pi/2,
           4: np.pi,
           6: np.pi/2}

# LATTICE
Lattice = Pbb.ChipStructure(structures)
Lattice.calculate_internal_transfer_function()
Lattice.calculate_transfer_function()
Lattice.set_heaters(heaters)
output_field = Lattice.calculate_output(input_field)
output_power = np.abs(output_field[0])**2

# PLOT
plt.figure(0)
plt.plot(np.angle(output_field[0]))
plt.plot(np.abs(output_field[0])**2)

# Plotting Power Output
plt.figure(1)
plt.plot(frequencies, output_power)
plt.grid()

# Plotting Power Output Fourier Transform
plt.figure(2)
x = np.linspace(0, filter_order, filter_order+1)
output_power_fft = np.fft.fft(output_power) / n_points
plt.scatter(x, output_power_fft[0:filter_order+1])
plt.grid()
plt.show()


