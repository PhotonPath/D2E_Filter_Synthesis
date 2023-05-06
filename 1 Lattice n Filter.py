"""
Code Data 2023/04/11
Author Mattia
"""

import Photonic_building_block as Pbb
import matplotlib.pyplot as plt
import numpy as np

# Input data
c = 300
FSR = 5 # THz
n_points = 100
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
}
dL = c/FSR/ng

# Lattice block element
Lattice_Order = 5
x = np.linspace(0, Lattice_Order, Lattice_Order+1)
Coupler = [Pbb.Coupler([1.5, 1.6], [coupler_value, coupler_value])] * (Lattice_Order * 2 + 2)
Balance_trait = []
Unbalance_trait = []
for i in range(Lattice_Order+1):
    Balance_trait += [Pbb.Balanced_propagation(neff, ng, 1.55, losses_parameter, 0)]
for i in range(Lattice_Order):
    Unbalance_trait += [Pbb.Unbalanced_propagation(neff, ng, 1.55, losses_parameter, 0, dL)]
coupling_losses = 0

# Output calculation
Lattice = Pbb.Chip_structure([Coupler, Balance_trait, Unbalance_trait], ['C', 'B', 'C', 'U'] * Lattice_Order + ['C', 'B', 'C'], coupling_losses)
heater_order = ['B', 'U'] * Lattice_Order + ['B']
heaters = np.zeros(Lattice_Order * 2 + 1)
heaters[0] = np.pi/2
heaters[2] = np.pi
heaters[4] = np.pi/2
Lattice.set_heaters(heaters, heater_order)
S = Lattice.calculate_S_matrix(wavelengths)
input_field = np.array([[1, 0], [0,  0]])
output_power = Pbb.calculate_outputs(input_field, S, dB=False)[:, 0]
output_field = Pbb.calculate_field_outputs(input_field, S)[:, 0]
plt.plot(np.angle(output_field))
plt.plot(np.abs(output_field)**2)
plt.show()

# Plotting Power Output
# plt.figure(1)
# plt.plot(frequencies, output_power)
# plt.grid()
# plt.show()

# Plotting Power Output Fourier Transform
# plt.figure(2)
# output_power_fft = np.fft.fft(output_power) / n_points
# plt.scatter(x, output_power_fft[0:Lattice_Order+1])
# plt.grid()
# plt.show()


