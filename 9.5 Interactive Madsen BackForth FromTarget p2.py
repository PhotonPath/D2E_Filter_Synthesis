from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
import numpy as np

"""
Code Data 2023/04/20
Author Mattia
"""

import Photonic_building_block as Pbb
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss

# Function for BACKFORTH MADSEN

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
    if abs(t_k) <= 1:
        return np.arccos(t_k)
    elif t_k > 1:
        return 0
    else:
        return np.pi

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

# GUI

class PlotGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        # Geometry
        self.secondary_window = tk.Toplevel(self)
        self.dimensions = (600, 600)
        self.title("Controller")
        self.secondary_window.title("Lattice")
        self.geometry(f"{self.dimensions[0]}x{self.dimensions[1]}")
        self.secondary_window.geometry(f"{self.dimensions[0]}x{self.dimensions[1]}")
        self.resizable(False, False)
        self.secondary_window.resizable(False, False)

        # Canvas Circles
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.fig_subplot = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(expand=True, fill="both")

        # Canvas Drawing
        self.fig2 = Figure(figsize=(5, 4), dpi=100)
        self.fig2_subplot = self.fig2.add_subplot(111)
        self.canvas_plot = FigureCanvasTkAgg(self.fig2, master=self.secondary_window)
        self.canvas_plot.get_tk_widget().pack(expand=True, fill="both")

        # Scrollbar K
        self.canvas_down = tk.Canvas(self, height=20)
        self.canvas_down.pack(side=tk.BOTTOM, fill='x')
        self.frame_left = tk.Frame(self.canvas_down, height = 30)
        self.frame_right = tk.Frame(self.canvas_down,  height = 30, width=500, bg="red")
        self.frame_left.pack(side=tk.LEFT, fill='x')
        self.frame_right.pack(side=tk.RIGHT, fill='x', expand=True)

        self.scrollbar = tk.Scrollbar(self.frame_right, orient='horizontal', command=self.update_k, width=30)
        self.scrollbar.set(0.5, 0.5)
        self.scrollbar.pack(side=tk.BOTTOM, fill='x')
        self.scrollbar2 = tk.Scrollbar(self.frame_right, orient='horizontal', command=self.update_k_slope, width=30)
        self.scrollbar2.set(0.5, 0.5)
        self.scrollbar2.pack(side=tk.BOTTOM, fill='x')
        self.scrollbar_text = tk.Label(self.frame_left, height=2, width=15, text="K Value 0.50")
        self.scrollbar_text.pack(side=tk.BOTTOM)
        self.scrollbar_text2 = tk.Label(self.frame_left, height=2, width=15, text="K Slope 0.00")
        self.scrollbar_text2.pack(side=tk.BOTTOM)

        # Circles
        self.lattice_order = 5
        self.n_target_points = self.lattice_order+2
        self.selected_target = 0
        self.target_function = None
        self.is_target_selected = False
        self.target_position = [[t*self.dimensions[0], v*self.dimensions[1]] for (t, v) in zip(np.linspace(0, 1, self.n_target_points), np.linspace(0, 1, self.n_target_points))]
        self.target_position[0]  = [0, self.dimensions[1]/2]
        self.target_position[-1] = [self.dimensions[0], self.dimensions[1]/2]

        # Lattice Stuff
        self.n_points = 500
        self.c = 299.792458  # um * THz
        self.dL = 30  # um
        self.wavelength_neff = 1.55  # um
        self.neff = 1.46  # @ wavelength_neff
        self.ng = 1.52  # @ wavelength_neff
        self.FSR = self.c / self.dL / self.ng  # THz
        self.frequencies = np.linspace(192, 192 + self.FSR, self.n_points)
        self.normalized_frequencies = np.linspace(0, 1, self.n_points, endpoint=False)
        self.wavelengths = self.c / self.frequencies
        self.K = 0.5
        self.K_slope = 0 # Wavelength dependent coupler
        self.input_field = np.array([[1, 0], [0, 0]])
        self.losses_propagation_parameter = {
            'A': 0,
            'B': 0,
            'C': 0,
            'D': 0,
            'wl1': 1.5,
            'wl2': 1.6
        }
        self.losses_coupling = 0
        self.heater_order = ['B', 'U'] * self.lattice_order + ['B']
        self.lattice = self.generate_lattice()

        # Event
        self.draw_all_target_points()
        self.bind('<Motion>', self.move_target)
        self.bind('<Button-1>', self.update_selected_circle_status)

        # Plot
        self.update_plot()

    def update_selected_circle_status(self, event):
        if "scrollbar" not in str(event.widget):
            self.is_target_selected = not self.is_target_selected
            if self.is_target_selected:
                difference = np.inf
                for target_idx, target_position in enumerate(self.target_position):
                    diff = (target_position[0] - (event.x - 75)/0.75) ** 2
                    if diff < difference:
                        self.selected_target = target_idx
                        difference = diff

    def update_plot(self):
        self.fig2_subplot.clear()
        Target_dB = np.polyval(self.target_function, self.normalized_frequencies)
        Target_Field_amplitude = np.sqrt(10 ** (Target_dB / 10))
        Target_Field_phase = -ss.hilbert(np.log(Target_Field_amplitude))
        Target_Field = Target_Field_amplitude * np.exp(1j * np.imag(Target_Field_phase))
        Target_Field_cut, Target_As = z_cut(Target_Field, self.lattice_order)

        # INVERT TARGET AS
        original_roots = np.roots(Target_As[::-1])
        roots = 1 / np.abs(original_roots) * np.exp(1j * np.angle(original_roots))
        target_as = np.poly(roots)[::-1]
        Target_As = target_as * np.sum(np.abs(Target_As)) / np.sum(np.abs(target_as))

        g = self.generate_output_power(Target_As)
        self.fig2_subplot.plot(self.frequencies, 10*np.log10(np.abs(Target_Field) ** 2), label="Target Profile")
        self.fig2_subplot.plot(self.frequencies, 10*np.log10(g), label="Lattice Profile")
        self.fig2_subplot.grid()
        self.fig2_subplot.legend()
        self.canvas_plot.draw()

    def move_target(self, event):
        if self.is_target_selected:
            self.fig_subplot.clear()
            if self.selected_target == 0 or self.selected_target == self.n_target_points - 1:
                self.target_position[0][1] = (event.y - 55) / 0.7
                self.target_position[-1][1] = (event.y - 55) / 0.7
            else:
                self.target_position[self.selected_target][1] = (event.y - 55) / 0.7
            self.draw_all_target_points()
            self.update_plot()

    def draw_all_target_points(self):
        for target_position in self.target_position:
            self.fig_subplot.scatter(target_position[0]/self.dimensions[0], target_position[1]/self.dimensions[1]*-10, s=50, color ='blue')

        self.target_function = np.polyfit(np.array(self.target_position)[:, 0]/self.dimensions[0], np.array(self.target_position)[:, 1]/self.dimensions[1]*-10, self.lattice_order+1)
        self.fig_subplot.plot(self.normalized_frequencies, np.polyval(self.target_function, self.normalized_frequencies))
        self.fig_subplot.set_xlim(0, 1)
        self.fig_subplot.set_ylim(-10, 0)
        self.fig_subplot.set_xlabel("Normalized Frequencies")
        self.fig_subplot.set_ylabel("Transfer Function dB")
        self.fig_subplot.grid()
        self.canvas.draw()

    def generate_lattice(self):
        # LATTICE GENERATION
        Couplers = []
        Balance_traits = []
        Unbalance_traits = []
        for idc in range(2 * self.lattice_order + 2):
            Couplers += [Pbb.Coupler([self.c/self.frequencies[-1], self.c/self.frequencies[0]], (self.K - self.K_slope, self.K + self.K_slope))]
        for idb in range(self.lattice_order + 1):
            Balance_traits += [Pbb.Balanced_propagation(self.neff, self.ng, self.wavelength_neff, self.losses_propagation_parameter, 0)]
        for idu in range(self.lattice_order):
            Unbalance_traits += [Pbb.Unbalanced_propagation(self.neff, self.ng, self.wavelength_neff, self.losses_propagation_parameter, 0, self.dL)]
        return Pbb.Chip_structure([Couplers, Balance_traits, Unbalance_traits], ['C', 'B', 'C', 'U'] * self.lattice_order + ['C', 'B', 'C'], self.losses_coupling)

    def update_k(self, action, value, end_action=None):
        if end_action is None:
            self.K = (float(value)+0.01) * 0.98
            self.K_slope = (float( self.scrollbar2.get()[0]) - 0.5) * 2 * self.K if self.K < 0.5 else (float(float( self.scrollbar2.get()[0])) - 0.5) * 2 * (1 - self.K)
            self.scrollbar.set(value, value)
            self.scrollbar_text.config(text = f"K Value {np.round(self.K, 2)}")
            self.scrollbar_text2.config(text = f"K Slope {np.round(self.K_slope, 2)}")
            self.lattice = self.generate_lattice()
            self.update_plot()

    def update_k_slope(self, action, value, end_action=None):
        if end_action is None:
            self.K_slope = (float(value)-0.5)*2*self.K if self.K < 0.5 else (float(value)-0.5)*2*(1-self.K)
            self.scrollbar2.set(value, value)
            self.scrollbar_text2.config(text = f"K Slope {np.round(self.K_slope, 2)}")
            self.lattice = self.generate_lattice()
            self.update_plot()

    def generate_output_power(self, target_field):
        # MADSEN ALGORITHM B PART -> find the correct target B
        Target_As, Target_Bs = find_valid_b(self.lattice_order, target_field, to_plot=False)

        # MADSEN ALGORITHM for a LATTICE formed by only couplers and unbalance (BACK)
        K_estimate, Phis_estimate, As_estimate, Bs_estimate = reverse_transfer_function(Target_As, Target_Bs)

        # BEYOND MADSEN -> Re-adjust the unbalance phases to compensate the coupler and CBC difference
        afs, bfs, phases = find_correction_phase(self.K, K_estimate, Phis_estimate)

        # NEFF COMPENSATION
        Neff_shift_phis = np.zeros(len(phases))
        Neff_shift_phis[1::2] = np.ones(self.lattice_order) * 2 * np.pi * ((self.neff - self.ng) * self.dL / self.wavelength_neff + self.frequencies[0] / self.FSR)
        # Neff_shift_phis[0] = np.pi
        # Neff_shift_phis[-1] = np.pi

        # LATTICE OUTPUT CALCULATION
        self.lattice.set_heaters(phases-Neff_shift_phis, self.heater_order)
        S = self.lattice.calculate_S_matrix(self.wavelengths)
        output_power = Pbb.calculate_outputs(self.input_field, S, dB=False)[:, 0]
        return output_power

GUI = PlotGUI()
GUI.mainloop()