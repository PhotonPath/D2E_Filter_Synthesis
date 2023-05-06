"""
Code Data 2023/04/20
Authors: Mattia, Doug

Synthesis using BACKFORTH function from MADSEN
"""
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
import Photonic_building_block as Pbb
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss


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

        # Windows
        self.dimensions = (600, 600)
        self.title("Controller")
        self.geometry(f"{self.dimensions[0]}x{self.dimensions[1]}+0+0")
        self.resizable(False, False)

        self.secondary_window = tk.Toplevel(self)
        self.secondary_window.title("Lattice")
        self.secondary_window.geometry(f"{self.dimensions[0]}x{self.dimensions[1]}+600+0")
        self.secondary_window.resizable(False, False)

        # Canvas Circles
        self.canvas_dimensions = (400, 400)
        self.canvas = tk.Canvas(self, width=self.canvas_dimensions[0], height=self.canvas_dimensions[1],
                                bg="white")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        # Canvas Drawing
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.fig_subplot = self.fig.add_subplot(111)
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.secondary_window)
        self.canvas_plot.get_tk_widget().grid(row=0, column=0, sticky="nesw")

        # Circles
        self.lattice_order = 5
        self.n_circles = self.lattice_order
        self.selected_circle = 0
        self.is_circle_selected = False
        self.circles_position = [((0.45 * np.cos(t) + 1) * self.canvas_dimensions[0] / 2,
                                  (0.45 * np.sin(v) + 1) * self.canvas_dimensions[1] / 2)
                                 for (t, v) in zip(np.linspace(0, np.pi * 2,
                                                               self.n_circles, endpoint=False),
                                                   np.linspace(0, np.pi * 2,
                                                               self.n_circles, endpoint=False))]

        # Event
        self.draw_all_circles()
        self.canvas.bind('<Motion>', self.move_circle)
        self.canvas.bind('<Button-1>', self.update_selected_circle_status)

        # Lattice Stuff
        self.n_points = 500
        self.c = 299.792458  # um * THz
        self.dL = 30  # um
        self.wavelength_neff = 1.55  # um
        self.neff = 1.49  # @ wavelength_neff
        self.ng = 1.52  # @ wavelength_neff
        self.FSR = self.c / self.dL / self.ng  # THz
        self.frequencies = np.linspace(192, 192 + self.FSR, self.n_points)
        self.normalized_frequencies = np.linspace(0, 1, self.n_points, endpoint=False)
        self.wavelengths = self.c / self.frequencies
        self.K = 0.5  # No wavelength dependent coupler
        self.input_field = np.array([[1, 0], [0, 0]])
        self.losses_propagation_parameter = {
            'A': 0.4,  # dB/cm
            'B': 4655.664,  # um ** -2
            'C': 3.614e-04,  # um ** 2
            'D': 0.05,  # dB/cm
            'wl1': 1.502293,  # um
            'wl2': 1.511449  # um
        }
        self.losses_coupling = -0.5  # dB
        self.heater_order = ['B', 'U'] * self.lattice_order + ['B']
        self.lattice = self.generate_lattice()

        # Input K value in scroll bar
        self.canvas_down = tk.Canvas(self)
        self.canvas_down.grid(row=1, column=0, columnspan=2, sticky="nsew")
        self.scrollbar_text = tk.Label(self.canvas_down, text="K Value 0.50")
        self.scrollbar_text.pack(side="left")
        self.scrollbar = tk.Scrollbar(self.canvas_down,
                                      orient='horizontal', command=self.update_k)
        self.scrollbar.set(0.5, 0.5)
        self.scrollbar.pack(side="left", fill="x", expand=True)

        # Create the editable parameters column
        param_frame = tk.Frame(self)
        param_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Input lattice order
        self.lattice_order_text = tk.Label(param_frame, text="Lattice order", width=10, anchor="w", justify="left")
        self.lattice_order_text.grid(row=0, column=0, sticky="NSEW")
        self.lattice_order_input = tk.Entry(param_frame, width=10)
        self.lattice_order_input.insert(0, "{}".format(self.lattice_order))
        self.lattice_order_input.grid(row=1, column=0, padx=5, pady=5)
        self.lattice_order_input.bind('<Return>', self.update_pars)

        # Input neff
        self.neff_text = tk.Label(param_frame, text="neff", width=5, anchor="w", justify="left")
        self.neff_text.grid(row=2, column=0, sticky="NSEW")
        self.neff_input = tk.Entry(param_frame, width=10)
        self.neff_input.insert(0, "{}".format(self.neff))
        self.neff_input.grid(row=3, column=0, padx=5, pady=5)
        self.neff_input.bind('<Return>', self.update_pars)

        # Input dL
        self.dL_text = tk.Label(param_frame, text="dL", width=5, anchor="w", justify="left")
        self.dL_text.grid(row=4, column=0, sticky="NSEW")
        self.dL_input = tk.Entry(param_frame, width=10)
        self.dL_input.insert(0, "{}".format(self.dL))
        self.dL_input.grid(row=5, column=0, padx=5, pady=5)
        self.dL_input.bind('<Return>', self.update_pars)

        # Input ng
        self.ng_text = tk.Label(param_frame, text="ng", width=5, anchor="w", justify="left")
        self.ng_text.grid(row=6, column=0, sticky="NSEW")
        self.ng_input = tk.Entry(param_frame, width=10)
        self.ng_input.insert(0, "{}".format(self.ng))
        self.ng_input.grid(row=7, column=0, padx=5, pady=5)
        self.ng_input.bind('<Return>', self.update_pars)

        # Input coupling losses
        self.losses_coupling_text = tk.Label(param_frame, text="Coupling losses", width=15, anchor="w", justify="left")
        self.losses_coupling_text.grid(row=8, column=0, sticky="NSEW")
        self.losses_coupling_input = tk.Entry(param_frame, width=10)
        self.losses_coupling_input.insert(0, "{}".format(self.losses_coupling))
        self.losses_coupling_input.grid(row=9, column=0, padx=5, pady=5)
        self.losses_coupling_input.bind('<Return>', self.update_pars)

        # Button to reset circle positions
        self.reset_button = tk.Button(param_frame, text="Reset Positions", command=self.reset_circles)
        self.reset_button.grid(row=10, column=0, padx=5, pady=5)

        # Plot
        self.update_plot()

    def update_selected_circle_status(self, event):
        if "scrollbar" not in str(event.widget):
            self.is_circle_selected = not self.is_circle_selected
            if self.is_circle_selected:
                difference = np.inf
                for circle_idx, circle_position in enumerate(self.circles_position):
                    diff = (circle_position[0] - event.x) ** 2 + (circle_position[1] - event.y) ** 2
                    if diff < difference:
                        self.selected_circle = circle_idx
                        difference = diff

    def update_plot(self):
        self.fig_subplot.clear()
        circles_complex_data = [(x*4/self.canvas_dimensions[0]-2+1j * (y*4/self.canvas_dimensions[1]-2))
                                for (x, y) in self.circles_position]
        circles_polynomial_from_roots = np.poly(circles_complex_data)
        circles_polynomial_from_roots = circles_polynomial_from_roots[::-1]
        phase = np.angle(circles_polynomial_from_roots[0])
        circles_polynomial_from_roots *= np.exp(-1j*phase)
        circles_polynomial_from_roots /= np.sum(np.abs(circles_polynomial_from_roots))
        f = field_from_z_coefficients(circles_polynomial_from_roots, self.n_points)
        g = self.generate_output_power(circles_polynomial_from_roots)
        self.fig_subplot.plot(self.frequencies, 10*np.log10(np.abs(f) ** 2), label="Target Profile")
        self.fig_subplot.plot(self.frequencies, 10*np.log10(g), label="Lattice Profile")
        self.fig_subplot.grid()
        self.fig_subplot.legend()
        self.canvas_plot.draw()

    def move_circle(self, event):
        if self.is_circle_selected:
            self.canvas.delete("all")
            self.circles_position[self.selected_circle] = (event.x, event.y)
            self.draw_all_circles()
            self.update_plot()

    def draw_circle_color(self, x, y, radius, fill_color=None):
        if fill_color is not None:
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=fill_color)
        else:
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius)

    def draw_all_circles(self):
        for circle_position in self.circles_position:
            self.draw_circle_color(circle_position[0], circle_position[1], 5, "red")
        self.draw_circle_color(self.canvas_dimensions[0]/2,
                               self.canvas_dimensions[1]/2,
                               self.canvas_dimensions[0]/4)

    def generate_lattice(self):
        # LATTICE GENERATION
        Couplers = []
        Balance_traits = []
        Unbalance_traits = []
        for idc in range(2 * self.lattice_order + 2):
            Couplers += [Pbb.Coupler([1.5, 1.6], (self.K, self.K))]
        for idb in range(self.lattice_order + 1):
            Balance_traits += [Pbb.Balanced_propagation(self.neff,
                                                        self.ng,
                                                        self.wavelength_neff,
                                                        self.losses_propagation_parameter, 0)]
        for idu in range(self.lattice_order):
            Unbalance_traits += [Pbb.Unbalanced_propagation(self.neff,
                                                            self.ng,
                                                            self.wavelength_neff,
                                                            self.losses_propagation_parameter,
                                                            0, self.dL)]
        return Pbb.Chip_structure([Couplers,
                                   Balance_traits,
                                   Unbalance_traits],
                                  ['C', 'B', 'C', 'U'] * self.lattice_order + ['C', 'B', 'C'],
                                  self.losses_coupling)

    def update_k(self, action, value, end_action=None):
        if end_action is None:
            self.K = (float(value)+0.01) * 0.98
            self.scrollbar.set(value, value)
            self.scrollbar_text.config(text = f"K Value {np.round(self.K, 2)}")
            self.lattice = self.generate_lattice()
            self.update_plot()

    def update_pars(self, event):
        self.lattice_order = int(self.lattice_order_input.get())
        self.n_circles = self.lattice_order
        self.reset_circles()
        self.draw_all_circles()

        self.neff = float(self.neff_input.get())
        self.ng = float(self.ng_input.get())
        self.dL = float(self.dL_input.get())
        self.losses_coupling = float(self.losses_coupling_input.get())

        self.heater_order = ['B', 'U'] * self.lattice_order + ['B']
        self.lattice = self.generate_lattice()
        self.update_plot()

    def reset_circles(self):
        self.canvas.delete("all")
        self.circles_position = [((0.45 * np.cos(t) + 1) * self.canvas_dimensions[0] / 2,
                                  (0.45 * np.sin(v) + 1) * self.canvas_dimensions[1] / 2)
                                 for (t, v) in zip(np.linspace(0, np.pi * 2,
                                                               self.n_circles,endpoint=False),
                                                   np.linspace(0, np.pi * 2,
                                                               self.n_circles, endpoint=False))]
        self.draw_all_circles()
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

        # LATTICE OUTPUT CALCULATION
        self.lattice.set_heaters(phases-Neff_shift_phis, self.heater_order)
        S = self.lattice.calculate_S_matrix(self.wavelengths)
        output_power = Pbb.calculate_outputs(self.input_field, S, dB=False)[:, 0]

        return output_power

GUI = PlotGUI()
GUI.mainloop()