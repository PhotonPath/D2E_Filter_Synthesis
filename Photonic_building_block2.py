"""
Block to simulate all lattice elements
Author: Mattia Conti
Date
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class Coupler:
    def __init__(self, ref_wl, ref_k):
        """
        This photonic block simulates the behaviour of a coupler

        INPUT1 ----         ----- OUTPUT1 (BAR)
                   >=======<
        INPUT2 ----         ----- OUTPUT2 (CROSS)

        General case: K = 1 - (k_ / δ sin(δ z + phi)) ^ 2 -> See Morichetti's book
        Analyzing simulation, by using k_ = δ, the error is under 0.2 %, and it is difficult to know the parameters
        with such an accuracy, so it is used the case k_ = δ In that case: K = cos (k_ z + phi) ^ 2
        Considering a linear dependence of the parameter involve in wavelength,
        the formula can be described by two parameter a, b:
        K = cos(a + b(λ - λ0))^2
        a and b are derived, given two reference wavelengths, at λ0 and at another wavelength
        :param ref_wl:   # Reference Wavelengths λ0 and λ1. THEY MUST BE CLOSE ENOUGH TO BE ON THE SAME PERIOD
        :param ref_k:    # Reference Coupler Ratio K evaluated at λ0 and λ1
        """
        # INPUT PARAMETERS
        self.ref_wl = ref_wl
        self.ref_K = ref_k
        self.a = 0
        self.b = 0
        self.calculate_ab()

        # TRANSFER FUNCTION
        self.S = []

    def calculate_ab(self):
        # OLD
        # self.a = np.arccos(np.sqrt(self.ref_K[0]))
        # self.b = (np.arccos(np.sqrt(self.ref_K[1])) - self.a) / (self.ref_wl[1] - self.ref_wl[0])
        # NEW
        self.a = np.arcsin(np.sqrt(self.ref_K[0]))
        self.b = (np.arcsin(np.sqrt(self.ref_K[1])) - self.a) / (self.ref_wl[1] - self.ref_wl[0])

    def calculate_S_matrix(self, wl):
        x = self.a + self.b * (wl - self.ref_wl[0])
        # OLD
        # self.S = np.array([[np.sin(x),    -1j * np.cos(x)],
        #                   [-1j * np.cos(x), np.sin(x)]])
        # NEW
        self.S = np.array([[np.cos(x),    -1j * np.sin(x)],
                          [-1j * np.sin(x), np.cos(x)]])
        return self.S


class Balanced_propagation:
    def __init__(self, neff, ng, wl0, losses_parameter, length):
        """
        This photonic block simulates the behaviour of two separated waveguides, with the same length.
        Heater changes only the up waveguide phase.

        INPUT1 --------[HEATER]-------- OUTPUT1 (BAR)

        INPUT2 ------------------------ OUTPUT2 (CROSS)

        :param neff: # Effective refractive index of the waveguide
        :param ng: # Group refractive index of the waveguide
        :param wl0:  # Central wavelength used in calculation, in um
        :param losses_parameter: # Losses parameter used to calculate propagation losses.
        :param length: # Length of the propagation trait, in cm
        """
        # INPUT PARAMETERS
        self.neff = neff
        self.ng = ng
        self.wl0 = wl0
        self.losses_parameter = losses_parameter
        self.L = length

        # HEATER CONTRIBUTION
        self.dphi_heater = 0                         # Phase variation introduced by the heater

        # CALCULATED PARAMETERS
        self.alpha = 0                               # Propagation losses, in dB/cm
        self.alpha_neper = 0                         # Propagation losses, in neper/cm
        self.dn = 0
        self.beta = 0
        self.calculate_dn()

        # TRANSFER FUNCTION
        self.S = []

    def calculate_dn(self):
        # Derivative of effective refractive index as a function of wavelength, in 1 / um
        self.dn = -(self.ng - self.neff) / self.wl0

    def set_heater(self, dphi):
        self.dphi_heater = dphi

    def get_heater(self):
        return self.dphi_heater

    def preliminary_losses(self, wl):
        self.alpha = (self.losses_parameter['A'] - self.losses_parameter['D']) / 2 * (
                    (1 / (1 + self.losses_parameter['B'] * (wl - self.losses_parameter['wl1']) ** 2))
                    + np.exp(-(wl - self.losses_parameter['wl2']) ** 2 / self.losses_parameter['C'])) \
                     + self.losses_parameter['D']
        self.alpha_neper = self.alpha * np.log(10) / 20

        n = self.neff + self.dn * (wl - self.wl0)  # Calculate the effective index at current wavelength
        self.beta = 2 * np.pi * n / wl  # Wave vector

    def calculate_S_matrix(self, wl):
        self.preliminary_losses(wl)
        self.S = np.array([[np.exp(-1j * (self.beta * self.L * 1e4 + self.dphi_heater))
                            * np.exp(-self.alpha_neper * self.L), np.zeros(len(wl))],
                           [np.zeros(len(wl)), np.exp(-1j * (self.beta * (self.L * 1e4)))
                            * np.exp(-self.alpha_neper * self.L)]])
        return self.S


class Unbalanced_propagation(Balanced_propagation):
    def __init__(self, neff, ng, wl0, losses_parameter, length, dl):
        """
        This photonic block simulates the behaviour of two separated waveguides, with a different in length of dL.
        If dL is positive, upper part is longer
        Heaters change only the up waveguide phase

        INPUT1 -----------------[HEATER]-------------------- OUTPUT1 (BAR)

        INPUT2 ------------------------ OUTPUT2 (CROSS)
                                       <-------------------> is the dL

        :param neff: # Effective refractive index of the waveguide
        :param ng: # Group refractive index of the waveguide
        :param wl0:  # Central wavelength used in calculation, in um
        :param losses_parameter: # Losses paremter used to calculate propagation losses.
        :param length: # Length of the propagation trait, in cm
        :param dl: # Difference in length between upper and lower waveguide, in um
        """
        # THE UNBALANCED PROPAGATION INHERIT METHODS FROM THE BALANCED PROPAGATION CLASS
        super().__init__(neff, ng, wl0, losses_parameter, length)
        self.dL = dl

    def calculate_S_matrix(self, wl):
        self.preliminary_losses(wl)
        self.S = np.array(
            [[np.exp(-1j * (self.beta * (self.L * 1e4 + self.dL) + self.dphi_heater))
              * np.exp(-self.alpha_neper * (self.L + self.dL / 1e4)), np.zeros(len(wl))],
             [np.zeros(len(wl)), np.exp(-1j * (self.beta * (self.L * 1e4)))
              * np.exp(-self.alpha_neper * self.L)]])

        return self.S


class Interrupt:
    def __init__(self, input_choice, output_choice):
        """
        This photonic block selects an input and lets only that input to exit from the wanted output.
        The following drawing is only one of the four possible examples.

        INPUT1 -------------X

        INPUT2 ------------------------ OUTPUT1 (BAR)

                                        OUTPUT2 (CROSS)

        :param input_choice: # an integer that selects the input that passes the Interrupt block.
                             0 from up waveguide, 1 from down waveguide.
        :param output_choice: # an integer that selects the outputs that exits from the Interrupt block.
                             0 from up waveguide, 1 from down waveguide.
        """

        # PARAMETERS
        self.input_choice = input_choice
        self.output_choice = output_choice

        # TRANSFER FUNCTION
        self.S = []

    def calculate_S_matrix(self, wl):
        n = len(wl)
        if self.input_choice == 0 and self.output_choice == 0:
            self.S = np.array([[np.ones(n), np.zeros(n)],
                               [np.zeros(n), np.zeros(n)]])
        elif self.input_choice == 1 and self.output_choice == 0:
            self.S = np.array([[np.zeros(n), np.ones(n)],
                               [np.zeros(n), np.zeros(n)]])
        elif self.input_choice == 0 and self.output_choice == 1:
            self.S = np.array([[np.zeros(n), np.zeros(n)],
                               [np.ones(n), np.zeros(n)]])
        elif self.input_choice == 1 and self.output_choice == 1:
            self.S = np.array([[np.zeros(n), np.zeros(n)],
                               [np.zeros(n), np.ones(n)]])

        return self.S


class Ring:
    def __init__(self, ref_wl, ref_k, neff, ng, wl0, losses_parameter, ring_excess_losses, ring_l):
        """
        This photonic block simulates the behaviour of a ring resonator.
        Inputs and outputs refers to the following drawings.

        INPUT1 ---->>>------------------ OUTPUT2
                         ******
                      ***      ***
                    **            **
                   *                *
                   *                *
                   *                *
                    **            **
                      ***      ***
                         ******
        OUTPUT1 ---<<<------------------- INPUT2

        # BAR = DROP & CROSS = THROUGH

        The block considers a constant loss for round trip (not wavelength dependent)
        The block considers a coupler coefficient linearly dependent by wavelength
        The block considers same coupling for top and bottom waveguide
        The block does not consider the additional phase of ring coupling

        :param ref_wl: # Reference Wavelengths λ0 and λ1. THEY MUST BE CLOSE ENOUGH TO BE ON THE SAME PERIOD
        :param ref_k:  # Reference Coupler Ratio K evaluated at λ0 and λ1
        :param neff: # Effective refractive index of the waveguide
        :param ng: # Group refractive index of the waveguide
        :param wl0: # Central wavelength used in calculation, in um
        :param losses_parameter: # Losses paremter used to calculate propagation losses.
        :param ring_excess_losses: # Excess loss that are introduced every time the light is coupled in the ring
        :param ring_l: # length of the ring, in um
        """

        # INPUT PARAMETERS
        self.ref_wl = ref_wl
        self.ref_K = ref_k
        self.neff = neff
        self.ng = ng
        self.wl0 = wl0
        self.losses_parameter = losses_parameter
        self.ring_excess_losses = ring_excess_losses
        self.ring_l = ring_l

        # CALCULATED PARAMETERS
        self.a = 0
        self.b = 0
        self.calculate_ab()

        self.dn = 0
        self.calculate_dn()

        self.alpha = 0
        self.alpha_neper = 0
        self.dphi_heater = 0

        # TRANSFER FUNCTION
        self.S = []

    def calculate_ab(self):
        self.a = np.arccos(np.sqrt(self.ref_K[0]))
        self.b = (np.arccos(np.sqrt(self.ref_K[1])) - self.a) / (self.ref_wl[1] - self.ref_wl[0])

    def calculate_dn(self):
        # Derivative of effective refractive index as a function of wavelength, in 1 / um
        self.dn = -(self.ng - self.neff) / self.wl0

    def calculate_losses(self, wl):
        self.alpha = (self.losses_parameter['A'] - self.losses_parameter['D']) / 2 * (
                    (1 / (1 + self.losses_parameter['B'] * (wl - self.losses_parameter['wl1']) ** 2))
                    + np.exp(-(wl - self.losses_parameter['wl2']) ** 2 / self.losses_parameter['C'])) \
                     + self.losses_parameter['D']
        self.alpha_neper = self.alpha * np.log(10) / 20 / 10000

    def set_heater(self, dphi):
        self.dphi_heater = dphi

    def calculate_S_matrix(self, wl):
        self.calculate_losses(wl)

        # Calculate the effective index at different wavelengths
        n = self.neff + self.dn * (wl - self.wl0)

        # Calculate the propagation constant
        beta = 2 * np.pi * n / wl

        # Calculate the coupling coefficient at different wavelengths
        K = np.cos(self.a + self.b * (wl - self.ref_wl[0])) ** 2

        # Ring excess losses
        alpha = np.exp(self.ring_excess_losses * np.log(10) / 20)
        prop_term = np.exp(1j * (beta * self.ring_l + self.dphi_heater)) * np.exp(-self.alpha_neper * self.ring_l)
        semi_propagation_term = np.exp(1j * (beta * self.ring_l + self.dphi_heater) / 2) * np.exp(-self.alpha_neper
                                                                                                  * self.ring_l / 2)

        self.S = np.array([[- K * alpha ** 2 * semi_propagation_term/(1 - (1 - K)*alpha**2*prop_term),
                            -alpha*np.sqrt(1 - K)*(1 - alpha**2 * prop_term)/(1 - (1 - K)*alpha**2*prop_term)],
                           [-alpha*np.sqrt(1 - K)*(1 - alpha**2 * prop_term)/(1 - (1 - K)*alpha**2*prop_term),
                            - K * alpha ** 2 * semi_propagation_term/(1 - (1 - K)*alpha**2*prop_term), ]])

        return self.S


class Chip_facet:
    def __init__(self, cl):
        """
        This photonic block simulates the facet between an input fiber and the chip.
        This introduced a constant coupling losses.
        :param cl: Constant coupling loss per facet in dB.
        """
        self.cl = cl
        self.S = []

    def calculate_S_matrix(self, wl):
        n = len(wl)
        field_loss = 10**(self.cl / 20)
        self.S = np.array([[np.ones(n) * field_loss, np.zeros(n)],
                          [np.zeros(n), np.ones(n) * field_loss]])
        return self.S


class Chip_structure:
    def __init__(self, structures, sorting_elements, cl=0):
        """
        This chip structure join together the previous photonic blocks.
        :param structures: list of [couplers, balanced traits, unbalanced traits, interrupts, rings] met by the light.
        All coupler blocks are in the first element, all balanced traits in the second element and so on...
        If more than one coupler block is present, they must be sorted from the first coupler met by the light.
        :param sorting_elements: letter array to understand the order of the elements. Ex ['C','B','C'] for Balance MZI
        :param cl: coupling losses per facet of the chip in dB
        """

        n = len(structures)
        self.couplers = structures[0]               # 'C'
        self.n_components = len(structures[0])

        if n > 1:
            self.bal_traits = structures[1]         # 'B'
            self.n_components += len(structures[1])
        else:
            self.bal_traits = []

        if n > 2:
            self.unbal_traits = structures[2]       # 'U'
            self.n_components += len(structures[2])
        else:
            self.unbal_traits = []

        if n > 3:
            self.interrupts = structures[3]         # 'I'
            self.n_components += len(structures[3])
        else:
            self.interrupts = []

        if n > 4:
            self.rings = structures[4]              # 'R'
            self.n_components += len(structures[4])
        else:
            self.rings = []

        self.chip_interface = Chip_facet(cl)
        self.sorting_elements = sorting_elements

        number_of_heater = len(self.bal_traits) + len(self.unbal_traits) + len( self.rings)
        self.conversion = np.ones(number_of_heater)
        self.phase_0 = np.zeros(number_of_heater)

        self.S = []

    def set_voltage_phase_parameter(self, conversion, phase0):
        self.conversion = conversion
        self.phase_0 = phase0

    def set_heaters(self, dphis, heaters_order):
        B_number = 0
        U_number = 0
        R_number = 0
        for idx, ido in enumerate(heaters_order):
            if ido == 'B':
                self.bal_traits[B_number].set_heater(dphis[idx])
                B_number += 1
            elif ido == 'U':
                self.unbal_traits[U_number].set_heater(dphis[idx])
                U_number += 1
            elif ido == 'R':
                self.rings[R_number].set_heater(dphis[idx])
                R_number += 1

    def get_heaters(self):
        dphis = []
        elements = []
        B_number = 0
        U_number = 0
        R_number = 0
        for ido in self.sorting_elements:
            if ido == 'B':
                dphis += [self.bal_traits[B_number].get_heater()]
                B_number += 1
                elements += ido
            elif ido == 'U':
                dphis += [self.unbal_traits[U_number].get_heater()]
                U_number += 1
                elements += ido
            elif ido == 'R':
                dphis += [self.rings[R_number].get_heater()]
                R_number += 1
                elements += ido
        return elements, dphis

    def set_heaters_by_voltage(self, voltage_square, heaters_order):
        dphis = voltage_square * self.conversion + self.phase_0
        self.set_heaters(dphis, heaters_order)

    def calculate_S_matrix(self, wl):
        C_number = 0
        B_number = 0
        U_number = 0
        I_number = 0
        R_number = 0
        S_matrices = []

        for ido in self.sorting_elements:
            if ido == 'C':
                S_matrices += [self.couplers[C_number].calculate_S_matrix(wl)]
                C_number += 1
            elif ido == 'B':
                S_matrices += [self.bal_traits[B_number].calculate_S_matrix(wl)]
                B_number += 1
            elif ido == 'U':
                S_matrices += [self.unbal_traits[U_number].calculate_S_matrix(wl)]
                U_number += 1
            elif ido == 'I':
                S_matrices += [self.interrupts[I_number].calculate_S_matrix(wl)]
                I_number += 1
            elif ido == 'R':
                S_matrices += [self.rings[R_number].calculate_S_matrix(wl)]
                R_number += 1

        S_CL = self.chip_interface.calculate_S_matrix(wl)

        S = S_CL.transpose()
        for ido in reversed(range(self.n_components)):
            S = S_matrices[ido].transpose() @ S
        S = S_CL.transpose() @ S

        self.S = S.transpose()
        return self.S

    def calculate_S_matrix_fast(self, wl):
        # Initialization part
        C_number = 0
        B_number = 0
        U_number = 0
        I_number = 0
        R_number = 0

        S_CL = self.chip_interface.calculate_S_matrix(wl)

        # S calculated matrix
        S = S_CL.transpose()
        for ido in reversed(self.sorting_elements):
            if ido == 'C':
                S_matrix = self.couplers[-1-C_number].S
                C_number += 1
            elif ido == 'B':
                S_matrix = self.bal_traits[-1-B_number].calculate_S_matrix(wl)
                B_number += 1
            elif ido == 'U':
                S_matrix = self.unbal_traits[-1-U_number].calculate_S_matrix(wl)
                U_number += 1
            elif ido == 'I':
                S_matrix = self.interrupts[-1-I_number].S
                I_number += 1
            elif ido == 'R':
                S_matrix = self.rings[-1-R_number].calculate_S_matrix(wl)
                R_number += 1
            S = S_matrix.T @ S
        self.S = S.transpose()

        # Return
        return self.S

class Lattice(Chip_structure):
    def __init__(self, coupler_args, waveguide_args, mzi_args, prop_loss_args, lattice_order):
        """

        :param coupler_args: contains a dictionary with two voices.
                            'wavelength_coupler_reference' contains the reference wavelength λ0 and λ1 in um
                            'coupler_ratio_at_reference' contains the reference coupler ration K at λ0 and λ1
        :param waveguide_args: contains a dictionary with four voices.
                            'neff' the effective refractive index
                            'ng' the group refractive index
                            'wl0' the reference wavelength λ0 in um
                            'cl' the coupling losses per facet in dB
        :param mzi_args: contains a dictionary with three voices.
                            'L_bal' The length of the balanced waveguide (even approximated) in cm
                            'L_unbal' The length of the unbalanced waveguide (even approximated) in cm
                            'dL' The Length of the unbalanced trait (precise) in um
        :param prop_loss_args: contains a dictionary with 6 parameters used to calculate the propagation losses.
                            'A', 'B', 'C', 'D', 'wl1', 'wl2'
        :param lattice_order: an integer that correspond to the order of the lattice
        """
        couplers = [Coupler(coupler_args['wavelength_coupler_reference'],
                            coupler_args['coupler_ratio_at_reference'])] * (lattice_order + 1) * 2

        balanced_traits = []
        unbalanced_traits = []
        for idx in range(lattice_order):
            balanced_traits += [Balanced_propagation(waveguide_args['neff'],
                                                     waveguide_args['ng'],
                                                     waveguide_args['wl0'],
                                                     prop_loss_args,
                                                     mzi_args['L_bal'])]
            unbalanced_traits += [Unbalanced_propagation(waveguide_args['neff'],
                                                         waveguide_args['ng'],
                                                         waveguide_args['wl0'],
                                                         prop_loss_args,
                                                         mzi_args['L_unbal'],
                                                         mzi_args['dL'])]
        balanced_traits += [Balanced_propagation(waveguide_args['neff'],
                                                 waveguide_args['ng'],
                                                 waveguide_args['wl0'],
                                                 prop_loss_args,
                                                 mzi_args['L_bal'])]

        order = ['C', 'B', 'C', 'U'] * lattice_order + ['C', 'B', 'C']

        self.lattice_order = lattice_order
        self.sorting_heaters = ['B', 'U'] * lattice_order + ['B']

        super().__init__([couplers, balanced_traits, unbalanced_traits], order, waveguide_args['cl'])


BAR = 1
CROSS = 2
BAR_CROSS = 3


def plot(wls, inputs, s_matrix, show=True, what_to_plot=BAR_CROSS, label=""):
    Outputs = calculate_outputs(inputs, s_matrix)
    plt.figure(1)
    if what_to_plot == BAR:
        plt.plot(wls, Outputs[:, 0], label='Bar' + label)
    elif what_to_plot == CROSS:
        plt.plot(wls, Outputs[:, 1], label='Cross' + label)
    elif what_to_plot == BAR_CROSS:
        # ax = plt.gca()
        # color = next(ax._get_lines.prop_cycler)['color']
        # plt.plot(wls, Outputs[:, 0], label='Bar' + label, color=color)
        # plt.plot(wls, Outputs[:, 1], label='Cross' + label, color=color)

        plt.plot(wls, Outputs, label=('Bar' + label, 'Cross' + label))
    plt.title("Single Polarization")
    plt.xlabel('Wavelength (um)')
    plt.ylabel('Transmission (dB)')
    plt.legend()
    if show:
        plt.grid()
        plt.show()


def plot_birefringence(wls, inputs, s_matrix_te, s_matrix_tm, p_te=0.5, show=True, what_to_plot=BAR_CROSS, label=""):
    Outputs = calculate_outputs_birefringence(inputs, s_matrix_te, s_matrix_tm, p_te)
    plt.figure(2)
    if what_to_plot == BAR:
        plt.plot(wls, Outputs[:, 0], label='Bar' + label)
    elif what_to_plot == CROSS:
        plt.plot(wls, Outputs[:, 1], label='Cross' + label)
    elif what_to_plot == BAR_CROSS:
        plt.plot(wls, Outputs, label=('Bar' + label, 'Cross' + label))
    plt.title("Scrambled Polarization")
    plt.xlabel('Wavelength (um)')
    plt.ylabel('Transmission (dB)')
    plt.legend()
    if show:
        plt.grid()
        plt.show()


def calculate_outputs(inputs, s_matrix, dB=True):
    Outputs = (inputs.T @ s_matrix.T).T
    Outputs_field = Outputs[:, 0].T
    if dB:
        Outputs_power_dB = 10 * np.log10(np.abs(Outputs_field) ** 2)
        return Outputs_power_dB
    else:
        Outputs_power_linear = np.abs(Outputs_field) ** 2
        return Outputs_power_linear


def calculate_field_outputs(inputs, s_matrix):
    Outputs = (inputs.transpose() @ s_matrix.transpose()).transpose()
    Outputs_field = Outputs[:, 0].transpose()
    return Outputs_field


def calculate_outputs_birefringence(inputs, s_matrix_te, s_matrix_tm, p_te=0.5, dB=True):
    Outputs_TE_linear = calculate_outputs(inputs, s_matrix_te, dB=False)
    Outputs_TM_linear = calculate_outputs(inputs, s_matrix_tm, dB=False)
    outputs_lin = Outputs_TE_linear * p_te + Outputs_TM_linear * (1 - p_te)

    if dB:
        Outputs_power_dB = 10 * np.log10(outputs_lin)
        return Outputs_power_dB
    else:
        return outputs_lin
