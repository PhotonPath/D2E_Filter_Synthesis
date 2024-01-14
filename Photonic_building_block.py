"""
Block to simulate all lattice elements
Author: Mattia Conti
Date 1/12/2024
"""

import numpy as np

class OpticalBlock:
    def __init__(self, wavelengths, **kwargs):
        """
        This is the general optical block. All other blocks should inherit from this block.
        :param wavelengths: wavelengths where the transfer function should be evaluated.
        :param params: all optical parameter useful to calculate the transfer function.
        """

        # Parameters
        self.__dict__.update(kwargs)
        self.wavelengths = wavelengths
        self.n_elements = len(self.wavelengths)

        # Transfer function
        self.S00 = np.array([])
        self.S01 = np.array([])
        self.S10 = np.array([])
        self.S11 = np.array([])
        self.outputs = [np.zeros(len(self.wavelengths)), np.zeros(len(self.wavelengths))]
        # Matrix convention
        # [s00 s01]
        # [s10 s11]

    def calculate_transfer_function(self):
        self.S00 = np.ones(self.n_elements)
        self.S01 = np.zeros(self.n_elements)
        self.S10 = np.zeros(self.n_elements)
        self.S11 = np.zeros(self.n_elements)

    def calculate_output(self, inputs):
        self.outputs[0] = inputs[0] * self.S00 + inputs[1] * self.S01
        self.outputs[1] = inputs[0] * self.S10 + inputs[1] * self.S11
        return self.outputs

class Coupler(OpticalBlock):
    def __init__(self, **kwargs):
        """
        This photonic block simulates the behaviour of a coupler

        INPUT1 ----         ----- OUTPUT1 (BAR)
                   >=======<
        INPUT2 ----         ----- OUTPUT2 (CROSS)

        sin(k0)**2 is the power in output if only input1 is present at wavelength0.
        Use k0 = np.pi/4 to have a 3dB coupler.
        Use k1 != from 0 to implement a first order dependence by wavelength.
        Use k2 != from 0 to implement a second order dependence by wavelength.

        :param kwargs:
        > wavelength0   = wavelength where value are calculated
        > k0            = value of k at wavelength0
        > k1            = derivative of k at wavelength0
        > k2            = second derivative of k at wavelength0
        """
        super().__init__(**kwargs)

        # Parameters
        self.k0 = kwargs.get('k0', 0.5)
        self.k1 = kwargs.get('k1', 0)
        self.k2 = kwargs.get('k2', 0)
        self.wavelength0 = kwargs.get('wavelength0', 1.55)
        self.name = "Coupler"

        # Internal parameters
        self.k = 0
        self.calculate_internal_parameters()

    def calculate_internal_parameters(self):
        self.k = self.k0 + self.k1 * (self.wavelengths - self.wavelength0) + self.k2 * (self.wavelengths - self.wavelength0) ** 2

    def calculate_transfer_function(self):
        self.S00 = np.cos(self.k)     + 0j
        self.S01 = -1j*np.sin(self.k) + 0j
        self.S10 = -1j*np.sin(self.k) + 0j
        self.S11 = np.cos(self.k)     + 0j

class DoubleWaveguide(OpticalBlock):
    """
    This photonic block simulates the behaviour of two separated waveguides, with a different in length of dL.
    If dL is positive, upper part is longer
    Heaters change only the up waveguide phase

    INPUT1 -----------------[HEATER]-------------------- OUTPUT1 (BAR)

    INPUT2 ------------------------                      OUTPUT2 (CROSS)

                                  <-------------------> is the dL

    Use neff1 != 0 to control first order derivative of effective refractive index.
    Use neff2 != 0 to control second order derivative of effective refractive index.
    Use ng != 0 as an alternative of first/second order derivative. It's the group index
    of the waveguide. Internally it calculates the neff1 to achieve this ng.

    Use 'A', 'B', 'C', 'D', 'λ1', 'λ2' to calculate propagation losses.
    If not specified, no propagation losses are applied.

    If you want to have a balance trait, leave a dL = 0 or not specify a dL.
    :param kwargs:
    > wavelength0   = wavelength where value are calculated
    > neff0         = effective refractive index at wavelength0
    > neff1         = derivative of the effective refractive index at wavelength0
    > neff2         = second derivative of the effective refractive index at wavelength0
    > L             = length of the propagation trait, in cm
    > dL            = length of unbalance between upper and lower trait, in um. Positive means longer upper trait.
    > A             = parameter of propagation losses formula
    > B             = parameter of propagation losses formula
    > C             = parameter of propagation losses formula
    > D             = parameter of propagation losses formula
    > wavelength1   = parameter of propagation losses formula
    > wavelength2   = parameter of propagation losses formula
    > heater_value  = shift in phase introduced by the heater
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Parameters
        self.neff0 = kwargs.get('neff0', 1.5)
        self.neff1 = kwargs.get('neff1', 0)
        self.neff2 = kwargs.get('neff2', 0)
        self.ng = kwargs.get('ng', None)
        self.L = kwargs.get('L', 0)
        self.dL = kwargs.get('dL', 0)
        self.A =  kwargs.get('A', 0)
        self.B =  kwargs.get('B', 1)
        self.C =  kwargs.get('C', 1)
        self.D =  kwargs.get('D', 0)
        self.wavelength0 = kwargs.get('wavelength0', 1.55)
        self.wavelength1 = kwargs.get('wavelength1', 1.50)
        self.wavelength2 = kwargs.get('wavelength2', 1.51)
        self.heater_value = kwargs.get('heater_value', 0)
        self.name = "DoubleWaveguide"

        # Internal Parameters
        self.alpha = 0
        self.alpha_lin = 0
        self.neff = 0
        self.beta = 0

        # Internal transfer function
        self.T00 = np.zeros(self.n_elements)
        self.T01 = np.zeros(self.n_elements)
        self.T10 = np.zeros(self.n_elements)
        self.T11 = np.zeros(self.n_elements)

        self.calculate_internal_parameters()
        self.calculate_internal_transfer_function()

    def calculate_internal_parameters(self):
        # Refractive index
        if self.ng is not None:
            self.neff1 = (self.neff0 - self.ng) / self.wavelength0
        self.neff = self.neff0 + self.neff1 * (self.wavelengths - self.wavelength0) + self.neff2 * (self.wavelengths - self.wavelength0) ** 2
        self.beta = 2 * np.pi * self.neff / self.wavelengths

        # Losses
        self.alpha = (self.A - self.D) / 2 * ((1 / (1 + self.B * (self.wavelengths - self.wavelength1) ** 2))
                                              + np.exp(-(self.wavelengths - self.wavelength2) ** 2 / self.C)) + self.D
        self.alpha_lin = self.alpha * np.log(10) / 20

    def calculate_internal_transfer_function(self):
        if self.dL >= 0:
            self.T00 = np.exp(-1j * self.beta * (self.L * 1e4 + self.dL) - self.alpha_lin * (self.L + self.dL / 1e4))
            self.T11 = np.exp(-1j * self.beta * self.L * 1e4 - self.alpha_lin * self.L)
        else:
            self.T00 = np.exp(-1j * self.beta * self.L * 1e4 - self.alpha_lin * self.L)
            self.T11 = np.exp(-1j * self.beta * (self.L * 1e4 + self.dL) - self.alpha_lin * (self.L + self.dL / 1e4))

        self.T01 = np.zeros(self.n_elements)
        self.T10 = np.zeros(self.n_elements)

        self.S00 = self.T00 + 0j
        self.S01 = self.T01 + 0j
        self.S10 = self.T10 + 0j
        self.S11 = self.T11 + 0j

    def calculate_transfer_function(self):
        self.S00 = self.T00 * np.exp(-1j * self.heater_value)

    def calculate_ng(self):
        self.ng = self.neff0 - self.neff1 * self.wavelength0
        return self.ng

    def set_heater(self, heater_value):
        self.heater_value = heater_value
        self.calculate_transfer_function()

class RingResonator(OpticalBlock):
    def __init__(self, **kwargs):
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

        Use neff1 != 0 to control first order derivative of effective refractive index.
        Use neff2 != 0 to control second order derivative of effective refractive index.
        Use ng != 0 as an alternative of first/second order derivative. It's the group index
        of the waveguide. Internally it calculates the neff1 to achieve this ng.
        Use k1 != from 0 to implement a first order dependence by wavelength of ring coupler.
        Use k2 != from 0 to implement a second order dependence by wavelength of ring coupler.
        Use gamma0 to introduce losses per round trip in dB.
        Use gamma1 != from 0 to implement a first order dependence by wavelength of losses.
        Use gamma2 != from 0 to implement a second order dependence by wavelength of losses.
        :param kwargs:
        > wavelength0   = wavelength where value are calculated
        > neff0         = effective refractive index at wavelength0
        > neff1         = derivative of the effective refractive index at wavelength0
        > neff2         = second derivative of the effective refractive index at wavelength0
        > radius        = length of radius of the ring, in um
        > k0            = value of k at wavelength0
        > k1            = derivative of k at wavelength0
        > k2            = second derivative of k at wavelength0
        > gamma0        = value of round trip losses in dB at wavelength0
        > gamma1        = derivative of round trip losses at wavelength0
        > gamma2        = second derivative of round trip losses at wavelength0
        > heater_value  = shift in phase introduced by the heater
        """
        super().__init__(**kwargs)

        # Parameters
        self.ng = kwargs.get('ng', None)
        self.neff0 = kwargs.get('neff0', 1.5)
        self.neff1 = kwargs.get('neff1', 0)
        self.neff2 = kwargs.get('neff2', 0)
        self.radius = kwargs.get('radius', 1)
        self.gamma0 = kwargs.get('gamma0', 1)
        self.gamma1 = kwargs.get('gamma1', 0)
        self.gamma2 = kwargs.get('gamma2', 0)
        self.k0 = kwargs.get('k0', 0.5)
        self.k1 = kwargs.get('k1', 0)
        self.k2 = kwargs.get('k2', 0)
        self.wavelength0 = kwargs.get('wavelength0', 1.55)
        self.heater_value = kwargs.get('heater_value', 0)
        self.name = "RingResonator"

        # Internal Parameters
        self.circumference = self.radius * 2 * np.pi
        self.gamma_lin = 0
        self.gamma = 0
        self.neff = 0
        self.beta = 0
        self.k = 0

        self.calculate_internal_parameters()

    def calculate_internal_parameters(self):
        # Refractive index
        if self.ng is not None:
            self.neff1 = (self.neff0 - self.ng) / self.wavelength0
        self.neff = self.neff0 + self.neff1 * (self.wavelengths - self.wavelength0) + self.neff2 * (self.wavelengths - self.wavelength0) ** 2
        self.beta = 2 * np.pi * self.neff / self.wavelengths

        # Coupler
        self.k = self.k0 + self.k1 * (self.wavelengths - self.wavelength0) + self.k2 * (self.wavelengths - self.wavelength0) ** 2

        # Losses
        self.gamma = self.gamma0 + self.gamma1 * (self.wavelengths - self.wavelength0) + self.gamma2 * (self.wavelengths - self.wavelength0) ** 2
        self.gamma_lin = self.gamma * np.log(10) / 20

    def calculate_transfer_function(self):
        prop_term = self.gamma_lin + 1j * (self.beta * self.circumference + self.heater_value)
        self.S00 = np.sqrt(1-self.k**2) * (1 - np.exp(-prop_term)) / (1 - np.exp(-prop_term) * (1-self.k**2))
        self.S11 = self.S00.copy()
        self.S01 = self.k**2 * np.sqrt(np.exp(-prop_term)) / (1 - np.exp(-prop_term) * (1-self.k**2))
        self.S10 = self.S01.copy()

    def set_heater(self, heater_value):
        self.heater_value = heater_value
        self.calculate_transfer_function()

    def calculate_ng(self):
        self.ng = self.neff0 - self.neff1 * self.wavelength0
        return self.ng

class WaveguideChoice(OpticalBlock):
    def __init__(self, **kwargs):
        """
        This photonic block selects an input and lets only that input to exit from the wanted output.
        The following drawing is only one of the four possible examples.

        INPUT1 -------------X

        INPUT2 ------------------------ OUTPUT1 (BAR)

                                        OUTPUT2 (CROSS)
        :param kwargs:
        >input_choice: # an integer that selects the input that passes the Interrupt block.
                       0 from up waveguide, 1 from down waveguide.
        >output_choice: # an integer that selects the outputs that exits from the Interrupt block.
                       0 from up waveguide, 1 from down waveguide.
        """
        super().__init__(**kwargs)

        # PARAMETERS
        self.input_choice = kwargs.get('input_choice', 0)
        self.output_choice = kwargs.get('output_choice', 0)

    def calculate_transfer_function(self):
        self.S00 = np.ones(self.n_elements) * (1 - self.input_choice) * (1 - self.output_choice) + 0j
        self.S01 = np.ones(self.n_elements) * self.input_choice * (1 - self.output_choice) + 0j
        self.S10 = np.ones(self.n_elements) * (1 - self.input_choice) * self.output_choice + 0j
        self.S11 = np.ones(self.n_elements) * self.input_choice * self.output_choice + 0j

class WaveguideFacet(OpticalBlock):
    """
    This photonic block simulates the behaviour of light entering/exiting from a waveguide.
    It generates constant losses.

    :param kwargs:
    > coupling_losses: coupling_losses in dB
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Parameters
        self.coupling_losses = kwargs.get('coupling_losses', 1)
        self.field_losses = 0
        self.name = "WaveguideFacet"

        self.calculate_internal_parameters()

    def calculate_internal_parameters(self):
        self.field_losses = 10 ** (self.coupling_losses / 20)

    def calculate_transfer_function(self):
        self.S00 = np.ones(self.n_elements) * self.field_losses + 0j
        self.S01 = np.zeros(self.n_elements)                    + 0j
        self.S10 = np.zeros(self.n_elements)                    + 0j
        self.S11 = np.ones(self.n_elements) * self.field_losses + 0j

class ChipStructure:
    """
    It's a chip that put together different optical block together.
    You have to provide just the list of structures in a dictionary.
    :param structures = is a dictionary. The key is the index associated with the order when
    the optical block consider is meet from the waveguide facet. The item is the optical block.

    """
    def __init__(self, structures):

        # Parameters
        self.structures = structures

        # Transfer Function
        self.S00 = []
        self.S01 = []
        self.S10 = []
        self.S11 = []

        # Outputs
        self.outputs = [0, 0]

    def calculate_internal_transfer_function(self):
        for idx in range(len(self.structures)):
            self.structures[idx].calculate_transfer_function()
    def calculate_transfer_function(self):
        for idx in reversed(range(len(self.structures))):
            if idx == len(self.structures) - 1:
                self.S00 = self.structures[idx].S00
                self.S01 = self.structures[idx].S01
                self.S10 = self.structures[idx].S10
                self.S11 = self.structures[idx].S11
            else:
                s00 = self.structures[idx].S00 * self.S00 + self.structures[idx].S01 * self.S10
                s01 = self.structures[idx].S00 * self.S01 + self.structures[idx].S01 * self.S11
                s10 = self.structures[idx].S10 * self.S00 + self.structures[idx].S11 * self.S10
                s11 = self.structures[idx].S10 * self.S01 + self.structures[idx].S11 * self.S11
                self.S00 = s00
                self.S01 = s01
                self.S10 = s10
                self.S11 = s11

    def calculate_output(self, inputs):
        self.outputs[0] = inputs[0] * self.S00 + inputs[1] * self.S10
        self.outputs[1] = inputs[0] * self.S01 + inputs[1] * self.S11
        return self.outputs

    def set_heaters(self, heater_values):
        for idx, heater_value in heater_values.items():
            self.structures[idx].set_heater(heater_value)