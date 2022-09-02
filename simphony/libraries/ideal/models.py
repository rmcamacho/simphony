# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
simphony.libraries.ideal.models
===============================
This module contains the ideal library which contains ideal PIC components.

"""
import numpy as np
from typing import Optional, Tuple
from simphony import Model


class BeamSplitter(Model):
    """This class represents an ideal lossy or lossless beam splitter.
    
    Parameters
    ----------
    theta :
        The transmittivity angle of the beam splitter. The amplitude 
        transmission of the beamsplitter :math:`t = \cos(\theta)`. By default 
        :math:`\theta = \frac{\pi}{4} which gives a 50/50 power splitting ratio.
    phi :
        The reflection phase angle of the beamsplitter. The phase of the 
        reflected port :math:`r = \sin(\theta) e^{j\phi}`. By default 
        :math:`\phi = \frac{\pi}{2}`.
    T0 :
        Power transmission coefficient of the first input port. By default there 
        is full power transmission T0 = 1.
    T1 :
        Power transmission coefficient of the second input port. By default 
        there is full power transmission T1 = 1.
    T2 :
        Power transmission coefficient of the first output port. By default 
        there is full power transmission T2 = 1.
    T3 :
        Power transmission coefficient of the second output port. By default 
        there is full power transmission T3 = 1.
    pins :
        Tuple of the names of the input and output ports. By default the pin 
        names are given by pins = ("in1", "in2", "out1", "out2").
    """
    pin_count = 4

    def __init__(
        self, 
        theta: float = np.pi / 4, 
        phi: float = np.pi / 2,
        T0: float = 1,
        T1: float = 1, 
        T2: float = 1, 
        T3: float =1, 
        pins: Optional[Tuple[str,...]] = ("in1", "in2", "out1", "out2")
    ) -> None:
        self.phi = phi
        self.theta = theta
        self.T0 = T0
        self.T1 = T1
        self.T2 = T2
        self.T3 = T3
        self.pins = pins
        super().__init__()
        self.rename_pins(*pins)

    def s_parameters(self, freqs: "np.array") -> "np.ndarray":
        t = np.cos(self.theta)
        r = np.sin(self.theta) * np.exp(1j * self.phi)
        rp = -np.conj(r)

        # fmt: off
        smatrix = np.array(
            [
                [0,                            0,                             t * np.sqrt(self.T0*self.T2), -rp * np.sqrt(self.T0*self.T3)],
                [0,                            0,                            -r * np.sqrt(self.T1*self.T2),  t * np.sqrt(self.T1*self.T3)],
                [t * np.sqrt(self.T0*self.T2), rp * np.sqrt(self.T1*self.T2), 0,                             0],
                [r * np.sqrt(self.T0*self.T3), t * np.sqrt(self.T1*self.T3),  0,                             0]
            ]
        )
        # fmt: on

        self.value = (
            (1 - self.T0)
            * np.sqrt(self.T2)
            * np.sqrt(self.T3)
            * np.cos(self.theta)
            * np.sin(self.theta)
        ) - (
            (1 - self.T1)
            * np.sqrt(self.T2)
            * np.sqrt(self.T3)
            * np.cos(self.theta)
            * np.sin(self.theta)
        )

        # the s-matrix is frequency independent,
        # so just return it for each frequency
        return np.repeat(smatrix, len(freqs)).reshape((len(freqs), 4, 4))