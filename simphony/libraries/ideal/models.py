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
        T: Tuple[float, float, float, float] = (1.0,1.0,1.0,1.0),
        pins: Optional[Tuple[str,...]] = ("in1", "in2", "out1", "out2")
    ) -> None:
        self.phi = phi
        self.theta = theta
        self.T0, self.T1, self.T2, self.T3 = T
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
                [0,                            0,                             t * np.sqrt(self.T0*self.T2), rp * np.sqrt(self.T0*self.T3)],
                [0,                            0,                            r * np.sqrt(self.T1*self.T2),  t * np.sqrt(self.T1*self.T3)],
                [t * np.sqrt(self.T0*self.T2), rp * np.sqrt(self.T1*self.T2), 0,                             0],
                [r * np.sqrt(self.T0*self.T3), t * np.sqrt(self.T1*self.T3),  0,                             0]
            ]
        )
        # fmt: on

        #value seems to be an indicator of non-idealities, like if theta is not 0 or 90, and if
        # the transmissions aren't lossless and aren't equal. Why do you want this number?
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
        return np.repeat(smatrix, len(freqs)).reshape((len(freqs), self.pin_count, self.pin_count))

class Waveguide(Model):
    """
    This class represents an ideal lossy or lossless waveguide
    
    Parameters
    ----------
    
    T0:
        Power transmission coefficient of the first input port. By default there is full power transmission, so T0 = 1
    T1:
        Power transmission coefficient of the first output port. By default there is full power transmission, so T1 = 1
    length:
        The length of the waveguide, which determines transmission time and phase change.
    """
    pin_count = 2
    
    def __init__(
        self, 
        #name: str = "", *, 
        #freq_range: Optional[Tuple[Optional[float], Optional[float]]] = None, 
        length: int = 0,
        T: Tuple[float,...] = (1.0,1.0),
        pins: Optional[Tuple[str,...]] = ("in1", "out1")
        ) -> None:
        self.length = length
        self.pins = pins
        self.T0, self.T1 = T
        super().__init__()
        self.rename_pins(*pins)
        
    #how is freqs used, passed in, accessed?
    def s_parameters(self, freqs: "np.array") -> "np.array":

        #I need to do the phase change calculation based on frequency, how do
        #I access freqs and then return an smatrix for each? Do I call
        # return Function(), where the function calculates it based on frequency?

        
        #for freq in freqs:
        k = 2*np.pi/freqs
        phase_change = np.exp(1j *k*self.length)
        smatrix = np.zeros((len(freqs), self.pin_count, self.pin_count),dtype = 'complex_')
        s01 = self.T0*phase_change
        s10 = self.T1*phase_change
        smatrix[:, 0, 1] = s01
        smatrix[:, 1, 0] = s10

        
        
        #returns the 3D array of smatrices
        return smatrix

class YBranch(Model):
    """
    This class represents an ideal lossy or lossless waveguide
    
    Parameters
    ----------
    
    T0:
        Power transmission coefficient of the first input port. By default there is full power transmission, so T0 = 1
    T1:
        Power transmission coefficient of the first output port. By default there is full power transmission, so T1 = 1
    T2:
        Power transmission coefficient of the second output port. By default there is full power transmission, so T2 = 1

    """
    pin_count = 3
    
    def __init__(
        self, 
        #name: str = "", *, 
        #freq_range: Optional[Tuple[Optional[float], 
        #Optional[float]]] = None, 
        #pins: Optional[List[Pin]] = None
        T: Tuple[float,...] = (1.0,1.0,1.0),
        pins: Optional[Tuple[str,...]] = ("in1", "out1", "out2"),
        #passing in parameters... 
        splittingRatio: int = .5      
        ) -> None:
        self.pins = pins
        self.T0, self.T1, self.T2 = T
        self.splittingRatio = splittingRatio
        super().__init__()
        self.rename_pins(*pins)

    def s_parameters(self, freqs: "np.array") -> "np.array":
        #3x3 array here, assume no cross coupling. The only coupling is from 1 to 2 and 1 to 3 and vice versa, not 2 to 3 and vice versa
        smatrix = (
            [
                [0,                                         self.splittingRatio*self.T0*self.T1,    (1- self.splittingRatio)*self.T0*self.T2],
                [self.splittingRatio*self.T1*self.T0,       0,                                      0],
                [(1-self.splittingRatio)*self.T2*self.T0,   0,                                      0]
            ]
        )
        
        # the s-matrix is frequency independent,
        # so just return it for each frequency
        return np.repeat(smatrix, len(freqs)).reshape((len(freqs), self.pin_count, self.pin_count))


class NinetyDegreeOpticalHybrid(Model):
    """
    This class represents an ideal lossy or lossless 90 degree optical hybrid. 
    The 90 degree optical hybrid takes one input, and splits that input into 
    four output signals, all phase shifted by 90 degrees from each other, so with 
    shifts of 0, j, -1, and -j 
    
    Parameters
    ----------
    
    T0 :
        Power transmission coefficient of the first input port. By default there 
        is full power transmission T0 = 1.
    T1 :
        Power transmission coefficient of the first output port. By default 
        there is full power transmission T1 = 1.
    T2 :
        Power transmission coefficient of the second output port. By default 
        there is full power transmission T2 = 1.
    T3 :
        Power transmission coefficient of the third output port. By default 
        there is full power transmission T3 = 1.
    T4 :
        Power transmission coefficient of the fourth output port. By default 
        there is full power transmission T4 = 1.

    """
    pin_count = 5
    
    def __init__(
        self, 
        #name: str = "", 
        #freq_range: Optional[Tuple[Optional[float], 
        #Optional[float]]] = None, 
        T: tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0),
        pins: Optional[Tuple[str,...]] = ("in1", "out1", "out2", "out3", "out4"),
        length = 0
        ) -> None:
        self.T0, self.T1, self.T2, self.T3, self.T4 = T
        self.pins = pins
        super().__init__()
        self.rename_pins(*pins)
    
    def s_parameters(self, freqs: "np.array") -> "np.ndarray":
        
        #to be a passive device it must not change depending on the direction you put light in. Must it be hermitian?
        #no cross coupling
        smatrix = (
            [
                [0,                     self.T0*self.T1,    -self.T0*self.T2*np.j,   -self.T0*self.T3,   self.T0*self.T4*np.j],
                [self.T1*self.T0,       0,                  0,                      0,                  0],
                [self.T2*self.T0*np.j,  0,                  0,                      0,                  0],
                [-self.T3*self.T0,      0,                  0,                      0,                  0],
                [-self.T4*self.T0*np.j, 0,                  0,                      0,                  0]
            ]
        )
        
        # the s-matrix is frequency independent,
        # so just return it for each frequency
        return np.repeat(smatrix, len(freqs)).reshape((len(freqs), self.pin_count, self.pin_count))
        
    
        