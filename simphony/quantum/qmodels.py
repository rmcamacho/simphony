# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
simphony.quantum.qmodels
========================

This module contains QuantumMixin for simphony models.
"""

import numpy as np
from simphony.exceptions import ShapeMismatchError
from simphony.libraries.ideal.models import BeamSplitter
from simphony.pins import PinList, Pin
from simphony.tools import xxpp_to_xpxp, xpxp_to_xxpp
from typing import List, Optional, Tuple
from scipy.optimize import least_squares
from enum import Enum, auto

import pandas as pd


def compose_qstate(*args: "QuantumState") -> "QuantumState":
    """
    Combines the quantum states of the input ports into a single quantum state.
    """
    N = 0
    mean_list = []
    cov_list = []
    pin_list = []
    for qstate in args:
        if not isinstance(qstate, QuantumState):
            raise TypeError("Input must be a QuantumState.")
        N += qstate.N
        mean_list.append(qstate.means)
        cov_list.append(qstate.cov)
        pin_list += qstate.pins
    means = np.concatenate(mean_list)
    covs = np.zeros((2*N, 2*N), dtype=float)
    left = 0
    # TODO: Currently puts states into xpxp, but should change to xxpp
    for qstate in args:
        rowcol = qstate.N*2 + left
        covs[left:rowcol, left:rowcol] = qstate.cov
        left = rowcol
    return QuantumState(means, covs, pin_list)

class QMode(Enum):
    """
    This class provides the enumeration for different mode types in a quantum
    model.
    """
    INPUT = auto() # normal input mode
    OUTPUT = auto() # normal output mode
    VACUUM = auto() # vacuum input mode
    LOSS = auto() # vacuum loss output mode

class QuantumState:
    """
    Represents a quantum state in a quantum model as a covariance matrix.

    All quantum states are represented in the xpxp convention.
    TODO: switch to xxpp convention

    Parameters
    ----------
    N :
        The number of modes in the quantum state.
    means :
        The means of the X and P quadratures of the quantum state. For example, 
        a coherent state :math:`\alpha = 3+4i` has means defined as 
        :math:`\begin{bmatrix} 3 & 4 \end{bmatrix}'. The shape of the means must
        be 2 * N.
    cov :
        The covariance matrix of the quantum state. For example, any coherent 
        state has a covariance matrix of :math:`\begin{bmatrix} 1/4 & 0 \\ 0 & 
        1/4 \end{bmatrix}`. The shape of the matrix must be 2 * N x 2 * N.
    pins :
        The pins to which the quantum state is connected.
    convention :
        The convention of the means and covariance matrix. Default is 'xpxp'.
    """
    def __init__(self, means: np.ndarray, cov: np.ndarray,  pins: PinList, convention: str='xpxp') -> None:
        self.N = len(pins)
        if means.shape != (2 * self.N,):
            raise ShapeMismatchError("The shape of the means must be 2 * N.")
        if cov.shape != (2 * self.N, 2 * self.N):
            raise ShapeMismatchError("The shape of the covariance matrix must \
                 be 2 * N x 2 * N.")
        self.means = means
        self.cov = cov
        self.pins = pins
        self.convention = convention

    def to_xpxp(self) -> None:
        """
        Converts the means and covariance matrix to the xpxp convention.
        """
        if self.convention == 'xxpp':
            self.means = xxpp_to_xpxp(self.means)
            self.cov = xxpp_to_xpxp(self.cov)
            self.convention = 'xpxp'
    
    def to_xxpp(self) -> None:
        """
        Converts the means and covariance matrix to the xxpp convention.
        """
        if self.convention == 'xpxp':
            self.means = xpxp_to_xxpp(self.means)
            self.cov = xpxp_to_xxpp(self.cov)
            self.convention = 'xxpp'

    # TODO: Add alternative methods for combining quantum states at the class level

class CoherentState(QuantumState):
    """
    Represents a coherent state in a quantum model as a covariance matrix.

    Parameters
    ----------
    alpha :
        The complex amplitude of the coherent state.
    pin : 
        The pin to which the coherent state is connected.
    """
    def __init__(self, alpha: complex, pin: "Pin") -> None:
        self.alpha = alpha
        self.N = 1
        self.means = np.array([alpha.real, alpha.imag])
        self.cov = np.array([[1/4, 0], [0, 1/4]])
        self.pins = PinList([pin])
   
class SqueezedState(QuantumState):
    """
    Represents a squeezed state in a quantum model as a covariance matrix.

    Parameters
    ----------
    r :
        The squeezing parameter of the squeezed state.
    phi :
        The squeezing phase of the squeezed state.
    pin :
        The pin to which the squeezed state is connected.
    alpha:
        The complex displacement of the squeezed state. Default is 0.
    """
    def __init__(self, r: float, phi: float, pin: "Pin", alpha: complex=0) -> None:
        self.r = r
        self.phi = phi
        self.N = 1
        self.means = np.array([alpha.real, alpha.imag])
        c, s = np.cos(phi/2), np.sin(phi/2)
        rot_mat = np.array([[c, -s], [s, c]])
        self.cov = rot_mat @ \
            ((1/4) * np.array([[np.exp(-2*r), 0], [0, np.exp(2*r)]])) @ \
            rot_mat.T
        self.pins = PinList([pin])

class TwoModeSqueezed(QuantumState):
    """
    Represents a two mode squeezed state in a quantum model as a covariance matrix.

    This state is described by three parameters: a two-mode squeezing parameter r, 
    and the two initial thermal occupations n_a and n_b.

    Parameters
    ----------
    r :
        The two-mode squeezing parameter of the two mode squeezed state.
    n_a :
        The initial thermal occupation of the first mode.
    n_b :
        The initial thermal occupation of the second mode.
    pin_a :
        The pin to which the first mode is connected.
    pin_b :
        The pin to which the second mode is connected.
    """
    def __init__(self, r: float, n_a: float, n_b: float, pin_a: "Pin", pin_b: "Pin") -> None:
        self.r = r
        self.n_a = n_a
        self.n_b = n_b
        self.N = 2
        self.means = np.array([0, 0, 0, 0])
        ca = (n_a + 1/2) * np.cosh(r)**2 + (n_b + 1/2) * np.sinh(r)**2
        cb = (n_b + 1/2) * np.cosh(r)**2 + (n_a + 1/2) * np.sinh(r)**2
        cab = (n_a + n_b + 1) * np.sinh(r) * np.cosh(r)
        self.means = np.array([0, 0, 0, 0])
        self.cov = np.array([[ca, 0, cab, 0], [0, cb, 0, cab], [cab, 0, cb, 0], [0, cab, 0, ca]]) / 2
        self.pins = PinList([pin_a, pin_b])


class QuantumMixin:
    """
    This class indicates a model as a quantum simulation compatible. 
    
    It contains the algorithms to convert a any classical model into a quantum 
    model by adding extra ports for vacuum inputs as loss channels. 
    """
    def __init__(self, *args, mixing: Optional[float] = None, **kwargs):
        self.mixing = mixing
        self._q_s_params = None
        super().__init__(*args, **kwargs)
        self.input_modes = None
        self.output_modes = None
        self.vacuum_modes = None
        self.loss_modes = None
        self.quantum_state = None

    def update_quantum_state(self, state: QuantumState) -> QuantumState:
        """
        This method updates the entire quantum state of the model. It takes the 
        input quantum states and combines it with the vacuum input modes.

        Parameters
        ----------
        state :
            The input quantum state.
        """
        n_inputs = state.N
        assert n_inputs == len(self.input_modes), "The number of input modes \
            must be equal to the number of input ports."
        # combine the input modes with the vacuum modes
        n_vacuum = len(self.vacuum_modes)
        n_modes = n_inputs + n_vacuum
        means = np.zeros(2 * n_modes)
        cov = np.zeros((2 * n_modes, 2 * n_modes))
        # update the means
        for ii, ind in enumerate(self.input_modes):
            means[2 * ind] = state.means[2 * ii]
            means[2 * ind + 1] = state.means[2 * ii + 1]
        # update the cov
        for ii, ind in enumerate(self.input_modes):
            for jj, jnd in enumerate(self.input_modes):
                cov[2*ind, 2*jnd] = state.cov[2*ii, 2*jj]
                cov[2*ind, 2*jnd+1] = state.cov[2*ii, 2*jj+1]
                cov[2*ind+1, 2*jnd] = state.cov[2*ii+1, 2*jj]
                cov[2*ind+1, 2*jnd+1] = state.cov[2*ii+1, 2*jj+1]
        
        # update the non-zero elements of the cov for the vacuum fluctuations
        for ii, ind in enumerate(self.vacuum_modes):
            cov[2*ind, 2*ind] = 1/4
            cov[2*ind+1, 2*ind+1] = 1/4
        # update the quantum state
        self.quantum_state = QuantumState(n_modes, means, cov)
        return self.quantum_state
    
    def _update_io(self, modes: List[QMode]) -> None:
        """
        This method updates the input and output modes of the model.
        """
        self.input_modes = []
        self.output_modes = []
        self.vacuum_modes = []
        self.loss_modes = []
        for ii, mode in enumerate(modes):
            if mode == QMode.INPUT:
                self.input_modes.append(ii)
            elif mode == QMode.OUTPUT:
                self.output_modes.append(ii)
            elif mode == QMode.VACUUM:
                self.vacuum_modes.append(ii)
            elif mode == QMode.LOSS:
                self.loss_modes.append(ii)
        assert len(self.vacuum_modes) == len(self.loss_modes), "The number of \
            vacuum modes must be equal to the number of loss modes."

    @staticmethod
    def _two_mode_loss(x, s_matrix, kappa):
        """
        This method contains the lossy model of a two mode beamsplitter.

        .. math:: 
            \begin{bmatrix} 
            t \sqrt{T_0 T_2} & -\kappa \exp{-i \phi} \sqrt{T_1 T_2} \\
            \kappa \exp{i \phi} \sqrt{T_0 T_3} & t \sqrt{T_1 T_3}
            \end{bmatrix}

        Parameters
        ----------
        x : list
            Tranmsission coeffiecients of the beamsplitter ports. Ordered as 
            follows: [T0, T1, T2, T3]
        s_matrix : np.ndarray
            The classical S-matrix of the beamsplitter for comparison.
        kappa : float
            The cross coupling amplitude coefficient.

        Returns
        -------
        np.ndarray
            Returns the difference between the parameterized s_matrix and the 
            given classical s_matrix.
        """
        t = np.sqrt(1-kappa**2)
        T0, T1, T2, T3 = x
        parameterized = np.array(
            [
                [t * np.sqrt(T0*T2), kappa * np.sqrt(T1*T2)],
                [kappa * np.sqrt(T0*T3), t * np.sqrt(T1*T3)]
            ]
        )
        return (parameterized - s_matrix).flatten()

    def convert_to_quantum(self, freqs: np.ndarray, mixing: Optional[float] = None) -> "np.ndarray":
        """
        This method returns the quantum S-parameters of the model.
        """
        # check if s_parameters() is implemented
        if not hasattr(self, "s_parameters"):
            raise NotImplementedError("s_parameters() is not implemented for this model.")
        S = self.circuit.s_parameters(freqs)
        n_freqs,n_ports,_,_ = S.shape
        new_n_ports = n_ports*3
        quantum_S = np.zeros((n_freqs, new_n_ports, new_n_ports, 2), dtype=float)
        for ii in range(len(freqs)):
            #TODO: generalize to any number of ports
            s_amp = S[ii][2:, :2, 0]
            s_phase = S[ii][2:, :2, 1]
            if mixing is None:
                # TODO: implement the algorithm to find the mixing parameter
                mixing = [0.5]
            mixing = mixing[0]
            kappa = np.sqrt(mixing)
            ans = least_squares(
                QuantumMixin._two_mode_loss, 
                (0.9, 0.9, 0.9, 0.9), # arbitrary initial guess 
                args=(s_amp, kappa),
                method='trf', 
                bounds=((0,0,0,0), (1,1,1,1)), 
                gtol=1e-15
            )
            # # TODO: different way to check for lossless case
            # for i, T in enumerate(ans.x):
            #     ans.x[i] = round(T, 7)
            T0, T1, T2, T3 = ans.x
            
            # print(ans.x)
            self.weights = np.array([T0, T1, T2, T3])
            bs_phi = s_phase[1,0]
            # print(bs_phi)
            bs = BeamSplitter(np.arcsin(kappa), bs_phi)
            bs0 = BeamSplitter(np.arccos(np.sqrt(T0)), \
                pins=("input", "vacuum", "out1", "loss"))
            bs1 = BeamSplitter(np.arccos(np.sqrt(T1)), \
                pins=("input", "vacuum", "out1", "loss"))
            bs2 = BeamSplitter(np.arccos(np.sqrt(T2)), \
                pins=("in1", "vacuum", "output", "loss"))
            bs3 = BeamSplitter(np.arccos(np.sqrt(T3)), \
                pins=("in1", "vacuum", "output", "loss"))
            bs.multiconnect(bs0.pins["out1"], bs1.pins["out1"], bs2, bs3)
            #TODO: algorithm to find mode types
            modes = [QMode.INPUT]*(len(bs.circuit.pins))
            for pin in bs.circuit.pins:
                index = bs.circuit.get_pin_index(pin)
                # print(index, pin.name)
                if pin.name == "input":
                    modes[index] = QMode.INPUT
                elif pin.name == "output":
                    modes[index] = QMode.OUTPUT
                elif pin.name == "vacuum":
                    modes[index] = QMode.VACUUM
                elif pin.name == "loss":
                    modes[index] = QMode.LOSS
                else:
                    raise ValueError("Invalid Quantum Circuit Pin Name")
            self._update_io(modes)
            quantum_S[ii] = bs.circuit.s_parameters(np.array([0]))
        self._q_s_params = quantum_S
        return quantum_S

    def quantum_transform(self, input: QuantumState, freqs: np.ndarray, alt=False) -> QuantumState:
        """
        This method applies the quantum transformation of the circuit to the 
        input quantum state.
        """
        
        if alt:
            smatrix = self.circuit.s_parameters(freqs)
            self._update_io([QMode.INPUT, QMode.INPUT, QMode.OUTPUT, QMode.OUTPUT])
        else:
            smatrix = self.convert_to_quantum(freqs)

        if input.N != len(self.input_modes):
            raise ValueError("The number of input modes does not match the \
                number of input ports.")
        vacuum_modes = [CoherentState(0, pin=port) for port in self.vacuum_modes]
        if len(vacuum_modes) > 0:
            new_input = compose_qstate(input, *vacuum_modes)
        else:
            new_input = input

        transforms = []
        qstates = []
        for freq_ind in range(len(freqs)):
            s_freq = smatrix[freq_ind]
            transform = np.zeros((len(self.input_modes + self.vacuum_modes)*2, len(self.output_modes + self.loss_modes)*2))
            step = len(self.input_modes + self.vacuum_modes)
            n_outputs = len(self.output_modes)
            n_total = len(self.output_modes + self.loss_modes)
            losses = np.zeros(n_total*2)
            kappas = np.zeros(n_total*2)
            for ii, si in enumerate(self.output_modes + self.loss_modes):
                for jj, sj in enumerate(self.input_modes + self.vacuum_modes):
                    S = s_freq[si, sj]
                    re = S[0] * np.cos(S[1])
                    im = S[0] * np.sin(S[1])

                    transform[ii, jj] = re
                    transform[ii, jj + step] = -im
                    transform[ii + step, jj] = im
                    transform[ii + step, jj + step] = re

                if alt:
                    K = 1 - np.sum(np.square(s_freq[si, :, 0]))
                    kappas[ii] = K
                    kappas[ii + step] = K

                    losses[ii] = self.value * (-1) ** (ii)
                    losses[ii + step] = self.value * (-1) ** (ii + 1)

            new_input.to_xxpp()
            output_means = transform @ new_input.means.T
            if alt:
                output_cov = (
                    transform @ new_input.cov @ transform.T
                    + 1 / 4 * np.diag(kappas) 
                    + 1 / 4 * np.diag(losses)[::-1]
                )
            else:
                output_cov = transform @ new_input.cov @ transform.T
                # Loss modes will always be the last modes in the circuit.
                # We need to remove them from the output means and cov
                indices = np.arange(2*n_total)
                droplist = indices[np.r_[n_outputs:n_total, n_total+n_outputs:2*n_total]]
                output_means = np.delete(output_means, droplist, axis=0)
                output_cov = np.delete(output_cov, droplist, axis=0)
                output_cov = np.delete(output_cov, droplist, axis=1)

                # TODO: Possibly implement tolerance for small numbers
                # convert small numbers to zero
                # output_means[abs(output_means) < 1e-10] = 0
                # output_cov[abs(output_cov) < 1e-10] = 0

            qstates.append(
                QuantumState(
                    output_means, 
                    output_cov, 
                    pins=[self.pins["in1"], self.pins["in2"]],
                    convention='xxpp'
                )
            )
            transforms.append(transform)

        return (transforms, qstates)

            
            
# TODO: Decide if this should be moved to examples
class QuantumBeamSplitter(QuantumMixin, BeamSplitter):
    """
    This class represents a quantum lossy or lossless beam splitter. Uses the
    ideal Beamsplitter class as a base class and adds the quantum conversion
    methods from the QuantumMixin class.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, mixing=kwargs.get('theta', None))
