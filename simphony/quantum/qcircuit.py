# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
simphony.quantum.qcircuit
========================

This module contains the quantum circuit model.
"""
from typing import Optional, Tuple

from simphony import Model
from simphony.layout import Circuit
from simphony.quantum import QuantumMixin

class Qcircuit(QuantumMixin, Model):
    """This class represents a quantum circuit.

    It adds functionality to the ``Circuit`` class to support quantum simulations.    
    """
    pin_count = 4
    def __init__(self, *args, circuit: Circuit, pins: Optional[Tuple[str,...]]=None, **kwargs) -> None:
        """Initializes a quantum circuit.

        Parameters
        ----------
        circuit :
            The circuit to convert to a quantum circuit
        """
        super().__init__(*args, **kwargs)
        self.ckt = circuit
        self.pin_count = len(circuit.pins)
        if pins is not None:
            self.rename_pins(*pins)

    def s_parameters(self, freqs: "np.array") -> "np.ndarray":
        return self.ckt.s_parameters(freqs)