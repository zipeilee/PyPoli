
"""
PyPoli - JAX-only Pauli Propagation library for quantum circuits

Based on PauliPropagation.jl implementation

Features:
- Pauli string algebra (PauliString, PauliSum)
- Quantum gates (single-qubit and two-qubit)
- Circuit construction and Pauli propagation
- Automatic differentiation with JAX
- Truncation support for large circuits
- Batch expectation value calculation
"""
from .core import PauliString, PauliSum
from .gates import (
    I, X, Y, Z, H, S, T,
    RX, RY, RZ,
    RXX, RYY, RZZ, CNOT, CZ
)
from .circuits import Circuit
from .propagation import (
    propagate, expectation_value,
    jit_expectation_value, batch_expectation_value
)
from .jit_propagation import make_jit_loss_fn, JITPropagator
from .ansatz import (
    VariationalAnsatz,
    HardwareEfficient,
    HardwareEfficientAnsatz
)

__all__ = [
    # Core Pauli algebra
    'PauliString', 'PauliSum',

    # Gates
    'I', 'X', 'Y', 'Z', 'H', 'S', 'T',
    'RX', 'RY', 'RZ', 'RXX', 'RYY', 'RZZ', 'CNOT', 'CZ',

    # Circuits
    'Circuit',

    # Propagation and expectation
    'propagate', 'expectation_value',
    'jit_expectation_value', 'batch_expectation_value',
    'make_jit_loss_fn', 'JITPropagator',

    # Ansatz implementations
    'VariationalAnsatz',
    'HardwareEfficient',
    'HardwareEfficientAnsatz'
]

__version__ = '0.1.0'
__author__ = 'PyPoli Team'
