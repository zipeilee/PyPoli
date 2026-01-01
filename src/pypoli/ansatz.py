"""
Ansatz module for PyPoli - Contains pre-built circuit ansatzes

Currently implemented:
- Hardware-efficient ansatz with RX-RZ-RX layers and alternating CZ gates
"""
from .circuits import Circuit
from .gates import RX, RZ, CZ


class HardwareEfficient:
    """
    Hardware-efficient ansatz implementation with RX-RZ-RX layers and alternating CZ gates.

    Structure:
    - Odd layers: RX(theta) - RZ(phi) - RX(gamma) on all qubits, followed by CZ gates on even qubit pairs (0-1, 2-3, etc.)
    - Even layers: RX(theta) - RZ(phi) - RX(gamma) on all qubits, followed by CZ gates on odd qubit pairs (1-2, 3-4, etc.)

    The ansatz is parameterized with shape: (num_layers, num_qubits, 3) for the single-qubit rotations.
    """

    def __init__(self, num_qubits: int, num_layers: int):
        """
        Initialize the hardware-efficient ansatz.

        Args:
            num_qubits: Number of qubits in the ansatz
            num_layers: Number of layers in the ansatz
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers

    def generate_circuit(self, params) -> Circuit:
        """
        Generate the Circuit object from the provided parameters.

        Args:
            params: Parameters for the ansatz with shape (num_layers, num_qubits, 3)

        Returns:
            Circuit: PyPoli Circuit object representing the ansatz
        """
        circuit = Circuit()

        for layer in range(self.num_layers):
            # Single-qubit gates: RX(theta) - RZ(phi) - RX(gamma)
            for qubit in range(self.num_qubits):
                theta, phi, gamma = params[layer, qubit]
                circuit.add_gate(RX(qubit, theta))
                circuit.add_gate(RZ(qubit, phi))
                circuit.add_gate(RX(qubit, gamma))

            # Two-qubit CZ gates with alternating pattern
            if self.num_qubits > 1:
                if layer % 2 == 0:  # Even layer: CZ on (0-1, 2-3, 4-5, etc.)
                    for qubit in range(0, self.num_qubits - 1, 2):
                        circuit.add_gate(CZ(qubit, qubit + 1))
                else:  # Odd layer: CZ on (1-2, 3-4, 5-6, etc.)
                    for qubit in range(1, self.num_qubits - 1, 2):
                        circuit.add_gate(CZ(qubit, qubit + 1))

        return circuit


class VariationalAnsatz:
    """
    Base class for variational ansatzes - provides common interface for all ansatz types.

    This class should be inherited by all new ansatz implementations.
    """

    def __init__(self, num_qubits: int, num_layers: int):
        """
        Initialize a variational ansatz.

        Args:
            num_qubits: Number of qubits in the ansatz
            num_layers: Number of layers in the ansatz
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers

    def generate_circuit(self, params) -> Circuit:
        """
        Generate the Circuit object from the provided parameters.

        Args:
            params: Parameters for the ansatz

        Returns:
            Circuit: PyPoli Circuit object representing the ansatz
        """
        raise NotImplementedError("generate_circuit must be implemented by subclass")


# Alias for backward compatibility and convenience
HardwareEfficientAnsatz = HardwareEfficient
