"""
Ansatz module for PyPoli - Contains pre-built circuit ansatzes

Currently implemented:
- Hardware-efficient ansatz with RX-RZ-RX layers and alternating CZ gates
"""
from .circuits import Circuit
from .gates import RX, RZ, CZ, Parameter


class HardwareEfficient:
    """
    Hardware-efficient ansatz implementation with RX-RZ-RX layers and alternating CZ gates.
    ...
    """

    def __init__(self, num_qubits: int, num_layers: int):
        self.num_qubits = num_qubits
        self.num_layers = num_layers

    def generate_circuit(self, params=None) -> Circuit:
        """
        Generate the Circuit object.
        
        Args:
            params: Optional parameters. 
                    If provided (shape: num_layers, num_qubits, 3), returns a flat circuit with bound values.
                    If None, returns a structured Circuit object optimized for JIT compilation (scan).
                    
        Returns:
            Circuit: PyPoli Circuit object.
        """
        if params is not None:
            # Eager/Flat mode
            circuit = Circuit()
            for layer in range(self.num_layers):
                for qubit in range(self.num_qubits):
                    theta, phi, gamma = params[layer, qubit]
                    circuit.add_gate(RX(qubit, theta))
                    circuit.add_gate(RZ(qubit, phi))
                    circuit.add_gate(RX(qubit, gamma))

                if self.num_qubits > 1:
                    if layer % 2 == 0:
                        for qubit in range(0, self.num_qubits - 1, 2):
                            circuit.add_gate(CZ(qubit, qubit + 1))
                    else:
                        for qubit in range(1, self.num_qubits - 1, 2):
                            circuit.add_gate(CZ(qubit, qubit + 1))
            return circuit
        
        else:
            # Structured/JIT mode
            # We define a "Block" that consists of 2 layers (Even + Odd) to capture the alternating pattern.
            
            if self.num_layers % 2 != 0:
                # Warning or handle? Let's just create blocks for num_layers // 2
                pass
                
            block_template = Circuit()
            
            # Use a sentinel parameter object that indicates "this is a parameter slot"
            # We don't care about names anymore, just order.
            
            # Layer 0 (Even)
            for q in range(self.num_qubits):
                # 3 params per qubit
                block_template.add_gate(RX(q, Parameter("")))
                block_template.add_gate(RZ(q, Parameter("")))
                block_template.add_gate(RX(q, Parameter("")))
                
            if self.num_qubits > 1:
                for qubit in range(0, self.num_qubits - 1, 2):
                    block_template.add_gate(CZ(qubit, qubit + 1))
                    
            # Layer 1 (Odd)
            for q in range(self.num_qubits):
                block_template.add_gate(RX(q, Parameter("")))
                block_template.add_gate(RZ(q, Parameter("")))
                block_template.add_gate(RX(q, Parameter("")))

            if self.num_qubits > 1:
                for qubit in range(1, self.num_qubits - 1, 2):
                    block_template.add_gate(CZ(qubit, qubit + 1))
                    
            # Return structured circuit
            # Depth is num_layers // 2
            return Circuit.from_layer(block_template, self.num_layers // 2)



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
