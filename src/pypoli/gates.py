"""
Quantum gates implementation with Pauli action rules (JAX-only)
"""
import jax
import jax.numpy as jnp
from typing import List, Any

from .core import PauliString

# Lookup tables for gate actions on Pauli operators
# For each gate, we define a lookup table that maps:
#   (input_pauli_bits) -> (output_pauli_bits, phase)
# Pauli bit encoding (2 bits per Pauli): I=00(0), X=01(1), Y=10(2), Z=11(3)
clifford_map = {
    # I gate: identity on all Paulis
    'I': [(0, 1), (1, 1), (2, 1), (3, 1)],

    # X gate (Pauli-X)
    'X': [(0, 1), (1, 1), (2, -1), (3, -1)],

    # Y gate (Pauli-Y)
    'Y': [(0, 1), (1, -1), (2, 1), (3, -1)],

    # Z gate (Pauli-Z)
    'Z': [(0, 1), (1, -1), (2, -1), (3, 1)],

    # H gate (Hadamard)
    'H': [(0, 1), (3, 1), (2, -1), (1, 1)],

    # S gate (Phase)
    'S': [(0, 1), (2, 1), (1, -1), (3, 1)],

    # CNOT gate (control, target)
    # 2 qubits: each Pauli combination is 4 bits
    # Input bits: (target << 2) | control
    'CNOT': [
        (0x00, 1),   # I-I -> I-I
        (0x01, 1),   # X-I -> X-X
        (0x02, 1),   # Y-I -> Y-X
        (0x03, 1),   # Z-I -> Z-I
        (0x04, 1),   # I-X -> I-X
        (0x05, 1),   # X-X -> X-I
        (0x06, 1),   # Y-X -> Y-I
        (0x07, 1),   # Z-X -> Z-X
        (0x08, 1),   # I-Y -> Z-Y
        (0x09, 1),   # X-Y -> Y-Z
        (0x0A, -1),  # Y-Y -> X-Z
        (0x0B, 1),   # Z-Y -> I-Y
        (0x0C, 1),   # I-Z -> Z-Z
        (0x0D, -1),  # X-Z -> Y-Y
        (0x0E, 1),   # Y-Z -> X-Y
        (0x0F, 1),   # Z-Z -> I-Z
    ],

    # CZ gate (control, target)
    # 2 qubits: each Pauli combination is 4 bits
    'CZ': [
        (0x00, 1),   # I-I -> I-I
        (0x01, 1),   # X-I -> X-Z
        (0x02, 1),   # Y-I -> Y-Z
        (0x03, 1),   # Z-I -> Z-I
        (0x04, 1),   # I-X -> Z-X
        (0x05, 1),   # X-X -> Y-Y
        (0x06, -1),  # Y-X -> X-Y
        (0x07, 1),   # Z-X -> I-X
        (0x08, 1),   # I-Y -> Z-Y
        (0x09, -1),  # X-Y -> Y-X
        (0x0A, 1),   # Y-Y -> X-X
        (0x0B, 1),   # Z-Y -> I-Y
        (0x0C, 1),   # I-Z -> I-Z
        (0x0D, 1),   # X-Z -> X-I
        (0x0E, 1),   # Y-Z -> Y-I
        (0x0F, 1),   # Z-Z -> Z-Z
    ]
}

# Update CNOT/CZ maps with correct values from script
# Script Output Mapping for CNOT:
# 0x00 (I-I) -> 0x00 (I-I)
# 0x01 (X-I) -> 0x05 (X-X)  (Wait, 0x01 is X-I? My script loop: i=Control, j=Target. i | (j<<2). X is 1. I is 0. X-I is 1 | 0 = 1. Correct.)
# 0x02 (Y-I) -> 0x06 (Y-X)
# 0x03 (Z-I) -> 0x03 (Z-I)
# 0x04 (I-X) -> 0x04 (I-X)
# 0x05 (X-X) -> 0x01 (X-I)
# 0x06 (Y-X) -> 0x02 (Y-I)
# 0x07 (Z-X) -> 0x07 (Z-X)
# 0x08 (I-Y) -> 0x0B (Z-Y)
# 0x09 (X-Y) -> 0x0E (Y-Z)
# 0x0A (Y-Y) -> 0x0D (X-Z) (Coeff -1)
# 0x0B (Z-Y) -> 0x08 (I-Y)
# 0x0C (I-Z) -> 0x0F (Z-Z)
# 0x0D (X-Z) -> 0x0A (Y-Y) (Coeff -1)
# 0x0E (Y-Z) -> 0x09 (X-Y)
# 0x0F (Z-Z) -> 0x0C (I-Z)

clifford_map['CNOT'] = [
    (0x00, 1),   # I-I -> I-I
    (0x05, 1),   # X-I -> X-X
    (0x06, 1),   # Y-I -> Y-X
    (0x03, 1),   # Z-I -> Z-I
    (0x04, 1),   # I-X -> I-X
    (0x01, 1),   # X-X -> X-I
    (0x02, 1),   # Y-X -> Y-I
    (0x07, 1),   # Z-X -> Z-X
    (0x0B, 1),   # I-Y -> Z-Y
    (0x0E, 1),   # X-Y -> Y-Z
    (0x0D, -1),  # Y-Y -> X-Z
    (0x08, 1),   # Z-Y -> I-Y
    (0x0F, 1),   # I-Z -> Z-Z
    (0x0A, -1),  # X-Z -> Y-Y
    (0x09, 1),   # Y-Z -> X-Y
    (0x0C, 1),   # Z-Z -> I-Z
]
# Wait, the index of the list IS the input bits.
# So I must place the result at the correct index.
cnot_list = [None] * 16
cnot_list[0x00] = (0x00, 1)
cnot_list[0x01] = (0x05, 1)
cnot_list[0x02] = (0x06, 1)
cnot_list[0x03] = (0x03, 1)
cnot_list[0x04] = (0x04, 1)
cnot_list[0x05] = (0x01, 1)
cnot_list[0x06] = (0x02, 1)
cnot_list[0x07] = (0x07, 1)
cnot_list[0x08] = (0x0B, 1)
cnot_list[0x09] = (0x0E, 1)
cnot_list[0x0A] = (0x0D, -1)
cnot_list[0x0B] = (0x08, 1)
cnot_list[0x0C] = (0x0F, 1)
cnot_list[0x0D] = (0x0A, -1)
cnot_list[0x0E] = (0x09, 1)
cnot_list[0x0F] = (0x0C, 1)
clifford_map['CNOT'] = cnot_list

# Script Output Mapping for CZ:
# 0x00 (I-I) -> 0x00
# 0x01 (X-I) -> 0x0D (X-Z)
# 0x02 (Y-I) -> 0x0E (Y-Z)
# 0x03 (Z-I) -> 0x03
# 0x04 (I-X) -> 0x07 (Z-X)
# 0x05 (X-X) -> 0x0A (Y-Y)
# 0x06 (Y-X) -> 0x09 (X-Y) (Coeff -1)
# 0x07 (Z-X) -> 0x04 (I-X)
# 0x08 (I-Y) -> 0x0B (Z-Y)
# 0x09 (X-Y) -> 0x06 (Y-X) (Coeff -1)
# 0x0A (Y-Y) -> 0x05 (X-X)
# 0x0B (Z-Y) -> 0x08 (I-Y)
# 0x0C (I-Z) -> 0x0C
# 0x0D (X-Z) -> 0x01 (X-I)
# 0x0E (Y-Z) -> 0x02 (Y-I)
# 0x0F (Z-Z) -> 0x0F

cz_list = [None] * 16
cz_list[0x00] = (0x00, 1)
cz_list[0x01] = (0x0D, 1)
cz_list[0x02] = (0x0E, 1)
cz_list[0x03] = (0x03, 1)
cz_list[0x04] = (0x07, 1)
cz_list[0x05] = (0x0A, 1)
cz_list[0x06] = (0x09, -1)
cz_list[0x07] = (0x04, 1)
cz_list[0x08] = (0x0B, 1)
cz_list[0x09] = (0x06, -1)
cz_list[0x0A] = (0x05, 1)
cz_list[0x0B] = (0x08, 1)
cz_list[0x0C] = (0x0C, 1)
cz_list[0x0D] = (0x01, 1)
cz_list[0x0E] = (0x02, 1)
cz_list[0x0F] = (0x0F, 1)
clifford_map['CZ'] = cz_list


class Parameter:
    """A symbolic parameter for parameterized gates."""
    def __init__(self, name: str):
        self.name = name
    
    def __repr__(self):
        return f"Parameter('{self.name}')"

class Gate:
    """
    Base class for all quantum gates.
    """

    def __init__(self, qubits: tuple):
        """
        Initialize a gate acting on the given qubits.

        Args:
            qubits: Tuple of qubit indices this gate acts on
        """
        self.qubits = qubits
        # Subclasses with parameters should populate this list with their parameter names
        # in the order they appear in the constructor args.
        # e.g. RX(q, theta) -> self.params = [theta]
        # But wait, theta might be a float or a Parameter object.
        # We need to distinguish.
        self.params = [] 

    def bind_parameters(self, param_values: list) -> 'Gate':
        """
        Return a new gate with parameters bound to values.
        Takes a list of values and consumes them in order.
        Returns the new gate.
        Note: The caller is responsible for slicing the correct values.
        Actually, for simplicity, let's just pass the map?
        No, user wants implicit binding.
        
        So we pass a list of values. But how does the gate know WHICH values?
        The caller (Circuit.propagate) iterates over gates and parameters simultaneously.
        """
        return self

    def pauli_action(self, pauli_str) -> List:

        """
        Action of this gate on a Pauli string.

        Returns:
            List[PauliString]: Result of the action (multiple terms if gate is
                             non-diagonal in the Pauli basis)
        """
        raise NotImplementedError(f"pauli_action not implemented for {type(self).__name__}")


# Single-qubit gates
class I(Gate):
    """Identity gate"""

    def __init__(self, qubit: int):
        super().__init__((qubit,))
        self.map = clifford_map['I']

    def pauli_action(self, pauli_str) -> List:
        lookup_bits = pauli_str.get_pauli_bit(self.qubits)
        new_pauli_bits, phase = self.map[lookup_bits]
        new_pauli_str = pauli_str.set_pauli_bit(new_pauli_bits, self.qubits)
        new_pauli_str.coefficient *= phase
        return [new_pauli_str]


class X(Gate):
    """Pauli X gate"""

    def __init__(self, qubit: int):
        super().__init__((qubit,))
        self.map = clifford_map['X']

    def pauli_action(self, pauli_str) -> List:
        lookup_bits = pauli_str.get_pauli_bit(self.qubits)
        new_pauli_bits, phase = self.map[lookup_bits]
        new_pauli_str = pauli_str.set_pauli_bit(new_pauli_bits, self.qubits)
        new_pauli_str.coefficient *= phase
        return [new_pauli_str]


class Y(Gate):
    """Pauli Y gate"""

    def __init__(self, qubit: int):
        super().__init__((qubit,))
        self.map = clifford_map['Y']

    def pauli_action(self, pauli_str) -> List:
        lookup_bits = pauli_str.get_pauli_bit(self.qubits)
        new_pauli_bits, phase = self.map[lookup_bits]
        new_pauli_str = pauli_str.set_pauli_bit(new_pauli_bits, self.qubits)
        new_pauli_str.coefficient *= phase
        return [new_pauli_str]


class Z(Gate):
    """Pauli Z gate"""

    def __init__(self, qubit: int):
        super().__init__((qubit,))
        self.map = clifford_map['Z']

    def pauli_action(self, pauli_str) -> List:
        lookup_bits = pauli_str.get_pauli_bit(self.qubits)
        new_pauli_bits, phase = self.map[lookup_bits]
        new_pauli_str = pauli_str.set_pauli_bit(new_pauli_bits, self.qubits)
        new_pauli_str.coefficient *= phase
        return [new_pauli_str]


class H(Gate):
    """Hadamard gate"""

    def __init__(self, qubit: int):
        super().__init__((qubit,))
        self.map = clifford_map['H']

    def pauli_action(self, pauli_str) -> List:
        lookup_bits = pauli_str.get_pauli_bit(self.qubits)
        new_pauli_bits, phase = self.map[lookup_bits]
        new_pauli_str = pauli_str.set_pauli_bit(new_pauli_bits, self.qubits)
        new_pauli_str.coefficient *= phase
        return [new_pauli_str]


class S(Gate):
    """Phase gate (S = diag(1, i))"""

    def __init__(self, qubit: int):
        super().__init__((qubit,))
        self.map = clifford_map['S']

    def pauli_action(self, pauli_str) -> List:
        lookup_bits = pauli_str.get_pauli_bit(self.qubits)
        new_pauli_bits, phase = self.map[lookup_bits]
        new_pauli_str = pauli_str.set_pauli_bit(new_pauli_bits, self.qubits)
        new_pauli_str.coefficient *= phase
        return [new_pauli_str]


class T(Gate):
    """T gate (T = diag(1, exp(iπ/4)))"""

    def __init__(self, qubit: int):
        super().__init__((qubit,))
        self.theta = jnp.pi / 4

    def pauli_action(self, pauli_str) -> List:
        """Pauli action of T gate on a Pauli string."""
        # T commutes with I and Z.
        # X -> X cos(pi/4) + Y sin(pi/4)
        # Y -> Y cos(pi/4) - X sin(pi/4)
        
        pauli_bits = pauli_str.get_pauli_bit(self.qubits)

        cos_val = jnp.cos(self.theta)
        sin_val = jnp.sin(self.theta)

        if pauli_bits == PauliString._pauli_to_bit['I'] or pauli_bits == PauliString._pauli_to_bit['Z']:
            return [pauli_str]
        elif pauli_bits == PauliString._pauli_to_bit['X']:
            # X -> X cos - i sin (Z X) ? No.
            # T X Tdag = X cos + Y sin
            x_paulis = pauli_str.set_pauli_bit(PauliString._pauli_to_bit['X'], self.qubits)
            y_paulis = pauli_str.set_pauli_bit(PauliString._pauli_to_bit['Y'], self.qubits)
            return [
                type(pauli_str)(x_paulis.paulis, x_paulis.coefficient * cos_val),
                type(pauli_str)(y_paulis.paulis, y_paulis.coefficient * sin_val)
            ]
        elif pauli_bits == PauliString._pauli_to_bit['Y']:
            # T Y Tdag = Y cos - X sin
            y_paulis = pauli_str.set_pauli_bit(PauliString._pauli_to_bit['Y'], self.qubits)
            x_paulis = pauli_str.set_pauli_bit(PauliString._pauli_to_bit['X'], self.qubits)
            return [
                type(pauli_str)(y_paulis.paulis, y_paulis.coefficient * cos_val),
                type(pauli_str)(x_paulis.paulis, x_paulis.coefficient * (-sin_val))
            ]
        return [pauli_str] # Should not happen


# Single-qubit rotation gates
class RX(Gate):
    """Rotation around X-axis: exp(-iθ X / 2)"""

    def __init__(self, qubit: int, theta: Any):
        super().__init__((qubit,))
        
        # Auto-convert string to Parameter
        if isinstance(theta, str):
            theta = Parameter(theta)
            
        self.theta = theta # Can be float, JAX array, or Parameter
        self.generator_mask = PauliString._pauli_to_bit['X']
        if isinstance(theta, Parameter):
            self.params = [theta]

    def bind_parameters(self, param_values: list) -> 'RX':
        if isinstance(self.theta, Parameter):
            # Consume one value
            return RX(self.qubits[0], param_values[0])
        return self

    def pauli_action(self, pauli_str) -> List:
        pauli_bits = pauli_str.get_pauli_bit(self.qubits)

        if pauli_bits == self.generator_mask or pauli_bits == 0: # Commutes with X and I
            return [pauli_str]

        # For non-commuting cases (Y, Z)
        # P -> P cos(theta) - i sin(theta) (X P)
        
        # If theta is a Parameter, we cannot compute cos/sin yet.
        # This method assumes bound parameters (concrete values or JAX tracers).
        if isinstance(self.theta, Parameter):
            raise ValueError(f"Cannot apply gate with unbound parameter: {self.theta}")
            
        theta_val = jnp.asarray(self.theta, dtype=jnp.float64)
        cos_val = jnp.cos(theta_val)
        sin_val = jnp.sin(theta_val)

        if pauli_bits == PauliString._pauli_to_bit['Y']:
            # X Y = iZ. -i sin (iZ) = sin Z.
            # Y -> Y cos + Z sin
            y_paulis = pauli_str.set_pauli_bit(PauliString._pauli_to_bit['Y'], self.qubits)
            z_paulis = pauli_str.set_pauli_bit(PauliString._pauli_to_bit['Z'], self.qubits)
            return [
                type(pauli_str)(y_paulis.paulis, y_paulis.coefficient * cos_val),
                type(pauli_str)(z_paulis.paulis, z_paulis.coefficient * sin_val)
            ]
        elif pauli_bits == PauliString._pauli_to_bit['Z']:
            # X Z = -iY. -i sin (-iY) = -sin Y.
            # Z -> Z cos - Y sin
            z_paulis = pauli_str.set_pauli_bit(PauliString._pauli_to_bit['Z'], self.qubits)
            y_paulis = pauli_str.set_pauli_bit(PauliString._pauli_to_bit['Y'], self.qubits)
            return [
                type(pauli_str)(z_paulis.paulis, z_paulis.coefficient * cos_val),
                type(pauli_str)(y_paulis.paulis, y_paulis.coefficient * (-sin_val))
            ]
        return [pauli_str]


class RY(Gate):
    """Rotation around Y-axis: exp(-iθ Y / 2)"""

    def __init__(self, qubit: int, theta: Any):
        super().__init__((qubit,))
        
        # Auto-convert string to Parameter
        if isinstance(theta, str):
            theta = Parameter(theta)
            
        self.theta = theta
        self.generator_mask = PauliString._pauli_to_bit['Y']
        if isinstance(theta, Parameter):
            self.params = [theta]

    def bind_parameters(self, param_values: list) -> 'RY':
        if isinstance(self.theta, Parameter):
            return RY(self.qubits[0], param_values[0])
        return self

    def pauli_action(self, pauli_str) -> List:
        pauli_bits = pauli_str.get_pauli_bit(self.qubits)

        if pauli_bits == self.generator_mask or pauli_bits == 0:
            return [pauli_str]

        if isinstance(self.theta, Parameter):
            raise ValueError(f"Cannot apply gate with unbound parameter: {self.theta}")
            
        theta_val = jnp.asarray(self.theta, dtype=jnp.float64)
        cos_val = jnp.cos(theta_val)
        sin_val = jnp.sin(theta_val)

        if pauli_bits == PauliString._pauli_to_bit['X']:
            # Y X = -iZ. -i sin (-iZ) = -sin Z
            # X -> X cos - Z sin
            x_paulis = pauli_str.set_pauli_bit(PauliString._pauli_to_bit['X'], self.qubits)
            z_paulis = pauli_str.set_pauli_bit(PauliString._pauli_to_bit['Z'], self.qubits)
            return [
                type(pauli_str)(x_paulis.paulis, x_paulis.coefficient * cos_val),
                type(pauli_str)(z_paulis.paulis, z_paulis.coefficient * (-sin_val))
            ]
        elif pauli_bits == PauliString._pauli_to_bit['Z']:
            # Y Z = iX. -i sin (iX) = sin X
            # Z -> Z cos + X sin
            z_paulis = pauli_str.set_pauli_bit(PauliString._pauli_to_bit['Z'], self.qubits)
            x_paulis = pauli_str.set_pauli_bit(PauliString._pauli_to_bit['X'], self.qubits)
            return [
                type(pauli_str)(z_paulis.paulis, z_paulis.coefficient * cos_val),
                type(pauli_str)(x_paulis.paulis, x_paulis.coefficient * sin_val)
            ]
        return [pauli_str]


class RZ(Gate):
    """Rotation around Z-axis: exp(-iθ Z / 2)"""

    def __init__(self, qubit: int, theta: Any):
        super().__init__((qubit,))
        
        # Auto-convert string to Parameter
        if isinstance(theta, str):
            theta = Parameter(theta)
            
        self.theta = theta
        self.generator_mask = PauliString._pauli_to_bit['Z']
        if isinstance(theta, Parameter):
            self.params = [theta]

    def bind_parameters(self, param_values: list) -> 'RZ':
        if isinstance(self.theta, Parameter):
            return RZ(self.qubits[0], param_values[0])
        return self

    def pauli_action(self, pauli_str) -> List:
        pauli_bits = pauli_str.get_pauli_bit(self.qubits)

        if pauli_bits == self.generator_mask or pauli_bits == 0:
            return [pauli_str]

        if isinstance(self.theta, Parameter):
            raise ValueError(f"Cannot apply gate with unbound parameter: {self.theta}")
            
        theta_val = jnp.asarray(self.theta, dtype=jnp.float64)
        cos_val = jnp.cos(theta_val)
        sin_val = jnp.sin(theta_val)

        if pauli_bits == PauliString._pauli_to_bit['X']:
            # Z X = iY. -i sin (iY) = sin Y
            # X -> X cos + Y sin
            x_paulis = pauli_str.set_pauli_bit(PauliString._pauli_to_bit['X'], self.qubits)
            y_paulis = pauli_str.set_pauli_bit(PauliString._pauli_to_bit['Y'], self.qubits)
            return [
                type(pauli_str)(x_paulis.paulis, x_paulis.coefficient * cos_val),
                type(pauli_str)(y_paulis.paulis, y_paulis.coefficient * sin_val)
            ]
        elif pauli_bits == PauliString._pauli_to_bit['Y']:
            # Z Y = -iX. -i sin (-iX) = -sin X
            # Y -> Y cos - X sin
            y_paulis = pauli_str.set_pauli_bit(PauliString._pauli_to_bit['Y'], self.qubits)
            x_paulis = pauli_str.set_pauli_bit(PauliString._pauli_to_bit['X'], self.qubits)
            return [
                type(pauli_str)(y_paulis.paulis, y_paulis.coefficient * cos_val),
                type(pauli_str)(x_paulis.paulis, x_paulis.coefficient * (-sin_val))
            ]
        return [pauli_str]


# Two-qubit gates
class RXX(Gate):
    """Rotation around XX-axis: exp(-iθ XX / 2)"""

    def __init__(self, qubit1: int, qubit2: int, theta: Any):
        super().__init__((qubit1, qubit2))
        
        if isinstance(theta, str):
            theta = Parameter(theta)
            
        self.theta = theta
        if isinstance(theta, Parameter):
            self.params = [theta]

    def bind_parameters(self, param_values: list) -> 'RXX':
        if isinstance(self.theta, Parameter):
            return RXX(self.qubits[0], self.qubits[1], param_values[0])
        return self

    def pauli_action(self, pauli_str) -> List:
        if isinstance(self.theta, Parameter):
            raise ValueError(f"Cannot apply gate with unbound parameter: {self.theta}")
        # ... rest of implementation assumes bound theta ...
        q0, q1 = self.qubits
        p0 = pauli_str.paulis.get(q0, 'I')
        p1 = pauli_str.paulis.get(q1, 'I')
        
        # Check commutation with XX
        # Anti-commutes if odd number of anti-commuting factors
        # X anti-commutes with Y, Z.
        anti_count = 0
        if p0 in ('Y', 'Z'): anti_count += 1
        if p1 in ('Y', 'Z'): anti_count += 1
        
        if anti_count % 2 == 0:
            # Commutes
            return [pauli_str]
        
        # Anti-commutes: P -> P cos - i sin (XX P)
        cos_val = jnp.cos(self.theta)
        sin_val = jnp.sin(self.theta)
        
        term1 = type(pauli_str)(pauli_str.paulis, pauli_str.coefficient * cos_val)
        
        # Calculate XX * P
        # Create XX PauliString
        xx_pauli = PauliString({q0: 'X', q1: 'X'}, 1.0)
        prod = xx_pauli * pauli_str
        
        # Result is P cos - i sin (prod)
        # prod has some coefficient (phase).
        # We want -i * sin * prod.coefficient
        new_coeff = -1j * sin_val * prod.coefficient
        
        term2 = type(pauli_str)(prod.paulis, new_coeff)
        
        return [term1, term2]


class RYY(Gate):
    """Rotation around YY-axis: exp(-iθ YY / 2)"""

    def __init__(self, qubit1: int, qubit2: int, theta: Any):
        super().__init__((qubit1, qubit2))
        
        if isinstance(theta, str):
            theta = Parameter(theta)
            
        self.theta = theta
        if isinstance(theta, Parameter):
            self.params = [theta]

    def bind_parameters(self, param_values: list) -> 'RYY':
        if isinstance(self.theta, Parameter):
            return RYY(self.qubits[0], self.qubits[1], param_values[0])
        return self

    def pauli_action(self, pauli_str) -> List:
        if isinstance(self.theta, Parameter):
            raise ValueError(f"Cannot apply gate with unbound parameter: {self.theta}")
            
        q0, q1 = self.qubits
        p0 = pauli_str.paulis.get(q0, 'I')
        p1 = pauli_str.paulis.get(q1, 'I')
        
        # Check commutation with YY
        # Y anti-commutes with X, Z
        anti_count = 0
        if p0 in ('X', 'Z'): anti_count += 1
        if p1 in ('X', 'Z'): anti_count += 1
        
        if anti_count % 2 == 0:
            return [pauli_str]
            
        cos_val = jnp.cos(self.theta)
        sin_val = jnp.sin(self.theta)
        
        term1 = type(pauli_str)(pauli_str.paulis, pauli_str.coefficient * cos_val)
        
        yy_pauli = PauliString({q0: 'Y', q1: 'Y'}, 1.0)
        prod = yy_pauli * pauli_str
        new_coeff = -1j * sin_val * prod.coefficient
        
        term2 = type(pauli_str)(prod.paulis, new_coeff)
        
        return [term1, term2]


class RZZ(Gate):
    """Rotation around ZZ-axis: exp(-iθ ZZ / 2)"""

    def __init__(self, qubit1: int, qubit2: int, theta: Any):
        super().__init__((qubit1, qubit2))
        
        if isinstance(theta, str):
            theta = Parameter(theta)
            
        self.theta = theta
        if isinstance(theta, Parameter):
            self.params = [theta]

    def bind_parameters(self, param_values: list) -> 'RZZ':
        if isinstance(self.theta, Parameter):
            return RZZ(self.qubits[0], self.qubits[1], param_values[0])
        return self

    def pauli_action(self, pauli_str) -> List:
        if isinstance(self.theta, Parameter):
            raise ValueError(f"Cannot apply gate with unbound parameter: {self.theta}")
            
        q0, q1 = self.qubits
        p0 = pauli_str.paulis.get(q0, 'I')
        p1 = pauli_str.paulis.get(q1, 'I')
        
        # Check commutation with ZZ
        # Z anti-commutes with X, Y
        anti_count = 0
        if p0 in ('X', 'Y'): anti_count += 1
        if p1 in ('X', 'Y'): anti_count += 1
        
        if anti_count % 2 == 0:
            return [pauli_str]
            
        cos_val = jnp.cos(self.theta)
        sin_val = jnp.sin(self.theta)
        
        term1 = type(pauli_str)(pauli_str.paulis, pauli_str.coefficient * cos_val)
        
        zz_pauli = PauliString({q0: 'Z', q1: 'Z'}, 1.0)
        prod = zz_pauli * pauli_str
        new_coeff = -1j * sin_val * prod.coefficient
        
        term2 = type(pauli_str)(prod.paulis, new_coeff)
        
        return [term1, term2]


class CNOT(Gate):
    """CNOT gate (control, target)"""

    def __init__(self, control_qubit: int, target_qubit: int):
        super().__init__((control_qubit, target_qubit))
        self.map = clifford_map['CNOT']

    def pauli_action(self, pauli_str) -> List:
        lookup_bits = pauli_str.get_pauli_bit(self.qubits)
        new_pauli_bits, phase = self.map[lookup_bits]
        new_pauli_str = pauli_str.set_pauli_bit(new_pauli_bits, self.qubits)
        new_pauli_str.coefficient *= phase
        return [new_pauli_str]


class CZ(Gate):
    """CZ gate (control, target)"""

    def __init__(self, control_qubit: int, target_qubit: int):
        super().__init__((control_qubit, target_qubit))
        self.map = clifford_map['CZ']

    def pauli_action(self, pauli_str) -> List:
        lookup_bits = pauli_str.get_pauli_bit(self.qubits)
        new_pauli_bits, phase = self.map[lookup_bits]
        new_pauli_str = pauli_str.set_pauli_bit(new_pauli_bits, self.qubits)
        new_pauli_str.coefficient *= phase
        return [new_pauli_str]
