"""
Core Pauli algebra implementation (JAX-only)
"""
import jax
import jax.numpy as jnp
from typing import Dict, List, Union, Any


class Parameter:
    """A symbolic parameter for parameterized gates."""
    def __init__(self, name: str):
        self.name = name
    
    def __repr__(self):
        return f"Parameter('{self.name}')"

class PauliString:
    """
    JAX-compatible Pauli string implementation.

    Represents a Pauli string as a dictionary of qubit-index to Pauli operator.
    """

    # Pauli to bit encoding: 2 bits per Pauli
    # I = 00 (0), X = 01 (1), Y = 10 (2), Z = 11 (3)
    _pauli_to_bit = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
    _bit_to_pauli = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}

    def __init__(self, paulis: Dict[int, str], coefficient: Any = 1.0):
        """
        Initialize a PauliString.

        Args:
            paulis: Dictionary mapping qubit indices to Pauli operators ('I', 'X', 'Y', 'Z')
            coefficient: Complex coefficient for this Pauli string
        """
        # Store only non-identity operators
        self.paulis = {q: p for q, p in paulis.items() if p != 'I'}
        self.coefficient = jnp.asarray(coefficient, dtype=jnp.complex64)

    def __repr__(self):
        if not self.paulis:
            return f"PauliString(I, {self.coefficient})"
        terms = [f"{p}{q}" for q, p in sorted(self.paulis.items())]
        return f"PauliString({'·'.join(terms)}, {self.coefficient})"

    def __mul__(self, other: Union['PauliString', float, complex]):
        """
        Multiply two Pauli strings or a Pauli string by a scalar.
        """
        if isinstance(other, PauliString):
            # Combine Pauli strings
            new_paulis = {}
            phase = self.coefficient * other.coefficient

            # Combine operators for all qubits
            all_qubits = set(self.paulis.keys()) | set(other.paulis.keys())

            for q in all_qubits:
                p1 = self.paulis.get(q, 'I')
                p2 = other.paulis.get(q, 'I')

                # Pauli multiplication rules
                if p1 == 'I':
                    new_p = p2
                elif p2 == 'I':
                    new_p = p1
                elif p1 == p2:
                    new_p = 'I'
                    # Phase remains the same for same operators
                elif (p1, p2) == ('X', 'Y'):
                    new_p = 'Z'
                    phase *= 1j
                elif (p1, p2) == ('Y', 'X'):
                    new_p = 'Z'
                    phase *= -1j
                elif (p1, p2) == ('Y', 'Z'):
                    new_p = 'X'
                    phase *= 1j
                elif (p1, p2) == ('Z', 'Y'):
                    new_p = 'X'
                    phase *= -1j
                elif (p1, p2) == ('Z', 'X'):
                    new_p = 'Y'
                    phase *= 1j
                elif (p1, p2) == ('X', 'Z'):
                    new_p = 'Y'
                    phase *= -1j
                else:
                    raise ValueError(f"Invalid Pauli operators: {p1} and {p2}")

                if new_p != 'I':
                    new_paulis[q] = new_p

            return PauliString(new_paulis, phase)
        else:
            # Scalar multiplication
            return PauliString(self.paulis, self.coefficient * other)

    def __rmul__(self, other: Union[float, complex]):
        return self * other

    def get_pauli_bit(self, qinds):
        """
        Get the bit encoding of the Pauli operators acting on the specified qubits.

        Args:
            qinds: List or tuple of qubit indices

        Returns:
            int: Combined bit encoding of the Pauli operators on the specified qubits.
                The operator on qubit qinds[k] occupies bits 2k and 2k+1.
        """
        bits = 0
        for i, q in enumerate(sorted(qinds)):
            pauli = self.paulis.get(q, 'I')
            bits |= self._pauli_to_bit[pauli] << (2 * i)
        return bits

    def set_pauli_bit(self, new_bits, qinds):
        """
        Create a new PauliString with the specified qubits set to the Pauli operators
        encoded in new_bits.

        Args:
            new_bits: int encoding the new Pauli operators (2 bits per qubit)
            qinds: List or tuple of qubit indices, sorted in ascending order

        Returns:
            PauliString: New PauliString with the modified Pauli operators.
        """
        new_paulis = self.paulis.copy()

        # Create a sorted list of qubit indices to ensure correct bit order
        sorted_qinds = sorted(qinds)

        # Update each qubit's Pauli operator
        for i, q in enumerate(sorted_qinds):
            # Extract 2 bits for this qubit
            pauli_bit = (new_bits >> (2 * i)) & 3
            pauli = self._bit_to_pauli[pauli_bit]

            if pauli == 'I':
                # Remove from dictionary if it's I
                new_paulis.pop(q, None)
            else:
                # Update or add the Pauli operator
                new_paulis[q] = pauli

        # Return new PauliString with same coefficient
        return PauliString(new_paulis, self.coefficient)

    @staticmethod
    def _bitpaulimultiply(p1_bits: int, p2_bits: int) -> tuple[int, int]:
        """
        Multiply two Pauli strings represented as bit patterns.

        Args:
            p1_bits: Bit pattern for first Pauli string
            p2_bits: Bit pattern for second Pauli string

        Returns:
            Tuple of (result_bits, im_exponent): Resulting Pauli string bits and imaginary exponent
        """
        # Pauli multiplication rules using bitwise operations
        # For each 2-bit Pauli segment:
        # - X*Y = iZ    (01 * 10 = 11 with im^1)
        # - Y*Z = iX    (10 * 11 = 01 with im^1)
        # - Z*X = iY    (11 * 01 = 10 with im^1)
        # - Y*X = -iZ   (10 * 01 = 11 with im^3)
        # - Z*Y = -iX   (11 * 10 = 01 with im^3)
        # - X*Z = -iY   (01 * 11 = 10 with im^3)
        # - Same Pauli or with I: result remains same with im^0 or im^2 (which is -1)

        result_bits = 0
        im_exponent = 0
        mask = 0b11  # 2 bits for Pauli operator

        while p1_bits != 0 or p2_bits != 0:
            # Extract the current Pauli operators
            pauli1 = p1_bits & mask
            pauli2 = p2_bits & mask

            if pauli1 == 0:  # I & something
                product = pauli2
                exponent = 0
            elif pauli2 == 0:  # something & I
                product = pauli1
                exponent = 0
            elif pauli1 == pauli2:  # same Pauli
                product = 0  # I
                exponent = 2  # (-1) factor
            else:
                # Different non-I Paulis
                if (pauli1 == 1 and pauli2 == 2) or (pauli1 == 2 and pauli2 == 3) or (pauli1 == 3 and pauli2 == 1):
                    # X*Y=iZ, Y*Z=iX, Z*X=iY
                    if pauli1 == 1 and pauli2 == 2:
                        product = 3  # Z
                    elif pauli1 == 2 and pauli2 == 3:
                        product = 1  # X
                    else:  # Z*X
                        product = 2  # Y
                    exponent = 1
                else:
                    # Y*X=-iZ, Z*Y=-iX, X*Z=-iY
                    if pauli1 == 2 and pauli2 == 1:
                        product = 3  # Z
                    elif pauli1 == 3 and pauli2 == 2:
                        product = 1  # X
                    else:  # X*Z
                        product = 2  # Y
                    exponent = 3

            result_bits |= product
            im_exponent = (im_exponent + exponent) % 4

            # Shift to next qubit
            p1_bits >>= 2
            p2_bits >>= 2
            result_bits <<= 2

        # Shift back the last one
        result_bits >>= 2

        return result_bits, im_exponent

    @staticmethod
    def _calculatesignexponent(pauli1_bits: int, pauli2_bits: int) -> int:
        """
        Calculate the exponent of the imaginary unit when multiplying two Pauli strings.

        This is a faster implementation based on bitwise operations.
        """
        im_exponent = 0
        mask = 0b11

        while pauli1_bits != 0 or pauli2_bits != 0:
            p1 = pauli1_bits & mask
            p2 = pauli2_bits & mask

            if p1 != 0 and p2 != 0 and p1 != p2:
                # Calculate the relative order
                if (p1 == 1 and p2 == 2) or (p1 == 2 and p2 == 3) or (p1 == 3 and p2 == 1):
                    im_exponent = (im_exponent + 1) % 4
                else:
                    im_exponent = (im_exponent + 3) % 4

            pauli1_bits >>= 2
            pauli2_bits >>= 2

        return im_exponent

    def __add__(self, other: 'PauliString'):
        """
        Add two Pauli strings to form a PauliSum.
        """
        return PauliSum([self, other])


class PauliSum:
    """
    JAX-compatible Pauli sum implementation.

    Represents a sum of PauliString objects.
    """

    def __init__(self, nqubits: int = 0, pauli_strings: List[PauliString] = None):
        """
        Initialize a PauliSum.

        Args:
            nqubits: Number of qubits (for convenience, not strictly enforced)
            pauli_strings: List of PauliString objects to initialize with
        """
        self.nqubits = nqubits
        self.pauli_strings = pauli_strings if pauli_strings is not None else []

    def __repr__(self):
        return " + \n".join(map(str, self.pauli_strings))

    def __add__(self, other: Union['PauliString', 'PauliSum']):
        """
        Add another PauliString or PauliSum to this PauliSum.
        """
        if isinstance(other, PauliString):
            return PauliSum(self.nqubits, self.pauli_strings + [other])
        elif isinstance(other, PauliSum):
            return PauliSum(max(self.nqubits, other.nqubits),
                          self.pauli_strings + other.pauli_strings)
        else:
            raise TypeError(f"Cannot add PauliSum with {type(other)}")

    def add(self, pauli_symbols: Union[str, list], qubits: Union[int, list], coefficient: float = 1.0):
        """
        Add a Pauli term to the PauliSum.

        Args:
            pauli_symbols: Single Pauli symbol (string like 'X', 'Y', 'Z') or list for multi-qubit
            qubits: Single qubit index or list of qubit indices
            coefficient: Coefficient for this term
        """
        # Convert to list if single value
        if isinstance(pauli_symbols, str):
            pauli_symbols = [pauli_symbols]
        if isinstance(qubits, int):
            qubits = [qubits]

        # Validate input
        if len(pauli_symbols) != len(qubits):
            raise ValueError(f"Number of Pauli symbols ({len(pauli_symbols)}) must match number of qubits ({len(qubits)})")

        # Create Pauli dictionary
        pauli_dict = {q: p for p, q in zip(pauli_symbols, qubits)}

        # Create PauliString and add to the sum
        pauli_str = PauliString(pauli_dict, coefficient)
        self.pauli_strings.append(pauli_str)

    def __iadd__(self, other: Union['PauliString', 'PauliSum']):
        """
        In-place addition for PauliSum.
        """
        if isinstance(other, PauliString):
            self.pauli_strings.append(other)
        elif isinstance(other, PauliSum):
            self.pauli_strings.extend(other.pauli_strings)
            self.nqubits = max(self.nqubits, other.nqubits)
        else:
            raise TypeError(f"Cannot add PauliSum with {type(other)}")
        return self

    def __mul__(self, other: Union[float, complex]):
        """
        Multiply all terms in the PauliSum by a scalar.
        """
        return PauliSum([term * other for term in self.pauli_strings])

    def __rmul__(self, other: Union[float, complex]):
        return self * other

    def __len__(self):
        return len(self.pauli_strings)
