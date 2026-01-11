"""
Core Pauli algebra implementation (JAX-only)
Dual-storage architecture: bit encoding for CPU, JAX arrays for GPU/AD
"""
import jax
import jax.numpy as jnp
from typing import Dict, List, Union, Any, Tuple


class Parameter:
    """A symbolic parameter for parameterized gates."""
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"Parameter('{self.name}')"


class PauliString:
    """
    JAX-compatible Pauli string with dual-storage architecture.

    Storage:
    - self.bits (int): Bit encoding for CPU-efficient operations
    - Each qubit uses 2 bits: I=00(0), X=01(1), Y=10(2), Z=11(3)
    - Qubit i occupies bits [2*i, 2*i+1]

    JAX Array Conversion:
    - to_array(): Converts to uint8 array for GPU/AD operations
    - from_array(): Creates from JAX array
    """

    # Pauli to bit encoding: 2 bits per Pauli
    # I = 00 (0), X = 01 (1), Y = 10 (2), Z = 11 (3)
    _pauli_to_bit = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
    _bit_to_pauli = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}

    def __init__(self, paulis: Union[Dict[int, str], int], coefficient: Any = 1.0, nqubits: int = None):
        """
        Initialize a PauliString.

        Args:
            paulis: Dictionary mapping qubit indices to Pauli operators,
                   OR integer bit encoding (if nqubits is provided)
            coefficient: Complex coefficient for this Pauli string
            nqubits: Total number of qubits (required if paulis is int)
        """
        self.coefficient = jnp.asarray(coefficient, dtype=jnp.complex64)

        if isinstance(paulis, int):
            # Initialize from bit encoding
            self.bits = paulis
            if nqubits is None:
                raise ValueError("nqubits must be provided when initializing from bits")
            self.nqubits = nqubits
        else:
            # Initialize from dictionary (backward compatible)
            if nqubits is None:
                nqubits = max(paulis.keys()) + 1 if paulis else 1
            self.nqubits = nqubits
            self.bits = self._dict_to_bits(paulis, nqubits)

    @staticmethod
    def _dict_to_bits(paulis: Dict[int, str], nqubits: int) -> int:
        """Convert dictionary to bit encoding."""
        bits = 0
        for q, p in paulis.items():
            if p != 'I':
                bits |= PauliString._pauli_to_bit[p] << (2 * q)
        return bits

    def to_dict(self) -> Dict[int, str]:
        """Convert bit encoding back to dictionary (for debugging/compatibility)."""
        paulis = {}
        for q in range(self.nqubits):
            pauli_bit = (self.bits >> (2 * q)) & 3
            if pauli_bit != 0:
                paulis[q] = self._bit_to_pauli[pauli_bit]
        return paulis

    def to_array(self) -> jnp.ndarray:
        """
        Convert to JAX array for GPU/AD operations.

        Returns:
            jnp.ndarray: uint8 array of shape (nqubits,)
                         Values: 0=I, 1=X, 2=Y, 3=Z
        """
        # Extract 2-bit values for each qubit
        arr = jnp.zeros(self.nqubits, dtype=jnp.uint8)
        for q in range(self.nqubits):
            arr = arr.at[q].set((self.bits >> (2 * q)) & 3)
        return arr

    @classmethod
    def from_array(cls, arr: jnp.ndarray, coefficient: Any = 1.0) -> 'PauliString':
        """
        Create PauliString from JAX array.

        Args:
            arr: uint8 array of shape (nqubits,), values 0-3
            coefficient: Complex coefficient

        Returns:
            PauliString with bit encoding
        """
        nqubits = len(arr)
        bits = 0
        # Convert JAX array to Python int for bit encoding
        arr_list = list(arr) if hasattr(arr, '__iter__') else arr
        for q, val in enumerate(arr_list):
            if val != 0:
                bits |= int(val) << (2 * q)
        return cls(bits, coefficient, nqubits)

    @property
    def paulis(self) -> Dict[int, str]:
        """Property for backward compatibility - converts bits to dict."""
        return self.to_dict()

    def __repr__(self):
        paulis = self.to_dict()
        if not paulis:
            return f"PauliString(I, {self.coefficient})"
        terms = [f"{p}{q}" for q, p in sorted(paulis.items())]
        return f"PauliString({'·'.join(terms)}, {self.coefficient})"

    def __mul__(self, other: Union['PauliString', float, complex]):
        """
        Multiply two Pauli strings or a Pauli string by a scalar.
        Uses fast bit operations for Pauli multiplication.
        """
        if isinstance(other, PauliString):
            # Ensure both have same nqubits
            max_nq = max(self.nqubits, other.nqubits)
            self_bits = self.bits
            other_bits = other.bits

            # Use the fast bit multiplication method
            result_bits, im_exponent = self._bitpaulimultiply(self_bits, other_bits)

            # Calculate phase from imaginary exponent and coefficients
            phase_factors = [1, 1j, -1, -1j]  # i^0, i^1, i^2, i^3
            phase = phase_factors[im_exponent]
            new_coefficient = self.coefficient * other.coefficient * phase

            return PauliString(result_bits, new_coefficient, max_nq)
        else:
            # Scalar multiplication
            return PauliString(self.bits, self.coefficient * other, self.nqubits)

    def __rmul__(self, other: Union[float, complex]):
        return self * other

    def get_pauli_bit(self, qinds) -> int:
        """
        Get the bit encoding of the Pauli operators acting on the specified qubits.
        Now O(k) where k = len(qinds) with simple bit operations.

        Args:
            qinds: List or tuple of qubit indices

        Returns:
            int: Combined bit encoding of the Pauli operators on the specified qubits.
                The operator on qubit qinds[k] occupies bits 2k and 2k+1.
        """
        bits = 0
        for i, q in enumerate(qinds):
            # Extract 2 bits for qubit q and place them at position 2*i
            pauli_bit = (self.bits >> (2 * q)) & 3
            bits |= pauli_bit << (2 * i)
        return bits

    def set_pauli_bit(self, new_bits: int, qinds) -> 'PauliString':
        """
        Create a new PauliString with the specified qubits set to the Pauli operators
        encoded in new_bits. Now O(k) where k = len(qinds).

        Args:
            new_bits: int encoding the new Pauli operators (2 bits per qubit)
            qinds: List or tuple of qubit indices (no need to sort!)

        Returns:
            PauliString: New PauliString with the modified Pauli operators.
        """
        # Clear the bits at specified qubit positions
        new_bits_value = self.bits
        for q in qinds:
            # Clear 2 bits at position 2*q
            mask = ~(3 << (2 * q))
            new_bits_value &= mask

        # Set the new bits
        for i, q in enumerate(qinds):
            # Extract 2 bits for this qubit from new_bits
            pauli_bit = (new_bits >> (2 * i)) & 3
            new_bits_value |= pauli_bit << (2 * q)

        return PauliString(new_bits_value, self.coefficient, self.nqubits)

    @staticmethod
    def _bitpaulimultiply(p1_bits: int, p2_bits: int) -> Tuple[int, int]:
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

        shift = 0
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

            result_bits |= product << shift
            im_exponent = (im_exponent + exponent) % 4

            # Shift to next qubit
            p1_bits >>= 2
            p2_bits >>= 2
            shift += 2

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
        return PauliSum(pauli_strings=[self, other])

    def __eq__(self, other: 'PauliString') -> bool:
        """Check equality of PauliStrings."""
        if not isinstance(other, PauliString):
            return False
        return self.bits == other.bits and self.nqubits == other.nqubits

    def __hash__(self) -> int:
        """Hash for using PauliString as dict key."""
        return hash((self.bits, self.nqubits))


class PauliSum:
    """
    JAX-compatible Pauli sum implementation.

    Represents a sum of PauliString objects.
    """

    def __init__(self, nqubits: int = 0, pauli_strings: List[PauliString] | None = None):
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
        return PauliSum(pauli_strings=[term * other for term in self.pauli_strings])

    def __rmul__(self, other: Union[float, complex]):
        return self * other

    def __len__(self):
        return len(self.pauli_strings)
