"""
Quantum circuits implementation (JAX-only)
"""
from typing import List, Any
import jax.numpy as jnp
from .gates import Gate


class Circuit:
    """
    Quantum circuit implementation with Pauli propagation support.
    """

    def __init__(self, gates: List[Gate] = None):
        """
        Initialize a quantum circuit.

        Args:
            gates: List of gates to add to the circuit initially
        """
        self.gates = gates if gates is not None else []

    def add_gate(self, gate: Gate):
        """
        Add a gate to the circuit.

        Args:
            gate: Gate to add to the circuit
        """
        self.gates.append(gate)

    def append(self, gate: Gate):
        """
        Alias for add_gate to support list-like append interface.
        """
        self.add_gate(gate)

    def extend(self, gates: list):
        """
        Add multiple gates to the circuit at once.

        Args:
            gates: List of Gate objects to add
        """
        for gate in gates:
            self.add_gate(gate)

    def propagate(self, pauli_obj, truncation=None, max_weight=None, min_abs_coeff=None) -> List:
        """
        Propagate a Pauli string or Pauli sum through the circuit with optional truncation after each gate.

        This implements the Heisenberg picture propagation:
        - Gates are applied in reverse order
        - The action of each gate is its conjugate action

        Args:
            pauli_obj: PauliString or PauliSum to propagate (Heisenberg picture)
            truncation: DEPRECATED: Use min_abs_coeff or max_weight instead.
                        None, float threshold, or callable that takes a list of PauliString
                        and returns a truncated list. If float, terms with abs(coefficient)
                        below threshold are removed.
            max_weight: Maximum Pauli weight (number of non-identity Paulis) to keep
            min_abs_coeff: Minimum absolute coefficient magnitude to keep

        Returns:
            List[PauliString]: List of Pauli strings after propagation
        """
        # Handle both PauliString and PauliSum
        from .core import PauliString, PauliSum

        if isinstance(pauli_obj, PauliSum):
            # Propagate each term in the sum
            propagated = []
            for term in pauli_obj.pauli_strings:
                propagated.extend(self.propagate(term, truncation=truncation, max_weight=max_weight, min_abs_coeff=min_abs_coeff))
            return propagated

        # Start with the input Pauli object
        current_terms = [pauli_obj]

        # Apply gates in reverse order for Heisenberg picture (as in PauliPropagation.jl)
        for gate in reversed(self.gates):
            next_terms = []
            for term in current_terms:
                next_terms.extend(gate.pauli_action(term))

            # Apply truncation after each gate
            filtered = []
            for term in next_terms:
                keep = True

                # Apply weight truncation if specified
                if max_weight is not None:
                    term_weight = len(term.paulis)  # Number of non-identity Paulis
                    if term_weight > max_weight:
                        keep = False

                # Apply coefficient magnitude truncation if specified
                if keep and min_abs_coeff is not None:
                    if jnp.abs(term.coefficient) < min_abs_coeff:
                        keep = False

                # Apply deprecated truncation parameter if still used
                if keep and truncation is not None:
                    if isinstance(truncation, float):
                        if jnp.abs(term.coefficient) < truncation:
                            keep = False
                    elif callable(truncation):
                        # This approach is not efficient but maintains backward compatibility
                        if term not in truncation([term]):
                            keep = False

                if keep:
                    filtered.append(term)

            current_terms = filtered

        return current_terms

    def __repr__(self):
        gates_repr = "\n".join(f"  {gate}" for gate in self.gates)
        return f"Circuit with {len(self.gates)} gates:\n{gates_repr}"