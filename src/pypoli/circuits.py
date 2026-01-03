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
                    # Use jax.lax.cond or similar for tracing-safe condition if needed,
                    # but for simple structure-changing operations (filtering list),
                    # we cannot filter based on traced values inside JIT.
                    #
                    # However, term.coefficient is a JAX Tracer when inside JIT.
                    # Python's `if` cannot evaluate a Tracer.
                    #
                    # SOLUTION: We cannot drop terms based on traced coefficient values during JIT compilation
                    # because the structure of the computation graph (which terms exist) depends on values.
                    #
                    # If we are inside JIT (tracer), we CANNOT remove terms from the list based on value.
                    # We can only zero them out (masking), but they still exist in the graph.
                    #
                    # BUT, Pauli propagation is inherently dynamic structure.
                    # This means we CANNOT JIT compile the `propagate` function if truncation depends on dynamic values.
                    #
                    # Strategy:
                    # 1. If we are not JITing (concrete values), we filter.
                    # 2. If we are JITing (tracers), we CANNOT filter list length.
                    #    We can only multiply by a mask: coeff = coeff * (abs(coeff) >= threshold).
                    #    This keeps the term but with 0 coefficient.
                    
                    is_tracer = hasattr(term.coefficient, 'aval') or hasattr(term.coefficient, 'tracer')
                    
                    if not is_tracer:
                        # Concrete value execution (eager mode)
                        if jnp.abs(term.coefficient) < min_abs_coeff:
                            keep = False
                    else:
                        # JIT/Traced execution
                        # We cannot change the list structure based on values.
                        # We keep the term but zero out its coefficient if it's small.
                        # Note: This defeats the speedup purpose of truncation (reducing terms),
                        # but allows JIT to run.
                        # To truly get speedup, one must not JIT the propagation structure logic,
                        # or use static parameters that don't change.
                        #
                        # However, for hybrid training, parameters change, so coefficients change.
                        # So the set of Pauli strings would change dynamically.
                        # JAX JIT requires static graph structure.
                        #
                        # Conclusion: Dynamic truncation based on coefficients is incompatible with JAX JIT
                        # if we want to actually remove terms from the list to save compute.
                        #
                        # For this specific error, we will use a mask for Tracers.
                        mask = (jnp.abs(term.coefficient) >= min_abs_coeff).astype(term.coefficient.dtype)
                        term.coefficient = term.coefficient * mask
                        # We keep the term (it has 0 coeff now if truncated)
                        keep = True

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

            # Deduplication / Merging of terms
            # This is crucial for performance to prevent exponential growth of redundant terms.
            # We group by Pauli string signature (dict keys) and sum coefficients.
            # In JIT mode, this "sum" will be a sum of Tracers.
            
            merged_terms = {}
            for term in filtered:
                # PauliString needs to be hashable or have a unique string rep.
                # Assuming str(term) or term.paulis (frozenset/tuple) is a good key.
                # PauliString.paulis is a dict {qubit: 'X/Y/Z'}. We can make it a tuple of sorted items.
                
                # Create a canonical key for the Pauli operator part (ignoring coefficient)
                # Sort by qubit index
                sorted_items = tuple(sorted(term.paulis.items()))
                key = sorted_items 
                
                if key in merged_terms:
                    # Sum coefficients
                    merged_terms[key].coefficient = merged_terms[key].coefficient + term.coefficient
                else:
                    # Store the term (we clone it to be safe, though not strictly necessary if we don't mutate in place elsewhere)
                    merged_terms[key] = term
            
            current_terms = list(merged_terms.values())

        return current_terms

    def __repr__(self):
        gates_repr = "\n".join(f"  {gate}" for gate in self.gates)
        return f"Circuit with {len(self.gates)} gates:\n{gates_repr}"