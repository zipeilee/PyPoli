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

    def __init__(self, gates: List[Gate] | None = None, structure=None, params=None):
        """
        Initialize a quantum circuit.

        Args:
            gates: List of gates to add to the circuit initially
            structure: Internal structure info (e.g. for repeated layers)
            params: Bound parameters for the circuit (JAX array)
        """
        self.gates = gates if gates is not None else []
        self.structure = structure # {'type': 'repeated_layer', 'layer_circuit': Circuit, 'depth': int}
        self.params = params

    @classmethod
    def from_layer(cls, layer_circuit: 'Circuit', depth: int):
        """
        Create a Circuit from a repeated layer.
        
        Args:
            layer_circuit: The circuit template for one layer (with Parameters)
            depth: Number of repetitions
        """
        # We don't expand gates here. We store the structure.
        return cls(gates=[], structure={'type': 'repeated_layer', 'layer_circuit': layer_circuit, 'depth': depth})

    def bind(self, params) -> 'Circuit':
        """
        Bind parameters to the circuit.
        
        Args:
            params: JAX array of parameters.
            
        Returns:
            A new Circuit object with bound parameters.
        """
        # Return a shallow copy with new params
        # We share gates/structure to save memory
        return Circuit(gates=self.gates, structure=self.structure, params=params)

    def add_gate(self, gate: Gate):
        """
        Add a gate to the circuit.

        Args:
            gate: Gate to add to the circuit
        """
        if self.structure:
            raise ValueError("Cannot add gates to a structured Circuit (e.g. created via from_layer).")
        self.gates.append(gate)


    def add(self, gate: Gate):
        """
        Alias for add_gate.
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

    def propagate(self, pauli_obj, truncation=None, max_weight=None, min_abs_coeff=None, damping=0.0, params=None) -> List:
        """
        Propagate a Pauli string or Pauli sum through the circuit.
        
        Args:
            pauli_obj: PauliString or PauliSum to propagate
            max_weight: Maximum Pauli weight to keep
            min_abs_coeff: Minimum absolute coefficient to keep
            damping: Damping factor for soft truncation. 
                     Condition: |coeff| * 10^(-damping * weight) >= min_abs_coeff
            params: Optional parameters. If None, uses bound self.params.
            
        Note:
            If the circuit is parameterized, parameters must be bound via .bind(params) before calling propagate.
        """
        from .core import PauliString, PauliSum

        # Use bound parameters if not provided explicitly
        if params is None:
            params = self.params

        # Standard Python propagation implementation (JIT-traceable but slow compile if large)
        if isinstance(pauli_obj, PauliSum):
            # Propagate each term in the sum
            propagated = []
            for term in pauli_obj.pauli_strings:
                propagated.extend(self.propagate(term, truncation=truncation, max_weight=max_weight, min_abs_coeff=min_abs_coeff, damping=damping))
            return propagated

        if isinstance(pauli_obj, list):
            # Propagate each term in the list
            propagated = []
            for term in pauli_obj:
                propagated.extend(self.propagate(term, truncation=truncation, max_weight=max_weight, min_abs_coeff=min_abs_coeff, damping=damping))
            return propagated

        # Base case: Single PauliString
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
                    # Use bit counting for faster weight calculation
                    term_weight = term.bits.bit_count() // 2  # Each non-I Pauli uses 2 bits
                    if term_weight > max_weight:
                        keep = False

                # Apply coefficient magnitude truncation if specified
                if keep and min_abs_coeff is not None:
                    # Damping Truncation Logic
                    # If damping > 0, we require larger coefficients for larger weights.
                    # Threshold = min_abs_coeff * 10^(damping * weight)
                    
                    current_threshold = min_abs_coeff
                    if damping > 0:
                        # Use bit counting for faster weight calculation
                        term_weight = term.bits.bit_count() // 2
                        current_threshold = min_abs_coeff * (10.0 ** (damping * term_weight))
                    
                    # ... (rest of logic using current_threshold instead of min_abs_coeff)
                    
                    is_tracer = hasattr(term.coefficient, 'aval') or hasattr(term.coefficient, 'tracer')
                    
                    if not is_tracer:
                        # Concrete value execution (eager mode)
                        if jnp.abs(term.coefficient) < current_threshold:
                            keep = False
                    else:
                        # JIT/Traced execution
                        # We cannot change the list structure based on values.
                        mask = (jnp.abs(term.coefficient) >= current_threshold).astype(term.coefficient.dtype)
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
            # We group by Pauli string signature (bits) and sum coefficients.

            merged_terms = {}
            for term in filtered:
                # Use bits directly as key - much faster than dict conversion
                key = (term.bits, term.nqubits)

                if key in merged_terms:
                    # Sum coefficients
                    merged_terms[key].coefficient = merged_terms[key].coefficient + term.coefficient
                else:
                    merged_terms[key] = term

            current_terms = list(merged_terms.values())

        return current_terms

    def __repr__(self):
        gates_repr = "\n".join(f"  {gate}" for gate in self.gates)
        return f"Circuit with {len(self.gates)} gates:\n{gates_repr}"