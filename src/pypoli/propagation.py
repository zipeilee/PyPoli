"""
Pauli propagation and expectation value calculation (JAX-only)

Core principle: This library implements observable-centric Pauli propagation.
- No initial state parameter: All calculations assume the initial state is the vacuum state |000...0⟩
- The input to expectation_value is the observable (PauliString/PauliSum) to be measured
- The library computes ⟨0| U† O U |0⟩ where U is the circuit and O is the observable
"""
import jax
import jax.numpy as jnp
from typing import List, Any


def propagate(circuit, pauli_observable, truncation=None, max_weight=None, min_abs_coeff=None, damping=0.0) -> List:
    """
    Propagate an observable (PauliString or PauliSum) through a circuit in the Heisenberg picture.

    Gates are applied in reverse order, and the result is U† O U where U is the circuit.

    Args:
        circuit: Circuit to propagate through
        pauli_observable: PauliString or PauliSum representing the observable to propagate
        truncation: DEPRECATED: Use min_abs_coeff or max_weight instead.
                    None, float threshold, or callable that takes a list of PauliString
                    and returns a truncated list. If float, terms with abs(coefficient)
                    below threshold are removed.
        max_weight: Maximum Pauli weight (number of non-identity Paulis) to keep
        min_abs_coeff: Minimum absolute coefficient magnitude to keep
        damping: Damping factor for soft truncation. Condition: |coeff| * 10^(-damping * weight) >= min_abs_coeff

    Returns:
        List[PauliString]: Propagated and possibly truncated Pauli strings
    """
    return circuit.propagate(pauli_observable, truncation=truncation, max_weight=max_weight, min_abs_coeff=min_abs_coeff, damping=damping)


def expectation_value(circuit_or_propagated, observable=None, truncation=None, max_weight=None, min_abs_coeff=None, damping=0.0) -> Any:
    """
    Calculate the expectation value of an observable for the vacuum state |000...0⟩

    The function can be called in two ways:
    1. expectation_value(circuit, observable) - propagates and calculates expectation
    2. expectation_value(propagated_terms) - calculates expectation from pre-propagated terms

    Since the initial state is always vacuum, the expectation value is simply the coefficient
    of the identity Pauli string in the propagated observable.

    Args:
        circuit_or_propagated: Circuit to apply or pre-propagated list of PauliString terms
        observable: PauliString or PauliSum representing the observable to measure (required if first arg is Circuit)
        truncation: DEPRECATED: Use min_abs_coeff or max_weight instead.
                    None, float threshold, or callable for truncation (only applied if circuit is passed)
        max_weight: Maximum Pauli weight (number of non-identity Paulis) to keep during propagation
        min_abs_coeff: Minimum absolute coefficient magnitude to keep during propagation
        damping: Damping factor for soft truncation. Condition: |coeff| * 10^(-damping * weight) >= min_abs_coeff

    Returns:
        float: Expectation value ⟨0| U† O U |0⟩
    """
    from .core import PauliString, PauliSum
    from .circuits import Circuit

    # Check if first argument is a circuit (propagate first)
    if isinstance(circuit_or_propagated, Circuit):
        if observable is None:
            raise ValueError("observable must be provided when calling expectation_value with a Circuit")
        propagated = circuit_or_propagated.propagate(observable, truncation=truncation, max_weight=max_weight, min_abs_coeff=min_abs_coeff, damping=damping)
    else:
        propagated = circuit_or_propagated

    # For vacuum state, expectation is the coefficient of identity or Z-only terms
    # <0|Z|0> = 1, <0|I|0> = 1. <0|X|0> = 0, <0|Y|0> = 0.
    expectation = 0.0
    for term in propagated:
        # Check if the term contains only 'Z' (or 'I' which is not stored)
        is_diagonal = True
        for p in term.paulis.values():
            if p != 'Z':
                is_diagonal = False
                break
        
        if is_diagonal:
            expectation += term.coefficient

    # Take the real part since expectation value should be real
    return jnp.real(expectation)


def batch_expectation_value(circuit, observables, truncation=None, max_weight=None, min_abs_coeff=None, damping=0.0) -> List:
    """
    Calculate expectation values for multiple observables in batch.

    Args:
        circuit: Circuit to apply
        observables: List of PauliString or PauliSum observables to measure
        truncation: DEPRECATED: Use min_abs_coeff or max_weight instead.
                    None, float threshold, or callable for truncation
        max_weight: Maximum Pauli weight (number of non-identity Paulis) to keep during propagation
        min_abs_coeff: Minimum absolute coefficient magnitude to keep during propagation
        damping: Damping factor for soft truncation. Condition: |coeff| * 10^(-damping * weight) >= min_abs_coeff

    Returns:
        List[float]: Expectation values for each observable
    """
    return [expectation_value(circuit, obs, truncation=truncation, max_weight=max_weight, min_abs_coeff=min_abs_coeff, damping=damping) for obs in observables]

# JIT compilation of expectation_value
# Note: This requires static arguments for non-Pytree objects (Circuit, PauliString)
# For parameterized circuits, use jax.jit on a function that creates the circuit and calls expectation_value
jit_expectation_value = jax.jit(expectation_value, static_argnums=(0, 1))
