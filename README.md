# PyPoli - Pauli Propagation for Python

PyPoli is a Python library for efficient classical simulation of quantum circuits using Pauli propagation. It provides seamless integration with JAX and Flax for automatic differentiation, making it suitable for quantum machine learning applications.

## Features

- **Pauli Propagation**: Efficient simulation of quantum circuits using Pauli string propagation in the Heisenberg picture.
- **Automatic Differentiation**: JAX/Flax compatible for gradient-based optimization.
- **Core Quantum Gates**: Support for I, X, Y, Z, H, S, RX, RY, RZ, CNOT, CZ, etc.
- **Vacuum-Centric**: Optimized for calculations starting from the $|00\dots0\rangle$ state.
- **Truncation Support**: Option to truncate small terms during propagation to manage complexity.

## Installation

```bash
pip install -e .
```

## Basic Usage

PyPoli uses the Heisenberg picture. You define a circuit $U$ and an observable $O$, and the library computes the propagated observable $O' = U^\dagger O U$. The expectation value is then $\langle 0| O' |0\rangle$.

```python
import pypoli
from pypoli import Circuit, PauliString, H, X, RZZ
import jax.numpy as jnp

# Create a circuit
circuit = Circuit()
circuit.add_gate(H(0))
circuit.add_gate(X(1))
circuit.add_gate(RZZ(0, 1, jnp.pi / 2))

# Define an observable to measure (e.g., Z on qubit 0)
observable = PauliString({0: 'Z'}, 1.0)

# Propagate the observable through the circuit
# This computes U_dag * observable * U
propagated = circuit.propagate(observable)

# Compute expectation value for the vacuum state |00...0>
exp_val = pypoli.expectation_value(propagated)
print(f"Expectation value: {exp_val}")
```

## JAX Integration

```python
import jax
import jax.numpy as jnp
from pypoli import Circuit, RX, RY, RZZ, PauliString, expectation_value

# Define a parameterized circuit
def create_circuit(params):
    circuit = Circuit()
    circuit.add_gate(RX(0, params[0]))
    circuit.add_gate(RY(1, params[1]))
    circuit.add_gate(RZZ(0, 1, params[2]))
    return circuit

# Loss function
def loss(params):
    circuit = create_circuit(params)
    # Observable: Z0 * Z1
    observable = PauliString({0: 'Z', 1: 'Z'}, 1.0)
    
    # Propagate and measure
    propagated = circuit.propagate(observable)
    exp_val = expectation_value(propagated)
    
    # Target value: 0.7
    return jnp.abs(exp_val - 0.7) ** 2

# Optimize
params = jnp.array([0.0, 0.0, 0.0])
grad = jax.grad(loss)(params)
print(f"Gradient: {grad}")
```

## Documentation

- **Core**: `PauliString`, `Circuit`, `expectation_value`
- **Gates**: `I`, `X`, `Y`, `Z`, `H`, `S`, `RX`, `RY`, `RZ`, `RXX`, `RYY`, `RZZ`, `CNOT`, `CZ`

## License

MIT License
