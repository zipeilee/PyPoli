# PyPoli - JAX-only Pauli Propagation Library

PyPoli是一个纯JAX实现的Pauli传播库，基于PauliPropagation.jl的设计原则，专注于高效的可观测值中心Pauli传播，支持自动微分和硬件高效Ansatz。

## 核心设计原则
1. **无初始状态假设**：所有计算都在真空态|000...0⟩上执行
2. **可观测值中心**：在海森堡绘景中传播Pauli字符串可观测值通过量子电路
3. **海森堡绘景**：门以相反顺序应用（U = U_n ... U_2 U_1 → U† O U = U_1† U_2† ... U_n† O U_n ... U_2 U_1）
4. **直接期望值计算**：将传播和期望值计算结合到一个操作中
5. **自动微分**：完整的JAX支持用于基于梯度的优化
6. **截断支持**：系数大小和权重截断以提高效率


## Installation
```bash
# Install from source
cd PyPoli
pip install -e .
```

## Import Statement
```python
from pypoli import PauliString, PauliSum, Circuit
from pypoli import I, X, Y, Z, H, S, T, RX, RY, RZ, RXX, RYY, RZZ, CNOT, CZ
from pypoli import propagate, expectation_value, batch_expectation_value
# Ansatz imports
from pypoli import VariationalAnsatz, HardwareEfficient, HardwareEfficientAnsatz
```

## Ansatz Module
PyPoli includes a pre-built hardware-efficient ansatz implementation for convenience.

### HardwareEfficient
The HardwareEfficient ansatz features:
- **Single-qubit gates**: RX(theta) - RZ(phi) - RX(gamma) sequence on all qubits per layer
- **Two-qubit gates**: Alternating CZ gates pattern:
  - Even layers: CZ on qubit pairs (0-1, 2-3, 4-5, ...)
  - Odd layers: CZ on qubit pairs (1-2, 3-4, 5-6, ...)

**Parameter requirements**: Shape must be (num_layers, num_qubits, 3) where:
- params[layer][qubit][0] = theta (RX)
- params[layer][qubit][1] = phi (RZ)
- params[layer][qubit][2] = gamma (RX)

**Usage**:
```python
import jax.numpy as jnp

# Initialize ansatz
num_qubits = 4
num_layers = 2
ansatz = HardwareEfficient(num_qubits, num_layers)

# Generate random parameters
params = jnp.random.rand(num_layers, num_qubits, 3)

# Create circuit
circuit = ansatz.generate_circuit(params)
print(circuit)
```

**Extending for custom ansatz**:
All ansatzes inherit from `VariationalAnsatz` base class. To create your own:

```python
from pypoli import VariationalAnsatz, Circuit, RX, RY

class MyCustomAnsatz(VariationalAnsatz):
    def generate_circuit(self, params):
        circuit = Circuit()
        # Implement your custom gate sequence here
        # Example: RX-RY layers with CNOT
        for layer in range(self.num_layers):
            for qubit in range(self.num_qubits):
                circuit.add_gate(RX(qubit, params[layer, qubit, 0]))
                circuit.add_gate(RY(qubit, params[layer, qubit, 1]))

        return circuit

# Usage
ansatz = MyCustomAnsatz(3, 2)
circuit = ansatz.generate_circuit(jnp.random.rand(2, 3, 2))
```

## Change Log

### Version 0.1.0 - Current Implementation

#### Improvements to Core Pauli Algebra:
1. **Bit Encoding System**:
   - Implemented bit encoding for Pauli strings: I=00(0), X=01(1), Y=10(2), Z=11(3)
   - Added `get_pauli_bit()` method to retrieve bit patterns for specified qubits
   - Added `set_pauli_bit()` method to update Pauli strings from bit patterns
   - Faster than dictionary lookups for gate operations

2. **Bitwise Pauli Multiplication**:
   - Added `_bitpaulimultiply()` for efficient bit-level Pauli string multiplication
   - Added `_calculatesignexponent()` to compute imaginary phase factors
   - Uses bitwise operations instead of string comparisons for speed

#### Gate System Improvements:
1. **Clifford Gates with Lookup Tables**:
   - Rewrote I, X, Y, Z, H, S, CNOT, CZ gates to use precomputed lookup tables
   - Lookup tables map input Pauli bit patterns to output patterns and phases
   - Dramatically faster than conditional-based gate action

2. **Pauli Rotation Gates Optimization**:
   - Updated RX, RY, RZ to use bitwise operations for commutativity checks
   - Uses precomputed bit patterns instead of string comparisons
   - Maintains parameterized behavior while improving performance

## Core API

### 1. PauliString
Represents a single Pauli string.

**Usage**:
```python
# Identity operator
I = PauliString({}, 1.0)

# Z0 operator
Z0 = PauliString({0: 'Z'}, 1.0)

# X0⊗Y1⊗Z2 operator
XYZ = PauliString({0: 'X', 1: 'Y', 2: 'Z'}, 0.5+0.3j)
```

### 2. PauliSum
Represents a linear combination of Pauli strings.

**Usage**:
```python
# X0 + Z1 + 0.5 X2
sum_pauli = PauliString({0: 'X'}) + PauliString({1: 'Z'}) + 0.5 * PauliString({2: 'X'})
```

### 3. Circuit
Represents a quantum circuit.

**Initialization**:
```python
# Empty circuit
circuit = Circuit()

# Circuit with initial gates
circuit = Circuit([H(0), X(1), CNOT(0, 1)])
```

**Adding Gates**:
```python
# Single gate
circuit.add_gate(RY(0, jnp.pi/4))

# List-like append
circuit.append(RZ(1, jnp.pi/2))

# Multiple gates using extend
gates = [RX(q, jnp.random.rand()) for q in range(3)]
circuit.extend(gates)

# For loop construction
for qubit in range(4):
    circuit.add_gate(H(qubit))
    circuit.add_gate(RY(qubit, params[qubit]))

for qubit in range(3):
    circuit.add_gate(CNOT(qubit, qubit+1))
```

**Propagation**:
```python
# Propagate an observable through the circuit
observable = PauliString({0: 'Z', 1: 'X'})
propagated = circuit.propagate(observable, max_weight=6, min_abs_coeff=1e-4)
```

**Propagation Parameters**:
- `max_weight`: Maximum Pauli weight to keep (number of non-identity operators)
- `min_abs_coeff`: Minimum coefficient magnitude to keep
- These truncations are applied after each gate for efficiency

### 4. Gates
PyPoli supports the following gates:

**Single-qubit gates**:
- `I(qubit)`: Identity
- `X(qubit)`: Pauli X
- `Y(qubit)`: Pauli Y
- `Z(qubit)`: Pauli Z
- `H(qubit)`: Hadamard
- `S(qubit)`: Phase gate
- `T(qubit)`: T gate

**Single-qubit rotation gates**:
- `RX(qubit, theta)`: Rotation around X-axis
- `RY(qubit, theta)`: Rotation around Y-axis
- `RZ(qubit, theta)`: Rotation around Z-axis

**Two-qubit gates**:
- `RXX(qubit1, qubit2, theta)`: Rotation around XX-axis
- `RYY(qubit1, qubit2, theta)`: Rotation around YY-axis
- `RZZ(qubit1, qubit2, theta)`: Rotation around ZZ-axis
- `CNOT(control, target)`: CNOT gate

### 5. Expectation Value Calculation
Calculates ⟨0|U† O U|0⟩ for an observable O and circuit U.

**Usage**:
```python
# Single expectation value
exp_val = expectation_value(circuit, PauliString({0: 'Z'}))

# With truncation
exp_val = expectation_value(circuit, PauliString({0: 'Z'}), max_weight=4)

# Batch calculation
observables = [PauliString({q: 'Z'}) for q in range(3)]
exp_vals = batch_expectation_value(circuit, observables)
```

### 6. Automatic Differentiation
Full JAX support for gradient calculation.

**Usage**:
```python
# Parameterized circuit
def create_circuit(params):
    circuit = Circuit()
    circuit.add_gate(RX(0, params[0]))
    circuit.add_gate(RZZ(0, 1, params[1]))
    circuit.add_gate(RY(1, params[2]))
    return circuit

# Loss function
def loss(params):
    circuit = create_circuit(params)
    return expectation_value(circuit, PauliString({0: 'Z', 1: 'X'}))

# Initial parameters
params = jnp.array([0.0, 0.0, 0.0])

# Compute gradient
grad = jax.grad(loss)(params)

# JIT compilation for speed
loss_jit = jax.jit(loss)
exp_val = loss_jit(params)
```

## Examples

### 1. Hardware-Efficient Ansatz Construction
```python
num_qubits = 4
num_layers = 2
params = jnp.random.rand(num_layers, num_qubits, 2)

circuit = Circuit()

for layer in range(num_layers):
    # Single-qubit rotations
    for qubit in range(num_qubits):
        circuit.add_gate(RX(qubit, params[layer, qubit, 0]))
        circuit.add_gate(RY(qubit, params[layer, qubit, 1]))

    # Two-qubit entangling gates
    for qubit in range(num_qubits - 1):
        circuit.add_gate(CNOT(qubit, qubit + 1))

    # Circular CZ gate for periodic boundary conditions
    circuit.add_gate(CZ(num_qubits - 1, 0))
```

### 2. Simple Expectation Calculation
```python
# Circuit: |+> state preparation and X measurement
circuit = Circuit([H(0)])
O = PauliString({0: 'X'})
exp_val = expectation_value(circuit, O)  # Should be 1.0
```

### 3. Truncation Example
```python
# Complex circuit with many terms
circuit = Circuit([H(0), H(1), H(2), RXX(0,1,jnp.pi/4), RYY(1,2,jnp.pi/3)])

# Without truncation
exp_full = expectation_value(circuit, PauliString({0: 'Z', 2: 'Z'}))

# With truncation
exp_trunc = expectation_value(circuit, PauliString({0: 'Z', 2: 'Z'}), max_weight=2, min_abs_coeff=1e-3)

print(f"Full expectation: {exp_full}")
print(f"Truncated expectation: {exp_trunc}")
print(f"Relative difference: {abs(exp_full - exp_trunc)/abs(exp_full)}")
```

## Documentation
- **File structure**:
  - `core.py`: PauliString and PauliSum implementation
  - `gates.py`: Quantum gate definitions
  - `circuits.py`: Circuit implementation with propagation
  - `propagation.py`: Expectation value calculation
- **Key functions**:
  - `circuit.add_gate(gate)`: Add a gate to the circuit
  - `circuit.append(gate)`: List-like gate addition
  - `circuit.extend(gates)`: Add multiple gates at once
  - `circuit.propagate(observable)`: Propagate observable through circuit
  - `expectation_value(circuit, observable)`: Calculate expectation value
  - `batch_expectation_value(circuit, observables)`: Batch expectation calculation

## Performance Tips
- Use `jax.jit()` on expectation_value for large circuits
- Apply truncation to reduce the number of terms during propagation
- Use list comprehension and `extend()` for fast circuit construction
- Take advantage of automatic differentiation for optimization tasks

## Future Improvements
- Support for more two-qubit gates (CZ, SWAP)
- Noisy channel support
- Symmetry-aware truncation
- Visualization tools
