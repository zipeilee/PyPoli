
import jax
import jax.numpy as jnp
import sys
sys.path.insert(0, './src')

from pypoli import PauliString, PauliSum

print("## 1. Basic Pauli String Operations")
# Create Pauli strings
I = PauliString({}, 1.0)  # Identity
X0 = PauliString({0: 'X'}, 1.0)  # X on qubit 0
Z1 = PauliString({1: 'Z'}, 2.0)  # Z on qubit 1 with coefficient 2

print(f"Identity: {I}")
print(f"X on qubit 0: {X0}")
print(f"Z on qubit 1: {Z1}")

# Pauli string multiplication
X0Z1 = X0 * Z1
print(f"\nX0 * Z1 = {X0Z1}")

# Pauli string addition (PauliSum)
sum_pauli = X0 + Z1 + X0Z1
print(f"\nX0 + Z1 + X0Z1 = {sum_pauli}")

print("\n## 2. Quantum Gates and Circuits")
from pypoli import Circuit, X, Y, Z, H, RX, RY, RZ, RXX, RYY, RZZ

# Create a simple circuit
circuit = Circuit()
circuit.add_gate(H(0))
circuit.add_gate(X(1))
circuit.add_gate(RZZ(0, 1, jnp.pi / 2))
circuit.add_gate(RX(0, jnp.pi / 4))

print(f"Circuit:\n{circuit}")

# Propagate an observable through the circuit (Heisenberg picture)
# Observable: Z on qubit 0
observable = PauliString({0: 'Z'}, 1.0)
propagated = circuit.propagate(observable)
print(f"\nPropagated terms for Z0: {len(propagated)}")

print("\n## 3. Expectation Value Calculation")
from pypoli import expectation_value, batch_expectation_value

# Define an observable
observable = PauliString({0: 'Z', 1: 'X'}, 1.0)

# Propagate and calculate expectation value
# This computes <0| U_dag O U |0>
propagated_obs = circuit.propagate(observable)
exp_val = expectation_value(propagated_obs)
print(f"Expectation value <Z0 X1>: {exp_val}")

# Batch calculation with multiple observables
observables = [
    PauliString({0: 'Z'}, 1.0),
    PauliString({1: 'X'}, 1.0),
    PauliString({0: 'Z', 1: 'X'}, 1.0),
    PauliString({0: 'Y', 1: 'Y'}, 1.0)
]

exp_vals = batch_expectation_value(circuit, observables)
print(f"\nBatch expectation values:")
for i, val in enumerate(exp_vals):
    print(f"  <{observables[i].paulis}>: {val}")

print("\n## 4. Automatic Differentiation (JAX)")
# Define a parameterized circuit
def create_circuit(params):
    """
    params: [theta0, theta1, theta2]
    """
    circuit = Circuit()
    circuit.add_gate(RX(0, params[0]))
    circuit.add_gate(RY(1, params[1]))
    circuit.add_gate(RZZ(0, 1, params[2]))
    return circuit

# Define loss function
def loss(params, target=0.7):
    circuit = create_circuit(params)
    # Propagate the observable Z0 Z1
    propagated = circuit.propagate(PauliString({0: 'Z', 1: 'Z'}, 1.0))
    exp_val = expectation_value(propagated)
    print(f"DEBUG: exp_val in loss = {exp_val}")
    return jnp.abs(exp_val - target) ** 2

# Initialize parameters and compute gradient
initial_params = jnp.array([0.0, 0.0, 0.0])
grad_loss = jax.grad(loss)

print(f"Initial loss: {loss(initial_params)}")
print(f"Gradient: {grad_loss(initial_params)}")

print("\n## 5. Truncation for Efficiency")
# Propagate with and without truncation
circuit_complex = Circuit([H(0), H(1), H(2), 
                          RXX(0,1,jnp.pi/4), RYY(1,2,jnp.pi/3), RZZ(0,2,jnp.pi/6)])

observable = PauliString({0: 'X', 2: 'Z'}, 1.0)

# Without truncation
prop_full = circuit_complex.propagate(observable)

# With threshold truncation
trunc_threshold = 0.05
prop_trunc = circuit_complex.propagate(observable, min_abs_coeff=trunc_threshold)

print(f"Without truncation: {len(prop_full)} terms")
print(f"With truncation (threshold {trunc_threshold}): {len(prop_trunc)} terms")

# Check expectation value difference
exp_full = expectation_value(prop_full)
exp_trunc = expectation_value(prop_trunc)

print(f"\nExpectation value difference: {jnp.abs(exp_full - exp_trunc):.6f}")

print("\n## 6. Training a Parameterized Circuit")
# Define a training circuit
def create_train_circuit(params):
    circuit = Circuit()
    circuit.add_gate(RX(0, params[0]))
    circuit.add_gate(RZ(0, params[1]))
    circuit.add_gate(RX(0, params[2]))
    return circuit

# Loss function: maximize <Z> for qubit 0 (target: |1> state)
def train_loss(params):
    circuit = create_train_circuit(params)
    propagated = circuit.propagate(PauliString({0: 'Z'}, 1.0))
    exp_val = expectation_value(propagated)
    return -exp_val  # Negative because we want to minimize loss but maximize <Z>

# Training loop
params = jnp.array([0.0, 0.0, 0.0])
# Use simple gradient descent update since jax.optimizers is deprecated/moved
learning_rate = 0.1

print("Training a circuit to prepare |1> state...")
print("=" * 50)

for step in range(15):
    grads = jax.grad(train_loss)(params)
    params = params - learning_rate * grads
    current_loss = train_loss(params)
    
    # Compute expectation value for monitoring
    exp_val_Z = -current_loss

    if step % 3 == 0:
        print(f"Step {step:2d}: Loss = {current_loss:.6f}, <Z> = {exp_val_Z:.6f}")

print(f"\nTraining complete. Final <Z> = {exp_val_Z:.6f}")
