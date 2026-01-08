"""
10-qubit TFIM (Transverse Field Ising Model) VQE example using PyPoli's Hardware Efficient Ansatz.
Clean, minimal implementation using direct PyPoli API.

Hamiltonian: H = -J sum_{i=0 to 9} Z_i Z_{i+1} - h sum_{i=0 to 9} X_i
with periodic boundary conditions (Z_10 = Z_0) and J = h = 1.0
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import time
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("./src"))

from pypoli import PauliString, PauliSum, Circuit
from pypoli import X, Z, RX, RZ, CZ
from pypoli import expectation_value
from pypoli import HardwareEfficient

# Enable x32 for performance (matching current project configuration)
jax.config.update("jax_enable_x64", False)

def generate_tfim_hamiltonian(num_qubits: int, J: float = 1.0, h: float = 1.0) -> PauliSum:
    """
    Generate TFIM Hamiltonian with periodic boundary conditions.
    """
    hamiltonian = PauliSum()

    # ZZ terms (adjacent coupling)
    for i in range(num_qubits):
        j = (i + 1) % num_qubits
        hamiltonian += PauliString({i: 'Z', j: 'Z'}, -J)

    # X terms (transverse field)
    for i in range(num_qubits):
        hamiltonian += PauliString({i: 'X'}, -h)

    return hamiltonian

def vqe_optimization(hamiltonian: PauliSum, num_qubits: int, num_layers: int,
                     learning_rate: float = 0.1, num_steps: int = 20,
                     max_weight: int = 2, min_abs_coeff: float = 1e-2):
    """
    Perform VQE optimization.
    """
    ansatz = HardwareEfficient(num_qubits, num_layers)
    key = jax.random.PRNGKey(0)
    params = jax.random.normal(key, (num_layers, num_qubits, 3)) * 0.1

    # Energy function
    def energy_fn(params):
        circuit = ansatz.generate_circuit(params)
        return expectation_value(
            circuit, hamiltonian,
            max_weight=max_weight,
            min_abs_coeff=min_abs_coeff
        )

    # Optimizer
    optimizer = optax.adam(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=lambda p: energy_fn(p),
        params=params,
        tx=optimizer
    )

    history = []
    start_time = time.time()

    print("Step    | Energy")
    print("--------|----------")

    for step in range(num_steps):
        loss, grads = jax.value_and_grad(energy_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        history.append(float(loss))

        if step % 5 == 0 or step == num_steps - 1:
            print(f"{step:6d} | {loss:.6f}")

    total_time = time.time() - start_time
    print(f"\nOptimization complete in {total_time:.2f} seconds")
    final_energy = state.apply_fn(state.params)
    print(f"Final energy: {final_energy:.8f}")

    return final_energy, state.params, history

def main():
    print("="*60)
    print("10-Qubit TFIM VQE with Hardware Efficient Ansatz")
    print("="*60)

    NUM_QUBITS = 10
    NUM_LAYERS = 2
    J = 1.0
    h = 1.0

    print(f"\nSystem parameters:")
    print(f"- Number of qubits: {NUM_QUBITS}")
    print(f"- Ansatz layers: {NUM_LAYERS}")
    print(f"- J = {J}, h = {h}")

    hamiltonian = generate_tfim_hamiltonian(NUM_QUBITS, J, h)
    print(f"\nHamiltonian terms: {len(hamiltonian.pauli_strings)} terms")

    print("\nStarting VQE optimization...")
    final_energy, final_params, history = vqe_optimization(
        hamiltonian, NUM_QUBITS, NUM_LAYERS,
        learning_rate=0.1, num_steps=20,
        max_weight=2, min_abs_coeff=1e-2
    )

    print("\n" + "="*60)
    print("VQE Results")
    print("="*60)
    print(f"Final energy: {final_energy:.8f}")

    classical_energy = -NUM_QUBITS
    print(f"\nClassical energy: {classical_energy:.8f}")
    print(f"Improvement: {abs(final_energy - classical_energy):.8f}")

if __name__ == "__main__":
    main()
