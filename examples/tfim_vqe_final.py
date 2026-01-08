"""
10-qubit TFIM (Transverse Field Ising Model) VQE example using PyPoli's Hardware Efficient Ansatz.
FINAL VERSION using direct PyPoli API (no custom basis projection).

Hamiltonian: H = -J sum_{i=0 to 9} Z_i Z_{i+1} - h sum_{i=0 to 9} X_i
with periodic boundary conditions (Z_10 = Z_0) and J = h = 1.0

Key Implementation:
- Direct use of PyPoli's built-in propagate and expectation_value
- HardwareEfficient ansatz with RX-RZ-RX layers and CZ entangling
- No custom basis projection (using pypoli's built-in truncation)
- Efficient for 10-qubit TFIM with small number of layers
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
from pypoli import I, X, Y, Z, H, S, T, RX, RY, RZ, RXX, RYY, RZZ, CNOT, CZ
from pypoli import propagate, expectation_value, batch_expectation_value
from pypoli import HardwareEfficient, HardwareEfficientAnsatz

# Enable x32 for performance (matching current project configuration)
jax.config.update("jax_enable_x64", False)

def generate_tfim_hamiltonian(num_qubits: int, J: float = 1.0, h: float = 1.0) -> PauliSum:
    """
    Generate TFIM Hamiltonian with periodic boundary conditions.
    """
    hamiltonian = PauliSum()

    # ZZ terms (adjacent coupling)
    for i in range(num_qubits):
        j = (i + 1) % num_qubits  # periodic boundary conditions
        pauli_str = PauliString({i: 'Z', j: 'Z'}, -J)
        hamiltonian += pauli_str

    # X terms (transverse field)
    for i in range(num_qubits):
        pauli_str = PauliString({i: 'X'}, -h)
        hamiltonian += pauli_str

    return hamiltonian

def vqe_optimization(hamiltonian: PauliSum, num_qubits: int, num_layers: int,
                     learning_rate: float = 0.1, num_steps: int = 50,
                     max_weight: int = 4, min_abs_coeff: float = 1e-3):
    """
    Perform VQE optimization using direct PyPoli API.
    """
    # Initialize ansatz
    ansatz = HardwareEfficient(num_qubits, num_layers)

    # Initialize random parameters with small values
    key = jax.random.PRNGKey(0)
    params = jax.random.normal(key, (num_layers, num_qubits, 3)) * 0.1

    # Define energy function
    def energy_fn(params):
        circuit = ansatz.generate_circuit(params)
        return expectation_value(
            circuit, hamiltonian,
            max_weight=max_weight,
            min_abs_coeff=min_abs_coeff
        )

    # Initialize optimizer
    optimizer = optax.adam(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=lambda p: energy_fn(p),
        params=params,
        tx=optimizer
    )

    # Optimization loop
    history = []
    start_time = time.time()

    print("Step    | Energy")
    print("--------|----------")

    for step in range(num_steps):
        loss, grads = jax.value_and_grad(energy_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        history.append(float(loss))

        if step % 5 == 0 or step == num_steps - 1:
            print(f"{step:6d} | {loss:.8f}")

    total_time = time.time() - start_time
    print(f"\nOptimization complete in {total_time:.2f} seconds")
    final_energy = state.apply_fn(state.params)
    print(f"Final energy: {final_energy:.8f}")

    return final_energy, state.params, history

def main():
    print("="*60)
    print("10-Qubit TFIM VQE with Hardware Efficient Ansatz")
    print("="*60)
    print("(FINAL VERSION - DIRECT PYPOLI API)")
    print("="*60)

    # System parameters
    NUM_QUBITS = 10
    NUM_LAYERS = 2  # Use small number of layers for speed
    J = 1.0
    h = 1.0

    print(f"\nSystem parameters:")
    print(f"- Number of qubits: {NUM_QUBITS}")
    print(f"- Ansatz layers: {NUM_LAYERS}")
    print(f"- Coupling strength J: {J}")
    print(f"- Transverse field h: {h}")

    # Generate TFIM Hamiltonian
    print("\nGenerating TFIM Hamiltonian...")
    hamiltonian = generate_tfim_hamiltonian(NUM_QUBITS, J, h)
    print(f"Hamiltonian terms: {len(hamiltonian.pauli_strings)} terms")

    # Perform VQE optimization
    print("\nStarting VQE optimization...")
    final_energy, final_params, history = vqe_optimization(
        hamiltonian, NUM_QUBITS, NUM_LAYERS,
        learning_rate=0.1, num_steps=50,
        max_weight=4, min_abs_coeff=1e-3
    )

    # Analyze results
    print("\n" + "="*60)
    print("VQE Results")
    print("="*60)
    print(f"Final ground state energy: {final_energy:.8f}")
    print(f"Optimization steps: {len(history)}")

    # Calculate reference energy (classical limit)
    print("\nReference values:")
    classical_energy = -NUM_QUBITS  # Classical ground state (all spins up)
    print(f"Classical ground state energy: {classical_energy:.8f}")
    print(f"Quantum improvement: {abs(final_energy - classical_energy):.8f}")

    # Save history
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(history, label='VQE Energy')
        plt.axhline(y=classical_energy, color='r', linestyle='--', label='Classical Energy')
        plt.xlabel('Optimization Step')
        plt.ylabel('Energy')
        plt.title('10-Qubit TFIM VQE Optimization (Direct API)')
        plt.legend()
        plt.grid(True)
        plt.savefig('tfim_vqe_optimization_final.png', dpi=300, bbox_inches='tight')
        print("\nOptimization plot saved to 'tfim_vqe_optimization_final.png'")
    except ImportError:
        print("\nMatplotlib not found, skipping plot generation")
    except Exception as e:
        print(f"\nError generating plot: {e}")

if __name__ == "__main__":
    main()
