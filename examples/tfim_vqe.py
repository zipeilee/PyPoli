"""
10-qubit TFIM (Transverse Field Ising Model) VQE example using PyPoli's Hardware Efficient Ansatz.

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
from pypoli import I, X, Y, Z, H, S, T, RX, RY, RZ, RXX, RYY, RZZ, CNOT, CZ
from pypoli import propagate, expectation_value, batch_expectation_value
from pypoli import HardwareEfficient, HardwareEfficientAnsatz

# Enable x32 for performance (matching current project configuration)
jax.config.update("jax_enable_x64", False)

def generate_tfim_hamiltonian(num_qubits: int, J: float = 1.0, h: float = 1.0) -> PauliSum:
    """
    Generate TFIM Hamiltonian with periodic boundary conditions.

    Args:
        num_qubits: Number of qubits
        J: Coupling strength between adjacent qubits
        h: Transverse field strength

    Returns:
        PauliSum: TFIM Hamiltonian
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
                     learning_rate: float = 0.1, num_steps: int = 500):
    """
    Perform VQE optimization to find the ground state energy of the Hamiltonian.

    Args:
        hamiltonian: Target Hamiltonian to optimize
        num_qubits: Number of qubits
        num_layers: Number of ansatz layers
        learning_rate: Learning rate for optimizer
        num_steps: Number of optimization steps

    Returns:
        tuple: (optimized_energy, optimized_params, history)
    """
    # Initialize ansatz
    ansatz = HardwareEfficient(num_qubits, num_layers)

    # Initialize random parameters
    key = jax.random.PRNGKey(0)
    params = jax.random.uniform(key, (num_layers, num_qubits, 3), minval=0, maxval=2*jnp.pi)

    # Define energy function
    @jax.jit
    def energy_fn(params):
        circuit = ansatz.generate_circuit(params)
        return expectation_value(circuit, hamiltonian)

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

    @jax.jit
    def update_step(state):
        def loss_fn(params):
            return energy_fn(params)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    for step in range(num_steps):
        state, loss = update_step(state)
        history.append(float(loss))

        if step % 50 == 0:
            print(f"Step {step:4d} | Energy: {loss:.8f}")

    total_time = time.time() - start_time
    print(f"\nOptimization complete in {total_time:.2f} seconds")
    print(f"Final energy: {state.apply_fn(state.params):.8f}")

    return state.apply_fn(state.params), state.params, history

def main():
    print("="*60)
    print("10-Qubit TFIM VQE with Hardware Efficient Ansatz")
    print("="*60)

    # System parameters
    NUM_QUBITS = 10
    NUM_LAYERS = 5
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
        learning_rate=0.1, num_steps=500
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
        plt.title('10-Qubit TFIM VQE Optimization')
        plt.legend()
        plt.grid(True)
        plt.savefig('tfim_vqe_optimization.png', dpi=300, bbox_inches='tight')
        print("\nOptimization plot saved to 'tfim_vqe_optimization.png'")
    except ImportError:
        print("\nMatplotlib not found, skipping plot generation")
    except Exception as e:
        print(f"\nError generating plot: {e}")

if __name__ == "__main__":
    main()
