"""
10-qubit TFIM (Transverse Field Ising Model) VQE example using PyPoli's Hardware Efficient Ansatz.
ULTRA-FAST VERSION with minimal JIT overhead.

Hamiltonian: H = -J sum_{i=0 to 9} Z_i Z_{i+1} - h sum_{i=0 to 9} X_i
with periodic boundary conditions (Z_10 = Z_0) and J = h = 1.0

Key Optimizations:
- Avoid unnecessary JIT compilation overhead
- Use minimal circuit construction
- Aggressive truncation to control term growth
- Precompute ansatz structure
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

def create_ansatz_circuit(params, num_qubits, num_layers):
    """
    Create a hardware efficient ansatz circuit directly (no class overhead).
    This is faster than using the HardwareEfficient class.
    """
    circuit = Circuit()

    for layer in range(num_layers):
        # Single-qubit gates: RX(theta) - RZ(phi) - RX(gamma)
        for qubit in range(num_qubits):
            theta, phi, gamma = params[layer, qubit]
            circuit.add_gate(RX(qubit, theta))
            circuit.add_gate(RZ(qubit, phi))
            circuit.add_gate(RX(qubit, gamma))

        # Two-qubit CZ gates with alternating pattern
        if num_qubits > 1:
            if layer % 2 == 0:  # Even layer: CZ on (0-1, 2-3, 4-5, etc.)
                for qubit in range(0, num_qubits - 1, 2):
                    circuit.add_gate(CZ(qubit, qubit + 1))
            else:  # Odd layer: CZ on (1-2, 3-4, 5-6, etc.)
                for qubit in range(1, num_qubits - 1, 2):
                    circuit.add_gate(CZ(qubit, qubit + 1))

    return circuit

def energy_fn(params, hamiltonian_terms, num_qubits, num_layers,
             max_weight=4, min_abs_coeff=1e-3):
    """
    Calculate energy without JIT overhead.
    """
    circuit = create_ansatz_circuit(params, num_qubits, num_layers)

    total_energy = 0.0

    for term in hamiltonian_terms:
        exp_val = expectation_value(
            circuit, term,
            max_weight=max_weight,
            min_abs_coeff=min_abs_coeff
        )
        total_energy += exp_val * term.coefficient

    return jnp.real(total_energy)

def vqe_optimization(hamiltonian: PauliSum, num_qubits: int, num_layers: int,
                     learning_rate: float = 0.1, num_steps: int = 100,
                     max_weight: int = 3, min_abs_coeff: float = 1e-2):
    """
    Perform VQE optimization with minimal JIT overhead.
    """
    # Extract individual terms from Hamiltonian
    hamiltonian_terms = hamiltonian.pauli_strings

    # Initialize random parameters with small values
    key = jax.random.PRNGKey(0)
    params = jax.random.normal(key, (num_layers, num_qubits, 3)) * 0.05

    # Initialize optimizer
    optimizer = optax.adam(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=lambda p: energy_fn(p, hamiltonian_terms, num_qubits, num_layers,
                                    max_weight, min_abs_coeff),
        params=params,
        tx=optimizer
    )

    # Optimization loop without JIT (to avoid compilation overhead)
    history = []
    start_time = time.time()

    print("Step    | Energy")
    print("--------|----------")

    for step in range(num_steps):
        # Calculate gradient and update
        def loss_fn(params):
            return energy_fn(params, hamiltonian_terms, num_qubits, num_layers,
                           max_weight, min_abs_coeff)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        history.append(float(loss))

        if step % 10 == 0 or step == num_steps - 1:
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
    print("(ULTRA-FAST VERSION)")
    print("="*60)

    # System parameters
    NUM_QUBITS = 10
    NUM_LAYERS = 2  # Very few layers for speed
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

    # Perform VQE optimization with aggressive truncation
    print("\nStarting VQE optimization...")
    final_energy, final_params, history = vqe_optimization(
        hamiltonian, NUM_QUBITS, NUM_LAYERS,
        learning_rate=0.1, num_steps=100,
        max_weight=3, min_abs_coeff=1e-2
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
        plt.title('10-Qubit TFIM VQE Optimization (Ultra-Fast)')
        plt.legend()
        plt.grid(True)
        plt.savefig('tfim_vqe_optimization_fast.png', dpi=300, bbox_inches='tight')
        print("\nOptimization plot saved to 'tfim_vqe_optimization_fast.png'")
    except ImportError:
        print("\nMatplotlib not found, skipping plot generation")
    except Exception as e:
        print(f"\nError generating plot: {e}")

if __name__ == "__main__":
    main()
