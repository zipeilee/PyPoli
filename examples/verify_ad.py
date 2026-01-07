
import jax
import jax.numpy as jnp
import sys
import os

# Add the source directory to the path to import pypoli
sys.path.append(os.path.abspath("./src"))

from pypoli import Circuit, PauliString, expectation_value
from pypoli import RX, RY, RZ, RZZ, CNOT, CZ, H

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

def run_vqe():
    print("Setting up VQE for TFIM...")
    # Problem Constants
    N = 4
    J = 1.0
    h = 1.0
    DEPTH = 3

    # Define Hamiltonian
    hamiltonian = []
    # Interaction terms Z_i Z_{i+1}
    for i in range(N - 1):
        hamiltonian.append(PauliString({i: 'Z', i+1: 'Z'}, -J))
    # Transverse field terms X_i
    for i in range(N):
        hamiltonian.append(PauliString({i: 'X'}, -h))

    # Helper function to build circuit (Standard Python function)
    def make_ansatz(params):
        circuit = Circuit()
        param_idx = 0
        
        # Initial rotations
        for i in range(N):
            circuit.add_gate(RY(i, params[param_idx]))
            param_idx += 1
            
        # Entangling layers
        for d in range(DEPTH):
            for i in range(N - 1):
                circuit.add_gate(CNOT(i, i+1))
            for i in range(N):
                circuit.add_gate(RY(i, params[param_idx]))
                param_idx += 1
        return circuit

    # Define Loss Function with Decorators
    # @jax.jit compiles the function for speed
    # @jax.value_and_grad automatically computes gradients
    @jax.jit
    @jax.value_and_grad
    def loss_fn(params):
        # 1. Build the circuit with current parameters
        #    (JIT unrolls this loop and traces the circuit structure)
        circuit = make_ansatz(params)
        
        # 2. Compute expectation value (Energy)
        total_energy = 0.0
        for term in hamiltonian:
            # expectation_value propagates the Pauli term through the circuit
            # and computes <0| U' O U |0>
            total_energy += expectation_value(circuit, term)
            
        return total_energy

    # Initialize parameters
    n_params = N + DEPTH * N
    key = jax.random.PRNGKey(42)
    params = jax.random.uniform(key, shape=(n_params,), minval=0, maxval=2*jnp.pi)

    print("Compiling and computing initial gradient...")
    # The first call triggers JIT compilation
    e_init, grad_init = loss_fn(params)
    
    print(f"Initial Energy: {e_init}")
    print(f"Gradient Norm: {jnp.linalg.norm(grad_init)}")
    
    # Optimization loop
    learning_rate = 0.1
    print("\nStarting optimization (5 steps)...")
    for i in range(5):
        energy, grads = loss_fn(params)
        params = params - learning_rate * grads
        print(f"Step {i+1}: Energy = {energy:.6f}")

if __name__ == "__main__":
    run_vqe()
