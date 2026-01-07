"""
Adaptive VQE Example using PyPoli.

This example demonstrates how to perform Adaptive VQE (ADAPT-VQE style) using PyPoli's
efficient JIT compilation. 

The key idea is:
1.  Define a pool of operators (or a template layer).
2.  Train the current circuit structure using JAX JIT.
3.  Dynamically grow the circuit (increase depth) based on convergence.
4.  Trigger JIT re-compilation for the new structure (which is fast due to O(1) compilation).

In this simplified example, we will iteratively increase the depth of a Hardware Efficient Ansatz
to minimize a target Hamiltonian expectation value.
"""

import jax
import jax.numpy as jnp
import optax
import numpy as np
import time

from pypoli import Circuit, PauliString, expectation_value
from pypoli.gates import RX, RZ, CNOT, CZ, Parameter

# Enable x64 for precision
# jax.config.update("jax_enable_x64", True)

def create_layer_template(n_qubits):
    """
    Creates a single layer template with parameterized gates.
    Structure: RX(th) -> RZ(ph) -> CNOT chain
    """
    layer = Circuit()
    # 1. Rotations
    for q in range(n_qubits):
        # Implicitly creating Parameters by passing strings
        layer.add_gate(RX(q, "theta")) 
        layer.add_gate(RZ(q, "phi"))
    
    # 2. Entanglement (Linear topology)
    for q in range(n_qubits - 1):
        layer.add_gate(CNOT(q, q+1))
        
    return layer

def main():
    print("=== PyPoli Adaptive VQE Example ===")
    
    # 1. Problem Setup
    N_QUBITS = 6
    # Target Hamiltonian: 1D Ising Model with Transverse Field
    # H = -J sum(Z_i Z_{i+1}) - h sum(X_i)
    # Let J=1, h=1
    target_hamiltonian = []
    
    # Interaction terms
    for i in range(N_QUBITS - 1):
        target_hamiltonian.append(PauliString({i: 'Z', i+1: 'Z'}, -1.0))
    # Field terms
    for i in range(N_QUBITS):
        target_hamiltonian.append(PauliString({i: 'X'}, -1.0))
        
    print(f"System: {N_QUBITS} Qubits, 1D Ising Model")
    print(f"Hamiltonian terms: {len(target_hamiltonian)}")
    
    # 2. Adaptive Loop
    # We start with depth 1 and grow until convergence
    MAX_DEPTH = 10
    current_depth = 1
    
    # Template layer (compiled once conceptually, reused structurally)
    layer_template = create_layer_template(N_QUBITS)
    
    # Initialize parameters
    # 2 params per qubit per layer (RX, RZ)
    params_per_layer = N_QUBITS * 2
    
    # We keep track of trained parameters to warm-start the next iteration
    # Shape: (current_depth, params_per_layer)
    # Start with random values
    key = jax.random.PRNGKey(42)
    # Increase scale to 0.1 to avoid being stuck at identity
    current_params = jax.random.normal(key, (current_depth, params_per_layer)) 
    
    optimizer = optax.adam(learning_rate=0.05)
    
    # Create PauliSum once outside the loop
    from pypoli import PauliSum
    hamiltonian_sum = PauliSum(pauli_strings=target_hamiltonian)
    
    print("\nStarting Adaptive Optimization loop...")
    
    for adapt_step in range(MAX_DEPTH):
        print(f"\n--- Adaptation Step {adapt_step + 1}: Depth {current_depth} ---")
        
        # A. Construct Circuit Structure
        # This is fast Python object creation
        circuit = Circuit.from_layer(layer_template, current_depth)
        
        # B. Define Loss Function
        # We define it inside the loop because 'circuit' object changes
        # But 'circuit' is captured as a static object (PyTree or closure) by JIT
        
        @jax.jit
        def loss_fn(params):
            # 1. Bind parameters
            # circuit.bind is lightweight
            bound_c = circuit.bind(params)
            
            # 2. Compute Expectation
            # Now we use batch propagation!
            # The entire Hamiltonian is propagated in ONE scan pass.
            
            return expectation_value(
                bound_c, 
                hamiltonian_sum, 
                max_weight=2, 
                min_abs_coeff=1e-5, # Lower threshold for low-weight terms
                damping=0.5         # Soft truncation: punishes high-weight terms
            )

        # C. Optimization Loop
        opt_state = optimizer.init(current_params)
        
        # Compile time measurement
        start_compile = time.time()
        # Trigger JIT with a dummy call or first step
        loss_val, grads = jax.value_and_grad(loss_fn)(current_params)
        grads.block_until_ready() # Wait for compilation
        compile_time = time.time() - start_compile
        
        print(f"JIT Compilation Time: {compile_time:.4f}s")
        print(f"Initial Energy: {loss_val:.6f}")
        
        # Training steps
        steps = 50
        t0 = time.time()
        for i in range(steps):
            loss_val, grads = jax.value_and_grad(loss_fn)(current_params)
            updates, opt_state = optimizer.update(grads, opt_state)
            current_params = optax.apply_updates(current_params, updates)
            
        t1 = time.time()
        print(f"Training Time ({steps} steps): {t1-t0:.4f}s")
        print(f"Final Energy: {loss_val:.6f}")
        
        # D. Check Convergence
        # If energy improvement is small, stop?
        # For this demo, we just grow.
        
        # E. Grow Circuit
        # We append a new layer initialized near identity but with some noise
        if adapt_step < MAX_DEPTH - 1:
            new_layer_params = jax.random.normal(key, (1, params_per_layer)) * 0.1
            current_params = jnp.concatenate([current_params, new_layer_params], axis=0)
            current_depth += 1
            
    print("\nOptimization Complete.")

if __name__ == "__main__":
    main()
