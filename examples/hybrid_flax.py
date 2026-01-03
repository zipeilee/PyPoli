
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import sys
import os
import time
import numpy as np

# Add src to path
sys.path.append(os.path.abspath("./src"))

from pypoli import Circuit, PauliString, expectation_value, PauliSum
from pypoli import RY, CNOT

# Enable x64 for precision
jax.config.update("jax_enable_x64", True)

class QuantumLayer(nn.Module):
    """
    A Flax module wrapping a quantum circuit using jax.lax.scan for fast compilation.
    
    Structure:
    1. Encoding: RY(x_i) on each qubit.
    2. Ansatz: Hardware Efficient Ansatz with trainable parameters.
    3. Measurement: Expectation of Z_i Z_{i+1} for all adjacent pairs.
    """
    n_qubits: int
    depth: int
    max_weight: int = 2  # Hardcoded for basis generation
    min_abs_coeff: float = 1e-3

    @nn.compact
    def __call__(self, x):
        # x shape: (batch_size, n_qubits)
        
        # Initialize trainable parameters for the ansatz
        params = self.param('circuit_params', 
                          nn.initializers.uniform(scale=2*jnp.pi), 
                          (self.depth, self.n_qubits))
        
        # --- Precompute Basis for Weight <= 2 ---
        # We need a fixed basis to use scan.
        # Basis includes Identity, all 1-body Paulis, all 2-body Paulis.
        
        from itertools import combinations
        
        # Helper to create PauliString
        def ps(qubit_paulis):
            # qubit_paulis: dict {qubit_idx: 'X'/'Y'/'Z'}
            return PauliString(qubit_paulis, 1.0)
            
        basis = []
        basis.append(ps({})) # Identity
        
        # 1-body
        for i in range(self.n_qubits):
            for p in ['X', 'Y', 'Z']:
                basis.append(ps({i: p}))
                
        # 2-body
        if self.max_weight >= 2:
            for i, j in combinations(range(self.n_qubits), 2):
                for p1 in ['X', 'Y', 'Z']:
                    for p2 in ['X', 'Y', 'Z']:
                        basis.append(ps({i: p1, j: p2}))
                        
        basis_len = len(basis)
        # print(f"Basis size: {basis_len}") # Cannot print in JIT
        
        # Create a mapping from "canonical signature" to index
        # Signature: tuple of sorted (qubit, pauli_char)
        basis_map = {}
        for idx, term in enumerate(basis):
            sig = tuple(sorted(term.paulis.items()))
            basis_map[sig] = idx

        # --- Define Scan Step Function ---
        def scan_step(carry, layer_param):
            # carry: coeff_vector (basis_len,)
            # layer_param: (n_qubits,) rotations for this layer
            
            # 1. Convert vector back to PauliSum (sparse representation for pypoli)
            # To make this JIT-able, we cannot iterate over the whole basis if it's too large,
            # BUT here we are inside the body of scan, which is compiled once.
            # However, pypoli.Circuit.propagate expects a list of PauliStrings.
            # We can construct this list. Since basis is static, this list length is static.
            # We only include terms where current coeff is non-zero? 
            # No, in JIT we must process all basis terms, but their coeffs might be zero.
            
            # OPTIMIZATION:
            # Instead of full basis -> propagate -> full basis,
            # we can rely on pypoli's logic.
            # But pypoli's logic is what we want to wrap in scan.
            # So we create a "Layer Circuit" using pypoli.
            
            # Construct the circuit for ONE layer
            layer_circuit = Circuit()
            # Entangling
            for i in range(self.n_qubits - 1):
                layer_circuit.add_gate(CNOT(i, i+1))
            # Rotations
            for i in range(self.n_qubits):
                layer_circuit.add_gate(RY(i, layer_param[i]))
            
            # Propagate each basis element through this layer
            # This produces a linear map (Matrix).
            # We can precompute this matrix? No, it depends on layer_param (rotations).
            # So we must compute it on the fly.
            
            # To do this efficiently in JAX:
            # We treat the coefficient vector as the state.
            # We need to apply the linear transformation defined by layer_circuit.
            
            # This is hard to vectorize perfectly with existing pypoli without rewriting pypoli.
            # BUT, we can use a loop over the basis. Since basis size ~400, it's not too bad for 1 layer.
            # Wait, 400 * 400 matrix is small, but computing it involves propagating 400 terms.
            # 400 terms * (10+9 gates) = 7600 gate applications per layer.
            # This is much better than propagating 25 layers deep * term growth.
            
            # Let's try to construct the output vector.
            new_coeffs = jnp.zeros(basis_len, dtype=jnp.float64)
            
            # We iterate over the input basis terms that have non-zero coefficients.
            # In JIT, we must iterate over ALL basis terms.
            # This might be slow if basis is large. 
            # 436 is acceptable.
            
            # Actually, pypoli's propagate function works on a List of PauliStrings.
            # We can reconstruct the "current state" as a list of PauliStrings with coeffs from 'carry'.
            
            # Reconstructing input terms (only symbolic, coeffs come from carry)
            input_terms = []
            for idx, term in enumerate(basis):
                # Create a NEW PauliString with coefficient from carry[idx]
                # We need to be careful: PauliString.coefficient should be a JAX Tracer.
                new_term = PauliString(term.paulis, carry[idx])
                input_terms.append(new_term)
            
            # Propagate through the layer
            # We use truncation inside propagate to keep things sparse if possible,
            # but here we are mapping back to basis, so we just want the result.
            # Max weight is already handled by our basis definition, but intermediate terms might grow.
            # We let pypoli handle intermediate growth, then we project back.
            
            propagated_terms = layer_circuit.propagate(
                PauliSum(input_terms), 
                max_weight=self.max_weight + 1 # Allow slight intermediate growth
            )
            
            # Project back to Basis
            # propagated_terms is a list of PauliStrings.
            # We sum their coefficients into the corresponding slots in new_coeffs.
            
            # We can't mutate new_coeffs in place in JAX easily.
            # We collect updates.
            
            # This part is tricky in JAX because propagated_terms structure depends on values if we use truncation.
            # BUT we set pypoli to NOT truncate by coefficient (min_abs_coeff=0 or None inside scan),
            # only by weight.
            # Wait, pypoli's propagate structure is static if truncation is static (max_weight).
            # So `propagated_terms` will have a fixed length and fixed Pauli structures for a given input basis.
            # Yes! Because `layer_circuit` topology is fixed.
            # The only variable is the rotation angles.
            # Rotation angles affect coefficients, not which Pauli strings are generated (structurally).
            # RY gate: Y -> Y, Z -> Z*cos + X*sin. Structure is fixed (sum of terms).
            
            # So `propagated_terms` is a static list of PauliStrings with Tracer coefficients.
            
            # We map these back to new_coeffs.
            updates_indices = []
            updates_values = []
            
            for term in propagated_terms:
                sig = tuple(sorted(term.paulis.items()))
                if sig in basis_map:
                    idx = basis_map[sig]
                    updates_indices.append(idx)
                    updates_values.append(term.coefficient)
                # Else: term is outside basis (truncated)
            
            if updates_indices:
                new_coeffs = new_coeffs.at[jnp.array(updates_indices)].add(jnp.array(updates_values))
            
            # Apply coefficient truncation (Masking)
            mask = (jnp.abs(new_coeffs) >= self.min_abs_coeff).astype(new_coeffs.dtype)
            new_coeffs = new_coeffs * mask
            
            return new_coeffs, None

        # --- Define Forward Function for One Example ---
        def single_circuit_forward(x_input, circuit_params):
            # 1. Encoding
            # Map input observable (Z_i Z_{i+1}) backwards through Encoding Layer first?
            # No, standard is: State |0> -> Enc(x) -> Ansatz -> Measure.
            # Heisenberg: Measure -> Ansatz^dag -> Enc^dag -> |0>.
            
            # We are propagating the Observable O.
            # O' = U^dag O U.
            # U = Ansatz * Enc.
            # O' = Enc^dag * Ansatz^dag * O * Ansatz * Enc.
            # In pypoli propagate, we add gates in reverse order of application.
            # So we add Ansatz gates (reversed), then Enc gates (reversed).
            
            # Here we use scan for Ansatz.
            # So we start with Observable O.
            # Propagate through Ansatz (reversed layers).
            # Then propagate through Enc.
            # Then take expectation with |0> (which is just the coeff of Identity term).
            
            # Measurement Loop
            expectations = []
            for i in range(self.n_qubits - 1):
                # Initial Observable: Z_i Z_{i+1}
                target_sig = tuple(sorted({i: 'Z', i+1: 'Z'}.items()))
                target_idx = basis_map.get(target_sig)
                
                if target_idx is None:
                    # Should not happen if basis covers 2-body
                    expectations.append(0.0)
                    continue
                
                # Initial coeff vector
                init_coeffs = jnp.zeros(basis_len, dtype=jnp.float64)
                init_coeffs = init_coeffs.at[target_idx].set(1.0)
                
                # SCAN through Ansatz Layers
                # We need to iterate layers in REVERSE order for Heisenberg propagation.
                # circuit_params shape: (depth, n_qubits)
                # We reverse it along axis 0
                rev_params = circuit_params[::-1]
                
                final_coeffs, _ = jax.lax.scan(scan_step, init_coeffs, rev_params)
                
                # Now propagate through Encoding Layer
                # This is just one layer, no need for scan
                # Reconstruct PauliSum
                input_terms = []
                for idx, term in enumerate(basis):
                    # Only add if coeff is non-negligible to save compute?
                    # Inside JIT we can't condition on value.
                    new_term = PauliString(term.paulis, final_coeffs[idx])
                    input_terms.append(new_term)
                
                enc_circuit = Circuit()
                for q in range(self.n_qubits):
                    enc_circuit.add_gate(RY(q, x_input[q]))
                
                # Propagate (Encoding layer is applied first in state prep, so last in Heisenberg)
                # Wait, pypoli propagate applies gates in reverse order of list.
                # Forward: Enc -> Ansatz
                # Reverse: Ansatz^dag -> Enc^dag
                # So we did Ansatz (via scan). Now Enc.
                final_terms = enc_circuit.propagate(PauliSum(input_terms), max_weight=self.max_weight)
                
                # Expectation value <0|P|0>
                # For PauliString P, <0|P|0> is non-zero only if P is purely Z or I.
                # Actually, pypoli might have a helper for this, but let's implement:
                # <0|X|0> = 0, <0|Y|0> = 0, <0|Z|0> = 1, <0|I|0> = 1.
                # So we just keep terms that only have Z or I.
                
                val = 0.0
                for term in final_terms:
                    is_diagonal = True
                    for p in term.paulis.values():
                        if p in ['X', 'Y']:
                            is_diagonal = False
                            break
                    if is_diagonal:
                        val += term.coefficient
                
                expectations.append(val)
            
            return jnp.array(expectations)

        # Vectorize over the batch dimension
        batch_forward = jax.vmap(single_circuit_forward, in_axes=(0, None))
        return batch_forward(x, params)

class HybridModel(nn.Module):
    """
    Hybrid Quantum-Classical Model.
    x -> QuantumLayer -> MLP -> y
    """
    n_qubits: int
    q_depth: int
    hidden_dim: int
    output_dim: int
    max_weight: int = 4  # Truncation parameter
    min_abs_coeff: float = 1e-3  # Truncation parameter

    @nn.compact
    def __call__(self, x):
        # x: (batch, n_qubits)
        
        # Quantum Layer
        # Output shape: (batch, n_qubits - 1)
        q_out = QuantumLayer(n_qubits=self.n_qubits, 
                           depth=self.q_depth,
                           max_weight=self.max_weight,
                           min_abs_coeff=self.min_abs_coeff)(x)
        
        # Classical MLP
        # We can treat q_out as features
        y = nn.Dense(features=self.hidden_dim)(q_out)
        y = nn.relu(y)
        y = nn.Dense(features=self.output_dim)(y)
        
        return y

def create_train_state(rng, model, sample_input, learning_rate):
    params = model.init(rng, sample_input)['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch_x, batch_y):
    def loss_fn(params):
        predictions = state.apply_fn({'params': params}, batch_x)
        loss = jnp.mean((predictions - batch_y) ** 2)
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def main():
    print("Initializing Hybrid Quantum-Classical Model...")
    
    # Hyperparameters for Deep Circuit & Truncation
    BATCH_SIZE = 32
    N_QUBITS = 10       # Increased to 10 qubits
    Q_DEPTH = 25        # Depth 25
    HIDDEN_DIM = 8
    OUTPUT_DIM = 1
    LEARNING_RATE = 0.01
    STEPS = 1000
    
    # Aggressive Truncation
    # For speed, we rely on WEIGHT truncation which is static and JIT-compatible
    # Coefficient truncation is less effective inside JIT because it can't remove terms
    MAX_WEIGHT = 2      # Reduced from 4 to 2 (Keep only up to 2-body terms)
    MIN_ABS_COEFF = 1.0 # Very Aggressive Truncation

    # Enable JIT for maximum speed
    # With MAX_WEIGHT=2, the number of terms is small enough that we don't need to physically remove terms.
    # Coefficient truncation will work via masking (setting small coeffs to 0).
    USE_JIT = True

    # 1. Generate Dummy Data
    # Task: Learn a simple function y = sum(x) (normalized)
    key = jax.random.PRNGKey(0)
    key, x_key, y_key = jax.random.split(key, 3)
    
    # X: Random angles in [0, pi]
    X_train = jax.random.uniform(x_key, (BATCH_SIZE, N_QUBITS), minval=0, maxval=jnp.pi)
    
    # Y: Target is related to input sum, creating a synthetic regression task
    # We normalize to be somewhat in range of reasonable outputs
    Y_train = jnp.sum(jnp.sin(X_train), axis=1, keepdims=True)
    
    print(f"Data shapes: X {X_train.shape}, Y {Y_train.shape}")
    print(f"Circuit Depth: {Q_DEPTH}, Truncation: MaxWeight={MAX_WEIGHT}, MinCoeff={MIN_ABS_COEFF}")
    
    # 2. Initialize Model
    model = HybridModel(n_qubits=N_QUBITS, 
                        q_depth=Q_DEPTH, 
                        hidden_dim=HIDDEN_DIM, 
                        output_dim=OUTPUT_DIM,
                        max_weight=MAX_WEIGHT,
                        min_abs_coeff=MIN_ABS_COEFF)
    
    state = create_train_state(key, model, X_train, LEARNING_RATE)
    
    print("\nStarting Training...")
    print("-" * 30)
    
    # 3. Training Loop
    if USE_JIT:
        train_step_fn = train_step
        print("Using JIT compilation (Warning: Coefficient truncation will be masked, not removed)")
    else:
        # We need to un-JIT train_step if it was decorated
        # Since train_step is defined with @jax.jit, we can't easily undo it unless we redefine
        # Let's redefine a non-jitted version for this test
        print("Using Eager Execution (Enables dynamic coefficient truncation)")
        
        def train_step_eager(state, batch_x, batch_y):
            def loss_fn(params):
                predictions = state.apply_fn({'params': params}, batch_x)
                loss = jnp.mean((predictions - batch_y) ** 2)
                return loss
            
            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss
            
        train_step_fn = train_step_eager

    start_time = time.time()
    for step in range(STEPS):
        step_start = time.time()
        state, loss = train_step_fn(state, X_train, Y_train)
        # Block until computation is done for accurate timing
        loss.block_until_ready()
        step_end = time.time()
        
        if step % 100 == 0:
            print(f"Step {step:3d} | Loss: {loss:.6f} | Time: {step_end - step_start:.4f}s")
            
    total_time = time.time() - start_time
    print("-" * 30)
    print(f"Final Loss: {loss:.6f}")
    print(f"Total Training Time: {total_time:.4f}s")
    print(f"Average Time per Step: {total_time/STEPS:.4f}s")
    
    # 4. Evaluation
    print("\nEvaluation:")
    preds = state.apply_fn({'params': state.params}, X_train)
    print(f"Sample Prediction: {preds[0][0]:.4f} vs Target: {Y_train[0][0]:.4f}")

if __name__ == "__main__":
    main()
