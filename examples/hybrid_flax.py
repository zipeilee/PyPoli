
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

from pypoli.core import PauliString, PauliSum, Parameter
from pypoli.gates import RY, CNOT
from pypoli.circuits import Circuit

# Use default precision (float32/complex64) for speed
jax.config.update("jax_enable_x64", False)

# Fix for Flax/Python 3.14 compatibility
if not hasattr(nn.Module, '__annotations__'):
    nn.Module.__annotations__ = {}

# Define a global cache and builder function outside the class
_QUANTUM_CACHE = {}

def get_quantum_layer_fn(n_qubits, depth, max_weight):
    cache_key = (n_qubits, depth, max_weight)
    if cache_key in _QUANTUM_CACHE:
        return _QUANTUM_CACHE[cache_key]
        
    print(f"Compiling Quantum Layer: Q={n_qubits}, D={depth}, W={max_weight}...")
    circuit = Circuit()
    for q in range(n_qubits):
        circuit.add(RY(q, Parameter(f"enc_{q}")))
    for d in range(depth):
        for i in range(n_qubits - 1):
            circuit.add(CNOT(i, i + 1))
        for q in range(n_qubits):
            circuit.add(RY(q, Parameter(f"ansatz_{d}_{q}")))
    from pypoli.jit_propagation import compile_expectation_fn
    
    # Define observables: Z_i Z_{i+1} for i in range(n_qubits-1)
    observables = []
    for i in range(n_qubits - 1):
        observables.append(PauliString({i: 'Z', i + 1: 'Z'}, 1.0))
        
    compute_expectations = compile_expectation_fn(circuit, n_qubits, max_weight, observables)

    _QUANTUM_CACHE[cache_key] = compute_expectations
    param_len = n_qubits + depth * n_qubits
    _ = compute_expectations(jnp.zeros((param_len,), dtype=jnp.float32))
    return compute_expectations

class QuantumLayer(nn.Module):
    """
    A Flax module wrapping a quantum circuit using pypoli's JIT compilation.
    """
    n_qubits: int
    depth: int
    max_weight: int = 2
    min_abs_coeff: float = 1e-3

    @nn.compact
    def __call__(self, x):
        # x shape: (batch_size, n_qubits)
        params = self.param('circuit_params', 
                          nn.initializers.uniform(scale=2*jnp.pi), 
                          (self.depth * self.n_qubits,))
        
        # Get the compiled function (Singleton/Cached)
        # This is NOT inline JIT anymore, it's a pre-compiled pure function.
        quantum_fn = get_quantum_layer_fn(self.n_qubits, self.depth, self.max_weight)
        
        # Helper to combine and run
        def single_forward(x_sample, p_ansatz):
            combined = jnp.concatenate([x_sample, p_ansatz])
            return quantum_fn(combined)
            
        return jax.vmap(single_forward, in_axes=(0, None))(x, params)

class HybridModel(nn.Module):
    """
    Hybrid Quantum-Classical Model.
    x -> QuantumLayer -> MLP -> y
    """
    n_qubits: int
    q_depth: int
    hidden_dim: int
    output_dim: int
    max_weight: int = 4
    min_abs_coeff: float = 1e-3

    @nn.compact
    def __call__(self, x):
        # x: (batch, n_qubits)
        
        # Quantum Layer
        q_out = QuantumLayer(n_qubits=self.n_qubits, 
                           depth=self.q_depth,
                           max_weight=self.max_weight,
                           min_abs_coeff=self.min_abs_coeff)(x)
        
        # Classical MLP
        y = nn.Dense(features=self.hidden_dim)(q_out)
        y = nn.relu(y)
        y = nn.Dense(features=self.output_dim)(y)
        
        return y

def create_train_state(rng, model, sample_input, learning_rate):
    params = model.init(rng, sample_input)['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)

@jax.jit(donate_argnames=('state',))
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
    print("Initializing Hybrid Quantum-Classical Model (Updated for pypoli v2)...")
    
    # Hyperparameters
    BATCH_SIZE = 32
    N_QUBITS = 10       
    Q_DEPTH = 10        # Reduced depth for demonstration speed
    HIDDEN_DIM = 8
    OUTPUT_DIM = 1
    LEARNING_RATE = 0.01
    STEPS = 500
    
    MAX_WEIGHT = 4
    MIN_ABS_COEFF = 1e-3

    # 1. Generate Dummy Data
    key = jax.random.PRNGKey(0)
    key, x_key, y_key = jax.random.split(key, 3)
    
    X_train = jax.random.uniform(x_key, (BATCH_SIZE, N_QUBITS), minval=0, maxval=jnp.pi)
    Y_train = jnp.sum(jnp.sin(X_train), axis=1, keepdims=True)
    
    print(f"Data shapes: X {X_train.shape}, Y {Y_train.shape}")
    
    # 2. Initialize Model
    model = HybridModel(n_qubits=N_QUBITS, 
                        q_depth=Q_DEPTH, 
                        hidden_dim=HIDDEN_DIM, 
                        output_dim=OUTPUT_DIM,
                        max_weight=MAX_WEIGHT,
                        min_abs_coeff=MIN_ABS_COEFF)
    
    print("Compiling Model...")
    t0 = time.time()
    # Initialize triggers compilation of the circuit structure
    state = create_train_state(key, model, X_train, LEARNING_RATE)
    print(f"Initialization/Compilation Done in {time.time()-t0:.2f}s")
    state, _ = train_step(state, X_train, Y_train)
    
    print("\nStarting Training...")
    print("-" * 30)
    
    start_time = time.time()
    for step in range(STEPS):
        step_start = time.time()
        state, loss = train_step(state, X_train, Y_train)
        loss.block_until_ready() # Ensure sync for timing
        
        if step % 50 == 0:
            print(f"Step {step:3d} | Loss: {loss:.6f} | Time: {time.time() - step_start:.4f}s")
            
    total_time = time.time() - start_time
    print("-" * 30)
    print(f"Final Loss: {loss:.6f}")
    print(f"Total Training Time: {total_time:.4f}s")
    
    # 4. Evaluation
    print("\nEvaluation:")
    preds = state.apply_fn({'params': state.params}, X_train)
    print(f"Sample Prediction: {preds[0][0]:.4f} vs Target: {Y_train[0][0]:.4f}")

if __name__ == "__main__":
    main()
