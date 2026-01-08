
import jax
import jax.numpy as jnp
import optax
import time
import warnings
from pypoli.jit_propagation import compile_expectation_fn
from pypoli.core import PauliString, PauliSum, Parameter
from pypoli.gates import RX, RZ, CNOT
from pypoli.circuits import Circuit

# Enable x64 for precision
jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore")

# --- 1. Problem Configuration ---
N_QUBITS = 10
MAX_WEIGHT = 6 # Increased from 4 to reduce truncation errors
DEPTH = 10
LEARNING_RATE = 0.05
STEPS = 200

print(f"=== Optimized Simple HEA VQE (Standard Circuit Mode) ===")
print(f"System: {N_QUBITS} Qubits, Weight {MAX_WEIGHT}, Depth {DEPTH}")

# --- 2. Define Hamiltonian ---
print("Constructing Hamiltonian...")
terms = []
# Ising Z terms: - Z_i Z_{i+1}
for i in range(N_QUBITS - 1):
    terms.append(PauliString({i: 'Z', i+1: 'Z'}, -1.0))
# Ising X terms: - X_i
for i in range(N_QUBITS):
    terms.append(PauliString({i: 'X'}, -1.0))
    
observable = PauliSum(pauli_strings=terms)

# --- 3. Define Circuit Structure (Standard Mode) ---
# We build a single Circuit object with Parameter placeholders.
print("Constructing Circuit...")
# Note: Circuit init doesn't take n_qubits, but it's inferred from gates or propagated.
# We just create an empty circuit.
circuit = Circuit()

for d in range(DEPTH):
    # Layer d
    # RX
    for q in range(N_QUBITS):
        # We assign a unique Parameter for each trainable gate
        # The JITPropagator will map them to the input array in order of appearance.
        p_name = f"l{d}_q{q}_rx"
        circuit.add(RX(q, Parameter(p_name)))
        
    # RZ
    for q in range(N_QUBITS):
        p_name = f"l{d}_q{q}_rz"
        circuit.add(RZ(q, Parameter(p_name)))
        
    # CNOT
    for i in range(N_QUBITS - 1):
        circuit.add(CNOT(i, i+1))

# --- 4. Create JIT Loss Function ---
print("Compiling JIT Loss Function...")
t0 = time.time()

# Now we use the unified compile_expectation_fn!
loss_fn = compile_expectation_fn(
    circuit=circuit,
    n_qubits=N_QUBITS,
    max_weight=MAX_WEIGHT,
    observables=observable
)
print(f"Compilation/Pre-computation Done in {time.time()-t0:.2f}s")

# --- 5. Optimization Loop ---

def main():
    print("\n--- Starting Training ---")
    
    # Total params: DEPTH * (2 * N_QUBITS)
    TOTAL_PARAMS = DEPTH * 2 * N_QUBITS
    
    key = jax.random.PRNGKey(42)
    # Flat parameter array
    params = jax.random.normal(key, (TOTAL_PARAMS,)) * 0.1
    
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)
    
    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    # Warmup
    print("JIT Compiling Step Function...")
    t0 = time.time()
    step(params, opt_state)
    print(f"JIT Finished in {time.time()-t0:.2f}s")
    
    t_start = time.time()
    for i in range(STEPS):
        params, opt_state, loss = step(params, opt_state)
        if i % 20 == 0:
            print(f"Step {i}: Energy = {loss:.6f}")
            
    print(f"Final Energy: {loss:.6f}")
    print(f"Training Time: {time.time()-t_start:.2f}s")

if __name__ == "__main__":
    main()
