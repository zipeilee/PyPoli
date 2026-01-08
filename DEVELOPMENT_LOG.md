# PyPoli Development Log

## Session: Fixes for Heisenberg Picture and Gate Logic

### 1. Issue Identification
The initial implementation had several discrepancies with the standard physics of Pauli propagation in the Heisenberg picture:
- **Gate Lookup Tables**: The `clifford_map` for CNOT and CZ gates contained incorrect mappings.
- **Rotation Gates**: The rotation gates (RX, RY, RZ, etc.) used half-angles (`theta/2`) incorrectly in the output coefficients for the Heisenberg picture transformation $P \to U^\dagger P U$.
- **Expectation Value**: The expectation value calculation incorrectly assumed only the Identity term contributed to the vacuum expectation value, ignoring $Z$ terms which also have $\langle 0|Z|0\rangle = 1$.

### 2. Implementation Changes

#### Gate Logic (`src/pypoli/gates.py`)
- **Generated Correct Tables**: Created `tools_gen_tables.py` to verify and generate correct Pauli mappings for CNOT and CZ gates.
- **Updated `clifford_map`**: Replaced the incorrect lookup tables with the verified ones.
- **Fixed Rotation Logic**: 
  - Corrected the formula for rotation gates to $P \to P \cos\theta - i \sin\theta [G, P]$ (where $G$ is the generator).
  - Fixed `RXX`, `RYY`, `RZZ` to correctly identify anti-commuting terms and apply rotations only to them.

#### Propagation Logic (`src/pypoli/propagation.py`)
- **Updated `expectation_value`**: 
  - Modified the function to sum coefficients of all diagonal terms (Identity and Z-only strings) since $\langle 0|Z|0\rangle = 1$.
  - Added support for passing `Circuit` objects directly (calls `propagate` internally).
  - Added `jit_expectation_value` for JAX compatibility.

### 3. Verification
- **Demo Script**: Created `demo.py` (and updated `pypoli_demo.ipynb`) to test the full pipeline.
- **Results**:
  - Validated that $\langle 0|Z|0\rangle = 1$ and $\langle 0|X|0\rangle = 0$.
  - Validated correct gradient descent training for state preparation.
  - Confirmed that truncation works as expected.

### 4. Conclusion
The library now correctly implements the Heisenberg picture propagation for quantum circuits starting from the vacuum state. The API is observable-centric, where users propagate an observable $O$ to compute $U^\dagger O U$, and the expectation value is taken with respect to $|0\rangle$.

## Session: JAX JIT Optimization and Hybrid Model Training (2026-01-03)

### 1. Goal
Implement a performant Hybrid Quantum-Classical Neural Network using `flax` and `optax`, ensuring the quantum simulation part (PyPoli) is compatible with JAX JIT compilation for efficient training.

### 2. Challenges
- **JIT Incompatibility**: The dynamic nature of Pauli propagation (term branching, filtering) conflicts with JAX JIT's requirement for static graph structures. Specifically, list lengths cannot change during JIT execution.
- **Loop Unrolling Explosion**: For deep circuits ($D=100$), standard JIT unrolls the loop, causing compilation time and memory usage to grow linearly with depth, making deep circuits untrainable.

### 3. Solutions

#### A. Coefficient Masking for JIT
- Modified `src/pypoli/circuits.py` to handle JAX Tracers in the `propagate` function.
- Implemented **Masking** instead of Removal: When running under JIT, coefficients below `min_abs_coeff` are set to zero (masked) rather than removing the term from the list. This maintains a static graph structure while suppressing noise.

#### B. Basis Projection with `jax.lax.scan`
- Implemented a "Basis Projection" strategy in `examples/hybrid_flax_benchmark.py`.
- **Pre-computation**: Defined a fixed basis of Pauli strings (e.g., all terms with Weight $\le 2$).
- **Vectorization**: Mapped the dynamic Pauli sum to a fixed-size coefficient vector $\mathbf{c} \in \mathbb{R}^{|\mathcal{B}|}$.
- **Scan**: Used `jax.lax.scan` to iterate over circuit layers. Since the state is now a fixed-size vector, JAX only compiles the single-step function once, reducing compilation complexity from $O(D)$ to $O(1)$.

### 4. Results
- **Compilation Speed**: Compilation time for a 25-layer circuit reduced from >2 minutes (timeout) to **<1 second**.
- **Execution Speed**: Average training step (forward + backward) takes **~0.6ms** for 10 qubits, 25 layers.
- **Convergence**: Successfully trained a hybrid model to fit a regression task ($Loss: 40.9 \to 0.97$) in 1000 steps (total time ~0.65s).
- **Scalability**: Verified that the approach scales to arbitrary depths without compilation overhead.

### 5. Documentation
- Created `JAX_Quantum_Optimization_Note.md` (Chinese & English) detailing the optimization strategy and theoretical background.

## Session: Hybrid Flax Integration & Performance Tuning (2026-01-08)

### 1. Issue Identification
The previous "Hybrid Flax" implementation suffered from severe performance degradation during training loops.
- **Re-JIT on Every Step**: The Flax `Module` was re-tracing the quantum computation graph on every iteration, leading to massive overhead (seconds per step instead of milliseconds).
- **Initialization Latency**: The first step of training had a huge pause due to unoptimized compilation triggers.
- **VMap Overhead**: Manual loop over batch dimension was inefficient compared to proper `jax.vmap`.

### 2. Implementation Changes

#### A. Global Compilation Cache (`examples/hybrid_flax.py`)
- **Decoupled Compilation**: Moved the circuit construction and JIT compilation logic *outside* the Flax Module into a global `get_quantum_layer_fn` with caching `_QUANTUM_CACHE`.
- **Singleton Pattern**: Ensures that for a given configuration `(n_qubits, depth, max_weight)`, the circuit is built and compiled exactly once per process.

#### B. Pure JAX Optimization
- **Pre-warming**: Added a dummy execution step immediately after JIT compilation to force XLA compilation before the training loop starts. This eliminates the "Step 0" lag.
- **Batch Propagation**: Implemented `jax.vmap` at the lowest level of the propagator step, allowing parallel processing of the entire batch of inputs.
- **DType Unification**: Standardized on `complex64` to reduce memory bandwidth and align with TPUs/GPUs.

#### C. Algorithmic Stability
- **Reverted to Pauli Propagation**: Explicitly rejected full state-vector simulation approaches in favor of the Truncated Pauli Propagation method (Heisenberg picture).
- **Weight Truncation**: Strictly enforced `max_weight` truncation to maintain $O(Poly(N))$ complexity instead of $O(2^N)$.

### 3. Results
- **Training Speed**: Training steps are now consistently fast (~milliseconds) after the initial pre-warm.
- **Stability**: Removed "re-compiling" log messages during the training loop.
- **Correctness**: Maintained the physical truncation logic requested by the user.
