"""
PyPoli CPU vs GPU Performance Comparison

This example demonstrates:
1. CPU-optimized operations using bit encoding
2. GPU-optimized operations using JAX arrays
3. Performance comparison between the two approaches
"""
import sys
sys.path.insert(0, 'src')

import time
import jax
import jax.numpy as jnp
from pypoli import PauliString, Circuit, RX, RY, RZ, RXX, RYY, RZZ, CNOT, H


# =============================================================================
# CPU-OPTIMIZED OPERATIONS (Bit Encoding)
# =============================================================================

def cpu_optimized_multiplication(n_qubits=10, n_operations=1000):
    """
    CPU-optimized: Uses native Python int bit operations
    - Fast for small-to-medium scale operations
    - No device transfer overhead
    - Direct bit manipulation
    """
    print("=== CPU-Optimized (Bit Encoding) ===")

    # Create Pauli strings using bit encoding
    # Each qubit uses 2 bits: I=00, X=01, Y=10, Z=11
    bits_x = sum(1 << (2*i) for i in range(n_qubits))  # X on all qubits
    bits_z = sum(3 << (2*i) for i in range(n_qubits))  # Z on all qubits

    p1 = PauliString(bits_x, 1.0, n_qubits)
    p2 = PauliString(bits_z, 1.0, n_qubits)

    print(f"Pauli 1 bits: {bin(p1.bits)}")
    print(f"Pauli 2 bits: {bin(p2.bits)}")
    # Calculate weight by counting set bits (2 bits per qubit)
    weight = bin(p1.bits).count('1') // 2
    print(f"Weight: {weight}")

    # Benchmark multiplication
    start = time.time()
    for _ in range(n_operations):
        result = p1 * p2
    elapsed = time.time() - start

    print(f"Multiplications: {n_operations}")
    print(f"Time: {elapsed:.4f} seconds")
    print(f"Ops/sec: {n_operations/elapsed:.0f}")

    return elapsed


def cpu_optimized_propagation(n_qubits=6, depth=4):
    """
    CPU-optimized circuit propagation using bit encoding
    """
    print("\n=== CPU Circuit Propagation ===")

    circuit = Circuit()
    for layer in range(depth):
        for q in range(n_qubits):
            circuit.add_gate(H(q))  # Clifford gates don't increase terms
        for q in range(n_qubits - 1):
            circuit.add_gate(CNOT(q, q + 1))

    # Observable with full weight
    obs_bits = sum(1 << (2*i) for i in range(n_qubits))
    observable = PauliString(obs_bits, 1.0, n_qubits)

    start = time.time()
    result = circuit.propagate(observable)
    elapsed = time.time() - start

    print(f"Qubits: {n_qubits}, Depth: {depth}")
    print(f"Input weight: {n_qubits}")
    print(f"Output terms: {len(result)}")
    print(f"Time: {elapsed:.4f} seconds")

    return elapsed


# =============================================================================
# GPU-OPTIMIZED OPERATIONS (JAX Arrays)
# =============================================================================

def gpu_array_conversion(n_qubits=10):
    """
    GPU: Demonstrates JAX array conversion for GPU processing
    - Fast for batch operations on GPU
    - Compatible with JAX transformations
    """
    print("\n=== GPU JAX Array Conversion ===")

    # Create Pauli string using bit encoding
    bits_x = sum(1 << (2*i) for i in range(n_qubits))
    pauli = PauliString(bits_x, 1.0, n_qubits)

    # Convert to JAX array for GPU processing
    arr = pauli.to_array()
    print(f"JAX array shape: {arr.shape}")
    print(f"JAX array dtype: {arr.dtype}")
    print(f"Device: {jax.devices()}")

    # Demonstrate GPU operations on the array
    start = time.time()

    # Vectorized operations (run on GPU if available)
    doubled = arr * 2  # Vectorized doubling
    masked = jnp.where(arr > 0, arr, 0)  # Conditional masking

    elapsed = time.time() - start
    print(f"GPU operations time: {elapsed:.6f}s")

    # Convert back to PauliString
    pauli_back = PauliString.from_array(masked, 2.0)
    print(f"Converted back: bits={bin(pauli_back.bits)}")

    return elapsed


def gpu_optimized_jit_circuit(n_qubits=4, n_repeats=50):
    """
    GPU-optimized with JIT compilation
    - First run: compilation (slow)
    - Subsequent runs: optimized execution (fast)
    """
    print("\n=== GPU JIT-Compiled Circuit ===")

    @jax.jit
    def circuit_fn(_params):
        """JIT-compiled circuit for GPU execution"""
        circuit = Circuit()
        for q in range(n_qubits):
            circuit.add_gate(H(q))  # Clifford gates only
        for q in range(n_qubits - 1):
            circuit.add_gate(CNOT(q, q + 1))

        obs_bits = 3 << (2 * 0)  # Z on qubit 0
        observable = PauliString(obs_bits, 1.0, n_qubits)
        result = circuit.propagate(observable)

        # Sum coefficients for expectation value
        coeffs = jnp.array([r.coefficient for r in result])
        total = jnp.sum(coeffs)
        return total

    params = jnp.ones(2 * n_qubits)

    # First run includes compilation
    start = time.time()
    _ = circuit_fn(params)
    first_run = time.time() - start

    # Subsequent runs use compiled version
    start = time.time()
    for _ in range(n_repeats):
        _ = circuit_fn(params)
    avg_time = (time.time() - start) / n_repeats

    print(f"Qubits: {n_qubits}")
    print(f"First run (with JIT compile): {first_run:.4f}s")
    print(f"Avg subsequent runs: {avg_time:.6f}s")
    if avg_time > 0:
        print(f"Speedup: {first_run/avg_time:.1f}x")

    return avg_time


def gpu_autodifferentiation_example(n_qubits=4):
    """
    GPU with automatic differentiation
    - Essential for variational quantum algorithms
    - Only works with JAX-compatible operations
    """
    print("\n=== GPU with Automatic Differentiation ===")

    @jax.jit
    def variational_circuit(params):
        """Parameterized circuit for optimization"""
        circuit = Circuit()
        for q in range(n_qubits):
            circuit.add_gate(RY(q, params[q]))
        for q in range(n_qubits - 1):
            circuit.add_gate(CNOT(q, q + 1))

        obs_bits = sum(3 << (2*i) for i in range(n_qubits))
        observable = PauliString(obs_bits, 1.0, n_qubits)
        result = circuit.propagate(observable, max_weight=n_qubits)

        # Expectation value - convert list to JAX array
        coeffs = jnp.array([r.coefficient for r in result])
        exp_val = jnp.real(jnp.sum(coeffs))
        return exp_val

    params = jnp.zeros(n_qubits)

    # Compute gradient
    grad_fn = jax.grad(variational_circuit)
    grad = grad_fn(params)

    print(f"Parameters: {params}")
    print(f"Gradients: {grad}")
    print(f"Gradient norm: {jnp.linalg.norm(grad):.6f}")

    # Simple gradient descent step
    learning_rate = 0.1
    new_params = params - learning_rate * grad
    new_exp_val = variational_circuit(new_params)

    print(f"After one step: {new_exp_val:.6f}")


# =============================================================================
# USAGE RECOMMENDATIONS
# =============================================================================

def print_recommendations():
    """Print usage recommendations for CPU vs GPU"""
    print("\n" + "="*60)
    print("USAGE RECOMMENDATIONS")
    print("="*60)

    print("""
┌─────────────────────┬────────────────────────────────────────┐
│ USE CPU (Bit Encoding) │ WHEN:                                │
├─────────────────────┼────────────────────────────────────────┤
│ • Small circuits     │ • < 20 qubits                         │
│ • Single operations  │ • Exploratory analysis                │
│ • Debugging          │ • Need fast iteration                 │
│ • Low overhead       │ • Memory constrained                  │
│                     │ • No batch operations                  │
└─────────────────────┴────────────────────────────────────────┘

┌─────────────────────┬────────────────────────────────────────┐
│ USE GPU (JAX Arrays)│ WHEN:                                │
├─────────────────────┼────────────────────────────────────────┤
│ • Large circuits     │ • > 20 qubits                         │
│ • Batch operations   │ • Many observables                    │
│ • Training/optim.    │ • Need gradients (VQA, QML)           │
│ • Repetitive eval.   │ • JIT compilation beneficial          │
│ • Parallelization    │ • GPU available                       │
└─────────────────────┴────────────────────────────────────────┘

CONVERSION METHODS:
    # CPU → GPU
    arr = pauli.to_array()           # bits → JAX array

    # GPU → CPU
    pauli = PauliString.from_array(arr, coeff)  # JAX array → bits

BEST PRACTICE:
    1. Start with CPU (bit encoding) for development
    2. Switch to GPU (JAX arrays) for:
       - Large-scale simulations
       - Training variational circuits
       - Batch expectation value calculations
    3. Always use @jax.jit for repeated circuit evaluations
""")


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def main():
    """Run comprehensive CPU vs GPU benchmark"""
    print("="*60)
    print("PyPoli CPU vs GPU Performance Comparison")
    print("="*60)
    print(f"JAX devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}\n")

    # CPU benchmarks
    cpu_mul_time = cpu_optimized_multiplication(n_qubits=10, n_operations=1000)
    cpu_circuit_time = cpu_optimized_propagation(n_qubits=8, depth=10)

    # GPU benchmarks
    gpu_array_time = gpu_array_conversion(n_qubits=10)
    gpu_jit_time = gpu_optimized_jit_circuit(n_qubits=4, n_repeats=50)

    # Gradient example
    gpu_autodifferentiation_example(n_qubits=4)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"CPU multiplication (1000 ops):  {cpu_mul_time:.4f}s")
    print(f"CPU circuit propagation:         {cpu_circuit_time:.4f}s")
    print(f"GPU array conversion:            {gpu_array_time:.6f}s")
    print(f"GPU JIT circuit (avg):           {gpu_jit_time:.6f}s")

    print_recommendations()


if __name__ == "__main__":
    main()
