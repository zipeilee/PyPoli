
import jax
import jax.numpy as jnp
import numpy as np
from pypoli.core import PauliString, PauliSum, Parameter
from pypoli.gates import RX, RZ, CNOT
from pypoli.circuits import Circuit
from pypoli.jit_propagation import make_jit_loss_fn, JITPropagator

jax.config.update("jax_enable_x64", True)

def test_debug():
    N_QUBITS = 10
    MAX_WEIGHT = 4
    
    # 1. Test Observable Mapping
    terms = []
    # 9 ZZ terms
    for i in range(N_QUBITS - 1):
        terms.append(PauliString({i: 'Z', i+1: 'Z'}, -1.0))
    # 10 X terms
    for i in range(N_QUBITS):
        terms.append(PauliString({i: 'X'}, -1.0))
        
    observable = PauliSum(pauli_strings=terms)
    
    # Manually check mapping
    prop = JITPropagator(N_QUBITS, MAX_WEIGHT)
    coeffs = prop.map_observable(observable)
    print(f"Coeffs Norm: {jnp.linalg.norm(coeffs)}")
    print(f"Non-zero elements: {jnp.sum(coeffs != 0)}")
    
    # Check Diagonal Mask
    diag_mask = prop.get_diagonal_mask()
    diag_energy = jnp.sum(jnp.where(diag_mask, coeffs, 0.0))
    print(f"Energy at Identity (Expectation of ZZ): {diag_energy}")
    # Should be -9.0
    
    # 2. Test CNOT Propagation (Identity Params)
    # Circuit with just CNOTs
    circuit = Circuit()
    for i in range(N_QUBITS - 1):
        circuit.add(CNOT(i, i+1))
        
    # Compile
    # We pass empty params? No, make_jit_loss_fn expects params matching parameterized gates.
    # CNOT has no params.
    loss_fn = make_jit_loss_fn(circuit, n_qubits=N_QUBITS, max_weight=MAX_WEIGHT, observable=observable)
    
    e_cnot = loss_fn(jnp.array([]))
    print(f"Energy after CNOTs (should be -9.0 if X->XX is ignored): {e_cnot}")
    
    # 3. Test Full Circuit with Zero Params
    circuit2 = Circuit()
    # RX(0)
    for q in range(N_QUBITS):
        circuit2.add(RX(q, Parameter(f"rx_{q}")))
    # CNOTs
    for i in range(N_QUBITS - 1):
        circuit2.add(CNOT(i, i+1))
        
    loss_fn2 = make_jit_loss_fn(circuit2, n_qubits=N_QUBITS, max_weight=MAX_WEIGHT, observable=observable)
    
    # Params = 0
    params = jnp.zeros(N_QUBITS)
    e_full = loss_fn2(params)
    print(f"Energy with RX(0) + CNOTs: {e_full}")
    
    # 4. Test Random Params
    key = jax.random.PRNGKey(42)
    rand_params = jax.random.normal(key, (N_QUBITS,)) * 0.1
    e_rand = loss_fn2(rand_params)
    print(f"Energy with RX(0.1) + CNOTs: {e_rand}")
    
    # 5. Full HEA Depth 10
    circuit_hea = Circuit()
    DEPTH = 10
    for d in range(DEPTH):
        for q in range(N_QUBITS): circuit_hea.add(RX(q, Parameter(f"l{d}q{q}rx")))
        for q in range(N_QUBITS): circuit_hea.add(RZ(q, Parameter(f"l{d}q{q}rz")))
        for i in range(N_QUBITS - 1): circuit_hea.add(CNOT(i, i+1))
        
    loss_fn_hea = make_jit_loss_fn(circuit_hea, n_qubits=N_QUBITS, max_weight=MAX_WEIGHT, observable=observable)
    total_params = DEPTH * 2 * N_QUBITS
    hea_params = jax.random.normal(key, (total_params,)) * 0.1
    e_hea = loss_fn_hea(hea_params)
    print(f"Full HEA Depth 10 Random Energy: {e_hea}")

if __name__ == "__main__":
    test_debug()
