
from pypoli import Circuit, PauliString, PauliSum
from pypoli.gates import Parameter
from pypoli.jit_propagation import Basis
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)

def test_full_hea():
    print("\n--- Testing Full HEA VQE Scenario ---")
    n_qubits = 6
    depth = 10
    
    # 1. Template
    layer = Circuit()
    from pypoli.gates import RX, RZ, CNOT
    for q in range(n_qubits):
        layer.add_gate(RX(q, "theta")) 
        layer.add_gate(RZ(q, "phi"))
    for q in range(n_qubits - 1):
        layer.add_gate(CNOT(q, q+1))
        
    circuit = Circuit.from_layer(layer, depth)
    
    # 2. Params
    params = jnp.zeros((depth, n_qubits * 2)) 
    bound_c = circuit.bind(params)
    
    # 3. Propagate Z0 Z1
    obs = PauliString({0: 'Z', 1: 'Z'}, 1.0)
    
    print("Propagating Z0Z1 through full HEA (params=0)...")
    res = bound_c.propagate(obs, max_weight=4, min_abs_coeff=1e-5, damping=0.0)
    
    print(f"Result terms count: {len(res)}")
    # Find coeff of Z0Z1
    z0z1_coeff = 0.0
    for term in res:
        if term.paulis == {0: 'Z', 1: 'Z'}:
            z0z1_coeff = term.coefficient
            break
            
    print(f"Z0Z1 Coeff: {z0z1_coeff}")
    
    # Also check Expectation Value
    from pypoli import expectation_value
    val = expectation_value(bound_c, obs, max_weight=4, min_abs_coeff=1e-5, damping=0.0)
    print(f"Expectation Value: {val}")

if __name__ == "__main__":
    # manual_scan_step()
    # test_circuit_prop()
    test_full_hea()
