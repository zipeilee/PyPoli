
from pypoli import Circuit, PauliString
from pypoli.gates import RX
from pypoli.jit_propagation import propagate_jit_coefficients, TensorBasis
import jax.numpy as jnp
import jax

# Disable x64 to match current env issues? No, let's keep it simple.
# jax.config.update("jax_enable_x64", True)

def test_simple_prop():
    print("\n--- Testing Simple Propagation ---")
    n_qubits = 1
    max_weight = 1
    
    # 1. Layer: RX(0)
    def layer_fn(p):
        return [RX(0, p[0])]
        
    # 2. Params: pi/2
    params = jnp.array([[jnp.pi/2]])
    
    # 3. Observable: Z0
    obs = PauliString({0: 'Z'}, 1.0)
    
    print("Propagating Z0 through RX(0, pi/2)...")
    coeffs, basis = propagate_jit_coefficients(
        layer_fn, params, obs, n_qubits, max_weight, min_abs_coeff=1e-5, damping_factor=0.0
    )
    
    print(f"Basis size: {basis.size}")
    for i, term in enumerate(basis.terms):
        c = coeffs[i]
        if jnp.abs(c) > 1e-5:
            print(f"Term {term}: {c}")

    # Expected: Y0 (or -Y0 depending on convention). RX(pi/2) Z RX(-pi/2) = Y?
    # R_X(th) Z R_X(-th) = Z cos - Y sin.
    # cos(pi/2)=0. sin(pi/2)=1. -> -Y.
    
    # Check Norm
    norm = jnp.linalg.norm(coeffs)
    print(f"Final Norm: {norm}")

if __name__ == "__main__":
    test_simple_prop()
