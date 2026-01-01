"""
Test script to verify the lookup table implementation of quantum gates
"""
import jax
import jax.numpy as jnp
from pypoli import PauliString, Circuit, I, X, Y, Z, H, S, CNOT, CZ

def test_single_qubit_gates():
    """Test single-qubit Clifford gates"""

    print("Testing single-qubit gates...")

    # Create a Pauli string for testing
    pauli_x = PauliString({0: 'X'})
    pauli_y = PauliString({0: 'Y'})
    pauli_z = PauliString({0: 'Z'})

    print(f"\nOriginal X0: {pauli_x}")

    # Test X gate
    result = X(0).pauli_action(pauli_x)
    print(f"X(0) * X0 * X(0)†: {result}")

    # Test Y gate
    result = Y(0).pauli_action(pauli_y)
    print(f"Y(0) * Y0 * Y(0)†: {result}")

    # Test Z gate on X
    result = Z(0).pauli_action(pauli_x)
    print(f"Z(0) * X0 * Z(0)†: {result}")

    # Test Hadamard on X
    result = H(0).pauli_action(pauli_x)
    print(f"H(0) * X0 * H(0)†: {result}")

    # Test Hadamard on Z
    result = H(0).pauli_action(pauli_z)
    print(f"H(0) * Z0 * H(0)†: {result}")

    # Test S on X
    result = S(0).pauli_action(pauli_x)
    print(f"S(0) * X0 * S(0)†: {result}")

    # Test S on Y
    result = S(0).pauli_action(pauli_y)
    print(f"S(0) * Y0 * S(0)†: {result}")

def test_two_qubit_gates():
    """Test two-qubit Clifford gates"""

    print("\n\nTesting two-qubit gates...")

    # Test CNOT gate
    cnot = CNOT(0, 1)  # control: 0, target: 1

    # Test CNOT on X0 (control qubit)
    x0 = PauliString({0: 'X'})
    result = cnot.pauli_action(x0)
    print(f"CNOT(0,1) * X0 * CNOT(0,1)†: {result}")

    # Test CNOT on X1 (target qubit)
    x1 = PauliString({1: 'X'})
    result = cnot.pauli_action(x1)
    print(f"CNOT(0,1) * X1 * CNOT(0,1)†: {result}")

    # Test CNOT on Y1 (target qubit)
    y1 = PauliString({1: 'Y'})
    result = cnot.pauli_action(y1)
    print(f"CNOT(0,1) * Y1 * CNOT(0,1)†: {result}")

    # Test CNOT on Z0 (control qubit)
    z0 = PauliString({0: 'Z'})
    result = cnot.pauli_action(z0)
    print(f"CNOT(0,1) * Z0 * CNOT(0,1)†: {result}")

    # Test CNOT on Z1 (target qubit)
    z1 = PauliString({1: 'Z'})
    result = cnot.pauli_action(z1)
    print(f"CNOT(0,1) * Z1 * CNOT(0,1)†: {result}")

    # Test CZ gate
    print("\nTesting CZ gate...")
    cz = CZ(0, 1)

    # Test CZ on X0 X1
    x0x1 = PauliString({0: 'X', 1: 'X'})
    result = cz.pauli_action(x0x1)
    print(f"CZ(0,1) * X0 X1 * CZ(0,1)†: {result}")

    # Test CZ on Y0 X1
    y0x1 = PauliString({0: 'Y', 1: 'X'})
    result = cz.pauli_action(y0x1)
    print(f"CZ(0,1) * Y0 X1 * CZ(0,1)†: {result}")

def test_circuit_propagation():
    """Test circuit propagation with lookup table gates"""

    print("\n\nTesting circuit propagation...")

    # Create a simple circuit: H(0) -> CNOT(0,1)
    circuit = Circuit()
    circuit.add_gate(H(0))
    circuit.add_gate(CNOT(0, 1))

    print(f"Circuit:\n{circuit}")

    # Test propagation of Z1
    z1 = PauliString({1: 'Z'})
    result = circuit.propagate(z1)
    print(f"\nPropagate Z1 through circuit:\n{result}")

    # Test expectation value calculation
    from pypoli import expectation_value

    exp_val = expectation_value(circuit, z1)
    print(f"\nExpectation value <0| U† Z1 U |0>: {exp_val}")

if __name__ == "__main__":
    test_single_qubit_gates()
    test_two_qubit_gates()
    test_circuit_propagation()
    print("\n\nAll tests completed!")
