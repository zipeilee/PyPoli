
import numpy as np

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

paulis = [I, X, Y, Z]
pauli_names = ['I', 'X', 'Y', 'Z']

def tensor(A, B):
    return np.kron(A, B)

# CNOT matrix (control=0, target=1)
# |00> -> |00>, |01> -> |01>, |10> -> |11>, |11> -> |10>
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

# CZ matrix
CZ = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
], dtype=complex)

# Single qubit gates
H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)

def decompose_pauli(matrix):
    # Decompose 2x2 or 4x4 matrix into Pauli string
    # M = sum c_i P_i
    # c_i = 1/N Tr(M P_i)
    dim = matrix.shape[0]
    if dim == 2:
        coeffs = []
        for i in range(4):
            c = np.trace(matrix @ paulis[i]) / 2
            coeffs.append(c)
        return coeffs
    elif dim == 4:
        coeffs = []
        for i in range(4):
            for j in range(4):
                P = tensor(paulis[i], paulis[j])
                c = np.trace(matrix @ P) / 4
                coeffs.append(((i, j), c))
        return coeffs

def check_gate_2q(U, name):
    print(f"Generating table for {name}")
    table = []
    # Input P is P_c \otimes P_t
    # We want U^\dagger P U
    # Because Heisenberg picture: O(t) = U^\dagger O U
    
    # Wait, the library says:
    # "Gates are applied in reverse order (U = U_n ... U_1 -> U† O U = U_1† ... U_n† O U_n ... U_1)"
    # "The action of each gate is its conjugate action"
    # Gate.pauli_action(term) returns list of terms.
    # If we have circuit U, and observable O.
    # We want U^\dagger O U.
    # If circuit has one gate G. U = G.
    # Result = G^\dagger O G.
    
    # Let's verify this interpretation.
    # If I have a state |psi> and observable O.
    # <psi| U^\dagger O U |psi>.
    # If O is Pauli P.
    # We map P -> G^\dagger P G.
    
    U_dag = U.conj().T
    
    for i in range(4): # Control Pauli
        for j in range(4): # Target Pauli
            input_P = tensor(paulis[i], paulis[j])
            
            # Calculate G^\dagger P G
            output_M = U_dag @ input_P @ U
            
            # Decompose output
            coeffs = decompose_pauli(output_M)
            
            # Find the non-zero term (Clifford gates map Pauli to Pauli)
            found = False
            for (out_i, out_j), c in coeffs:
                if abs(c) > 1e-10:
                    # Found the mapping
                    # Input bits: (j << 2) | i  (Target is bits 2-3, Control is bits 0-1)
                    # NOTE: library uses enumerate(sorted(qubits)). 
                    # If CNOT(0, 1), q0 is control, q1 is target.
                    # sorted: 0, 1.
                    # i=0 (q0/control): bits 0-1.
                    # i=1 (q1/target): bits 2-3.
                    input_bits = i | (j << 2)
                    output_bits = out_i | (out_j << 2)
                    
                    # Phase
                    # c can be 1, -1, 1j, -1j
                    print(f"  0x{input_bits:02X} ({pauli_names[i]}-{pauli_names[j]}) -> 0x{output_bits:02X} ({pauli_names[out_i]}-{pauli_names[out_j]}), coeff={c:.2f}")
                    found = True
            if not found:
                print(f"  ERROR: No mapping found for {pauli_names[i]}-{pauli_names[j]}")

def check_gate_1q(U, name):
    print(f"Generating table for {name}")
    U_dag = U.conj().T
    for i in range(4):
        input_P = paulis[i]
        output_M = U_dag @ input_P @ U
        
        coeffs = decompose_pauli(output_M)
        for idx, c in enumerate(coeffs):
            if abs(c) > 1e-10:
                print(f"  {pauli_names[i]} -> {pauli_names[idx]}, coeff={c:.2f}")

print("--- 1Q Gates ---")
check_gate_1q(I, 'I')
check_gate_1q(X, 'X')
check_gate_1q(Y, 'Y')
check_gate_1q(Z, 'Z')
check_gate_1q(H, 'H')
check_gate_1q(S, 'S')

print("\n--- 2Q Gates ---")
check_gate_2q(CNOT, 'CNOT')
check_gate_2q(CZ, 'CZ')
