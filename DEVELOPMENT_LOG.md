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
