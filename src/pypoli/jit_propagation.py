
import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Dict, Any, Union, Callable
from itertools import combinations
import time
from .core import PauliString, PauliSum, Parameter
from .circuits import Circuit
from .gates import Gate

# --- 1. Tensor Basis ---

class TensorBasis:
    def __init__(self, n_qubits: int, max_weight: int):
        self.n_qubits = n_qubits
        self.max_weight = max_weight
        
        # Generate Terms
        self.terms = []
        def ps(qubit_paulis): return PauliString(qubit_paulis, 1.0)
        
        # Identity
        self.terms.append(ps({}))
        # 1-body
        for i in range(n_qubits):
            for p in ['X', 'Y', 'Z']: self.terms.append(ps({i: p}))
        
        if max_weight >= 2:
            for i, j in combinations(range(n_qubits), 2):
                for p1 in ['X', 'Y', 'Z']:
                    for p2 in ['X', 'Y', 'Z']: self.terms.append(ps({i: p1, j: p2}))
                    
        if max_weight >= 3:
             for i, j, k in combinations(range(n_qubits), 3):
                for p1 in ['X', 'Y', 'Z']:
                    for p2 in ['X', 'Y', 'Z']:
                        for p3 in ['X', 'Y', 'Z']: self.terms.append(ps({i: p1, j: p2, k: p3}))
        
        if max_weight >= 4:
             for i, j, k, l in combinations(range(n_qubits), 4):
                for p1 in ['X', 'Y', 'Z']:
                    for p2 in ['X', 'Y', 'Z']:
                        for p3 in ['X', 'Y', 'Z']: 
                            for p4 in ['X', 'Y', 'Z']: self.terms.append(ps({i: p1, j: p2, k: p3, l: p4}))

        self.size = len(self.terms)
        
        # Build Integer Maps for fast lookup
        self.tensor = np.zeros((self.size, n_qubits), dtype=np.int8)
        self.int_keys = np.zeros(self.size, dtype=np.int64)
        self.int_map = {}
        self.map = {tuple(sorted(t.paulis.items())): i for i, t in enumerate(self.terms)}
        self.weights = np.zeros(self.size, dtype=np.int32)
        
        p_val_map = {'I':0, 'X':1, 'Y':2, 'Z':3}
        for i, t in enumerate(self.terms):
            val = 0
            w = 0
            for q, p in t.paulis.items():
                pv = p_val_map[p]
                self.tensor[i, q] = pv
                # Use Python int for calculation to avoid overflow
                val += pv * (4**q)
                if pv != 0: w += 1
            self.int_keys[i] = val
            self.int_map[val] = i
            self.weights[i] = w

# --- 2. CNOT Lookup Table ---

CNOT_TABLE = {}
def _build_cnot_table():
    # Helper to multiply single qubit Paulis: return (res, phase_code)
    # phase_code: 0=1, 1=i, 2=-1, 3=-i
    def mul(p1, p2): 
        if p1==0: return p2, 0
        if p2==0: return p1, 0
        if p1==p2: return 0, 0
        if p1==1 and p2==2: return 3, 1 # XY=iZ
        if p1==2 and p2==3: return 1, 1 # YZ=iX
        if p1==3 and p2==1: return 2, 1 # ZX=iY
        if p1==2 and p2==1: return 3, 3 # YX=-iZ
        if p1==3 and p2==2: return 1, 3 # ZY=-iX
        if p1==1 and p2==3: return 2, 3 # XZ=-iY
        return 0, 0

    # Complex multiplication for phase codes
    # 0->1, 1->i, 2->-1, 3->-i
    def mul_phase(ph1, ph2):
        # (i^a) * (i^b) = i^(a+b)
        return (ph1 + ph2) % 4

    for pc in range(4):
        for pt in range(4):
            # Map Pc
            res_c_c, res_c_t = 0, 0
            ph_c = 0
            if pc == 1: res_c_c, res_c_t = 1, 1 # X -> XX
            elif pc == 2: res_c_c, res_c_t = 2, 1 # Y -> YX
            elif pc == 3: res_c_c, res_c_t = 3, 0 # Z -> ZI
            elif pc == 0: res_c_c, res_c_t = 0, 0
            
            # Map Pt
            res_t_c, res_t_t = 0, 0
            ph_t = 0
            if pt == 1: res_t_c, res_t_t = 0, 1 # X -> IX
            elif pt == 2: res_t_c, res_t_t = 3, 2 # Y -> ZY
            elif pt == 3: res_t_c, res_t_t = 3, 3 # Z -> ZZ
            elif pt == 0: res_t_c, res_t_t = 0, 0
            
            # Combine
            final_c, ph1 = mul(res_c_c, res_t_c)
            final_t, ph2 = mul(res_c_t, res_t_t)
            
            total_phase = mul_phase(mul_phase(ph_c, ph_t), mul_phase(ph1, ph2))
            
            # Convert phase code to complex
            phase_val = 1.0
            if total_phase == 1: phase_val = 1j
            elif total_phase == 2: phase_val = -1.0
            elif total_phase == 3: phase_val = -1j
            
            CNOT_TABLE[(pc, pt)] = (final_c, final_t, phase_val)

_build_cnot_table()


# --- 3. JIT Propagator ---

class JITPropagator:
    """
    Optimized JIT Propagator using Integer Tensor Basis and Pre-computation.
    """
    def __init__(self, n_qubits: int, max_weight: int = 4):
        self.n_qubits = n_qubits
        self.max_weight = max_weight
        self.basis = TensorBasis(n_qubits, max_weight)
        
    def compile_layer_ops(self, gates: List[Gate]):
        """
        Compile a list of gates into sparse update operations.
        """
        ops = []
        p_idx = 0
        
        # Track parameters
        # We need to assign `p_idx` to parameterized gates.
        
        for gate in gates:
            g_type = type(gate).__name__
            qubits = gate.qubits
            
            # Check if parameterized (Either has Parameter object or is_param flag)
            is_param = False
            
            if hasattr(gate, 'theta'):
                if isinstance(gate.theta, Parameter):
                    is_param = True
                elif not isinstance(gate.theta, (int, float, complex)) and not isinstance(gate.theta, jnp.ndarray):
                    # Fallback for tracers or other types
                    is_param = True
            
            current_p_idx = -1
            if is_param:
                current_p_idx = p_idx
                p_idx += 1
            
            # --- Vectorized Compilation ---
            
            src_list = []
            dst_list = []
            phase_list = []
            ctype_list = [] # 0:const, 1:cos, 2:sin
            
            if g_type in ['RX', 'RY', 'RZ']:
                # Rotation
                q = qubits[0]
                col = self.basis.tensor[:, q]
                
                # Determine anti-commuting Pauli
                if g_type == 'RX': mask = (col == 2) | (col == 3)
                elif g_type == 'RY': mask = (col == 1) | (col == 3)
                elif g_type == 'RZ': mask = (col == 1) | (col == 2)
                else: mask = np.zeros_like(col, dtype=bool)
                
                indices = np.where(mask)[0]
                
                for i in indices:
                    p = int(col[i])
                    target_p = 0
                    ph = 1.0
                    
                    if g_type == 'RX': # X
                        if p == 2: target_p = 3; ph = 1.0
                        else: target_p = 2; ph = -1.0
                    elif g_type == 'RY': # Y
                        if p == 1: target_p = 3; ph = -1.0
                        else: target_p = 1; ph = 1.0
                    elif g_type == 'RZ': # Z
                        if p == 1: target_p = 2; ph = 1.0
                        else: target_p = 1; ph = -1.0
                        
                    # Calculate key
                    # Remove p, add target_p
                    key = self.basis.int_keys[i] - p*(4**q) + target_p*(4**q)
                    
                    dst = self.basis.size # Default to dummy (truncation)
                    if key in self.basis.int_map:
                        dst = self.basis.int_map[key]
                        
                    src_list.append(i)
                    dst_list.append(dst)
                    phase_list.append(ph)
                    ctype_list.append(1) # Mark as Rotation
            
                if src_list:
                    ops.append({
                        'type': 'rot',
                        'p_idx': current_p_idx,
                        'src': jnp.array(src_list, dtype=jnp.int32),
                        'dst': jnp.array(dst_list, dtype=jnp.int32),
                        'phase': jnp.array(phase_list, dtype=jnp.complex64)
                    })
                    
            elif g_type == 'CNOT':
                # CNOT
                c, t = qubits
                c_mult = 4**c
                t_mult = 4**t
                
                mask = (self.basis.tensor[:, c] == 1) | (self.basis.tensor[:, c] == 2) | \
                       (self.basis.tensor[:, t] == 2) | (self.basis.tensor[:, t] == 3)
                indices = np.where(mask)[0]
                
                src_rem = []
                src_add = []
                dst_add = []
                phase_add = []
                
                for idx in indices:
                    pc = int(self.basis.tensor[idx, c])
                    pt = int(self.basis.tensor[idx, t])
                    
                    new_pc, new_pt, ph = CNOT_TABLE[(pc, pt)]
                    
                    if new_pc == pc and new_pt == pt and abs(ph - 1.0) < 1e-9:
                        continue
                        
                    old_val = pc*c_mult + pt*t_mult
                    new_val = new_pc*c_mult + new_pt*t_mult
                    key = self.basis.int_keys[idx] - old_val + new_val
                    
                    dst = self.basis.size # Default to dummy (truncation)
                    if key in self.basis.int_map:
                        dst = self.basis.int_map[key]
                        
                    src_rem.append(idx)
                    src_add.append(idx)
                    dst_add.append(dst)
                    phase_add.append(ph)
                
                if src_rem:
                     ops.append({
                        'type': 'cnot',
                        'p_idx': -1,
                        'rem': jnp.array(src_rem, dtype=jnp.int32),
                        'src': jnp.array(src_add, dtype=jnp.int32),
                        'dst': jnp.array(dst_add, dtype=jnp.int32),
                        'phase': jnp.array(phase_add, dtype=jnp.complex64)
                    })

        return ops, p_idx

    def make_scan_step(self, layer_ops):
        """
        Create the JAX scan step function.
        """
        thresholds = 0.0 # TODO: Pass damping
        
        def scan_step(coeffs, layer_params_vals):
            current_coeffs = coeffs
            
            # Process gates in order
            for i, op in enumerate(layer_ops):
                # Debug Norm
                # if i % 50 == 0:
                #      jax.debug.print("Op {i} Norm: {n}", i=i, n=jnp.linalg.norm(current_coeffs))
            
                if op['type'] == 'rot':
                    # Rotation Logic
                    # P' = P + P(cos-1) + P'(sin*phase)
                    
                    src = op['src']
                    dst = op['dst']
                    phase = op['phase']
                    
                    theta = layer_params_vals[op['p_idx']]
                    cos_t = jnp.cos(theta)
                    sin_t = jnp.sin(theta)
                    
                    vals = current_coeffs[src]
                    
                    # 1. Cosine attenuation (in-place add)
                    delta_cos = vals * (cos_t - 1.0)
                    current_coeffs = current_coeffs.at[src].add(delta_cos)
                    
                    # 2. Sine mixing (cross add)
                    # Note: phase is real (+1/-1) for these Pauli rotations
                    delta_sin = vals * sin_t * phase
                    current_coeffs = current_coeffs.at[dst].add(delta_sin)
                    
                elif op['type'] == 'cnot':
                    # CNOT Logic
                    # P -> P_new * phase
                    
                    src_idx = op['src']
                    rem_idx = op['rem']
                    dst_idx = op['dst']
                    phase = op['phase']
                    
                    # Read values BEFORE clearing
                    vals_to_move = current_coeffs[src_idx]
                    
                    # Remove from old position (subtract self)
                    current_coeffs = current_coeffs.at[rem_idx].add(-1.0 * current_coeffs[rem_idx])
                    
                    # Add to new position
                    current_coeffs = current_coeffs.at[dst_idx].add(vals_to_move * phase)
            
            # Debug: Return norm
            # norm = jnp.linalg.norm(current_coeffs)
            return current_coeffs, None
            
        return scan_step
    
    def make_flat_step(self, layer_ops):
        """
        Create a flat execution function (no scan, just run ops).
        """
        def flat_step(coeffs, params_vals):
            current_coeffs = coeffs
            is_batch = (current_coeffs.ndim == 2)
            
            # Process gates in order
            for op in layer_ops:
                if op['type'] == 'rot':
                    src = op['src']
                    dst = op['dst']
                    phase = op['phase']
                    
                    theta = params_vals[op['p_idx']]
                    cos_t = jnp.cos(theta)
                    sin_t = jnp.sin(theta)
                    
                    if is_batch:
                        vals = current_coeffs[:, src]
                        delta_cos = vals * (cos_t - 1.0)
                        current_coeffs = current_coeffs.at[:, src].add(delta_cos)
                        delta_sin = vals * sin_t * phase
                        current_coeffs = current_coeffs.at[:, dst].add(delta_sin)
                    else:
                        vals = current_coeffs[src]
                        delta_cos = vals * (cos_t - 1.0)
                        current_coeffs = current_coeffs.at[src].add(delta_cos)
                        delta_sin = vals * sin_t * phase
                        current_coeffs = current_coeffs.at[dst].add(delta_sin)
                    
                elif op['type'] == 'cnot':
                    src_idx = op['src']
                    rem_idx = op['rem']
                    dst_idx = op['dst']
                    phase = op['phase']
                    
                    if is_batch:
                        vals_to_move = current_coeffs[:, src_idx]
                        current_coeffs = current_coeffs.at[:, rem_idx].add(-1.0 * current_coeffs[:, rem_idx])
                        current_coeffs = current_coeffs.at[:, dst_idx].add(vals_to_move * phase)
                    else:
                        vals_to_move = current_coeffs[src_idx]
                        current_coeffs = current_coeffs.at[rem_idx].add(-1.0 * current_coeffs[rem_idx])
                        current_coeffs = current_coeffs.at[dst_idx].add(vals_to_move * phase)
            
            return current_coeffs
        return flat_step

    def map_observable(self, observable):
        """
        Map observable to coefficient vector.
        """
        # Allocate size + 1 for dummy truncation bin
        coeffs = jnp.zeros(self.basis.size + 1, dtype=jnp.complex64)
        
        input_paulis = []
        if isinstance(observable, PauliString): input_paulis = [observable]
        elif isinstance(observable, PauliSum): input_paulis = observable.pauli_strings
        elif isinstance(observable, list): input_paulis = observable
        
        indices = []
        values = []
        for term in input_paulis:
            sig = tuple(sorted(term.paulis.items()))
            if sig in self.basis.map:
                indices.append(self.basis.map[sig])
                values.append(term.coefficient)
        
        if indices:
            coeffs = coeffs.at[jnp.array(indices)].add(jnp.array(values))
            
        return coeffs

    def get_diagonal_mask(self):
        """
        Return boolean mask for diagonal Pauli terms (I, Z).
        """
        mask = np.ones(self.basis.size + 1, dtype=bool)
        for i, term in enumerate(self.basis.terms):
            for p in term.paulis.values():
                if p in ['X', 'Y']:
                    mask[i] = False
                    break
        # Dummy bin is not diagonal (it's trash)
        mask[self.basis.size] = False
        return jnp.array(mask, dtype=jnp.bool_)

# --- 4. Public API ---

def compile_expectation_fn(
    circuit: Circuit,
    n_qubits: int,
    max_weight: int,
    observables: Union[PauliSum, List[PauliString], List[PauliSum]],
):
    """
    Compile a JIT-compatible function that computes expectation values.
    
    Args:
        circuit: The Quantum Circuit (with Parameters).
        n_qubits: Number of qubits.
        max_weight: Max Pauli weight for truncation.
        observables: Either a single Observable (returns scalar) or list (returns vector).
        
    Returns:
        forward_fn(params) -> result
        
        - If observables is single: result is scalar (float).
        - If observables is list: result is array (shape: [n_obs]).
        - params must be a flat array matching the order of Parameters in the circuit.
    """
    propagator = JITPropagator(n_qubits, max_weight)
    
    # 1. Compile Ops (Reverse for Heisenberg Picture)
    layer_ops, _ = propagator.compile_layer_ops(circuit.gates)
    flat_step = propagator.make_flat_step(layer_ops[::-1])
    diag_mask = propagator.get_diagonal_mask()
    
    # 2. Handle Observables
    is_list = isinstance(observables, list)
    
    if is_list:
        # Vector Output
        obs_coeffs = jnp.stack([propagator.map_observable(o) for o in observables])
        # vmap over observables (axis 0 of obs_coeffs), params are shared (None)
        batch_step = jax.vmap(flat_step, in_axes=(0, None))
        
        def _forward(params):
            final_vecs = batch_step(obs_coeffs, params)
            # Sum diagonal elements
            res = jnp.sum(jnp.where(diag_mask, final_vecs, 0.0), axis=1)
            return jnp.real(res)
            
    else:
        # Scalar Output
        init_coeffs = propagator.map_observable(observables)
        
        def _forward(params):
            final_coeffs = flat_step(init_coeffs, params)
            res = jnp.sum(jnp.where(diag_mask, final_coeffs, 0.0))
            return jnp.real(res)
            
    return jax.jit(_forward)

def make_jit_loss_fn(
    circuit: Union[Callable, Circuit], 
    n_qubits: int = None, 
    max_weight: int = 4, 
    observable = None,
    n_layers: int = None, # Optional if passing Circuit
):
    """
    Create a JIT-compiled loss function for VQE.
    Wrapper around compile_expectation_fn for backward compatibility.
    """
    
    # 1. Detect Mode
    is_legacy_mode = callable(circuit)
    
    if is_legacy_mode:
        if n_qubits is None or n_layers is None:
            raise ValueError("n_qubits and n_layers must be provided for function-based circuit structure.")
        return _make_jit_loss_fn_legacy(circuit, n_qubits, max_weight, observable, n_layers)
    
    # 2. Standard Mode
    if isinstance(circuit, Circuit):
        if n_qubits is None:
            n_qubits = circuit.n_qubits
    else:
        raise ValueError("Circuit must be a Circuit object or a callable.")
        
    forward_fn = compile_expectation_fn(circuit, n_qubits, max_weight, observable)
    return forward_fn

def _make_jit_loss_fn_legacy(
    circuit_structure_fn: Callable, 
    n_qubits: int, 
    max_weight: int, 
    observable,
    n_layers: int
):
    """
    Legacy implementation using scan and structure function.
    """
    propagator = JITPropagator(n_qubits, max_weight)
    init_coeffs = propagator.map_observable(observable)
    
    dummy_params = jnp.zeros(1000)
    try:
        gates = circuit_structure_fn(dummy_params)
    except Exception:
        raise ValueError("Could not inspect circuit structure.")
        
    layer_ops, params_per_layer = propagator.compile_layer_ops(gates)
    
    # Reverse ops for Heisenberg within the layer
    # And we scan layers in reverse.
    scan_step = propagator.make_scan_step(layer_ops[::-1])
    
    diag_mask = propagator.get_diagonal_mask()
    
    def loss_fn(params):
        rev_params = params[::-1]
        final_vec, _ = jax.lax.scan(scan_step, init_coeffs, rev_params)
        
        # Debug print
        # jax.debug.print("Layer Norms: {}", norms)
        
        energy = jnp.sum(jnp.where(diag_mask, final_vec, 0.0))
        return jnp.real(energy)

    return loss_fn
