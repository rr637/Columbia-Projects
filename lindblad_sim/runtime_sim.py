from qiskit import *
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
import time 
import gc
from qiskit import transpile


def get_xxz_chain_hamiltonian_dict(n_qubits: int, Delta_ZZ: float) -> dict:
    """
    Construct a dictionary representing the XXZ chain Hamiltonian.
    
    For a chain of n_qubits, we sum over nearest-neighbor (edge) interactions.
    Each key in the dictionary is a tuple of length n_qubits where each element
    is 'I', 'X', 'Y', or 'Z'. The value is the corresponding weight for that term.

    The Hamiltonian on a pair (i,j) is given by:
      H_ij =  [X_i X_j + Y_i Y_j] +(Δ_ZZ)Z_i Z_j]

    Parameters:
      n_qubits: Number of qubits (sites).
      Delta_ZZ: The anisotropy coefficient Δ_ZZ for the Z_i Z_j term.
      t: A scaling parameter for the interaction strength.
    
    Returns:
      A dictionary with keys as tuples representing Pauli strings (indexed from qubit 0 to n_qubits-1)
      and values as the cumulative coefficients for that term.
    """
    # Create a path graph for the 1D chain
    sites = nx.path_graph(n_qubits)
    
    # Coefficients for the interaction terms
    coeff_xxyy = 1
    coeff_zz   = Delta_ZZ 
    
    # Dictionary to hold the Pauli term and its coefficient
    hamiltonian_dict = {}

    for edge in sites.edges:
        i, j = edge
        
        # For each Pauli type, we create a list that represents an n_qubit operator.
        for op, coeff in [('X', coeff_xxyy), ('Y', coeff_xxyy), ('Z', coeff_zz)]:
            # Start with the identity on all qubits.
            pauli_list = ['I'] * n_qubits
            # Place the Pauli operator on qubit i and qubit j.
            pauli_list[i] = op
            pauli_list[j] = op
            # Convert the list to a tuple so it can be used as a dictionary key.
            key = ''.join(pauli_list)
            
            # Sum the coefficient if the same Pauli string appears from overlapping edges.
            if key in hamiltonian_dict:
                hamiltonian_dict[key] += coeff
            else:
                hamiltonian_dict[key] = coeff

    return hamiltonian_dict
print(get_xxz_chain_hamiltonian_dict(3,0.5))
def get_transverse_Ising_hamiltonian_dict(n_qubits: int, g, J) -> dict:
    """
    The Hamiltonian on pair (i,j) is
    -J(Z_i Z_j) + g(X_i + X_j)
    """
    # Create a path graph for the 1D chain
    sites = nx.path_graph(n_qubits)
    
    # Coefficients for the interaction terms
    coeff_x = g
    coeff_zz   = -J
    
    # Dictionary to hold the Pauli term and its coefficient
    hamiltonian_dict = {}

    for edge in sites.edges:
        i, j = edge
        
        op, coeff = ('Z', coeff_zz)
        pauli_list = ['I'] * n_qubits
        # Place the Pauli operator on qubit i and qubit j.
        pauli_list[i] = op
        pauli_list[j] = op
        # Convert the list to a tuple so it can be used as a dictionary key.
        key = ''.join(pauli_list)
        
        # Sum the coefficient if the same Pauli string appears from overlapping edges.
        if key in hamiltonian_dict:
            hamiltonian_dict[key] += coeff
        else:
            hamiltonian_dict[key] = coeff
    for i in range(n_qubits):
        op, coeff = ('X',coeff_x)
        pauli_list = ['I'] * n_qubits
        pauli_list[i] = op
        key = ''.join(pauli_list)
        
        # Sum the coefficient if the same Pauli string appears from overlapping edges.
        if key in hamiltonian_dict:
            hamiltonian_dict[key] += coeff
        else:
            hamiltonian_dict[key] = coeff

    return hamiltonian_dict


def exp_Z(qc, qr, pauli_idx, t=1) -> None:
    """
    Implementation of exp(iZZ...Zt) on qubits in pauli_idx, where t is the parameter.
    Adds to input circuit.
    :param qc: Quantumqc.
    :param qr: QuantumRegister.
    :param pauli_idx: the indexes from qr that correspond to entries not equal to I:
                      e.g. if we have XXII then pauli_idx = [0, 1].
    :param t: the parameter t in exp(iZZ...Z t).
    """
    # the first CNOTs
    for i in range(len(pauli_idx) - 1):
        qc.cx(qr[pauli_idx[i]], qr[pauli_idx[i + 1]])

    # Rz gate
    qc.rz(-2 * t, qr[pauli_idx[-1]])

    # the second CNOTs
    for i in reversed(range(len(pauli_idx) - 1)):
        qc.cx(qr[pauli_idx[i]], qr[pauli_idx[i + 1]])
  
def exp_pauli(pauli, qr, t=1) -> QuantumCircuit:
    """
    The qc for the exp(i P t), where P is the Pauli term, 
    t is the parameter.
    :param pauli: the string for the Pauli term: e.g. "XIXY".
    :param qr: QuantumRegister.
    :param t: the parameter t in exp(i P t).
    :return: Quantum Circuit that implements exp(i P t) or 
             control version of it.
    """
    
    pauli_qc = QuantumCircuit(qr)
    pre_post_qc = QuantumCircuit(qr)
    pauli_idexes = []

    for i in range(len(qr)):
        if pauli[i] == 'I':
            continue
        elif pauli[i] == 'Z':
            pauli_idexes.append(i)
        elif pauli[i] == 'X':
            pre_post_qc.h(qr[i])
            pauli_idexes.append(i)
        elif pauli[i] == 'Y':
            pre_post_qc.u(np.pi/2,np.pi / 2, np.pi / 2, qr[i])
            pauli_idexes.append(i)

    pauli_qc = pauli_qc.compose(pre_post_qc)
    exp_Z(pauli_qc, qr, pauli_idexes, t)
    pauli_qc = pauli_qc.compose(pre_post_qc)

    return pauli_qc 
def hamiltonian_simulation_FO_trotter(hamiltonian, qr, N, t=1):
    """
    The implementation of exp((-)iHt) (First Order), where H is the Hamiltonian 
    operator, t is the parameter.
    :param hamiltonian: dictionary of Pauli terms with their weights
    :param qr: QuantumRegister.
    :param t: the parameter t in exp(iHt).
    :param N: the Trotter number.
    :return: Quantum Ciruit that corresponds to exp(iHt) 
             or control version of it.
    """
    # print(hamiltonian)
    delta_t = t / N
    exp_hamiltonian = QuantumCircuit(qr)
    exp_delta_t = QuantumCircuit(qr)

    for pauli in hamiltonian:
        weight = hamiltonian[pauli]
        exp_delta_t = exp_delta_t.compose(exp_pauli(pauli, qr, -weight * delta_t))

    for i in range(N):
        exp_hamiltonian = exp_hamiltonian.compose(exp_delta_t)

    return exp_hamiltonian

def hamiltonian_simulation_SO_trotter(hamiltonian, qr, N, t=1):
    """
    The implementation of exp((-)iHt) (Second Order), where H is the Hamiltonian 
    operator, t is the parameter.
    :param hamiltonian: dictionary of Pauli terms with their weights: 
                        e.g. {"XZX": 2, "ZYI": 5, "IYZ": 7}.
    :param qr: QuantumRegister.
    :param control_qubit: the control Qubit from QuantumRegister 
                          other than qr.
    :param t: the parameter t in exp(iHt).
    :param N: the Trotter number.
    :return: QuantumCircuit that corresponds to exp(iHt) 
             or control version of it.
    """
    
    delta_t = t / N
    exp_hamiltonian = QuantumCircuit(qr)
    exp_delta_t = QuantumCircuit(qr)

    for pauli in hamiltonian:
        weight = hamiltonian[pauli]
        exp_delta_t = exp_delta_t.compose(exp_pauli(pauli, qr, -weight * delta_t/2))
    for pauli in reversed(list(hamiltonian)):
        weight = hamiltonian[pauli]
        exp_delta_t = exp_delta_t.compose(exp_pauli(pauli, qr, -weight * delta_t/2))

    for i in range(N):
        exp_hamiltonian =exp_hamiltonian.compose(exp_delta_t)

    return exp_hamiltonian


def _trotter_step(hamiltonian, qr, h: float, order: int, p: int = 2) -> QuantumCircuit:
    """
    Build a single Suzuki–Trotter step S_order(h) of even order 'order'.
    Recurses down to order=2 (Strang splitting).
    https://arxiv.org/pdf/2211.02691 3.1.1 'Suzuki's method'

    """
    if order == 2:
        # Strang / second order:
        circ = QuantumCircuit(qr)
        # half‑step forward
        for pauli, weight in hamiltonian.items():
            circ = circ.compose(exp_pauli(pauli, qr, -weight * h/2))
        # half‑step backward
        for pauli, weight in reversed(list(hamiltonian.items())):
            circ = circ.compose(exp_pauli(pauli, qr, -weight * h/2))
        return circ

    # otherwise build S_{order} from S_{order-2}
    n = order - 2
    # Suzuki “p=2” coefficient:
    s = 1.0 / (2*p - (2*p)**(1.0/(n+1)))
    # recursive pieces
    S_small = _trotter_step(hamiltonian, qr, s*h,     order-2, p)
    S_mid   = _trotter_step(hamiltonian, qr, (1-2*p*s)*h, order-2, p)

    # assemble:  (S_small)^p · S_mid · (S_small)^p
    circ = QuantumCircuit(qr)
    for _ in range(p):
        circ = circ.compose(S_small)
    circ = circ.compose(S_mid)
    for _ in range(p):
        circ = circ.compose(S_small)
    return circ

def hamiltonian_simulation_nth_order(hamiltonian, qr, N: int, t: float = 1.0, order: int = 2):
    """
    Full simulation exp(-i H t) via r=N trotter steps of size h=t/N,
    using a 2nd, 4th, 6th, … order Suzuki–Trotter decomposition.
    :param hamiltonian: dict mapping Pauli‑strings to real weights.
    :param qr:           your QuantumRegister (or list of qubits).
    :param N:            number of trotter repeats.
    :param t:            total time.
    :param order:        even integer ≥2, the global order of the scheme.
    """
    if order < 2 or order % 2 != 0:
        raise ValueError(f"`order` must be an even integer ≥ 2 (got order:{order}")
    h = t / N
    # build one step S_order(h)
    step = _trotter_step(hamiltonian, qr, h, order)

    # repeat it N times
    circ = QuantumCircuit(qr)
    for _ in range(N):
        circ = circ.compose(step)
    return circ

def get_ising_circ_FO(n: int, J: float, g: float, t: float, N: int):
  ham_ising = get_transverse_Ising_hamiltonian_dict(n,g,J)
  qr_ising = QuantumRegister(n)
  qc_ising = hamiltonian_simulation_FO_trotter(ham_ising,qr_ising,N,t)
  return qc_ising
def get_ising_circ_SO(n: int, J: float, g: float, t: float, N: int):
  ham_ising = get_transverse_Ising_hamiltonian_dict(n,g,J)
  qr_ising = QuantumRegister(n)
  qc_ising = hamiltonian_simulation_SO_trotter(ham_ising,qr_ising,N,t)
  return qc_ising

def get_ising_circ_nth_order(n,J,g,t,N,order):
  ham_ising = get_transverse_Ising_hamiltonian_dict(n,g,J)
  qr_ising = QuantumRegister(n)
  qc_ising = hamiltonian_simulation_nth_order(ham_ising,qr_ising,N,t,order)
  return qc_ising
def get_xxz_circ_FO(n: int, Delta_ZZ:float, t: float, N: int):
  ham_xxz = get_xxz_chain_hamiltonian_dict(n,Delta_ZZ)
  qr_xxz = QuantumRegister(n)
  qc_xxz = hamiltonian_simulation_FO_trotter(ham_xxz,qr_xxz,N,t)
  return qc_xxz
n = 5
Delta_ZZ = 0.5
g = 1
J = 1.2
t=1
N=100

import numpy as np
### QLIB IS A LIBRARY FOR QUANTUM DYNAMICS SIMULATION https://github.com/ishaanganti/qlib ###
from qlib.operators import * 
from qlib import oper, qprint
from qlib.utils import tensor
from qlib.states import state
from qlib import ket
from scipy.linalg import expm

def time_evolve_operator(H: oper, t: float):
    '''
    Returns the time evolution operator given a TIME-INDEPENDENT
    Hamiltonian H and a time t. The output is an 'oper' object
    per qlib classes.  
    '''
    H = H.to_matrix()
    U = expm(-1j * H * t)
    return oper(U)

def get_xxz_hamiltonian(n_qubits: int, delta: float) -> oper:
    '''
    Creates the Heisenberg XXZ Hamiltonian using qlib. 
    '''
    H_xxz = zeros(2**n_qubits)
    for k in range(n_qubits - 1):
        left_ids = identity(2**k)
        right_ids = identity(2**(n_qubits - k - 2))
        H_xxz += tensor(left_ids, pauli_x(), pauli_x(), right_ids)
        H_xxz += tensor(left_ids, pauli_y(), pauli_y(), right_ids)
        H_xxz += delta * tensor(left_ids, pauli_z(), pauli_z(), right_ids)
    return H_xxz

def get_ising_hamiltonian(n_qubits: int, J: float, g: float):
    '''
    Creates the Ising Hamiltonian using qlib. 
    '''
    H_ising = zeros(2**n_qubits)
    for k in range(n_qubits - 1):
        left_ids = identity(2**k)
        right_ids = identity(2**(n_qubits-k-2))
        H_ising += -J * tensor(left_ids, pauli_z(), pauli_z(), right_ids)
    for k in range(n_qubits):
        left_ids = identity(2**k)
        right_ids = identity(2**(n_qubits-k-1))
        H_ising += g * tensor(left_ids, pauli_x(), right_ids)
    return H_ising

psi0 = state(2**n, 0) 

# hamiltonians
H_xxz = get_xxz_hamiltonian(n, Delta_ZZ)
# print(Delta_ZZ)
H_ising = get_ising_hamiltonian(n, J, g)

# time evolved XXZ state
true_XXZ_state = time_evolve_operator(H=H_xxz, t=t) * psi0
# time evolved ising state
true_ising_state  = time_evolve_operator(H=H_ising, t=t) * psi0
# time evolution operator for Heisenberg XXZ model. 
U_xxz = time_evolve_operator(H_xxz, t)
# print(true_ising_state.vector)
# print(true_XXZ_state.vector)



# points = 20

# factor_map = {
#     1: 1000**(1/(points-1)),  
#     2:  800**(1/(points-1)),   
#     4:  200**(1/(points-1)),  
#     6:  100**(1/(points-1)),   
#     8:   50**(1/(points-1)),   
# }




orders      = [1, 2, 4, 6, 8]
# max_runtime = 4      # seconds
n_runs      = 3      # how many repeats to average

# Ns_list = [
#     [1, 2, 3, 5, 8, 12, 18, 26, 38, 50, 80, 116, 167, 241, 347, 504, 720, 1036],
#     [1, 2, 3, 5, 10, 12, 18, 29, 37, 53, 76, 92, 155, 221, 315, 448],
#     [1, 2, 3, 4, 6, 8, 11, 15, 19, 27, 33, 48, 58, 85, 102, 150],
#     [1, 2, 3, 4, 6, 8, 12, 15, 20, 26, 34, 44],
#     [1, 2, 3, 4, 5, 7, 9]
# ]

# containers for all orders:
# Ns_all    = []
# means_all = []
# stds_all  = []

# for order in orders:
#     Ns    = []
#     means = []
#     stds  = []
#     N = 1
#     factor = factor_map[order]
#     print(f"\n→ Processing order {order}-order")
#     while True:
#         # inner‐most loop: run the same (order,N) n_runs times
#         runtimes = []
#         for _ in range(n_runs):
#             gc.disable()

#             start = time.perf_counter()
            
#             # build & simulate
#             if order == 1:
#                 qc = get_ising_circ_FO(n, J, g, t, N)
#             else:
#                 qc = get_ising_circ_nth_order(n, J, g, t, N, order)
#             _ = Statevector(qc)
            
#             runtimes.append(time.perf_counter() - start)
#             gc.disable()

#         mean = np.mean(runtimes)
#         sd = np.std(runtimes)
#         print(f"  N={N:3d} → mean={mean:.3f}s, std={sd:.3f}s")
        
#         # stop as soon as the *mean* exceeds threshold
#         if mean > max_runtime:
#             break
        
#         # record results
#         Ns.append(N)
#         means.append(mean)
#         stds.append(sd)
        
#         # increment N (you can tweak these schedules as you like)
#         next_N = int(np.ceil(N * factor))
#         if next_N == N:
#             next_N = N + 1
#         N = next_N

#     # stash per-order arrays
#     Ns_all.append(np.array(Ns))
#     means_all.append(np.array(means))
#     stds_all.append(np.array(stds))



# Ns_list = [
#     [1, 2, 3, 5, 8, 12, 18, 26, 38, 50, 80, 116, 167, 241, 347, 504, 720, 1036],
#     [1, 2, 3, 5, 10, 12, 18, 29, 37, 53, 76, 92, 155, 221, 315, 448],
#     [1, 2, 3, 4, 6, 8, 11, 15, 19, 27, 33, 48, 58, 85, 102, 150],
#     [1, 2, 3, 4, 6, 8, 12, 15, 20, 26, 34, 44],
#     [1, 2, 3, 4, 5, 7, 9]
# ]

# means_all = []
# stds_all  = []
# for i in range(len(orders)):
#     means = []
#     stds  = []
    
    
#     print(f"\n→ Processing order {orders[i]}-order")
#     for N in Ns_list[i]:
#         # inner‐most loop: run the same (order,N) n_runs times
#         runtimes = []
#         for _ in range(n_runs):
#             gc.disable()

#             start = time.perf_counter()
            
#             # build & simulate
#             if orders[i] == 1:
#                 qc = get_ising_circ_FO(n, J, g, t, N)
#             else:
#                 qc = get_ising_circ_nth_order(n, J, g, t, N, orders[i])
#             _ = Statevector(qc)
            
#             runtimes.append(time.perf_counter() - start)
#             gc.disable()

#         mean = np.mean(runtimes)
#         sd = np.std(runtimes)
#         print(f"  N={N:3d} → mean={mean:.3f}s, std={sd:.3f}s")

#         means.append(mean)
#         stds.append(sd)
#     means_all.append(np.array(means))
#     stds_all.append(np.array(stds))


# ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# # --- Save just the (N, mean, std) for each order into one .npz --- #
# np.savez_compressed(
#     f"trotter_runtime_results-{ts}.npz",
#     orders = np.array(orders),
#     Ns     = np.array(Ns_list,    dtype=object),
#     means  = np.array(means_all, dtype=object),
#     stds   = np.array(stds_all,  dtype=object),
#     n_runs=n_runs
# )

# print("Saved summary to trotter_summary.npz")


def count_cnots(qc):
    # first make sure it’s in the standard basis
    basis_qc = transpile(qc, basis_gates=["u1","u2","u3","cx"], optimization_level=0)
    counts   = basis_qc.count_ops()
    return counts.get("cx", 0)
Ns_list = [
    [1, 2, 3, 5, 8, 12, 18, 26, 38, 50, 80, 116, 167, 241, 347, 504, 720, 1036],
    [1, 2, 3, 5, 10, 12, 18, 29, 37, 53, 76, 92, 155, 221, 315, 448],
    [1, 2, 3, 4, 6, 8, 11, 15, 19, 27, 33, 48, 58, 85, 102, 150],
    [1, 2, 3, 4, 6, 8, 12, 15, 20, 26, 34, 44],
    [1, 2, 3, 4, 5, 7, 9]
]

cx_count_all = []

for i in range(len(orders)):
    means = []
    stds  = []
    cx_counts = []
    
    print(f"\n→ Processing order {orders[i]}-order")
    for N in Ns_list[i]:
        # inner‐most loop: run the same (order,N) n_runs times
       
       
    

     
        
        # build & simulate
        if orders[i] == 1:
            qc = get_ising_circ_FO(n, J, g, t, N)
        else:
            qc = get_ising_circ_nth_order(n, J, g, t, N, orders[i])
      ``
        cx_count = count_cnots(qc)
        cx_counts.append(cx_count)
        print(f"  N={N:3d} → mean={cx_count:.3f}s")

    cx_count_all.append(cx_count_all)



ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- Save just the (N, mean, std) for each order into one .npz --- #
np.savez_compressed(
    f"trotter_cx_count_results-{ts}.npz",
    orders = np.array(orders),
    Ns     = np.array(Ns_list,    dtype=object),
    cx_counts  = np.array(cx_counts), dtype=object)

print("Saved summary to trotter_summary.npz")