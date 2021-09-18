"""
The following code was written by Harish Vallury - University of Melbourne
    -> In this work we explore the possibility of using the HQA to identify
        states from the heisenberg model
"""

from qiskit import *
from qiskit.providers.aer import StatevectorSimulator
import numpy as np
from scipy import sparse
import scipy.sparse.linalg as sp_linalg
from scipy.optimize import minimize as minimise
from multiprocessing import Pool
import pickle
import sys

# -----------------------Useful Functions----------------------------------------

PDICT = {'X': sparse.csr_matrix(np.array([[0,1],[1,0]])),
        'Y': sparse.csr_matrix(np.array([[0,-1j],[1j,0]])),
        'Z': sparse.csr_matrix(np.array([[1,0],[0,-1]])),
        'I': sparse.csr_matrix(np.eye(2))}
SB = {'I': '00', 'X': '01', 'Y': '10', 'Z': '11'}
BS = {'00': 'I', '01': 'X', '10': 'Y', '11': 'Z'}


def kronN(P):
    Pflip = P[::-1]
    output = PDICT[Pflip[0]]
    for p in Pflip[1:]:
        output = sparse.kron(PDICT[p],output,format='csr')
    return output


def H_matrix(H,L):
    output = sparse.csr_matrix((2**L,2**L), dtype=np.complex128)
    for P in H:
        output += H[P]*kronN(P)
    return output


def to_Pstring(P,L):
    b1,b2 = P
    b1s = np.binary_repr(b1,L)
    b2s = np.binary_repr(b2,L)
    Ps = ''
    for i in range(L):
        Ps += BS[f'{b1s[i]}{b2s[i]}']
    return Ps


def to_binarypair(Ps):
    P0,P1 = '',''
    for p in Ps:
        P0+=SB[p][0]
        P1+=SB[p][1]
    return (int(P0, base=2), int(P1, base=2))


# The eSWAP gate
def eswap(qc,theta,i,j):
    qc.cx(j,i)
    qc.crx(theta,i,j)
    qc.x(i)
    qc.rz(-theta/2,i)
    qc.x(i)
    qc.cx(j,i)


# The variational state
def varstate_eswap(qc, L, thetas, d=1):
    for i in range(L):
        qc.x(i)
    evens = [i for i in range(L-1) if i%2 == 0]
    # Singlet state
    for j in evens:
        qc.h(j)
        qc.cx(j,j+1)
    t = 0
    for layer in range(d):
        # eswap layer
        for j in evens[:-1]:
            eswap(qc,thetas[t]*2*np.pi,j+1,j+2)
            t += 1
        eswap(qc,thetas[t]*2*np.pi,0,L-1)
        t += 1
        for j in evens:
            eswap(qc,thetas[t]*2*np.pi,j,j+1)
            t += 1
    

# Evaluate the Hamiltonian expectation value given the trial state with parameters theta
def evalue_statevector(thetas, *args):
    L, D, H = args
    circ = QuantumCircuit(QuantumRegister(L, 'q'), ClassicalRegister(L, 'c'))
    varstate_eswap(circ,L,thetas,d=D)
    psi = execute(circ,StatevectorSimulator()).result().get_statevector(0)
    return (psi.conjugate().dot(H.dot(psi))).real/(4*L)


# Evaluate the fidelity (ground state overlap squared) given the trial state with parameters theta
def fidelity_statevector(thetas, *args):
    L, D, psi_0 = args
    circ = QuantumCircuit(QuantumRegister(L, 'q'), ClassicalRegister(L, 'c'))
    varstate_eswap(circ,L,thetas,d=D)
    psi = execute(circ,StatevectorSimulator()).result().get_statevector(0)
    return np.abs(psi.conj() @ psi_0)**2, psi


def process_set(s):
    H_expr = Hamiltonians[s]
    print(f'Generating Hamiltonian matrix ({s})...')
    H = H_matrix(H_expr,L)
    print(f'Diagonalising ({s})...')
    evals,evecs = sp_linalg.eigsh(H) 
    psi_0 = evecs[:,evals.argmin()]
    E0 = evals.min()/((4*L))
    
    D = 0
    print(f'Simulating D = 0 ({s})...')
    out = {0:{'D': 0, 'E0': E0, 'E': evalue_statevector([], L, D, H), 'F': fidelity_statevector([], L, D, psi_0), 'x': [], 'nfev': 0, 'nit': 0, 'x0': []}}
    
    for D in D_list:
        print(f'Minimising D = {D} ({s})...')
        x0 = np.random.rand(D*L)
        res = minimise(evalue_statevector, args = (L, D, H), x0 = x0, method = 'BFGS')
        thetas = list(res.x)
        F, statevec = fidelity_statevector(thetas, L, D, psi_0)
        out[D] = {'D': D, 'E0': E0, 'E': res.fun, 'F': F, 'x': thetas, 'nfev': res.nfev, 'nit': res.nit, 'x0': list(x0), 
                 'statevector':statevec}
        
    return out


def _free_filename(filename):
    indx = 1
    orig = filename
    while True:
        filename = orig.split('.pickle')[0] + '_ver{}.pickle'.format(indx)
        if not os.path.isfile(filename):
            break
        indx += 1
    return filename


def pickle_file(filename, ob):

    filename = _free_filename(filename)
    with open(filename, "wb") as f:
        pickle.dump(ob, f)


if __name__ == '__main__':

    n_qubits = int(sys.argv[1])
    num_Hamiltonians = int(sys.argv[2])
    if "o" in sys.argv[3]:
        D_list = [int(sys.argv[3][1:])]
    else:
        Dmax = int(sys.argv[3])
        D_list = list(range(1, Dmax+1))

    try:
        path_flag = True if sys.argv[4] == "True" else None
        print("Finding points along path={}.".format(path_flag))
    except:
        path_flag = False

    num_cores = 4
    
    # --------- Creating Hamiltonians --------------------------------------------------------
    L = n_qubits                            # number of qubits
    num_Hamiltonians = num_Hamiltonians     # size of random ensemble

    couplings = [(i,i+1) for i in range(L-1)]+[(L-1,0)]

    # Choosing path end points
    path_end_points = [np.random.rand(len(couplings)), 
                        np.random.rand(len(couplings))] # set to None otherwise
    

    # This generates the path if flagged
    if path_flag is not None:
        vec = (path_end_points[1] - path_end_points[0]) / num_Hamiltonians
        start = path_end_points[0]
        weights_list = [start + (vec*i) for i in range(num_Hamiltonians)]
    else:        
        weights = [np.random.rand(len(couplings)) for _ in range(num_Hamiltonians)]

    np.random.seed(1011780235)

    Hamiltonians = []
    Jlists = []
    for k in range(num_Hamiltonians):
        weights = weights_list[k]
        cwdict = {couplings[i]:weights[i] for i in range(L)}
        H_expr = {}
        for c in cwdict:
            for s in ('X','Y','Z'):
                P = 'I'*L
                P = f'{P[:c[0]]}{s}{P[c[0]+1:]}'
                P = f'{P[:c[1]]}{s}{P[c[1]+1:]}'
                H_expr[P] = cwdict[c]
        Hamiltonians.append(H_expr)
        Jlists.append(list(weights))
        
    # for i,Jlist in enumerate(Jlists):
    #     print(f"""Coupling set {i+1:0>2}: {' '.join([f"{J:.3f}" for J in Jlist])}""")

    with Pool(num_cores) as pool:
        results = pool.map(process_set, range(num_Hamiltonians))
    
    data = {}
    for i in range(num_Hamiltonians):
        data[i] = results[i]
        data[i]['Jlist'] = Jlists[i]

    pickle_file('state_files/data{}_n_qubits-{}.pickle'.format("_path" if path_flag is True else "", n_qubits), data)