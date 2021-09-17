
# Plotting
import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
from collections import defaultdict
import random
from scipy.stats import norm
from scipy.sparse import coo_matrix

# Pennylane
import pennylane as qml
from pennylane import numpy as np
from pennylane_cirq import ops as cirq_ops
import pennylane_qiskit

# Other tools
import time
import sys
import pickle
from multiprocessing import Pool
import os
import glob
import copy
import math
import pandas as pd

from qiskit import IBMQ
from qiskit.providers.aer.noise import NoiseModel

PAULIS = ['x', 'y', 'z']


n_qubits = 5


try:
    n_qubits = int(sys.argv[1])
    max_q_depth = int(sys.argv[2])
except:
    print("Taking {} qubits".format(n_qubits))

try:
    inp = sys.argv[5]
    if inp == 'noisy':
        NOISY = True
        dev = qml.device("cirq.mixedsimulator", wires=n_qubits)
    else:
        raise NameError('No such device')
except (qml._device.DeviceError, NameError, IndexError):
    print('No such device.')
    NOISY = False
    dev = qml.device("default.qubit", wires=n_qubits)
    dev_type = None


dev_type = 'default'
if __name__ == '__main__':

    dev_type = 'ibmq_sim'
    provider = IBMQ.load_account()
    backend = provider.get_backend('ibmq_vigo')
    noise_model = None
    noise_model = NoiseModel.from_backend(backend)


# backend = eval("provider.backends.{}".format('ibmq_ourense'))
# dev = qml.device("qiskit.ibmq", wires=n_qubits, backend='ibmq_qasm_simulator', provider=provider)
# dev = qml.device('qiskit.aer', wires=n_qubits)

print("Availability of gpu: {}".format(bool(torch.cuda.is_available())))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


PHASE_FLIP_ERROR = 0.02
BIT_FLIP_ERROR = 0.02


# ########################## Quantum Circuit ###############################################


def S_dag(wire):
    S_dag = np.array([[1, 0],[0, -1j]])
    qml.Hermitian(S_dag, wires=wire)


def H_layer(nqubits, wires_range=None):
    """Layer of single-qubit Hadamard gates.
    """
    if wires_range:
        ran = wires_range
    else:
        ran = [0, nqubits]

    for idx in range(*ran):
        qml.Hadamard(wires=idx)


def RY_layer(w, ancilla=False):
    """Layer of parametrized qubit rotations around the y axis.
    """
    anc = 0
    if ancilla:
        anc = 1
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx+anc)


def RZ_layer(w, ancilla=False):
    """Layer of parametrized qubit rotations around the Z axis.
    """
    anc = 0
    if ancilla:
        anc = 1
    for idx, element in enumerate(w):
        qml.RZ(element, wires=idx+anc)


def U3_layer(w):

    assert (len(w) / 3) == n_qubits
    i = 0
    n_q = 0
    while i < len(w):
        qml.U3(w[i], w[i+1], w[i+2], wires=n_q)
        n_q += 1
        i += 3


def entangling_layer(nqubits, ancilla=False, limit=None):
    """Layer of CNOTs followed by another shifted layer of CNOT.
    """
    # In other words it should apply something like :
    # CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT
    if limit != None:
        nqubits = limit
    init = 0
    if ancilla:
        init = 1
    for i in range(init, nqubits - 1 + init, 2):  # Loop over even indices: i=0,2,...N-2
        qml.CNOT(wires=[i, i + 1])
    for i in range(init + 1, nqubits - 1 + init, 2):  # Loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])


def vqe_gate(theta, wire1, wire2, pauli1, pauli2):
    if pauli1 == 'x':
        qml.Hadamard(wires=wire1)
    elif pauli1 == 'y':
        S_dag(wire1)
        qml.Hadamard(wires=wire1)

    if pauli2 == 'x':
        qml.Hadamard(wires=wire2)
    if pauli2 == 'y':
        S_dag(wire1)
        qml.Hadamard(wires=wire1)

    qml.CNOT(wires=[wire1, wire2])
    qml.RZ(theta, wires=wire2)
    qml.CNOT(wires=[wire1, wire2])

    if pauli2 == 'x':
        qml.Hadamard(wires=wire2)
    if pauli2 == 'y':
        qml.Hadamard(wires=wire1)
        S_dag(wire1)

    if pauli1 == 'x':
        qml.Hadamard(wires=wire1)
    elif pauli1 == 'y':
        qml.Hadamard(wires=wire1)
        S_dag(wire1)


def proj_meas_gen(n_qubits):
    size = 2**n_qubits
    for i in range(size):
        data = np.array([1], dtype=float)
        row = np.array([i])
        col = np.array([i])
        yield coo_matrix((data, (row, col)), shape=(size, size)).toarray()


###################################################################################


def reset_error(bitflip=0.02, phaseflip=0.02):

    global PHASE_FLIP_ERROR
    global BIT_FLIP_ERROR

    PHASE_FLIP_ERROR = phaseflip
    BIT_FLIP_ERROR = bitflip


def reset_q_num(n_qubits, dev_type='default', noisy=False, new_dev=None):
    global NOISY
    NOISY = noisy

    global dev

    if dev_type=='ibmq_sim':
        # dev = qml.device("qiskit.ibmq", wires=n_qubits, backend='ibmq_qasm_simulator', provider=provider)
        if noise_model:
            # dev = qml.device('qiskit.aer', wires=n_qubits, noise_model=noise_model)
            dev = pennylane_qiskit.aer.AerDevice(wires=n_qubits, shots=2048, backend='qasm_simulator', noise_model=noise_model)
        else:
            # dev = qml.device('qiskit.aer', wires=n_qubits)
            dev = pennylane_qiskit.aer.AerDevice(wires=n_qubits, shots=1024, backend='qasm_simulator')
    elif noisy:
        dev = qml.device("cirq.mixedsimulator", wires=n_qubits)
    elif dev_type == 'default':
        dev = qml.device("default.qubit", wires=n_qubits)

    if new_dev:
        dev = new_dev

    global q_net_ry_entan

    @qml.qnode(dev, interface="torch")
    def q_net_ry_entan(q_weights_flat, q_depth=None, op=None, n_qubits=None, limit=None):

        """ Where limit=(a,b) and 'a' is last q_depth to cut out and 'b' is how many qubits the first learning
            had. """

        # Reshape weights
        q_weights = q_weights_flat.reshape(q_depth, n_qubits)
        H_layer(n_qubits)
        if limit == None:
            limit = (-1, None)
        # Sequence of trainable variational layers
        for k in range(q_depth):
            if NOISY:
                for i in range(n_qubits):
                    cirq_ops.PhaseFlip(PHASE_FLIP_ERROR, wires=i)
                    cirq_ops.BitFlip(BIT_FLIP_ERROR, wires=i)
            if k < limit[0]:
                entangling_layer(n_qubits, limit=limit[1])
            else:
                entangling_layer(n_qubits, None)
            RY_layer(q_weights[k])

        """"# Uncomment for original
        exp_vals = qml.expval(qml.Hermitian(op, wires=list(range(n_qubits))))
        
        return exp_vals"""
        return qml.probs(wires=list(range(n_qubits)))

    global q_net_vqe

    @qml.qnode(dev, interface="torch")
    def q_net_vqe(q_weights, paulis=None, couplings=None, op=None):

        assert len(q_weights) == len(paulis) and len(paulis) == len(couplings)

        H_layer(n_qubits)

        for indx, pauli in enumerate(paulis):
            vqe_gate(q_weights[indx], couplings[indx][0], couplings[indx][1], pauli[0], pauli[1])

        """"# Uncomment for original
        exp_vals = qml.expval(qml.Hermitian(op, wires=list(range(n_qubits))))

        return exp_vals"""
        return qml.probs(wires=list(range(n_qubits)))

    global q_net_own

    @qml.qnode(dev, interface="torch")
    def q_net_own(q_weights, n_qubits=None, op=None):
        """ Note that we need at least 3 times the number of parameters as qubits """
        H_layer(n_qubits)
        RY_layer(q_weights[:n_qubits])
        if NOISY:
            for i in range(n_qubits):
                cirq_ops.PhaseFlip(PHASE_FLIP_ERROR, wires=i)
                cirq_ops.BitFlip(BIT_FLIP_ERROR, wires=i)
        n_q = 0
        i = 0
        n_weights = q_weights[n_qubits:]
        while i + 2 < len(n_weights):
            try:
                if n_q+1 >= n_qubits:
                    n_q = 0
                    continue
                qml.CNOT(wires=[n_q, n_q + 1])
                qml.U3(n_weights[i], n_weights[i + 1], n_weights[i + 2], wires=n_q)
                if i+5 >= len(n_weights):
                    break
                qml.U3(n_weights[i + 3], n_weights[i + 4], n_weights[i + 5], wires=n_q+1)
                if NOISY:
                    for jj in [0, 1]:
                        cirq_ops.PhaseFlip(PHASE_FLIP_ERROR, wires=n_q+jj)
                        cirq_ops.BitFlip(BIT_FLIP_ERROR, wires=n_q+jj)
            except IndexError:
                print('Warning OWN circuit parameter mismatch may have occured.')
                break
            i += 6
            n_q += 1

        """"# Uncomment for original
        exp_vals = qml.expval(qml.Hermitian(op, wires=list(range(n_qubits))))

        return exp_vals"""
        return qml.probs(wires=list(range(n_qubits)))


# This is actually setting up the gates

reset_q_num(n_qubits, dev_type=dev_type, noisy=NOISY)

# ########################## Model ###############################################


class DistributionEncoder(nn.Module):

    name = 'DistributionEncoder'

    def __init__(self, N, n_qubits, num_params, gate_type='ry', params=None):

        super().__init__()

        self.N = N
        self.n_qubits = n_qubits
        self.num_params = num_params
        self.gate_type = gate_type

        self.q_delta = 0.01

        if gate_type == 'ry':
            assert num_params % self.n_qubits == 0
            q_depth = num_params/self.n_qubits
            self.q_depth = int(q_depth)
            self.qcircuit = 'q_net_ry_entan'
        elif gate_type == 'vqe':
            self.couplings = staircase_coupling(num_params, self.n_qubits)
            self.type_coupling = 'staircase_coupling'
            self.paulis = fixed_paulis(num_params)
            self.qcircuit = 'q_net_vqe'

        elif gate_type == 'own':
            self.qcircuit = 'q_net_own'

        if isinstance(params, np.ndarray):
            self.params = nn.Parameter(torch.tensor(params))
        elif params == None:
            self.params = nn.Parameter(self.q_delta * torch.randn(num_params))
        else:
            self.params = nn.Parameter(torch.tensor(params))

        self.distr_info = None
        self.distr = None
        self.amplitude_distr = False
        self.limit = None
        self.train_dev = None

    def forward(self):
        if not hasattr(self, 'limit'):
            limit = None
        else:
            limit = self.limit

        obs_gen = proj_meas_gen(self.n_qubits)
        if self.gate_type == 'ry':
            op = next(obs_gen)
            q_out = q_net_ry_entan(self.params, q_depth=self.q_depth, op=op, n_qubits=self.n_qubits,
                                   limit=limit).unsqueeze(0)
            for op in obs_gen:
                break # get rid of this break for the original
                q_out_elem = q_net_ry_entan(self.params, q_depth=self.q_depth, op=op, n_qubits=self.n_qubits).unsqueeze(0)
                q_out = torch.cat((q_out, q_out_elem))

        elif self.gate_type == 'vqe':
            op = next(obs_gen)
            q_out = q_net_vqe(self.params, paulis=self.paulis, couplings=self.couplings, op=op).unsqueeze(0)
            for op in obs_gen:
                break  # get rid of this break for the original
                q_out_elem = q_net_vqe(self.params, paulis=self.paulis, couplings=self.couplings, op=op).unsqueeze(0)
                q_out = torch.cat((q_out, q_out_elem))

        elif self.gate_type == 'own':
            op = next(obs_gen)
            q_out = q_net_own(self.params, n_qubits=self.n_qubits, op=op).unsqueeze(0)
            for op in obs_gen:
                break  # get rid of this break for the original
                q_out_elem = q_net_own(self.params, n_qubits=self.n_qubits, op=op).unsqueeze(0)
                q_out = torch.cat((q_out, q_out_elem))
        return q_out

    def draw_circ(self):
        self()
        if self.gate_type == 'vqe':
            print(q_net_vqe.draw())
        elif self.gate_type == 'ry':
            print(q_net_ry_entan.draw())
        else:
            print(q_net_own.draw())

# ########################## Training functions ###############################################


def train_model(model, criterion, distr, num_iterations, loss_evol=None, step=0.001, amplitude_distr=False,
                stochastic_term=False, live_plot=False, save_per_iter=False):

    since = time.time()
    model.amplitude_distr = amplitude_distr
    optimizer = optim.Adam(model.parameters(), lr=step)

    if not amplitude_distr:
        distr = normalise_prob(distr)

    print("Training started using device: {}".format(dev))

    if not loss_evol:
        loss_evol = []

    if live_plot:
        fig, ax = plt.subplots()

    for ii in range(num_iterations):
        print('-----iteration {}--model:{}------'.format(ii, model))
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            output = model()
            if amplitude_distr:
                output = torch.sqrt(output)
            loss = criterion(output, distr.double())
            print("Loss", loss)
            loss_evol.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()
            if stochastic_term and len(loss_evol) > 11 and len(loss_evol)<num_iterations*0.6 and np.abs(loss_evol[-10]-loss_evol[-1])/loss_evol[0]<0.0001:
                print('Stochastic bump.')
                model.params = nn.Parameter(torch.tensor(model.params.detach().numpy() + stoch_unit_vector(model.num_params)*10/model.num_params))
                optimizer = optim.Adam(model.parameters(), lr=step)

            if live_plot:
                plt.plot(loss_evol)
                plt.show(block=False)
                plt.pause(0.1)

            if save_per_iter:
                save_model_and_loss(model, loss_evol, criterion=criterion, recreate=True)

    model.distr_info = distr_info(distr)
    model.distr = distr
    model.train_time = time.time() - since
    model.train_dev = dev
    print("Time taken to train model: {}".format(time.time() - since))

    return model, loss_evol


# ########################## Transferring functions ###############################################


def transfer_learning(model, new_qubit_size, add_params=0):
    """
    This function is transfering the learning that has been done for lower numbers of qubits to then train larger
    variational circuits, hopefully faster

    :param model: the old model which is a DistributionEncoder object
    :param new_qubit_size: the number of qubits in the new model
    :param num_ad_param: number of additional parameters that we allow our new model to have
    :return: a new variational circuit over new_qubit_size qubits which is a DistributionEncoder object
    """

    N = 2**new_qubit_size
    num_params = model.num_params + model.q_depth*(new_qubit_size - model.n_qubits) + add_params
    q_depth = int((num_params//new_qubit_size))
    gate_type = model.gate_type
    new_model = DistributionEncoder(N, new_qubit_size, num_params=q_depth*new_qubit_size,
                                    gate_type=gate_type)
    new_model.limit = (model.q_depth, model.n_qubits)
    transfer_parameters(model, new_model)
    transfer_distribution(model, new_model)

    return new_model


def transfer_distribution(model, new_model):

    distr_info_temp = model.distr_info

    mean = distr_info_temp[0]*(2**(new_model.n_qubits - model.n_qubits))
    std = distr_info_temp[1]*(2**(new_model.n_qubits - model.n_qubits))

    new_model.distr = normalise(gaussian_distr(2**new_model.n_qubits, std_dev=std, mean=mean))
    new_model.distr_info = distr_info(new_model.distr)


def transfer_parameters(model, new_model):

    assert model.gate_type == 'ry', "Only coded for RY architecture"

    if model.gate_type == 'ry':
        params = copy.copy(model.params)
        params.requires_grad = False
        params = list(params)
        diff_q = new_model.n_qubits - model.n_qubits
        new_params = []
        for q in range(model.q_depth):
            new_params = new_params + params[q*model.n_qubits:(q+1)*model.n_qubits]
            new_params = new_params + [torch.tensor(np.pi*0.5)]*diff_q

        extra_depth = int((new_model.num_params - len(new_params)) // new_model.n_qubits)
        print('{} extra layers added.'.format(extra_depth))
        new_params = new_params + list(np.random.rand(extra_depth*new_model.n_qubits)*new_model.q_delta)
    new_model.params = nn.Parameter(torch.tensor(new_params))


#############################################################################################

def stoch_unit_vector(dim):
    return normalise(np.random.rand(dim))


def distr_info(d):

    return [get_mean(d), get_std_dev(d)]


def normalise_prob(a):

    if len(a.shape) > 1:
        t = torch.Tensor()
        for elem in a:
            t = torch.cat((t,normalise(elem)))
        return t
    else:
        return a/torch.sum(a)


def normalise(a):

    if len(a.shape) > 1:
        t = torch.Tensor()
        for elem in a:
            t = torch.cat((t, normalise(elem)))
        return t
    else:
        if isinstance(a, np.ndarray):
            return a/np.sqrt(np.dot(a, a))

        return a/torch.sqrt(torch.dot(a, a))


def random_couplings(num_params, n_qubits):
    couplings = []
    while len(couplings) < num_params:
        c = [random.randint(0, n_qubits-1), random.randint(0, n_qubits-1)]
        if c[0] != c[1]:
            couplings.append(c)
    return couplings


def staircase_coupling(num_params, n_qubits):
    couplings = []
    i = 0
    while len(couplings) < num_params:
        if i < n_qubits - 1:
            couplings.append([i, i + 1])
            i += 1
        else:
            i = 0
    return couplings


def get_mean(v):
    N = len(v)
    x = torch.tensor(range(N))
    m = (x*v).sum().float() / v.sum()
    return m - (N/2)


def get_std_dev(v):
    N = len(v)
    x = torch.tensor(range(N))
    var = ((x*x*v).sum() / v.sum()) - ((x*v).sum() / v.sum())**2
    return var.float()**0.5


def random_paulis(num_params):
    return [PAULIS[random.randint(0, 2)] + PAULIS[random.randint(0, 2)] for _ in range(num_params)]


def fixed_paulis(num_params):
    return ['xz' for _ in range(num_params)]


def gaussian_distr(length, std_dev=1, mean=0, range_distr=None):
    if not range_distr:
        range_distr = [-length / 2, length / 2]

    x = np.linspace(range_distr[0], range_distr[1], length)

    return torch.from_numpy(norm.pdf(x, mean, std_dev)).float()


def log_norm_distr(length, mu=0.0, sig=1.0, range_distr=None, plot=False):

    if not range_distr:
        range_distr = [1, length+1]

    x = torch.tensor(np.linspace(range_distr[0], range_distr[1], length))
    prefac = 1.0/(x*sig*np.sqrt(2*np.pi))
    out = prefac*torch.exp(-((torch.log(x) - mu)**2)/(2.0*(sig**2)))

    if plot:
        plt.plot(x, out)
        plt.show()

    return out


def poly_distr(length, coeff, range_distr=None, plot=False):

    if not range_distr:
        range_distr = [1, length+1]

    x = torch.tensor(np.linspace(range_distr[0], range_distr[1], length))
    d = torch.zeros(length)
    for i in range(len(coeff)):
        d = d + coeff[i]*(x**i)

    if plot:
        plt.plot(x, normalise(d))
        plt.show()

    return normalise(d)


def MSELoss(input, target, weights=None):
    if weights == None:
        weights = 1

    out = torch.sum(weights * ((input - target) ** 2))

    return out * torch.conj(out)


def fidelity(input, target):

    return torch.sum(torch.conj(input) * target)


def check_model_compl(n_qubits, num_params, gate_type, criterion):

    if 'MSE' in str(criterion):
        criterion = 'mse'

    file = "model_{}_{}_{}_{}".format(gate_type, n_qubits, num_params, str(criterion))

    for filename in glob.glob("pickled_models/*.pickle*"):
        if file in filename:
            return True

    return False


def load_models(include=['']):
    root_dir = 'spartan_out'
    models = []
    model_filenames = []
    loss_filenames = []
    for filename in glob.iglob(root_dir + '**/*2020', recursive=True):
        required = False
        for crit in include:
            if crit in filename:
                required = True
        if not required:
            continue
        for model_folder in glob.iglob(filename + '**/*models', recursive=True):
            for model_file in glob.iglob(model_folder + '**/*', recursive=True):
                if 'model_' in model_file:
                    models.append(return_obj_from_pickle(model_file))
                    model_filenames.append(model_file)
                elif 'loss_' in model_file:
                    loss_filenames.append(model_file)
    print("Number of models:", len(models))
    return models, model_filenames, loss_filenames


def save_model_and_loss(model, loss=None, criterion='ukn', folder='pickled_models', save_loss=True,
                        model_filename=None, recreate=False):
    if 'MSE' in str(criterion):
        criterion = 'mse'
    i = 0
    filename = '{}/model_{}_{}_{}_{}_ver_{}.pickle'.format(folder, model.gate_type, model.n_qubits,
                                                                   model.num_params, criterion, i)

    while os.path.isfile(filename):
        i += 1
        filename = '{}/model_{}_{}_{}_{}_ver_{}.pickle'.format(folder, model.gate_type, model.n_qubits,
                                                                   model.num_params, criterion, i)
    if model_filename:
        filename = model_filename

    dbfile = open(filename, 'wb')
    # source, destination
    if recreate:
        model = DistributionEncoder(model.N, model.n_qubits, num_params=model.num_params,
                                    gate_type=model.gate_type, params=model.params.detach().numpy())
    pickle.dump(model, dbfile)
    dbfile.close()

    if not save_loss:
        return None

    if model_filename:
        a = model_filename.split('model_')
        dbfile = open(a[0] + "loss_evol_" + a[1], 'wb')
    else:
        dbfile = open('{}/loss_evol_{}_{}_{}_{}_ver_{}.pickle'.format(folder, model.gate_type, model.n_qubits,
                                                                       model.num_params, criterion, i), 'wb')
    # source, destination
    pickle.dump(loss, dbfile)
    dbfile.close()


def process(arguments):

    if len(arguments) == 9:
        ret_model = arguments[-1]
        arguments = arguments[:-1]
    else:
        ret_model = False

    n_qubits, num_params, gate_type, criterion, num_iterations, step, amplitude_distr, distr = arguments
    N = 2 ** n_qubits

    print("criterion:", criterion)
    model = DistributionEncoder(N, n_qubits, num_params=num_params, gate_type=gate_type)
    model, loss_evol = train_model(model, criterion, distr, num_iterations, step=step, amplitude_distr=amplitude_distr)

    save_model_and_loss(model, loss_evol, criterion=criterion)

    if ret_model:
        return model


def return_obj_from_pickle(filename):
    f = open(filename, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj


# ------------------------------- Analysis ------------------------------------------------------------ #


def loss_landscape_2D(model, indxs, distr, num_pix_len, plot=False, criterion=MSELoss):

    assert len(indxs) == 2, 'indxs must be given as a list of length 2 specifying two indices'
    if 'BCE' in str(criterion):
        distr = torch.tensor(distr).double()
    landscape = []
    actual_params = (float(model.params[indxs[0]].detach()),
                     float(model.params[indxs[1]].detach()))
    print(actual_params)
    for i in np.linspace(-np.pi*1.2, np.pi*1.2, num_pix_len):
        for j in np.linspace(-np.pi*1.2, np.pi*1.2, num_pix_len):
            print("param {} = {}, param {} = {}".format(indxs[0], i, indxs[1], j))
            model.params[indxs[0]] = i
            model.params[indxs[1]] = j
            d = model()
            landscape.append(float(criterion(d, distr)))
    landscape = np.array(landscape)
    model.params[indxs[0]] = actual_params[0]
    model.params[indxs[1]] = actual_params[1]
    print("actual values: p1={}, p2={}".format(*actual_params))

    landscape_mat = np.flip(landscape.reshape(num_pix_len, num_pix_len), axis=0)

    if plot:
        fig, ax = plt.subplots()
        im = ax.imshow(landscape_mat)
        plt.imshow(landscape_mat, extent=[-np.pi*1.2, np.pi*1.2, -np.pi*1.2, np.pi*1.2])
        fig.colorbar(im, label="Normalised sqrt(MSE)")

    return landscape_mat


# ------------------------------- Scripts ------------------------------------------------------------- #


def run_main_script(amplitude_distr=False, check_if_done=False):

    gate_types = ['ry', 'vqe', 'own']
    # gate_types = ['own', 'vqe']
    # gate_types = ['vqe']

    criterions = [MSELoss, nn.BCELoss()]
    try:
        mean_ = sys.argv[3]
        std_ = sys.argv[4]
        distr = gaussian_distr(2**n_qubits, mean=mean_, std_dev=std_)
    except:
        distr = gaussian_distr(2**n_qubits, std_dev=3)

    # distr = poly_distr(32, [200000, 8, -1000, 62, -1.0])
    num_iterations = 300
    step = 0.03

    N = 2**n_qubits

    all_arguments = []
    for criterion in criterions:
        for gate_type in gate_types:
            last_p = None
            for num_params in np.linspace(n_qubits, int(5*np.sqrt(N)), 12):
                num_params = int(num_params)
                if gate_type == 'ry':
                    num_params = num_params - int(num_params % n_qubits)
                    if last_p == num_params or num_params < n_qubits:
                        continue
                    last_p = num_params
                if gate_type == 'own':
                    f = int(math.ceil((num_params-n_qubits) / 3.0))
                    num_params = int(f*3) + n_qubits
                if check_if_done and check_model_compl(n_qubits, num_params, gate_type, criterion):
                    print("this is done:", n_qubits, num_params, gate_type, criterion, '\n')
                    continue

                all_arguments.append([n_qubits, num_params, gate_type, criterion, num_iterations, step,
                                      amplitude_distr, distr.clone().detach()])

    all_arguments_heading = "[n_qubits, num_params, gate_type, criterion,num_iterations, step]"
    print('Going to start all the jobs. Pooled jobs:\n' + '\n--> '.join([all_arguments_heading] +
                                        [str(i[:-1]) for i in all_arguments]))
    print('\n\n')
    print("Number of models: {}".format(len(all_arguments)))

    # p = Pool(10)
    # p.map(process, all_arguments)

    # Train specific model
    indx = 3
    print(all_arguments[indx])
    return process(all_arguments[indx] + [True])

    print('\n\n============================================================================\n\n')
    print("Jobs done:\n")


def optim_script(type_arch='vqe', amplitude_distr=False):

    N = 2 ** n_qubits
    gate_type = type_arch
    all_arguments = []
    criterion = MSELoss
    num_iterations = 200
    num_variations = 10
    step = 0.005
    distr = gaussian_distr(2 ** n_qubits, std_dev=3, mean=-3)

    for num_params in np.linspace(n_qubits, int(15*np.sqrt(N)), 30):
        num_params = int(num_params)
        for i in range(num_variations):
            all_arguments.append([n_qubits, num_params, gate_type, criterion, num_iterations, step,
                                  amplitude_distr, distr])

    all_arguments_heading = "[n_qubits, num_params, gate_type, criterion,num_iterations, step, ampli_distr]"
    print('Going to start all the jobs. Pooled jobs:\n' + '\n--> '.join([all_arguments_heading] +
                                                                    [str(i[:-1]) for i in all_arguments]))
    print("Overall {} jobs.".format(len(all_arguments)))
    print('\n\n')

    p = Pool(11)
    p.map(process, all_arguments)

    print('\n\n============================================================================\n\n')
    print("Jobs done:\n")


def optimise_distr_script(distr, num_params, amplitude_distr=False):
    """ So given a distribution, we create a model that can encode that particular distribution. Note
        the number of qubits is chosen in """
    N = 2 ** n_qubits
    gate_type = 'ry'
    criterion = MSELoss
    num_iterations = 500
    step = 0.001
    distr = normalise(distr)
    # The first True is for training for amplitudes while second true is for returning model
    return process([n_qubits, num_params, gate_type, criterion, num_iterations, step, amplitude_distr, distr, True])


def optimise_distr(distr, num_params, gate_type='ry', num_iterations=500, criterion=MSELoss,
                   step=0.001):
    distr = normalise(distr)
    amplitude_distr = True
    n_qubits = int(np.log2(len(distr)))
    return process([n_qubits, num_params, gate_type, criterion, num_iterations, step, amplitude_distr, distr, True])


def error_script(check_if_done=False):
    n_qubits = 5
    reset_q_num(n_qubits)
    criterion = MSELoss
    distr = normalise(gaussian_distr(2 ** n_qubits, std_dev=3, mean=0))
    num_iterations = 180
    N = 2 ** n_qubits
    step = 0.03
    num_params = 30

    reset_q_num(n_qubits, noisy=True)
    ry_models, ry_loss = [], []
    own_models, own_loss = [], []

    all_arguments = []
    for bitflip in np.linspace(0, 0.05, 20):
        for phaseflip in np.linspace(0, 0.05, 20):
            if check_if_done and check_error_model(bitflip, phaseflip):
                print("already done: {},{}".format(bitflip, phaseflip))
                continue
            all_arguments.append((N, n_qubits, num_params, bitflip, phaseflip, criterion, distr, num_iterations, step))

    p = Pool(30)
    p.map(error_process, all_arguments)

    ry_phase_error, ry_flip_error, own_phase_error, own_flip_error = [], [], [], []
    for filename in glob.iglob('error_models' + '**/*'):

        model = return_obj_from_pickle(filename)
        name = filename[13:]
        gate = name.split('_')[0]

        bitflip = float(name.split('-')[1])
        phaseflip = float(name.split('-')[-2])
        reset_error(bitflip=bitflip, phaseflip=phaseflip)

        if gate == 'ry':
            ry_models.append(model)
            ry_loss.append(criterion(model()[0], normalise_prob(distr)))
            ry_phase_error.append(phaseflip)
            ry_flip_error.append(bitflip)
        elif gate == 'own':
            own_models.append(model)
            own_loss.append(criterion(model()[0], normalise_prob(distr)))
            own_phase_error.append(phaseflip)
            own_flip_error.append(bitflip)

    ry_data = pd.DataFrame({'bitflip_error': ry_flip_error, 'phaseflip_error': ry_phase_error,
                            'ry_model':ry_models, 'loss':ry_loss})
    own_data = pd.DataFrame({'bitflip_error': own_flip_error, 'phaseflip_error': own_phase_error,
                             'own_model':own_models, 'loss':own_loss})

    data = pd.concat([ry_data, own_data], ignore_index=True)
    print(data)
    data.to_pickle('error_data1.pickle')


def barren_script():
    n_qubits = 5
    reset_q_num(n_qubits)
    criterion = MSELoss
    distr = normalise(gaussian_distr(2 ** n_qubits, std_dev=3, mean=0))
    num_iterations = 180
    N = 2 ** n_qubits
    step = 0.03
    gate = 'ry'

    bitflip = 0.01
    phaseflip = 0.01

    all_arguments = []
    for n_qubits in range(4, 8):
        for num_params in range(n_qubits, 16*n_qubits, n_qubits):
            distr = normalise(gaussian_distr(2 ** n_qubits, std_dev=3, mean=0))
            all_arguments.append((N, n_qubits, gate, num_params, bitflip, phaseflip, criterion, distr, num_iterations, step))

    print("Number of models:", len(all_arguments))
    p = Pool(30)
    p.map(barren_process, all_arguments)
    # barren_process(all_arguments[2])

    phase_error, flip_error, gate_type = [], [], []
    num_params_list = []
    loss = []
    n_qubit_list = []
    for filename in glob.iglob('error_models' + '**/*'):

        model = return_obj_from_pickle(filename)
        name = filename[13:]
        gate_type.append(model.gate_type)
        num_params_list.append(model.num_params)

        bitflip = float(name.split('-')[1])
        phaseflip = float(name.split('-')[-2])

        reset_q_num(model.n_qubits, noisy=True)
        reset_error(bitflip=bitflip, phaseflip=phaseflip)

        distr = normalise(gaussian_distr(2 ** model.n_qubits, std_dev=3, mean=0))
        loss.append(criterion(model()[0], normalise_prob(distr)))
        phase_error.append(phaseflip)
        flip_error.append(bitflip)
        n_qubit_list.append(model.n_qubits)

    data = pd.DataFrame({'n_qubits':n_qubit_list, 'gate_type':gate_type, 'num_params':num_params_list, 'bitflip_error': flip_error,
                            'phaseflip_error': phase_error, 'loss': loss})

    print(data)
    data.to_pickle('barren_data1.pickle')


def check_error_model(bitflip, phaseflip, gate_type='ry', num_params=None, n_qubits=None):
    if num_params:
        file = '{}_{}_{}_bitf-{:.6f}-phasef-{:.6f}-.pickle'.format(gate_type, n_qubits, num_params, bitflip, phaseflip)
    else:
        file = '{}_bitf-{:.6f}-phasef-{:.6f}-.pickle'.format(gate_type, bitflip, phaseflip)
    for filename in glob.glob("error_models/*.pickle*"):
        if file in filename:
            return True

    return False


def barren_process(arguments):

    N, n_qubits, gate_type, num_params, bitflip, phaseflip, criterion, distr, num_iterations, step = arguments
    reset_q_num(n_qubits, noisy=True)
    reset_error(bitflip=bitflip, phaseflip=phaseflip)
    m = DistributionEncoder(N, n_qubits, num_params=num_params, gate_type=gate_type)
    m.q_delta = 0.3
    m, _ = train_model(m, criterion, distr, num_iterations, step=step,
                       live_plot=False, stochastic_term=False)
    save_model_and_loss(m, save_loss=False,
                        model_filename='error_models/{}_{}_{}_bitf-{:.6f}-phasef-{:.6f}-.pickle'.format(gate_type, n_qubits, num_params, bitflip, phaseflip))


def error_process(arguments):

    N, n_qubits, num_params, bitflip, phaseflip, criterion, distr, num_iterations, step = arguments
    reset_q_num(n_qubits, noisy=True)
    print('bit_flip_error={},  phase_flip_error={}'.format(bitflip, phaseflip))
    reset_error(bitflip=bitflip, phaseflip=phaseflip)
    m = DistributionEncoder(N, n_qubits, num_params=num_params, gate_type='ry')
    m.q_delta = 0.3
    m, _ = train_model(m, criterion, distr, num_iterations, step=step,
                       live_plot=False, stochastic_term=False)

    save_model_and_loss(m, save_loss=False, model_filename='error_models/ry_bitf-{:.6f}-phasef-{:.6f}-.pickle'.format(bitflip, phaseflip))

    m = DistributionEncoder(N, n_qubits, num_params=num_params, gate_type='own')
    m.q_delta = 0.3
    m, _ = train_model(m, criterion, distr, num_iterations, step=step,
                       live_plot=False, stochastic_term=False)

    save_model_and_loss(m, save_loss=False, model_filename='error_models/own_bitf-{:.6f}-phasef-{:.6f}-.pickle'.format(bitflip, phaseflip))
