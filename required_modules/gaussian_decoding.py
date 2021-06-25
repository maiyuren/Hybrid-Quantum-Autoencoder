
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
import itertools

# Pennylane
import pennylane as qml
from pennylane import numpy as np
import pennylane_qiskit
from pennylane_cirq import ops as cirq_ops

# Other tools
import time
import sys
import os
import pickle
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import glob
import pandas as pd

from advanced_decoder import AdvancedDecoder, EnhancedDecoder

n_qubits = 10
max_q_depth = 10
SHOTS = 1000
SAMPLE = False

if __name__ == '__main__':
    try:
        n_qubits = int(sys.argv[1])
        max_q_depth = int(sys.argv[2])
    except IndexError:
        pass

dev = qml.device("default.qubit", wires=n_qubits, shots=SHOTS)

float_point_accuracy = 8
PHASE_FLIP_ERROR = 0.02
BIT_FLIP_ERROR = 0.02
NOISY = False

# ########################## Useful functions ###############################################


def gaussian_distr(length, std_dev=1, mean=0, range_distr=None):
    if not range_distr:
        range_distr = [-length / 2, length / 2]

    x = np.linspace(range_distr[0], range_distr[1], length)

    return torch.from_numpy(norm.pdf(x, mean, std_dev)).float()


def create_gaussian_distributions(N, list_mean=None, list_std=None, complex=False):

    if not isinstance(list_mean, np.ndarray) and not isinstance(list_mean, list):
        list_mean = np.arange(-N / 3.0, N / 3.0, N / 20.0)
    if not isinstance(list_std, np.ndarray) and not isinstance(list_std, list):
        list_std = np.linspace(N / 50.0, N / 3.0, 10)

    yield len(list_std)*len(list_mean)

    for mean in list_mean:
        for std in list_std:
            d = normalise(gaussian_distr(N, std_dev=std, mean=mean, range_distr=[-N / 2, N / 2]))
            if complex:
                d = d.numpy() * random_phases(N)
            yield d, mean, std


def ret_gaussian_distributions(N, list_mean=None, list_std=None):
    distributions = []
    std_deviations = []
    means = []
    distr_gen = create_gaussian_distributions(N, list_mean, list_std)
    size_distr = next(distr_gen)
    for d, m, s in distr_gen:
        distributions.append(d)
        means.append(m)
        std_deviations.append(s)

    return distributions, means, std_deviations


def sample_normalised_gaussian(N, range_mean=None, range_std_dev=None, range_distr=None, complex=False):
    if not isinstance(range_mean, np.ndarray) and not range_mean:
        range_mean = [-N/3, N/3]
    if not isinstance(range_std_dev, np.ndarray) and not range_std_dev:
        range_std_dev = [4, N/3]
        # range_std_dev = [N/15, N/5]
    m = (range_mean[1]-range_mean[0])*random.random() + range_mean[0]
    s = (range_std_dev[1]-range_std_dev[0])*random.random() + range_std_dev[0]
    d = normalise(gaussian_distr(N, mean=m, std_dev=s, range_distr=range_distr))
    if complex:
        d = d.numpy() * random_phases(N)

    return d, m, s


def sample_random_distributions(N):
    params_a = np.random.random(N)
    params_theta = np.random.random(N)
    d = sine_Fourier_series(N, params_a, params_theta)
    plt.plot(d)
    plt.show()
    return normalise(d)


#### The following section not used ########################################################################

def sine_Fourier_series(N, params_a, params_theta):
    range_distr = [-N / 2, N / 2]
    x = np.linspace(range_distr[0], range_distr[1], N)
    return params_a*np.sin(params_theta*x)


def gen_rand_gaussian(N, range_mean=None, range_std_dev=None, range_distr=None, complex=False):
    if complex:
        while True:
            d, m, s = sample_normalised_gaussian(N, range_mean, range_std_dev, range_distr)
            yield d.numpy() * random_phases(N), m, s
    else:
        while True:
            yield sample_normalised_gaussian(N, range_mean, range_std_dev, range_distr)


def gen_rand_distr(N):
    while True:
        sample_random_distributions(N)
        # Todo finish this


def random_phases(N):
    out = np.random.random(2*N) * (np.random.randint(2, size=2*N) * 2 - 1)
    return normalise_complex(out.view(np.complex128))

###########################################################################################################


def sigmoid_k(x, k):
    """ This specific sigmoid is scaled differently for purposes of this project: """

    # Have to be torch inputs
    return 1.0 / (1.0 + torch.exp(-k * x))


def normalise(a):
    if isinstance(a, np.ndarray):
        return a / np.sqrt(np.dot(a, a))
    if len(a.shape) > 1:
        t = torch.Tensor()
        for elem in a:
            t = torch.cat((t, normalise(elem)))
        return t
    else:
        return a/torch.sqrt(torch.dot(a, a))


def normalise_prob(a):

    if len(a.shape) > 1:
        t = torch.Tensor()
        for elem in a:
            t = torch.cat((t, normalise(elem)))
        return t
    else:
        return a/torch.sum(a)


def normalise_complex(a):
    assert isinstance(a, np.ndarray)

    return a/np.abs(a)


def mu_sig_to_out(mu, sig, N, reg_size=1):
    # The out here is the expectation value, NOT between 0 and 1
    if sig == None:
        if reg_size == 1:
            return torch.tensor([mu * 2.0 / N])

    if mu == None:
        if reg_size == 1:
            return torch.tensor([(2.0 * sig / N) - 1])

    assert abs(mu) <= N/2.0
    assert sig > 0

    n_qubits = int(np.log2(N))

    # Transform the mean
    mu = mu + (N/2.0)

    if reg_size == 'binary':
        b_mu = ("{0:0" + str(n_qubits) +"b}").format(int(mu))
        b_sig = ("{0:0" + str(n_qubits) +"b}").format(int(sig))
        mean_out = out_to_reg_out(b_mu, reg_size)
        std_out = out_to_reg_out(b_sig, reg_size)

        assert len(mean_out) + len(std_out) == 2 * n_qubits, "{} + {} != 2{}".format(len(mean_out),
                                                                                     len(std_out), n_qubits)
        return torch.cat((mean_out, std_out))

    out = torch.tensor([(2.0 * mu / N) - 1, (2.0 * sig / N) - 1])

    if reg_size == 1:
        return out
    mean_out = out_to_reg_out(out[0], reg_size)
    std_out = out_to_reg_out(out[1], reg_size)

    return torch.cat((mean_out, std_out))


def out_to_reg_out(x, reg_size=1):

    if reg_size == 'binary':
        neg = 0
        if x[0] == '-':
            # If negative then the first qubit is a 1 otherwise its 0
            x = x[1:]
            neg = 1
        out = torch.tensor([float(i) for i in x])
        out[0] = out[0] + neg
        return (out * 2) - 1

    if reg_size == 1:
        return x

    acc = 10 # ** (float_point_accuracy // reg_size)

    x = x * acc
    out = [int(x) / acc]
    x = abs(x) % 1
    for i in range(reg_size - 1):
        x = x * acc
        out.append(int(x) / acc)
        x = x % 1

    return torch.tensor(out)


def register_to_one(r, sig_fig):
    if r[0] < 0:
        negative = -1
    else:
        negative = 1

    out = ['0.'] + [str(float(i)).split('.')[-1][:sig_fig] for i in r]
    out = float(''.join(out))
    return torch.tensor(out)*negative


def out_to_mu_sig(out, N, reg_size=1, sample_num=None):
    """ out is in between 0 and 1 """

    n_qubits = int(np.log2(N))

    if sample_num:
        out = torch.round(out*sample_num)/sample_num

    if reg_size == 1:
        return (2*out[0]-1)*N/2.0, out[1]*N
        # return out[0]*N/2.0, (out[1]+1)*N/2.0
    elif reg_size == 'binary':
        return out_binary_weighting(out, n_qubits)

    sig_fig = 1 #(float_point_accuracy // reg_size)

    length = int(len(out)//2)
    mean = register_to_one(out[:length], sig_fig)
    std = register_to_one(out[length:], sig_fig)

    return mean*N/2.0, (std+1)*N/2.0


def out_binary_weighting(tr_out, n_qubits, trans_type='mean'):
    """ Returns two numbers of mu and sig for training the output. The input is soething that has been transformed
        to lie between 0 and 1 """
    weights = torch.tensor([2 ** (n_qubits - i - 1) for i in range(n_qubits)])

    if len(tr_out) == n_qubits*2:
        out_mu = torch.sum(weights*tr_out[:n_qubits]) + 1 - 2**(n_qubits-1)
        out_sig = torch.sum(weights*tr_out[n_qubits:])
        return out_mu, out_sig

    elif len(tr_out) == n_qubits:
        if trans_type == 'mean':
            return torch.sum(weights*tr_out) + 1 - 2**(n_qubits-1)
        elif trans_type == 'std':
            return torch.sum(weights*tr_out)


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


# ########################## Loss functions ###############################################

def MSELoss(input, target, weights=None):

    if weights == None:
        weights = 1

    return torch.sum(weights * ((input - target) ** 2))


# ########################## Quantum Circuit ###############################################

# Code obtained from quantum_transfer learning

def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates.
    """
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis.
    """
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)


def RX_layer(w):
    """Layer of parametrized qubit rotations around the x axis.
    """
    for idx, element in enumerate(w):
        qml.RX(element, wires=idx)


def random_r_layer(w, paulis):

    for idx, weight in enumerate(w):
        if paulis[idx] == 'x':
            qml.RX(weight, wires=idx)
        if paulis[idx] == 'y':
            qml.RY(weight, wires=idx)
        else:
            qml.RZ(weight, wires=idx)


def U2_layer(w):
    """Layer of parametrized qubit rotations with U2 rotations.
        """
    for idx in range(int(len(w)/2)):
        qml.U2(w[2*idx], w[2*idx + 1], wires=idx)


def entangling_layer(nqubits, couplings=None):
    """Layer of CNOTs followed by another shifted layer of CNOT.
    """
    if couplings:
        for coup in couplings:
            qml.CNOT(wires=coup)
    else:
        # In other words it should apply something like :
        # CNOT  CNOT  CNOT  CNOT...  CNOT
        #   CNOT  CNOT  CNOT...  CNOT
        for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
            qml.CNOT(wires=[i, i + 1])
        for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
            qml.CNOT(wires=[i, i + 1])


def set_random_paulis(n_qubits, q_depth):
    P = ['x', 'y', 'z']
    out = []
    for i in range(q_depth):
        out.append(random.choices(P, k=n_qubits))
    return out


def set_random_couplings(n_qubits, q_depth):

    q_array = list(range(n_qubits))
    out = []
    for i in range(q_depth):
        l = []
        for j in range(n_qubits):
            l.append(random.sample(q_array, 2))
        out.append(l)
    return out


def reset_q_num(n_qubits, set_gate_type='ry', dev_type='default', shots=1024):

    global dev
    if dev_type == 'default':
        dev = qml.device("default.qubit", wires=n_qubits, shots=SHOTS)
    elif dev_type == 'noisy':
        global NOISY
        NOISY = True
        dev = qml.device("cirq.mixedsimulator", wires=n_qubits)
    elif dev_type == 'aer':
        # dev = qml.device('qiskit.aer', wires=n_qubits, shots=shots)
        # dev = pennylane_qiskit.basic_aer.BasicAerDevice(wires=n_qubits, backend='qasm_simulator', shots=shots)
        dev = pennylane_qiskit.aer.AerDevice(wires=n_qubits, shots=shots, backend='qasm_simulator')
    else:
        assert NotImplemented

    if set_gate_type:
        global _gate_type
        _gate_type = set_gate_type

    global q_net

    @qml.qnode(dev, interface="torch")
    def q_net(q_weights_flat, n_qubits=None, amplitudes=None, q_depth=None, reg_size=None,
              paulis=None, couplings=None):
        qml.QubitStateVector(amplitudes, wires=list(range(n_qubits)))

        # Reshape weights
        if _gate_type == 'ry':
            q_weights = q_weights_flat.reshape(q_depth, n_qubits)
        elif _gate_type == 'u2':
            q_weights = q_weights_flat.reshape(q_depth, 2*n_qubits)
        elif _gate_type == 'ry_rx':
            q_weights = q_weights_flat.reshape(q_depth, n_qubits)
        elif _gate_type == 'random':
            q_weights = q_weights_flat.reshape(q_depth, n_qubits)
            assert couplings != None, "Need to define couplings for random circuit."
            assert paulis != None, "Need to define paulis for random circuit."

        H_layer(n_qubits)

        # Sequence of trainable variational layers
        _c = True
        for k in range(q_depth):
            if _gate_type == 'random':
                entangling_layer(n_qubits, couplings[k])
            else:
                entangling_layer(n_qubits)

            if _gate_type == 'ry':
                RY_layer(q_weights[k])
            elif _gate_type == 'u2':
                U2_layer(q_weights[k])
            elif _gate_type == 'ry_rx':
                if _c:
                    RY_layer(q_weights[k])
                    _c = False
                else:
                    RX_layer(q_weights[k])
                    _c = True
            elif _gate_type == 'random':
                random_r_layer(q_weights[k], paulis=paulis[k])
            if NOISY:
                for i in range(n_qubits):
                    cirq_ops.PhaseFlip(PHASE_FLIP_ERROR, wires=i)
                    cirq_ops.BitFlip(BIT_FLIP_ERROR, wires=i)

        if SAMPLE:
            out = tuple([qml.sample(qml.PauliZ(i)) for i in range(reg_size)])
            print(out)
            return out
        return tuple([qml.expval(qml.PauliZ(i)) for i in range(reg_size)])


reset_q_num(n_qubits, set_gate_type='ry')


def reset_noise(n_qubits, bitflip=0.02, phaseflip=0.02):

    global PHASE_FLIP_ERROR
    global BIT_FLIP_ERROR

    PHASE_FLIP_ERROR = phaseflip
    BIT_FLIP_ERROR = bitflip

    reset_q_num(n_qubits=n_qubits, dev_type='noisy', set_gate_type=None)


def reset_aer(n_qubits, shots=1024):

    global SHOTS
    SHOTS = shots

    reset_q_num(n_qubits=n_qubits, dev_type='aer', set_gate_type=None, shots=shots)


# ########################## Model ###############################################


class CharacteristicsDecoder(nn.Module):

    name = 'CharacteristicsDecoder'

    def __init__(self, input_size, n_qubits, q_depth, q_delta, reg_size=1, gate_type='ry'):
        """
        :param input_size:          input size
        :param n_qubits:            number of qubits
        :param q_depth:             number of layers of the parameterised rotations
        :param q_delta:             Initial spread of random quantum weights
        """
        super().__init__()

        self.input_size = input_size
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.q_delta = q_delta
        self.binary = False
        self.gate_type = gate_type

        if gate_type == 'ry':
            self.q_params = nn.Parameter(q_delta * torch.randn(self.q_depth * n_qubits))
        elif gate_type == 'u2':
            self.q_params = nn.Parameter(q_delta * torch.randn(2* self.q_depth * n_qubits))

        assert 2*reg_size <= n_qubits
        self.reg_size = reg_size

        self.model_info = {'input_size': input_size, 'n_qubits': n_qubits, 'q_depth': q_depth,
                           'q_delta': q_delta, 'reg_size': reg_size, 'model': 'CharacteristicsDecoder'}

        self.train_time = 0
        self.train_logistic_k = None
        self.last_state = None
        self.batch_size = None

    def forward(self, x, save_state=False):
        q_in = x
        q_out = torch.Tensor(0, 2*self.reg_size)
        # Because it comes as a batch (matrix) we cant just send the matrix in altogether like for the other layers
        #  but incase it has come by itself
        if len(q_in.shape) == 1:
            if isinstance(q_in, np.ndarray):
                q_in = np.array([q_in])
            else:
                q_in = q_in.unsqueeze(0)
        for elem in q_in:
            q_out_elem = q_net(self.q_params, n_qubits=self.n_qubits, amplitudes=elem,
                               q_depth=self.q_depth, reg_size=2*self.reg_size).float()
            if SAMPLE:
                q_out_elem = torch.mean(q_out_elem, axis=1)
            q_out = torch.cat((q_out, q_out_elem.unsqueeze(0)))

        if save_state:
            self.last_state = dev.state

        return q_out


class CharacteristicsSeparateDecoder(nn.Module):

    name = 'CharacteristicsSeparateDecoder'

    def __init__(self, input_size, n_qubits, q_depth, q_delta, reg_size=2, gate_type='ry', shots=None):
        """
                :param input_size:          input size
                :param n_qubits:            number of qubits
                :param q_depth:             number of layers of the parameterised rotations
                :param q_delta:             Initial spread of random quantum weights
        """
        super().__init__()

        self.input_size = input_size
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.q_delta = q_delta
        self.gate_type = gate_type
        self.train_shots = shots
        self.paulis = None # Only used for random gate type
        self.couplings = None

        if gate_type == 'ry':
            self.num_params = self.q_depth * n_qubits
        elif gate_type == 'u2':
            self.num_params = 2 * self.q_depth * n_qubits
        elif gate_type == 'ry_rx':
            self.num_params = self.q_depth * n_qubits
        elif gate_type == 'random':
            self.num_params = self.q_depth * n_qubits
            self.paulis = set_random_paulis(n_qubits, self.q_depth)
            self.couplings = set_random_couplings(n_qubits, self.q_depth)

        reset_q_num(self.n_qubits, set_gate_type=self.gate_type)
        if shots:
            # reset_aer(self.n_qubits, shots=shots)
            pass
        self.q_params_mean = nn.Parameter(q_delta * torch.randn(self.num_params))
        self.q_params_std = nn.Parameter(q_delta * torch.randn(self.num_params))

        if reg_size == 'binary':
            self.reg_size = n_qubits
            self.binary = True
        else:
            assert reg_size <= n_qubits
            self.reg_size = reg_size
            self.binary = False

        self.model_info = {'input_size':input_size, 'n_qubits':n_qubits, 'q_depth':q_depth,
                           'q_delta':q_delta, 'reg_size':reg_size, 'model': 'CharacteristicsSeparateDecoder'}

        self.train_time = 0
        self.distributions = None
        self.batch_size = None
        self.train_logistic_k = None
        self.train_regular_term = None
        self.train_p_fuzz = None
        self.last_state = [None, None]

    def forward(self, x, save_state=False, sample_num=None):
        q_in = x
        q_out = torch.Tensor(0, 2 * self.reg_size)
        # Because it comes as a batch (matrix) we cant just send the matrix in altogether like for the other layers
        #  but incase it has come by itself
        if len(q_in.shape) == 1:
            if isinstance(q_in, np.ndarray):
                q_in = np.array([q_in])
            else:
                q_in = q_in.unsqueeze(0)
        for elem in q_in:
            q_out_elem_mean = q_net(self.q_params_mean, n_qubits=self.n_qubits, amplitudes=elem,
                                    q_depth=self.q_depth, reg_size=self.reg_size,
                                    paulis=self.paulis, couplings=self.couplings).float()
            if save_state:
                m_state = dev.state
            q_out_elem_std = q_net(self.q_params_std, n_qubits=self.n_qubits, amplitudes=elem,
                                   q_depth=self.q_depth, reg_size=self.reg_size,
                                    paulis=self.paulis, couplings=self.couplings).float()
            if save_state:
                self.last_state = [m_state, dev.state]

            if SAMPLE:
                q_out_elem_mean = torch.mean(q_out_elem_mean, axis=1, dtype=torch.float32)
                q_out_elem_std = torch.mean(q_out_elem_std, axis=1, dtype=torch.float32)

            q_out_elem = torch.cat((q_out_elem_mean, q_out_elem_std)).unsqueeze(0)

            if sample_num:
                q_out_elem = q_out_elem + torch.tensor(np.random.random(self.n_qubits*2) - 0.5) / sample_num ** 0.5
                q_out_elem = q_out_elem.float()
            # Transformation of the output to be inbetween 0 and 1
            if self.train_logistic_k:
                q_out_elem = sigmoid_k(q_out_elem, self.train_logistic_k)
            else:
                q_out_elem = (q_out_elem + 1) / 2

            q_out = torch.cat((q_out, q_out_elem))

        # returns values between 0 and 1
        return q_out

# ########################## Training functions ###############################################


def train_model(model, criterion, optimizer, distr='set_gaussian', num_iterations=10, loss_evol=None,
                N=None, weights=None, logistic_k=None, convert_ouput=True, batch_size=1,
                noisy_train=False, complex_distr=False, shots=None, regular_term=None):
    reset_q_num(model.n_qubits, set_gate_type=model.gate_type)
    if noisy_train:
        reset_noise(model.n_qubits)
    elif shots:
        print("Number of shots set to:", shots)
        reset_aer(model.n_qubits, shots=shots)
    since = time.time()
    print("Training started:")
    print("Training device: {}".format(dev))
    model.train_logistic_k = logistic_k
    model.train_regular_term = regular_term
    if distr == 'sample_gaussian':
        check_sample = True
        if not loss_evol:
            loss_evol = []
    elif distr == 'set_gaussian':
        check_sample = False

        distr = create_gaussian_distributions(N)
        size_distr = next(distr)
        if not loss_evol:
            # One loss evolution for each type of gaussian
            loss_evol = [[] for _ in range(size_distr)]

    else:
        # This option is for when distr is the Distributions object in amplitude_encoding
        print('distributions set to Distributions object.')
        # distr = make_distr_generator(distr) # This code wont work because it gets stuck
                                                # in an infinite loop
    indx = 0
    distr = create_gaussian_distributions(N, complex=complex_distr)
    size_distr = next(distr)
    for ii in range(num_iterations):
        print('\n-----------iteration {}--model:{}---------------------------'.format(ii, model))
        try:
            obj = next(distr)
        except StopIteration:
            indx = 0
            distr = create_gaussian_distributions(N, complex=complex_distr)
            size_distr = next(distr)
            obj = next(distr)

        with torch.set_grad_enabled(True):
            loss = 0
            optimizer.zero_grad()
            for ob in [obj]+[sample_normalised_gaussian(model.input_size, complex=complex_distr)
                             for _ in range(batch_size-1)]:
                d, mean, std_dev = ob

                if model.binary:
                    reg_size = 'binary'
                else:
                    reg_size = model.reg_size
                    assert not logistic_k, "Logistic has only been programmed for use in binary"

                output = model(d)[0].float()
                print("Output:", output)
                # Convert out changes the output to a number so the loss is over a number than a register
                if convert_ouput and reg_size == 'binary':
                    mu, sig = out_binary_weighting(output, n_qubits=model.n_qubits)
                    print("Model:    m={} ---- s={}".format(mu, sig))
                    print("Expected: m={} ---- s={}".format(mean, std_dev))
                    loss += criterion(mu, mean) + criterion(sig, std_dev)
                elif weights == None or 'BCE' in str(criterion):

                    # output has already been transformed
                    loss += criterion(output, (mu_sig_to_out(mean, std_dev, N, reg_size) + 1) / 2)
                else:
                    print("Model output:    ", output)
                    print("Expected output: ", (mu_sig_to_out(mean, std_dev, N, reg_size) + 1) / 2)
                    print("Model: mean={}, std={}".format(*out_to_mu_sig(output, N, reg_size)))
                    print("Expected mean={}, std={}".format(mean, std_dev))
                    loss += criterion(output, (mu_sig_to_out(mean, std_dev, N, reg_size) + 1) / 2,
                                     weights=weights)
                if regular_term:
                    print("Loss before regular term:", loss, ", Model", model)
                    loss += -1 * regular_term * torch.sum(((output+1)/2) ** 2)

            print("Loss", loss, ", Model", model)
            loss_evol[indx].append(loss)
            loss.backward()
            optimizer.step()
            indx += 1

    model.batch_size = batch_size
    model.train_logistic_k = logistic_k
    model.train_time = time.time() - since
    print("Time taken to train model: {}".format(time.time()-since))
    return model, loss_evol


def make_distr_generator(f):
    while True:
        yield f.get()

# ########################## Testing functions ###############################################

def mse_error_mean(u, v):
    if not isinstance(u, torch.Tensor):
        u = torch.tensor(u)
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v)
    return ((u-v)**2).mean()


def test(model, num_test, distributions='gaussian_sample', means=None, stds=None, sample_num=None,
         noisy_test=False, range_mean=None, range_std=None, shots=None, outlier_removal=False):
    if not hasattr(model, 'paulis'):
        model.paulis = None
        model.couplings = None
    try:
        reset_q_num(model.n_qubits, set_gate_type=model.gate_type)
    except AttributeError:
        reset_q_num(model.n_qubits)
    if noisy_test:
        reset_noise(model.n_qubits)
    if shots:
        print("Number of shots set to:", shots)
        reset_aer(model.n_qubits, shots=shots)
    elif hasattr(model, 'train_shots') and model.train_shots:
        print("Number of shots set to:", model.train_shots)
        reset_aer(model.n_qubits, shots=model.train_shots)

    if model.binary:
        reg_size = 'binary'
    else:
        reg_size = model.reg_size
    print("Test device:", dev)
    if isinstance(distributions, list) or isinstance(distributions, torch.Tensor):
        distr_gen = ((d, means[indx],stds[indx]) for indx, d in enumerate(distributions))
    elif distributions == 'gaussian_sample':
        distr_gen = gen_rand_gaussian(model.input_size, range_mean=range_mean, range_std_dev=range_std)
    elif distributions == 'gaussian_specific':
        pass
    elif distributions == 'random_samples':
        distr_gen = gen_rand_distr(model.input_size)
    elif distributions == 'complex_gaussian':
        distr_gen = gen_rand_gaussian(model.input_size, complex=True)
    elif distributions == 'heatmap':
        distr_gen = create_gaussian_distributions(model.input_size)
        next(distr_gen)

    error_sum_mean = 0
    error_sum_std = 0
    results = []

    if distributions == 'heatmap':
        print("Starting heatmap.")
        h_map_mean, h_map_std = heat_map_mu_sig(model)
        return np.mean(h_map_mean), np.mean(h_map_std), np.array([h_map_mean, h_map_std])
    else:
        for i in tqdm(range(num_test)):
            d, m, s = next(distr_gen)
            out = model(d)[0]
            mu, sig = out_to_mu_sig(out, model.input_size, reg_size, sample_num=sample_num)

            m_er = mse_error_mean(mu, m)**0.5
            s_er = mse_error_mean(sig, s)**0.5
            error_sum_mean += m_er
            error_sum_std += s_er
            results.append([m, s, m_er.detach(), s_er.detach()])

    if outlier_removal:
        m = 1
        results = np.array(results)
        print("Average raw mean:", error_sum_mean/num_test)
        print("Average raw std:", error_sum_std/num_test)
        av_mean = np.mean(reject_outliers(results[:, 2], m=m))
        av_std = np.mean(reject_outliers(results[:, 3], m=m))
    else:
        av_mean = error_sum_mean/num_test
        av_std = error_sum_std/num_test

    print("The average mean sqrt(mse) is: {}".format(av_mean))
    print("The average std sqrt(mse) is: {}".format(av_std))

    return av_mean, av_std, results


def predict(model, distr, save_state=False, mu_sig=None):

    if model.binary:
        reg_size = 'binary'
    else:
        reg_size = model.reg_size

    out = model(distr, save_state=save_state)[0]
    mu, sig = out_to_mu_sig(out, model.input_size, reg_size)

    if mu_sig:
        m_er = mse_error_mean(mu, mu_sig[0]) ** 0.5
        s_er = mse_error_mean(sig, mu_sig[1]) ** 0.5
        return mu, sig, [m_er, s_er]
    return mu, sig

# ########################## Loading functions ###################################################


def load_models(include=['']):
    root_dir = 'spartan_out'
    models = []
    model_filenames = []
    loss_filenames = []
    for filename in glob.iglob(root_dir + '**/*', recursive=True):
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


def test_and_dataframe(num_test, models, model_filenames, loss_filenames, save_data=False,
                       test_type='gaussian_sample', circuit_sample_num=None, noisy_test=False,
                       p_range_mean=None, p_range_std=None, save_raw_results=False,
                       outlier_removal=False):
    mu_res, sig_res, log_k_list, q_depth_list, reg_size_list, n_qubit_list = [], [], [], [], [], []
    test_results, model_type_list, crit_list, shots_list = [], [], [], []
    for i, model in enumerate(models):
        print("Testing model: {}\n{} of {}".format(model_filenames[i], i, len(models)))
        reset_q_num(model.n_qubits)
        if p_range_mean:
            range_mean = (2**model.n_qubits)*np.array(p_range_mean)
        else:
            range_mean = None
        if p_range_std:
            range_std = (2**model.n_qubits)*np.array(p_range_std)
        else:
            range_std = None
        err_mu, err_sig, results = test(model, num_test, test_type, sample_num=circuit_sample_num,
                                        noisy_test=noisy_test, range_mean=range_mean,
                                        range_std=range_std, outlier_removal=outlier_removal)
        mu_res.append(err_mu)
        sig_res.append(err_sig)
        print('mu_error={} ----- sig_error={}'.format(err_mu, err_sig))
        if not hasattr(model, 'train_logistic_k') or model.train_logistic_k == None:
            log_k_list.append(-1)
        else:
            log_k_list.append(model.train_logistic_k)
        if hasattr(model, 'train_shots'):
            shots_list.append(model.train_shots)
        else:
            shots_list.append(None)
        q_depth_list.append(model.q_depth)
        reg_size_list.append(model.reg_size)
        n_qubit_list.append(model.n_qubits)
        test_results.append(results)
        model_type_list.append(model.name)
        if 'mse' in model_filenames[i]:
            crit_list.append('mse')
        else:
            crit_list.append('bce')

    data = pd.DataFrame({'model': models,
                         'model_type': model_type_list,
                         'n_qubits': n_qubit_list,
                         'Criterion': crit_list,
                         'q_depth': q_depth_list,
                         'reg_size': reg_size_list,
                         'logistic_k': log_k_list,
                         'train_shots': shots_list,
                         'mu_error': mu_res,
                         'sig_error': sig_res,
                         'model_file': model_filenames, 'loss_file': loss_filenames})

    if save_raw_results:
        data = data.assign(raw_test_results=test_results)

    if save_data:
        i = 0
        filename = 'test_data/test_data_ver_{}.pickle'.format(i)
        while os.path.isfile(filename):
            i += 1
            filename = 'test_data/test_data_ver_{}.pickle'.format(i)

        data.to_pickle(filename)

    return data

# ########################## Bookkeeping functions ###############################################


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def return_obj_from_pickle(filename):
    f = open(filename, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj


def add_distributions(d1, d2=None):
    if not d2:
        d2 = normalise(torch.rand(len(d1)))/6
    return normalise(d1+d2)


def ret_crit_name(criterion):

    if 'MSE' in str(criterion):
        return 'mse'
    elif 'BCE' in str(criterion):
        return 'bce'
    else:
        return str(criterion)


def restrict_models(models, model_files, loss_files, atr, atr_val):
    out_models = []
    out_model_files, out_loss_files = [], []
    for indx, m in enumerate(models):
        if eval("m." + atr) == atr_val:
            out_models.append(m)
            out_model_files.append(model_files[indx])
            out_loss_files.append(loss_files[indx])

    return out_models, out_model_files, out_loss_files


def save_model_and_loss(model, loss, n_qubits, q_depth, criterion):

    crit_name = ret_crit_name(criterion)
    i = 0
    t = model.model_info['model']
    filename = 'pickled_models/model_{}_{}_{}_{}_{}_logk_{}_ver_{}.pickle'.format(t, n_qubits, q_depth,
                                                                                crit_name, model.reg_size,
                                                                                model.train_logistic_k, i)

    while os.path.isfile(filename):
        i += 1
        filename = 'pickled_models/model_{}_{}_{}_{}_{}_logk_{}_ver_{}.pickle'.format(t, n_qubits, q_depth,
                                                                            crit_name, model.reg_size,
                                                                            model.train_logistic_k, i)

    dbfile = open(filename, 'wb')
    pickle.dump(model, dbfile)
    dbfile.close()

    dbfile = open('pickled_models/loss_evol_{}_{}_{}_{}_{}_logk_{}_ver_{}.pickle'.format(t, n_qubits, q_depth,
                                                                            crit_name, model.reg_size,
                                                                            model.train_logistic_k, i), 'wb')
    # source, destination
    pickle.dump(loss, dbfile)
    dbfile.close()


# Used for multiprocessing
def process(arguments):

    Model, input_size, N, reg_size, criterion, q_depth, gate_type, q_delta, num_iterations, \
    step, logistic_k, batch_size, noisy_train, weights = arguments

    print(" ------------- criterion: {}, q_depth: {} -------- ".format(criterion, q_depth))
    model = Model(input_size=input_size, q_depth=q_depth, n_qubits=n_qubits, q_delta=q_delta,
                  reg_size=reg_size, gate_type=gate_type)
    optimizer = optim.Adam(model.parameters(), lr=step)

    model, loss_evol = train_model(model, criterion, optimizer, num_iterations=num_iterations, N=N,
                                   weights=weights, logistic_k=logistic_k, batch_size=batch_size,
                                   noisy_train=noisy_train)
    save_model_and_loss(model, loss_evol, n_qubits=n_qubits, q_depth=q_depth, criterion=criterion)
    print(" Finished ---- criterion: {}, q_depth: {} -------- ".format(criterion, q_depth))


# ########################## Analysis functions ###############################################

def gaussian_grid(N, complex=False, fuzz=False):
    range_mean = np.arange(-N / 2.1, N / 2.1, N / 35.0)
    range_std = np.arange(N / 30.0, N / 3, N / 100)
    yield (len(range_std), len(range_mean))
    for std in range_std:
        for mean in range_mean:
            if complex:
                d = normalise(gaussian_distr(N, std_dev=std, mean=mean,
                                             range_distr=[-N / 2, N / 2])).numpy() * random_phases(N)
            elif fuzz:
                d = normalise(add_distributions(gaussian_distr(N, std_dev=std, mean=mean,
                                             range_distr=[-N / 2, N / 2]))).numpy() * random_phases(N)
            else:
                d = normalise(gaussian_distr(N, std_dev=std, mean=mean, range_distr=[-N / 2, N / 2]))
            yield d, mean, std


def heat_map_mu_sig(model, complex=False, fuzz=False, ret_entanglement=False):
    try:
        reset_q_num(model.n_qubits, set_gate_type=model.gate_type)
    except AttributeError:
        reset_q_num(model.n_qubits)
    try:
        if model.binary:
            reg_size = 'binary'
        else:
            reg_size = model.reg_size
    except AttributeError:
        reg_size = model.reg_size

    N = model.input_size
    d_gen = gaussian_grid(N, complex=complex, fuzz=fuzz)
    dim = next(d_gen)
    print("Dimensions:", dim)
    h_map_std, h_map_mean = [], []
    entan_mean_map, entan_std_map = [], []
    # Added the tqdm so get rid of it if it doesnt work
    for dd in tqdm(d_gen):
        d, m, s = dd
        # print(m, s)
        out = model(d, save_state=True)[0].detach().numpy()
        mu, sig = out_to_mu_sig(out, model.input_size, reg_size)
        m_er = mse_error_mean(mu, m) ** 0.5
        s_er = mse_error_mean(sig, s) ** 0.5
        h_map_mean.append(m_er.float())
        h_map_std.append(s_er.float())
        if ret_entanglement:
            m_state, std_state = model.last_state
            entan_mean_map.append(average_entan(model.n_qubits, m_state)[0])
            entan_std_map.append(average_entan(model.n_qubits, std_state)[0])

    h_map_mean = np.flip(np.array(h_map_mean).reshape(dim), axis=0)
    h_map_std = np.flip(np.array(h_map_std).reshape(dim), axis=0)

    if ret_entanglement:
        entan_mean_map = np.flip(np.array(entan_mean_map).reshape(dim), axis=0)
        entan_std_map = np.flip(np.array(entan_std_map).reshape(dim), axis=0)
        return h_map_mean, h_map_std, entan_mean_map, entan_std_map

    return h_map_mean, h_map_std


def meas_reg_entan(n_qubits, state, reg_q_list):
    """ Function for measuring the entanglement between registers """
    reg_q_list = sorted(reg_q_list)
    len_col = 2 ** len(reg_q_list)
    len_row = 2 ** (n_qubits - len(reg_q_list))
    mat = np.zeros((len_row, len_col), dtype=np.complex128)
    for indx in range(len(state)):
        bin_rep = indx_to_bin_list(n_qubits, indx)
        i, col, row = 0, [], []
        for i in range(len(bin_rep)):
            if i in reg_q_list:
                col.append(bin_rep[i])
            else:
                row.append(bin_rep[i])
        col = "".join(col)
        row = "".join(row)
        mat[int(row, 2)][int(col, 2)] = state[indx]

    _, s, _ = np.linalg.svd(mat, full_matrices=False)
    p = s*s
    return -1 * sum([pp*np.log2(pp) if pp!=0 else 0 for pp in p])


def average_entan(n_qubits, state):
    entan = []
    tot_entan = 0.0
    for i in range(1, n_qubits):
        en = meas_reg_entan(n_qubits, state, reg_q_list=list(range(i)))
        if np.isnan(en):
            print('Error')
            continue
        tot_entan = tot_entan + en
        entan.append(en)

    return tot_entan/(n_qubits-1), entan


def indx_to_bin_list(n_qubits, indx):
    out = list(bin(indx))[2:]
    return ['0'] * (n_qubits - len(out)) + out


def check_model_compl(n_qubits, model_type, reg_size, q_depth, criterion, logistic_k):

    crit_name = ret_crit_name(criterion)

    if reg_size == 'binary':
        reg_size = n_qubits

    file = 'model_{}_{}_{}_{}_{}_logk_{}_ver'.format(model_type, n_qubits, q_depth,
                                                                                crit_name, reg_size,
                                                                                logistic_k)
    for filename in glob.glob("pickled_models/*.pickle*"):
        if file in filename:
            return True

    return False

# #################################### Main ###############################################


def run_main_script(check_if_done=False, noisy_train=False, model=None):

    start = time.time()

    print("Number of CPUs: {} --------------------------------------------------------------\n\n".format(cpu_count()))

    input_size = 2 ** n_qubits
    N = input_size
    step = 0.008
    q_delta = 0.01
    reg_size = 'binary'
    gate_type = 'ry'
    num_iterations = 4000
    logistic_ks = [9]#[None, 11, 13, 1, 3, 5, 7, 9, 15, 20]
    batch_size = 2

    init_weight = 10
    if reg_size == 'binary':
        weights = torch.tensor([init_weight * (0.5) ** i for i in range(n_qubits)])
        weights = torch.cat((weights, weights))
    else:
        weights = torch.tensor([init_weight*(0.1)**i for i in range(reg_size)])
        weights = torch.cat((weights, weights))

    if reg_size == 'binary':
        Models = [CharacteristicsSeparateDecoder]
    else:
        Models = [CharacteristicsDecoder, CharacteristicsSeparateDecoder]
    if model:
        Models = [model]
    # criterions = [nn.BCELoss(weight=weights)]
    criterions = [MSELoss]
    all_arguments = []

    for Model in Models:
        for criterion in criterions:
            # for q_depth in np.linspace(n_qubits//3, max_q_depth, 10, endpoint=True):
            for q_depth in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
                for logistic_k in logistic_ks:
                    q_depth = int(q_depth)
                    if check_if_done and check_model_compl(n_qubits, Model.name, reg_size,
                                                           q_depth, criterion, logistic_k):
                        print("this is done:", n_qubits, Model.name, reg_size, q_depth, criterion,
                              logistic_k, '\n')
                        continue
                    all_arguments.append([Model, input_size, N, reg_size, criterion, q_depth, gate_type,
                                          q_delta, num_iterations, step, logistic_k, batch_size,
                                          noisy_train, weights])
    print('NUMBER OF MODELS: {}'.format(len(all_arguments)))
    all_arguments_heading = "[Model, input_size, N, reg_size, criterion, " \
                            "q_depth, q_delta, num_iterations, step, logistic_k, batch_size, noisy_train]"
    print('Going to start all the jobs. Pooled jobs:\n' + '\n--> '.join([all_arguments_heading]+[str(i[:-1]) for i in all_arguments]))
    print('\n\n')

    p = Pool(70)
    p.map(process, all_arguments)

    # indx = -6
    # print("Doing", all_arguments[indx])
    # process(all_arguments[indx])

    print('\n\n============================================================================\n\n')
    print("Jobs done:\n")
    print('All the jobs. Pooled jobs:\n' + '\n--> '.join([all_arguments_heading]+[str(i[:-1]) for i in all_arguments]))
    print("Overall time taken: {}".format(time.time()-start))


def batch_comparison_script():
    num_iterations = 15000
    all_arguments = []
    for b in range(1, 9):
        for n in [1, 'binary']:
            all_arguments.append([b, n, int(num_iterations/b)])

    print(all_arguments)

    p = Pool(30)
    p.map(batch_process, all_arguments)

    print("\n\n----------------- Batch script done. ---------------------------")


def batch_process(arguments):

    b, reg_size, num_iter = arguments
    single_model_script(batch=b, num_iter=num_iter, reg_s=reg_size)


def single_model_script(batch=None, num_iter=None, reg_s=None, set_logistic_k=None,
                        set_shots=None, regular_term=None):
    start = time.time()

    print("Number of CPUs: {} --------------------------------------------------------------\n\n".format(cpu_count()))
    n_qubits = 5
    input_size = 2 ** n_qubits
    N = input_size
    step = 0.01
    q_delta = 0.01
    reg_size = 'binary'
    gate_type = 'ry'
    num_iterations = 3500
    logistic_k = None
    Model = CharacteristicsSeparateDecoder
    # Model = AdvancedDecoder
    # Model = EnhancedDecoder
    criterion = MSELoss
    q_depth = 6
    batch_size = 2
    noisy_train = False
    complex_distr = False
    shots = None

    if set_shots:
        shots = set_shots
    if batch:
        batch_size = batch
    if num_iter:
        num_iterations = num_iter
    if reg_s:
        reg_size = reg_s
    if set_logistic_k:
        logistic_k = set_logistic_k

    init_weight = 10

    if reg_size == 'binary':
        weights = torch.tensor([init_weight * (0.5) ** i for i in range(n_qubits)])
        weights = torch.cat((weights, weights))
    else:
        weights = torch.tensor([init_weight * (0.1) ** i for i in range(reg_size)])
        weights = torch.cat((weights, weights))

    if reg_size == 'binary':
        assert Model.name != 'CharacteristicsDecoder'

    print(" ------------- criterion: {}, q_depth: {} -------- ".format(criterion, q_depth))
    model = Model(input_size=input_size, q_depth=q_depth, n_qubits=n_qubits, q_delta=q_delta,
                  reg_size=reg_size, gate_type=gate_type, shots=shots)
    optimizer = optim.Adam(model.parameters(), lr=step)

    model, loss_evol = train_model(model, criterion, optimizer, num_iterations=num_iterations, N=N,
                                   weights=weights, logistic_k=logistic_k, batch_size=batch_size,
                                   noisy_train=noisy_train, complex_distr=complex_distr,
                                   shots=shots, regular_term=regular_term)
    save_model_and_loss(model, loss_evol, n_qubits=n_qubits, q_depth=q_depth, criterion=criterion)
    print(" Finished ---- criterion: {}, q_depth: {} -------- ".format(criterion, q_depth))

    print("Overall time taken: {}".format(time.time() - start))


def run_test_script(required, num_test=100, test_type='gaussian_sample', circuit_sample_num=None,
                    noisy_test=False, p_range_mean=None, p_range_std=None,
                    outlier_removal=False):
    models, model_filenames, loss_filenames = load_models(required)
    data = test_and_dataframe(num_test, models, model_filenames, loss_filenames,
                              save_data=True, test_type=test_type, circuit_sample_num=circuit_sample_num,
                              noisy_test=noisy_test, p_range_mean=p_range_mean, p_range_std=p_range_std,
                              outlier_removal=outlier_removal)

    return data


def test_various_range(required, num_test=100, test_type='gaussian_sample',
                       p_mean_ranges=None, p_std_ranges=None, outlier_removal=False):
    dataframe_list = []
    if p_mean_ranges == None:
        p_mean_ranges = [ [-0.5, 0.5], [-0.4, 0.4], [-0.3, 0.3], [-0.2, 0.2], [-0.1, 0.1] ]
    if p_std_ranges == None:
        p_std_ranges = [ [0.005, 0.9], [0.005, 0.7], [0.02, 0.5], [0.03, 0.3], [0.04, 0.15] ]

    models, model_filenames, loss_filenames = load_models(required)
    assert len(p_mean_ranges) == len(p_std_ranges), "This is what it is programmed for."

    for i in range(len(p_std_ranges)):
        print("Range testing {} of {}.".format(i, len(p_std_ranges)))
        dataframe_list.append(test_and_dataframe(num_test, models, model_filenames, loss_filenames,
                                                 save_data=False, test_type=test_type,
                                                 noisy_test=False,
                                                 p_range_mean=p_mean_ranges[i],
                                                 p_range_std=p_std_ranges[i],
                                                 outlier_removal=outlier_removal))

    data = pd.concat(dataframe_list)
    data = data.assign(p_range_mean=list(itertools.chain.from_iterable([[i]*len(models) for i in p_mean_ranges])))
    data = data.assign(p_range_std=list(itertools.chain.from_iterable([[i]*len(models) for i in p_std_ranges])))

    i = 0
    filename = 'test_data/test_data_ver_{}.pickle'.format(i)
    while os.path.isfile(filename):
        i += 1
        filename = 'test_data/test_data_ver_{}.pickle'.format(i)

    data.to_pickle(filename)

    return data


def sample_com_process(argument):
    reg_term = 0.01

    shots, log_k = argument
    single_model_script(set_logistic_k=log_k, set_shots=shots, regular_term=reg_term)


def sample_comp_script():

    all_arguments = []
    for shots in [None, 50, 100, 200, 400, 600, 800, 1000, 1200]:
        for log_k in [None, 5, 11]:
            all_arguments.append([shots, log_k])

    print("Number of models:", len(all_arguments))
    print(all_arguments)
    print("\n\n")

    p = Pool(32)
    p.map(sample_com_process, all_arguments)


def continue_training(model_filename, num_iterations, batch_size=1):

    model = return_obj_from_pickle(model_filename)
    reset_q_num(model.n_qubits)

    start = time.time()
    criterion = MSELoss
    step = 0.01
    optimizer = optim.Adam(model.parameters(), lr=step)

    init_weight = 10
    if model.binary:
        weights = torch.tensor([init_weight * (0.5) ** i for i in range(model.n_qubits)])
        weights = torch.cat((weights, weights))
    else:
        weights = torch.tensor([init_weight * (0.1) ** i for i in range(model.reg_size)])
        weights = torch.cat((weights, weights))

    model, loss_evol = train_model(model, criterion, optimizer, num_iterations=num_iterations, N=model.N,
                                   weights=weights, logistic_k=model.train_logistic_k, batch_size=batch_size)
    save_model_and_loss(model, loss_evol, n_qubits=model.n_qubits, q_depth=model.q_depth, criterion=criterion)
    print(" Finished ---- criterion: {}, q_depth: {} -------- ".format(criterion, model.q_depth))

    print("Overall time taken: {}".format(time.time() - start))


if __name__ == '__main__':

    # ------------------------- Training scripts --------------------------------------------

    # single_model_script(regular_term=0.01)

    # run_main_script(check_if_done=False, noisy_train=False)

    sample_comp_script()

    # batch_comparison_script()

    # ----------------------------- Testing -------------------------------------------------
    # num_test = 300
    # required = ['jobs_date_26_09_2020']
    # data = run_test_script(required=required, num_test=num_test, outlier_removal=True)
    # data = test_various_range(required=required, num_test=num_test, outlier_removal=True)