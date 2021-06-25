
import sys
import pandas as pd
import itertools
import tqdm
import os
import qnn_gate_types_and_depth as vqc
from gaussian_decoding import create_gaussian_distributions, gen_rand_gaussian

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

# Other tools
import time
import pickle
from math import factorial


dev = None
NOISY = False


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


def ret_entan_states(N, entan_type):

    # Todo
    pass


def peturb_gaussian_distributions(N, degree=1, list_mean=None, list_std=None):

    assert degree >= 0

    gaussians, means, stds = ret_gaussian_distributions(N, list_mean, list_std)
    chars = [np.unique(np.array(means)), np.unique(np.array(stds))]

    height = np.max(np.array(gaussians[0]))/2
    # chars.append(np.linspace(-height/2, height/2, 5))
    chars.append(np.linspace(-height/4, height/2, 5))

    factor = N
    # factor = 2*N
    while degree > 0:
        chars.append(np.linspace(-height/factor, height/factor, 5))
        factor = factor*factor
        degree -= 1

    distributions = []
    out_char = []
    for comb in itertools.product(*chars):
        gaus = vqc.normalise(vqc.gaussian_distr(N, std_dev=comb[1], mean=comb[0]))

        distributions.append(vqc.normalise(restrict_pos(torch.tensor(gaus) * ret_poly(N, comb[2:]))))
        if torch.any(np.isnan(distributions[-1])):
            print('Error with mean={} and std={} and ret_poly_char={}'.format(comb[0], comb[1], comb[2:]))
            distributions.pop()
            continue
        out_char.append(distr_characterisation(distributions[-1]))
        # print(out_char[-1])
        # plt.plot(distributions[-1])
        # plt.show()

    # print(len(out_char), len(distributions))
    return distributions, out_char


def distr_characterisation(d, type='finance_problem', range_distr=None):

    N = len(d)
    if not range_distr:
        range_distr = [-N / 2, N / 2]

    x = torch.tensor(np.linspace(range_distr[0], range_distr[1]-1, N))

    if type == 'finance_problem':
        mean = sum((x*d)/d.sum())
        peak = x[torch.argmax(d)]
        return mean, peak
    else:
        print("Distribution characterisation not available.")
        exit(2)


def ret_poly(N, chars, range_distr=None):

    if not range_distr:
        range_distr = [-N / 2, N / 2]
    x = np.linspace(range_distr[0], range_distr[1], N)

    out = 0
    for i, c in enumerate(chars):
        out = out + c * x**i

    return torch.tensor(out)


def restrict_pos(a):

    if isinstance(a, np.ndarray) or isinstance(a, torch.Tensor):
        a[a < 0] = 0

    return a


def generate_look_up(chars, paired=False):
    if paired:
        return {k:v for v, k in enumerate(chars)}
    return {k:v for v, k in enumerate(zip(*chars))}


def distr_make_generator(d, c):
    for i in range(len(d)):
        yield d[i], c[i]


class Distributions:
    """
    Check OneNote for a thorough explanation
    """
    def __init__(self, in_size, N, num_distr=None, iterations=None, d_type='gaussian', list_mean=None, list_std=None,
                 order=None, distr=None, entan_type=None):

        self.char_size = in_size
        self.N = N
        self.x = torch.tensor(np.linspace(-N/2, (N/2)-1, N))
        self.type = d_type
        self.num_distr = num_distr
        self.num_iterations = iterations
        self.num_iter_left = iterations
        self.order = order
        self.d_type = d_type

        self.mean_list = list_mean
        self.std_list = list_std

        if d_type == 'gaussian':
            assert in_size == 2
        if d_type == 'poly_pert_gaussian':
            self.degree_perturb = in_size - 3

        if order == 'random':
            if d_type == 'gaussian':
                self.distr_list, self.temp_mean_list, self.temp_std_list = ret_gaussian_distributions(N, list_mean, list_std)
        elif order == 'weighted':
            if d_type == 'gaussian':
                self.distributions, self.mean_list, self.std_list = ret_gaussian_distributions(N, list_mean, list_std)
                self.tot_num_distr = len(self.mean_list)
                self.look_up_char = generate_look_up((self.mean_list, self.std_list))
            if d_type == 'poly_pert_gaussian':
                self.distributions, self.chara = peturb_gaussian_distributions(N, degree=self.degree_perturb,
                                                                               list_mean=list_mean,
                                                                               list_std=list_std)
                self.tot_num_distr = len(self.distributions)
                self.look_up_char = generate_look_up(self.chara, paired=True)

            self.loss_weighting = [1.0 / self.tot_num_distr] * self.tot_num_distr
            self.loss_account = [100] * self.tot_num_distr
            self.count_dist_train = np.array([0] * self.tot_num_distr)
        elif order == 'constant':
            self.distribution = distr
            self.m = vqc.get_mean(distr)
            self.s = vqc.get_std_dev(distr)
        else:
            if d_type == 'gaussian':
                self.distributions, mm, ss = ret_gaussian_distributions(N, list_mean, list_std)
                self.mean_list = mm
                self.std_list = ss
                self.look_up_char = generate_look_up((mm, ss))
                self.distr_gen = create_gaussian_distributions(N, list_mean, list_std)
                self.tot_num_distr = next(self.distr_gen)
            elif d_type == 'poly_pert_gaussian':
                self.distributions, self.chara = peturb_gaussian_distributions(N, degree=self.degree_perturb,
                                                                               list_mean=list_mean,
                                                                               list_std=list_std)
                self.distr_gen = distr_make_generator(self.distributions, self.chara)
                self.tot_num_distr = len(self.distributions)
                self.look_up_char = generate_look_up(self.chara, paired=True)
            elif d_type == 'entan_states':
                self.distributions, self.entan_classes = ret_entan_states(N, entan_type=entan_type)
                self.look_up_char = generate_look_up(self.entan_classes)
                self.distr_gen = distr_make_generator(self.distributions, self.entan_classes)

            self.loss_account = [100] * self.tot_num_distr
            self.count_dist_train = np.array([0] * self.tot_num_distr)

    def get(self, p_fuzz=None):
        # p_fuzz not implemented for this function

        if self.order == 'random':
            try:
                i = random.randint(0, len(self.distr_list))
                return self.distr_list.pop(i), self.temp_mean_list.pop(i), self.temp_std_list.pop(i)
            except IndexError:
                self.distr_list, self.temp_mean_list, self.temp_std_list = ret_gaussian_distributions(self.N, self.mean_list,
                                                                                            self.std_list)
                return self.get()
        elif self.order == 'weighted':
            indx = np.random.choice(self.tot_num_distr, p=self.loss_weighting)
            if self.d_type == 'gaussian':
                return self.distributions[indx], self.mean_list[indx], self.std_list[indx]
            elif self.d_type == 'poly_pert_gaussian':
                return self.distributions[indx], self.chara[indx]
        elif self.order == 'constant':
            return self.distribution, self.m, self.s
        else:
            try:
                return next(self.distr_gen)
            except (StopIteration, TypeError):
                print("New epoch")
                if self.d_type == 'gaussian':
                    self.distr_gen = create_gaussian_distributions(self.N, self.mean_list, self.std_list)
                    size_distr = next(self.distr_gen)
                    return self.get()
                elif self.d_type == 'poly_pert_gaussian':
                    self.distr_gen = distr_make_generator(self.distributions, self.chara)
                    return self.get()
                elif self.d_type == 'entan_states':
                    self.distr_gen = distr_make_generator(self.distributions, self.entan_classes)
                    return self.get()

    def get_random(self, p_fuzz=None):

        if self.d_type == 'gaussian':
            indx = np.random.randint(len(self.distributions))
            return self.distributions[indx], self.mean_list[indx], self.std_list[indx]
        if self.d_type == 'poly_pert_gaussian':
            indx = np.random.randint(len(self.distributions))
            if p_fuzz and np.random.random() < p_fuzz:
                dd = vqc.normalise(self.distributions[indx] * torch.tensor(np.random.random(self.N)))
                return dd, [sum((self.x*dd)/sum(dd)), self.x[torch.argmax(dd)]]
            return self.distributions[indx], self.chara[indx]
        if self.d_type == 'etan_states':
            indx = np.random.randint(len(self.distributions))
            return self.distributions[indx], self.entan_classes[indx]

    def save_last_loss(self, loss, chars):

        if self.order == None:
            return
        try:
            indx = self.look_up_char[chars]
        except AttributeError:
            print("Problem saving loss.")
            return
        self.loss_account[indx] = loss.detach().numpy()
        self.count_dist_train[indx] = self.count_dist_train[indx] + 1
        if self.order == 'weighted':
            self.calculate_weighting()

    def distr_from_cumul(self, c):

        assert len(c) == self.char_size

        k = np.linspace(-self.N/2, self.N/2, self.N)

        log_pk = np.zeros(self.N)
        for n in range(self.char_size):
            log_pk = log_pk + (c[n]*((-1j*k)**n))/factorial(n)
        pk = np.exp(log_pk)
        px = np.abs(np.fft.ifft(pk))
        return px

    def calculate_weighting(self):
        weight_power = 0.5
        weighting = (np.array(self.loss_account) ** weight_power) * \
                    (1 - self.count_dist_train/(1+sum(self.count_dist_train)))

        self.loss_weighting = weighting / sum(weighting)

    def ret_distr_info(self):

        if self.order == 'weighted':
            if self.d_type == 'poly_pert_gaussian':
                return pd.DataFrame({'characteristics': self.chara,
                                 'prob_weighting': self.loss_weighting,
                                 'last_loss': self.loss_account,
                                 'tot_trained_number': self.count_dist_train})
            else:
                return pd.DataFrame({'mean': self.mean_list, 'std': self.std_list,
                                     'prob_weighting': self.loss_weighting,
                                     'last_loss': self.loss_account,
                                     'tot_trained_number': self.count_dist_train})


class FullEncoder(nn.Module):

    name = 'FullEncoder'

    def __init__(self, in_size, n_qubits, num_params, gate_type='ry'):

        super().__init__()

        self.in_size = in_size
        self.n_qubits = n_qubits
        self.num_params = num_params
        self.gate_type = gate_type
        self.q_circ = None

        if gate_type == 'ry':
            print('nuber of qubits:', n_qubits)
            assert num_params % self.n_qubits == 0
            q_depth = num_params/self.n_qubits
            self.q_depth = int(q_depth)
            self.qcircuit = 'q_net_ry_entan'
        if gate_type == 'own':
            print("Number of qubits:", n_qubits)
            assert (num_params-n_qubits) % 3 == 0
            self.qcircuit = ''

        # Learning parameters
        self.c_layer1 = nn.Linear(in_size, 100)
        self.c_layer2 = nn.Linear(100, 80)
        self.c_layer3 = nn.Linear(80, num_params*2)
        self.c_out = nn.Linear(num_params*2, num_params)

        self.train_time = None
        self.distributions = None
        self.tot_loss = None
        self.training_error = None
        self.noisy_train = None
        self.batch_size = None

    def forward(self, x):
        x.to(vqc.device)
        x = torch.sigmoid(self.c_layer1(x))
        x = torch.sigmoid(self.c_layer2(x))
        x = torch.sigmoid(self.c_layer3(x))
        x = torch.tanh(self.c_out(x)) * np.pi / 2.0

        # obs_gen = vqc.proj_meas_gen(self.n_qubits)
        op = None
        if self.gate_type == 'ry':
            q_out = vqc.q_net_ry_entan(x, q_depth=self.q_depth, op=op, n_qubits=self.n_qubits).unsqueeze(0).to(vqc.device)
        if self.gate_type == 'own':
            q_out = vqc.q_net_own(x, n_qubits=self.n_qubits, op=op).unsqueeze(0).to(vqc.device)

        return q_out[0]

    def save_distr_obj(self, d):
        self.distributions = d

    def test(self, chars, distr_type='gaussian', return_distr=False, criterion=vqc.MSELoss):
        d = self(torch.tensor(chars)).detach()
        if distr_type == 'gaussian':
            distr = vqc.gaussian_distr(2**self.n_qubits, mean=chars[0], std_dev=chars[1])
        else:
            raise ValueError("This distribution type has not yet been programmed")

        if return_distr:
            return float(criterion(d, distr)), d.numpy()
        else:
            return float(criterion(d, distr))


def train_full_encoder(model, criterion, distributions, num_iterations, loss_evol=None, step=0.001, batch_size=1,
                       encoding_type='amplitudes'):

    since = time.time()
    print("Training started: noisy={}\ndevice={}\n".format(vqc.NOISY, vqc.dev))
    model.noisy_train = vqc.NOISY
    optimizer = optim.Adam(model.parameters(), lr=step)

    if not loss_evol:
        loss_evol = []

    if isinstance(distributions, np.ndarray):
        distributions = Distributions(model.in_size, 2**model.n_qubits, order='constant', distr=distributions)

    for ii in range(num_iterations):
        print('-----iteration {}--model:{}------'.format(ii, model))
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            d, m, s = distributions.get()
            # Probabilities are coming out of this model
            output = model(torch.tensor([m, s]))
            if encoding_type == 'amplitudes':
                output = output**0.5
                loss = criterion(output, vqc.normalise(d).double())
            elif encoding_type == 'probabilities':
                loss = criterion(output, vqc.normalise_prob(d).double())
            for i in range(batch_size-1):
                d, m, s = distributions.get()
                output = model(torch.tensor([m, s]))
                loss = loss + criterion(output, d.double())

            distributions.save_last_loss(loss, (m, s))

            print("Loss", loss)
            loss_evol.append(loss)
            loss.backward()
            optimizer.step()

    distributions.distr_gen = None # Cant pickle generators
    model.save_distr_obj(distributions)
    model.train_time = time.time() - since
    model.training_error = [vqc.PHASE_FLIP_ERROR, vqc.BIT_FLIP_ERROR]
    model.batch_size = batch_size
    print("Time taken to train model: {}".format(time.time() - since))

    return model, loss_evol


def gaussian_test_full_encoders(models, num_test, range_mean=None, range_std_dev=None, range_distr=None, complex=False):

    # Must make sure that in order to compare the encoders, they must have the same number of qubits
    n = models[0].n_qubits
    for m in models[1:]:
        assert m.n_qubits == n, "Models must have the same number of qubits for comparison"

    results = []
    distr_gen = gen_rand_gaussian(n, range_mean=range_mean, range_std_dev=range_std_dev,
                                        range_distr=range_distr, complex=complex)
    for ii in range(num_test):
        print("----- Test number {} -------".format(ii))
        distr, m, s = next(distr_gen)
        print("Mean = {},  Std = {}".format(m, s))
        loss = [mod.test([m, s]) for mod in models]
        results.append(loss)
        print("Loss = {}".format(loss))
    results = np.array(results)

    return results.mean(axis=0)


def risk_test_encoder(model, num_test, range_mean=None, range_std_dev=None, range_distr=None):
    """ Returns the risk which is the sqrt(MSE): defined in thesis. """

    tot_loss = 0
    loss_list = []
    distr_gen = gen_rand_gaussian(model.n_qubits, range_mean=range_mean, range_std_dev=range_std_dev,
                                  range_distr=range_distr)

    vqc.reset_q_num(model.n_qubits)
    for ii in tqdm.tqdm(range(num_test)):
        # print("----- Test number {} -------".format(ii))
        distr, m, s = next(distr_gen)
        loss_list.append(np.sqrt(model.test([m, s])))
        tot_loss += loss_list[-1]

    return tot_loss/num_test, loss_list


def test_dataframe_encoder(num_test, models, model_filenames, loss_filenames, save_data=True):

    n_qubit_list, risk_list, num_params_list, crit_list = [], [], [], []
    raw_test_results, batch_size_list = [], []
    for i, model in enumerate(models):
        print("Testing model: {}, noisy={}".format(model_filenames[i], model.noisy_train))
        vqc.reset_q_num(model.n_qubits, noisy=model.noisy_train)

        risk, results = risk_test_encoder(model, num_test)
        print("Risk=", risk)
        risk_list.append(risk)
        n_qubit_list.append(model.n_qubits)
        num_params_list.append(model.num_params)
        if 'mse' in model_filenames[i]:
            crit_list.append('mse')
        else:
            crit_list.append('bce')

        raw_test_results.append(results)
        try:
            batch_size_list.append(model.batch_size)
        except AttributeError:
            batch_size_list.append(None)

    data = pd.DataFrame({'model': models,
                         'n_qubits': n_qubit_list,
                         'Criterion': crit_list,
                         'num_params': num_params_list,
                         'risk': risk_list,
                         'batch_size':batch_size_list,
                         'raw_test_results': raw_test_results,
                         'model_file': model_filenames, 'loss_file': loss_filenames})

    if save_data:
        i = 0
        filename = 'test_data/test_data_ver_{}.pickle'.format(i)
        while os.path.isfile(filename):
            i += 1
            filename = 'test_data/test_data_ver_{}.pickle'.format(i)

        data.to_pickle(filename)

    return data


def state_to_distr(state):
    return abs(state)


def main_process(arguments):

    step = 0.001
    batch_size = 2
    n_qubits, in_size, num_params, gate_type, distributions, num_iterations, criterion = arguments

    model = FullEncoder(in_size, n_qubits, num_params, gate_type)
    vqc.reset_q_num(model.n_qubits, noisy=vqc.NOISY)
    model, loss_evol = train_full_encoder(model, criterion, distributions, num_iterations, step=step,
                                          batch_size=batch_size)

    vqc.save_model_and_loss(model, loss_evol, criterion=criterion)


def run_main_script(check_if_done=False, noisy=False):
    try:
        n_qubits = int(sys.argv[1])
    except IndexError:
        n_qubits = 6
    vqc.reset_q_num(n_qubits, noisy=noisy)
    N = 2 ** n_qubits
    gate_types = ['ry', 'own']
    criterion = vqc.MSELoss
    num_iterations = 6000

    input_size = 2 ** n_qubits
    in_size = 2
    distrib_train = 'weighted'
    # batch_size = None

    distributions = Distributions(in_size=in_size, N=N, num_distr=100, iterations=num_iterations, d_type='gaussian',
                                  order=distrib_train)
    all_arguments = []
    last_p = None
    for gate_type in gate_types:
        for num_params in np.linspace(n_qubits, int(8 * np.sqrt(N)), 10):
            num_params = int(num_params)
            if gate_type == 'ry':
                num_params = num_params - int(num_params % n_qubits)
                if last_p == num_params or num_params < n_qubits:
                    continue
                last_p = num_params
            if gate_type == 'own':
                f = int(vqc.math.ceil((num_params - n_qubits) / 3.0))
                num_params = int(f * 3) + n_qubits
            if check_if_done and vqc.check_model_compl(n_qubits, num_params, gate_type, criterion):
                print("this is done:", n_qubits, num_params, gate_type, criterion, '\n')
                continue
            all_arguments.append([n_qubits, in_size, num_params, gate_type, distributions,
                                  num_iterations, criterion])

    print("Number of models: {}".format(len(all_arguments)))
    print("[n_qubits, in_size, num_params, gate_type, distributions, num_iterations, criterion]")
    print(all_arguments)

    p = vqc.Pool(22)
    p.map(main_process, all_arguments)

    # main_process(all_arguments[-1])


def order_train_comp():

    orders = ['random', None, 'weighted']*5

    p = vqc.Pool(4)
    p.map(single_model, orders)


def single_model(order='weighted'):

    NOISY = False
    n_qubits = 6
    vqc.reset_q_num(n_qubits, noisy=NOISY)
    N = 2 ** n_qubits
    num_params = 36
    gate_type = 'ry'
    # criterion = nn.BCELoss()
    criterion = vqc.MSELoss
    num_iterations = 6000
    step = 0.001
    input_size = 2 ** n_qubits
    in_size = 2

    batch_size = 2

    distributions = Distributions(in_size=in_size, N=N, num_distr=100, iterations=num_iterations, d_type='gaussian',
                                  list_std=np.arange(1, 9, 2), order=order)

    print(distributions.order)
    # -------------------------------------------------------------------------------------

    model = FullEncoder(in_size, n_qubits, num_params, gate_type)
    model, loss_evol = train_full_encoder(model, criterion, distributions, num_iterations, step=step,
                                          batch_size=batch_size)
    vqc.save_model_and_loss(model, loss_evol, criterion=criterion)

    # -------------------------------------------------------------------------------------
    """ The following is for continuing to train a model (uncomment below and comment above for use) """


    # model = return_obj_from_pickle("pickled_models/model_ry_5_20_mse_ver_0.pickle")
    # loss_evol = return_obj_from_pickle("pickled_models/loss_evol_ry_5_20_mse_ver_0.pickle")
    #
    # model, loss_evol = train_full_encoder(model, criterion, distributions, num_iterations, step=step,
    #                                       batch_size=4, loss_evol=loss_evol)
    #
    # save_model_and_loss(model, loss_evol, criterion=criterion)


def testing_script(required, num_test):
    models, model_filenames, loss_filenames = vqc.load_models(required)
    data = test_dataframe_encoder(num_test, models, model_filenames, loss_filenames, save_data=True)

    return data


if __name__ == '__main__':

    # single_model()

    # run_main_script()

    # order_train_comp()

    num_test = 220
    required = ['jobs_date_05_09_2020']
    data = testing_script(required=required, num_test=num_test)

