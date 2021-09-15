
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from required_modules.amplitude_encoding import *
from required_modules import gaussian_decoding as gd
import tqdm

def standardize_data(X):
    '''
    This function standardize an array, its substracts mean value,
    and then divide the standard deviation.

    param 1: array
    return: standardized array
    '''
    rows, columns = X.shape

    standardizedArray = np.zeros(shape=(rows, columns))
    tempArray = np.zeros(rows)

    for column in range(columns):

        mean = np.mean(X[:, column])
        std = np.std(X[:, column])
        tempArray = np.empty(0)

        for element in X[:, column]:
            tempArray = np.append(tempArray, ((element - mean) / std))

        standardizedArray[:, column] = tempArray

    return standardizedArray


def reset_dec_ry_q_net(n_qubits):

    dev = qml.device("default.qubit", wires=n_qubits)

    global dec_ry_q_net

    @qml.qnode(dev, interface="torch")
    def dec_ry_q_net(q_weights_flat, amplitudes=None, amplitude_wires=None, q_depth=None,
                     reg_size=None, n_qubits=None):
        """ Going from quantum state to latent space. """
        if not amplitude_wires:
            amplitude_wires = list(range(n_qubits))

        qml.QubitStateVector(amplitudes, wires=amplitude_wires)

        # Reshape weights
        q_weights = q_weights_flat.reshape(q_depth, n_qubits)

        vqc.H_layer(n_qubits)

        # Sequence of trainable variational layers
        for k in range(q_depth):
            vqc.RY_layer(q_weights[k])
            vqc.entangling_layer(n_qubits)

        exp_vals = [qml.expval(qml.PauliZ(i)) for i in range(reg_size)]
        return tuple(exp_vals)


def reset_enc_train_ry_q_net(n_qubits):

    dev = qml.device("default.qubit", wires=n_qubits)

    global enc_train_ry_q_net

    @qml.qnode(dev, interface="torch")
    def enc_train_ry_q_net(q_weights_flat, amplitudes=None, init_rot=None, q_depth=None, n_qubits=None):
        """ Used for training that performs the swap test. The name says encoder but super confusing!!
            Everything however seems to work, just dont touch the code. """
        qml.QubitStateVector(amplitudes, wires=list(range(n_qubits + 1, n_qubits*2+1)))
        # Reshape weights
        q_weights = q_weights_flat.reshape(q_depth, n_qubits)

        # Change all these
        vqc.H_layer(n_qubits+1)
        # RY_layer(init_rot, ancilla=True)

        # Sequence of trainable variational layers
        for k in range(q_depth):
            vqc.RY_layer(q_weights[k], ancilla=True)
            vqc.entangling_layer(n_qubits, ancilla=True)

        # perform the SWAP test
        # The Hadammard has already been applied in the H_layer and is the 0 qubit
        for k in range(1, n_qubits + 1):
            qml.CSWAP(wires=[0, k, n_qubits + k])

        qml.Hadamard(wires=0)

        return qml.expval(qml.PauliZ(0))


def reset_enc_ry_q_net(n_qubits):

    dev = qml.device("default.qubit", wires=n_qubits)

    global enc_ry_q_net

    @qml.qnode(dev, interface="torch")
    def enc_ry_q_net(q_weights_flat, init_rot=None, q_depth=None, n_qubits=None, op=None):
        """ Goes from latent space to quantum state. """
        # Reshape weights
        q_weights = q_weights_flat.reshape(q_depth, n_qubits)

        vqc.H_layer(n_qubits)

        # Sequence of trainable variational layers
        for k in range(q_depth):
            vqc.RY_layer(q_weights[k])
            vqc.entangling_layer(n_qubits)

        return qml.probs(wires=list(range(n_qubits)))


def kronecker(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))


class HQA(nn.Module):

    name = 'HQA'
    def __init__(self, n_qubits, latent_size, num_params_enc, num_params_dec, gate_type='ry',
                 interwoven=False, adv_decoder=False):

        super().__init__()

        self.latent_size = latent_size
        self.n_qubits = n_qubits
        self.num_params_enc = num_params_enc
        self.num_params_dec = num_params_dec
        self.gate_type = gate_type
        self.q_circ = None
        self.interwoven = interwoven
        self.train_p_fuzz = None
        self.adv_decoder = adv_decoder

        # Why is this assertion here? Uncomment if something goes wrong later
        if interwoven:
            assert latent_size > 2 * n_qubits
        else:
            assert latent_size >= n_qubits

        if gate_type == 'ry':
            assert num_params_dec % self.latent_size == 0

        self.params_enc = nn.Parameter(0.1 * torch.randn(self.num_params_enc))
        self.params_dec = nn.Parameter(0.1 * torch.randn(self.num_params_dec))

        if adv_decoder:
            self.c_decoder_layer1 = nn.Linear(self.latent_size, 2*self.latent_size)
            self.c_decoder_layer_out = nn.Linear(2*self.latent_size, self.latent_size)

        # Classical learning parameters
        self.c_layer1 = nn.Linear(self.latent_size, 120)
        self.c_layer2 = nn.Linear(120, num_params_enc * 3)
        self.c_out = nn.Linear(num_params_enc * 3, num_params_enc)

        self.reset_q_circs()

        self.distributions = None
        self.train_time = None
        self.last_state = None
        self.regressors = None

        self.latent_vectors = None
        self.raw_latent_vectors = None
        self.mean_std_list = None
        self.characterists_list = None
        self.df_latent_vectors = None
        self.batch_size = None
        self.distr_type = None
        self.pca = None
        self.regular_term = None
        self._last_latent_vector = None

    def forward(self, x):

        amplitudes = x

        assert len(x) == 2 ** self.n_qubits

        if self.interwoven:
            amplitude_wires = list(range(0, 2 * self.n_qubits, 2))
        else:
            amplitude_wires = list(range(0, self.n_qubits))

        if self.gate_type == 'ry':
            q_depth = int(self.num_params_dec // self.latent_size)
            q_out = dec_ry_q_net(self.params_dec, amplitudes=amplitudes, amplitude_wires=amplitude_wires,
                                 q_depth=q_depth, reg_size=self.latent_size, n_qubits=self.latent_size)

        if self.adv_decoder:
            q_out = torch.sigmoid(self.c_decoder_layer1(q_out.float()))
            q_out = self.c_decoder_layer_out(q_out.float())

        self._last_latent_vector = q_out
        out = torch.sigmoid(self.c_layer1(q_out.float()))
        out = torch.sigmoid(self.c_layer2(out))
        out = torch.tanh(self.c_out(out)) * np.pi / 2.0

        if self.gate_type == 'ry':
            q_depth = int(self.num_params_enc // self.n_qubits)
            out = enc_train_ry_q_net(out, amplitudes=amplitudes, init_rot=None, q_depth=q_depth, n_qubits=self.n_qubits)

        return out

    def test(self, x):

        output = self(x)
        loss = vqc.MSELoss(1 - output, 0.0)

        return loss

    def encoder(self, z, plot=False): # Note the naming is different
        """ Going from latent space to the state vector """
        z = torch.tensor(z)
        out = torch.sigmoid(self.c_layer1(z.float()))
        out = torch.sigmoid(self.c_layer2(out))
        out = torch.tanh(self.c_out(out)) * np.pi / 2.0
        q_depth = int(self.num_params_enc // self.n_qubits)

        obs_gen = vqc.proj_meas_gen(self.n_qubits)
        if self.gate_type == 'ry':
            op = next(obs_gen)
            q_out = enc_ry_q_net(out, q_depth=q_depth, op=op, n_qubits=self.n_qubits)
            for op in obs_gen:
                break
                q_out_elem = enc_ry_q_net(out, q_depth=q_depth, op=op, n_qubits=self.n_qubits).unsqueeze(0)
                q_out = torch.cat((q_out, q_out_elem))

        if plot:
            plt.plot(q_out.detach().numpy())
            plt.show()
        return q_out

    def decoder(self, x): # Note that the naming is different 
        """ Going from state to latent vector """
        # Encoder section
        amplitudes = np.array(x)

        assert len(x) == 2 ** self.n_qubits

        if self.interwoven:
            amplitude_wires = list(range(0, 2 * self.n_qubits, 2))
        else:
            amplitude_wires = list(range(0, self.n_qubits))
        if self.gate_type == 'ry':
            q_depth = int(self.num_params_dec // self.latent_size)
            q_out = dec_ry_q_net(self.params_dec, amplitudes=amplitudes, amplitude_wires=amplitude_wires,
                                 q_depth=q_depth, reg_size=self.latent_size, n_qubits=self.latent_size)

        if self.adv_decoder:
            q_out = torch.sigmoid(self.c_decoder_layer1(q_out.float()))
            q_out = self.c_decoder_layer_out(q_out.float())

        return q_out

    def create_gaussian_latent_regressor(self, distributions=None):

        if distributions == None:
            distributions = create_gaussian_distributions(2 ** self.n_qubits)
            next(distributions)

        latent_vectors, X = [], []
        for d in distributions:
            distr, m, s = d
            print("------ m={}, s={} ---------".format(m, s))
            latent_vectors.append(self.decoder(distr).detach().numpy())
            X.append([m, s])

        X = np.array(X)
        self.mean_std_list = X
        self.characterists_list = X
        Y = np.array(latent_vectors)
        self.latent_vectors = Y
        self.raw_latent_vectors = Y
        self.regressors = [LinearRegression() for _ in range(len(Y[0, :]))]
        [reg.fit(X, Y[:, i]) for i, reg in enumerate(self.regressors)]

        self.dataframe_latent_points()

    def latent_encoder(self, x, plot=False):

        assert self.regressors != None, "Must create latent regressor."

        out = np.array([reg.predict([x])[0] for reg in self.regressors])

        assert len(out) == self.latent_size, "Error in regression. "

        q_out = self.encoder(out)
        if plot:
            plt.plot(np.arange(-0.5*(2**self.n_qubits), 0.5*(2**self.n_qubits)), q_out.detach().numpy())
            plt.show()
        return q_out

    def save(self, loss_evol):

        i = 0
        filename = 'pickled_models/hqa_model_ver_{}.pickle'.format(i)
        while os.path.isfile(filename):
            i += 1
            filename = 'pickled_models/hqa_model_ver_{}.pickle'.format(i)

        dbfile = open(filename, 'wb')
        pickle.dump(self, dbfile)
        dbfile.close()

        filename = 'pickled_models/hqa_loss_evol_ver_{}.pickle'.format(i)
        dbfile = open(filename, 'wb')
        pickle.dump(loss_evol, dbfile)
        dbfile.close()

    def reset_q_circs(self):
        reset_dec_ry_q_net(self.latent_size)
        reset_enc_train_ry_q_net(2 * self.n_qubits + 1)
        reset_enc_ry_q_net(self.n_qubits)

    def transition_states(self, states, iter_per_state, return_transition=False):
        if not return_transition:
            fig, ax = plt.subplots()
        out_list = []
        for s in range(len(states)-1):
            state1 = self.decoder(states[s])
            state2 = self.decoder(states[s + 1])
            tran_vec = (state2 - state1)/iter_per_state
            for i in range(iter_per_state + 1):
                out = self.encoder(state1 + i*tran_vec)
                if return_transition:
                    out_list.append(out.detach().numpy())
                else:
                    plt.cla()
                    plt.plot(np.arange(-0.5*(2**self.n_qubits), 0.5*(2**self.n_qubits)), out.detach().numpy())
                    plt.show(block=False)
                    plt.pause(0.1)

        return out_list

    def create_latent_vectors(self, distributions, characteristics):

        assert len(distributions) == len(characteristics)

        self.characterists_list, self.latent_vectors = [], []
        for i, d in tqdm.tqdm(enumerate(distributions)):
            # print("Creating ... i={}, chara={}".format(i, characteristics[i]))
            self.characterists_list.append(characteristics[i])
            self.latent_vectors.append(self.decoder(d).detach().numpy())

        self.characterists_list = np.array(self.characterists_list)
        self.latent_vectors = np.array(self.latent_vectors)
        self.raw_latent_vectors = self.latent_vectors

    def latent_landscape(self, p1, p2, p3=None, states=None):
        """ This function plots the states on the landscape of two parameters with index labels p1 & p2.
            states should be a DataFrame. """
        
        if states == None:
            states = self.df_latent_vectors


        p1_list, p2_list, p3_list = [], [], []
        for s in range(len(states)):
            p1_list.append(states.iloc[s].latent_vector[p1])
            p2_list.append(states.iloc[s].latent_vector[p2])
            if p3:
                p3_list.append(states.iloc[s].latent_vector[p3])

        self.df_latent_vectors = self.df_latent_vectors.assign(p1=p1_list)
        self.df_latent_vectors = self.df_latent_vectors.assign(p2=p2_list)
        if p3:
            self.df_latent_vectors = self.df_latent_vectors.assign(p3=p3_list)

    def pca_transform_latent_vectors(self, projection_size, print_importance=False):

        assert isinstance(self.latent_vectors, np.ndarray), "Need to create latent vectors."

        X = self.raw_latent_vectors
        X = standardize_data(X)
        covariance_matrix = np.cov(X.T)
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

        if print_importance:
            print("Variance explained by each principal component:\n"
                  "{}\n".format([(i/sum(eigen_values))*100 for i in eigen_values]))

        self.pca = PCA(n_components=projection_size)
        self.latent_vectors = self.pca.fit_transform(self.raw_latent_vectors)

    def dataframe_latent_points(self, d_type=None):

        assert isinstance(self.latent_vectors, np.ndarray), "Must create latent regressor."
        if d_type == 'gaussian':
            latent_points = pd.DataFrame({'mean' : self.characterists_list[:, 0], 'std': self.characterists_list[:, 1],
                                          'latent_vector': list(self.latent_vectors)})
        elif d_type == 'poly_pert_gaussian':
            latent_points = pd.DataFrame({'mean': self.characterists_list[:, 0], 'peak': self.characterists_list[:, 1],
                                          'latent_vector': list(self.latent_vectors)})

        elif d_type == 'heisenberg': # Characteristics need to be supplied in this form with depth as the first column followed by the coupling terms
            d = {"c{}".format(i): self.characterists_list[:, i] for i in range(1, len(self.characterists_list[0]))}
            d['depth'] = self.characterists_list[:, 0]
            d['latent_vector'] = list(self.latent_vectors)
            latent_points = pd.DataFrame(d)

        else:
            d = {"c{}".format(i): self.characterists_list[:, i] for i in range(len(self.characterists_list[0]))}
            d['latent_vector'] = list(self.latent_vectors)
            latent_points = pd.DataFrame(d)

        self.df_latent_vectors = latent_points


def train_hqa(model, distributions, num_iterations, loss_evol=None, batch_size=1,
               regular_term=None):

    since = time.time()
    print("Training started:")

    N = 2 ** model.n_qubits
    step = 0.005
    criterion = vqc.MSELoss

    model.distributions = distributions

    optimizer = optim.Adam(model.parameters(), lr=step)
    if not loss_evol:
        loss_evol = []
    for ii in range(num_iterations):
        print('-----iteration {}--model:{}------'.format(ii, model))
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()

            d = distributions[np.random.randint(len(distributions))]
            output = model(d)
            loss = criterion(1 - output, 0.0)
            if regular_term:
                loss += regular_term * torch.sum(model._last_latent_vector ** 2)
            for i in range(batch_size-1):
        
                d = distributions[np.random.randint(len(distributions))]

                output = model(d)
                loss = loss + criterion(1 - output, 0.0)
                if regular_term:
                    print("Loss before regular term:", loss)
                    loss += regular_term * torch.sum(model._last_latent_vector ** 2)

            print("Loss", loss)
            loss_evol.append(loss)
            loss.backward()
            optimizer.step()

    model.train_time = time.time() - since
    model.batch_size = batch_size
    model.regular_term = regular_term

    return model, loss_evol


def test_hqa(model, num_test, distributions):
    tot_loss = 0
    loss_list = []

    for ii in tqdm.tqdm(range(num_test)):
        # print("----- Test number {} -------".format(ii))
        d = distributions[np.random.randint(len(distributions))]
        loss_list.append(model.test(d).detach().numpy())
        tot_loss += loss_list[-1]

    print("Average Loss: {}".format(tot_loss/num_test))
    return tot_loss/num_test, loss_list


def test_dataframe_hqa(num_test, models, model_filenames, loss_filenames, distributions,
                           save_data=True):
    n_qubit_list, latent_size_list, loss_list, raw_test_results, d_type_list = [], [], [], [], []
    for i, model in enumerate(models):
        print("Testing model: {}".format(model_filenames[i]))
        model.reset_q_circs()
        loss, results = test_hqa(model, num_test, distributions)
        n_qubit_list.append(model.n_qubits)
        latent_size_list.append(model.latent_size)
        loss_list.append(loss)
        raw_test_results.append(results)
        try:
            d_type_list.append(model.distr_type)
        except AttributeError:
            d_type_list.append('gaussian')

    data = pd.DataFrame({'model': models,
                         'n_qubits': n_qubit_list,
                         'latent_size': latent_size_list,
                         'train_d_type': d_type_list,
                         'loss': loss_list,
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


def hqa_testing_script(required, num_test, distributions):

    models, model_filenames, loss_filenames = gd.load_models(required)
    data = test_dataframe_hqa(num_test, models, model_filenames, loss_filenames, distributions,
                               save_data=True)

    return data


def latent_space_process(argument):

    latent_size = argument
    main_script(latent_size=latent_size)


def latent_space_script():

    latent_sizes = [5, 6, 7, 8, 9, 10, 11, 12, 13]
    latent_sizes = latent_sizes * 3
    print("Number of models: ", len(latent_sizes))
    print(latent_sizes)
    print('\n\n')
    p = vqc.Pool(30)
    p.map(latent_space_process, latent_sizes)


def restrict_gaussian_script():
    n_qubits = 5
    d_type = 'gaussian'
    list_std = np.arange(2, 11, 1)
    list_mean = np.arange(-15, 15, 2)
    N = 2 ** n_qubits
    if d_type=='gaussian':
        in_size = 2
    elif d_type == 'poly_pert_gaussian':
        in_size = 4
    distributions = Distributions(in_size=in_size, N=N, num_distr=100, iterations=5000, d_type=d_type,
                                  list_std=list_std, list_mean=list_mean, order=None)

    main_script(set_distribution=distributions, n_qubits=n_qubits)


def regular_process(argument):

    r = argument
    main_script(d_type='gaussian', n_qubits=7, latent_size=12, regular_term=r)


def regular_script():

    r_list = [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625, 0.00078125,
              0.000390625, 0.0001953125]

    p = vqc.Pool(12)
    p.map(regular_process, r_list)


def main_script(latent_size=12, d_type='gaussian', n_qubits=5, set_distribution=None, regular_term=None):

    num_params_dec = 4 * latent_size
    num_params_enc = n_qubits * 5
    N = 2 ** n_qubits
    num_iterations = 5000
    distrib_train = None
    batch_size = 2
    interwoven = False
    p_fuzz = None

    if d_type=='gaussian':
        in_size = 2
    elif d_type == 'poly_pert_gaussian':
        in_size = 4
    elif d_type == 'entan_states':
        in_size = None

    distributions = Distributions(in_size=in_size, N=N, num_distr=100, iterations=num_iterations, d_type=d_type,
                                  list_std=None, list_mean=None, order=distrib_train)

    if set_distribution is not None:
        distributions = set_distribution

    model = HQA(n_qubits, latent_size, num_params_enc, num_params_dec, gate_type='ry', interwoven=interwoven,
                 adv_decoder=True)
    model.distr_type = d_type
    model, loss_evol = train_hqa(model, distributions, num_iterations, batch_size=batch_size, p_fuzz=p_fuzz,
                                  regular_term=regular_term)

    model.save(loss_evol)


def clustering_entan_states_script():

    return main_script(d_type='entan_states', n_qubits=7, latent_size=12)


if __name__ == '__main__':

    # Scripts ------------------------------------------------------------------------------------

    main_script()

    # main_script(d_type='gaussian', n_qubits=7, latent_size=12, regular_term=0.0001)

    # latent_space_script()

    # restrict_gaussian_script()

    # regular_script()

    # d = clustering_entan_states_script()

    # Testing -------------------------------------------------------------------------------------

    # num_test = 1000
    # required = ['jobs_date_03_05_2021']
    # hqa_testing_script(required=required, num_test=num_test, d_type=None, p_fuzz=None)

