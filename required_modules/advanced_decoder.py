
import sys
import gaussian_decoding as gd


class EnhancedDecoder(gd.nn.Module):

    name = 'EnhancedDecoder'

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
        gd.reset_q_num(n_qubits)
        self.q_depth = q_depth
        self.q_delta = q_delta
        self.gate_type = gate_type
        self.train_shots = shots
        if gate_type == 'ry':
            self.num_params = self.q_depth * n_qubits
            gd.reset_q_num(n_qubits, set_gate_type=gate_type)
        elif gate_type == 'u2':
            gd.reset_q_num(n_qubits, set_gate_type=gate_type)
            self.num_params = 2 * self.q_depth * n_qubits

        gd.reset_q_num(self.n_qubits, set_gate_type=self.gate_type)
        if shots:
            gd.reset_aer(self.n_qubits, shots=shots)

        if reg_size == 'binary':
            self.reg_size = n_qubits
            self.binary = True
        else:
            assert reg_size <= n_qubits
            self.reg_size = reg_size
            self.binary = False

        gd.reset_q_num(self.n_qubits, set_gate_type=self.gate_type)
        if shots:
            gd.reset_aer(self.n_qubits, shots=shots)
            # pass
        self.q_params_mean = gd.nn.Parameter(q_delta * gd.torch.randn(self.num_params))
        self.q_params_std = gd.nn.Parameter(q_delta * gd.torch.randn(self.num_params))

        classical_layer1_size = 100

        self.c_layer1_mean = gd.nn.Linear(n_qubits, classical_layer1_size)
        self.c_layer2_mean = gd.nn.Linear(classical_layer1_size, n_qubits * 2)
        self.c_out_mean = gd.nn.Linear(n_qubits * 2, self.reg_size)

        self.c_layer1_std = gd.nn.Linear(n_qubits, classical_layer1_size)
        self.c_layer2_std = gd.nn.Linear(classical_layer1_size, n_qubits * 2)
        self.c_out_std = gd.nn.Linear(n_qubits * 2, self.reg_size)

        if reg_size == 'binary':
            self.reg_size = n_qubits
            self.binary = True
        else:
            assert reg_size <= n_qubits
            self.reg_size = reg_size
            self.binary = False

        self.model_info = {'input_size': input_size, 'n_qubits': n_qubits, 'q_depth': q_depth,
                           'q_delta': q_delta, 'reg_size': reg_size, 'model': 'EnhancedDecoder'}

        self.train_time = 0
        self.distributions = None
        self.train_logistic_k = None
        self.train_p_fuzz = None
        self.batch_size = None
        self.last_state = [None, None]

    def forward(self, x, save_state=False, sample_num=None):
        q_in = x
        q_out = gd.torch.Tensor(0, 2 * self.reg_size)
        # q_out = q_out.to(self.device)
        # Because it comes as a batch (matrix) we cant just send the matrix in altogether like for the other layers
        #  but incase it has come by itself
        if len(q_in.shape) == 1:
            if isinstance(q_in, gd.np.ndarray):
                q_in = gd.np.array([q_in])
            else:
                q_in = q_in.unsqueeze(0)
        for elem in q_in:
            q_out_elem_mean = gd.q_net(self.q_params_mean, n_qubits=self.n_qubits, amplitudes=elem,
                                    q_depth=self.q_depth, reg_size=self.n_qubits).float()
            print("Out of ----- ", q_out_elem_mean)
            q_out_elem_mean = gd.torch.sigmoid(self.c_layer1_mean(q_out_elem_mean))
            q_out_elem_mean = gd.torch.sigmoid(self.c_layer2_mean(q_out_elem_mean))
            q_out_elem_mean = gd.torch.tanh(self.c_out_mean(q_out_elem_mean)) * gd.np.pi / 2.0
            if save_state:
                m_state = gd.dev.state

            q_out_elem_std = gd.q_net(self.q_params_std, n_qubits=self.n_qubits, amplitudes=elem,
                                   q_depth=self.q_depth, reg_size=self.n_qubits).float()
            q_out_elem_std = gd.torch.sigmoid(self.c_layer1_std(q_out_elem_std))
            q_out_elem_std = gd.torch.sigmoid(self.c_layer2_std(q_out_elem_std))
            q_out_elem_std = gd.torch.tanh(self.c_out_std(q_out_elem_std)) * gd.np.pi / 2.0
            if save_state:
                self.last_state = [m_state, gd.dev.state]

            q_out_elem = gd.torch.cat((q_out_elem_mean, q_out_elem_std)).unsqueeze(0)

            if sample_num:
                q_out_elem = q_out_elem + gd.torch.tensor(gd.np.random.random(16) - 0.5) / sample_num ** 0.5
                q_out_elem = q_out_elem.float()

            # Transformation of the output to be inbetween 0 and 1
            if self.train_logistic_k:
                q_out_elem = gd.sigmoid_k(q_out_elem, self.train_logistic_k)
            else:
                q_out_elem = (q_out_elem + 1) / 2

            q_out = gd.torch.cat((q_out, q_out_elem))

        return q_out


class AdvancedDecoder(gd.nn.Module):

    name = 'AdvancedDecoder'

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
        gd.reset_q_num(n_qubits)
        self.gate_type = gate_type
        self.train_shots = shots

        if isinstance(q_depth, int) or isinstance(q_depth, float):
            self.q_depth = [q_depth, q_depth]
        else:
            self.q_depth = q_depth

        if gate_type == 'ry':
            self.num_params = [self.q_depth[0]*n_qubits, self.q_depth[1]*n_qubits]
            gd.reset_q_num(n_qubits, set_gate_type=gate_type)
        elif gate_type == 'u2':
            self.num_params = [2 * self.q_depth[0]*n_qubits, 2 * self.q_depth[1]*n_qubits]
            gd.reset_q_num(n_qubits, set_gate_type=gate_type)

        if reg_size == 'binary':
            self.reg_size = n_qubits
            self.binary = True
        else:
            assert reg_size <= n_qubits
            self.reg_size = reg_size
            self.binary = False

        self.q_params_mean = gd.nn.Parameter(q_delta * gd.torch.randn(self.num_params[0]))
        self.q_params_std = gd.nn.Parameter(q_delta * gd.torch.randn(self.num_params[0]))

        classical_layer1_size = 100

        self.c_layer1_mean = gd.nn.Linear(n_qubits, classical_layer1_size)
        self.c_layer2_mean = gd.nn.Linear(classical_layer1_size, self.num_params[1] * 2)
        self.c_out_mean = gd.nn.Linear(self.num_params[1] * 2, self.num_params[1])

        self.c_layer1_std = gd.nn.Linear(n_qubits, classical_layer1_size)
        self.c_layer2_std = gd.nn.Linear(classical_layer1_size, self.num_params[1] * 2)
        self.c_out_std = gd.nn.Linear(self.num_params[1] * 2, self.num_params[1])

        self.model_info = {'input_size':input_size, 'n_qubits':n_qubits, 'q_depth':q_depth,
                           'q_delta':q_delta, 'reg_size':reg_size, 'model': 'AdvancedDecoder'}

        self.train_time = 0
        self.distributions = None
        self.batch_size = None
        self.train_logistic_k = None
        self.train_p_fuzz = None
        self.last_state = [None, None]

    def forward(self, x, save_state=False, sample_num=None):
        q_in = x
        q_out = gd.torch.Tensor(0, 2 * self.reg_size)
        # q_out = q_out.to(self.device)
        # Because it comes as a batch (matrix) we cant just send the matrix in altogether like for the other layers
        #  but incase it has come by itself
        if len(q_in.shape) == 1:
            if isinstance(q_in, gd.np.ndarray):
                q_in = gd.np.array([q_in])
            else:
                q_in = q_in.unsqueeze(0)
        for elem in q_in:
            q_out_elem_mean = gd.q_net(self.q_params_mean, n_qubits=self.n_qubits, amplitudes=elem,
                                    q_depth=self.q_depth[0], reg_size=self.n_qubits).float()
            # q_out_elem_mean.to(device)
            q_out_elem_mean = gd.torch.sigmoid(self.c_layer1_mean(q_out_elem_mean))
            q_out_elem_mean = gd.torch.sigmoid(self.c_layer2_mean(q_out_elem_mean))
            q_out_elem_mean = gd.torch.tanh(self.c_out_mean(q_out_elem_mean)) * gd.np.pi / 2.0
            q_out_elem_mean = gd.q_net(q_out_elem_mean, n_qubits=self.n_qubits, amplitudes=elem,
                                    q_depth=self.q_depth[1], reg_size=self.reg_size).float()
            if save_state:
                m_state = gd.dev.state

            q_out_elem_std = gd.q_net(self.q_params_std, n_qubits=self.n_qubits, amplitudes=elem,
                                   q_depth=self.q_depth[0], reg_size=self.n_qubits).float()
            # q_out_elem_mean.to(device)
            q_out_elem_std = gd.torch.sigmoid(self.c_layer1_std(q_out_elem_std))
            q_out_elem_std = gd.torch.sigmoid(self.c_layer2_std(q_out_elem_std))
            q_out_elem_std = gd.torch.tanh(self.c_out_std(q_out_elem_std)) * gd.np.pi / 2.0
            q_out_elem_std = gd.q_net(q_out_elem_std, n_qubits=self.n_qubits, amplitudes=elem,
                                    q_depth=self.q_depth[1], reg_size=self.reg_size).float()
            if save_state:
                self.last_state = [m_state, gd.dev.state]

            q_out_elem = gd.torch.cat((q_out_elem_mean, q_out_elem_std)).unsqueeze(0)

            if sample_num:
                q_out_elem = q_out_elem + gd.torch.tensor(gd.np.random.random(16) - 0.5) / sample_num ** 0.5
                q_out_elem = q_out_elem.float()

            # Transformation of the output to be inbetween 0 and 1
            if self.train_logistic_k:
                q_out_elem = gd.sigmoid_k(q_out_elem, self.train_logistic_k)
            else:
                q_out_elem = (q_out_elem + 1) / 2

            q_out = gd.torch.cat((q_out, q_out_elem))

        return q_out
