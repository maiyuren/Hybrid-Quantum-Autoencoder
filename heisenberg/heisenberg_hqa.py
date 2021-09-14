import sys 
sys.path.insert(0, "../")
sys.path.insert(0, "../required_modules/")
from HQA import HQA, train_hqa, test_hqa
import pickle
import pandas as pd
import numpy as np


def heisenberg_hqa_script(n_qubits, distributions):
    
    latent_size = n_qubits + 2 
    num_params_dec = 5 * latent_size
    num_params_enc = n_qubits * 5
    N = 2 ** n_qubits
    num_iterations = 500
    batch_size = 2
    interwoven = False
    adv_decoder = True
    regular_term = None 
    
    model = HQA(n_qubits, latent_size, num_params_enc, num_params_dec, gate_type='ry', interwoven=interwoven,
                 adv_decoder=adv_decoder)

    model, loss_evol = train_hqa(model, distributions, num_iterations, batch_size=batch_size, 
                                  regular_term=regular_term)
    
    model.save(loss_evol)
    return model 

def datafile_to_dataframe(filename):

    with open(filename, "rb") as f:
        data = pickle.load(f)

    Jlist, E0, E, F, x, nfev, nit, x0, statevector, h_indx, D = [],[],[],[],[],[],[],[],[],[],[]
    for ham_indx, dat in data.items():
        _Jlist = dat['Jlist']
        for depth, d in dat.items():
            if depth == 'Jlist' or depth == 0:
                continue
            assert depth == d['D'], "Something went wrong."
            D.append(d['D'])
            Jlist.append(_Jlist)
            E0.append(d["E0"])
            E.append(d["E"])
            F.append(d["F"])
            x.append(d['x'])
            nfev.append(d['nfev'])
            nit.append(d['nit'])
            x0.append(d['x0'])
            statevector.append(d['statevector'])
            h_indx.append(ham_indx)
            
    pd_data = pd.DataFrame({'hamiltonian_indx':h_indx, "depth":D, "Jlist":Jlist, "E0":E0, "E":E, 
                            "F":F, "x":x,'nfev':nfev, 'nit':nit, 'x0':x0, 'statevector':statevector})

    return pd_data


def run_heisenberg_hqa_from_datafile(filename):
    data = datafile_to_dataframe(filename)
    distributions = np.array(data.statevector.to_list())

    n_qubits = int(np.log2(len(distributions[0])))
    return heisenberg_hqa_script(n_qubits, distributions)


if __name__ == '__main__':

    run_heisenberg_hqa_from_datafile("state_files/data_n_qubits-6.pkl")