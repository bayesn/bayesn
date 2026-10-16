""" Slow end-to-end test of T21 training. This relies on the Foundation_DR1 submodule
in tests/test_files/training_data being initialised. If you did not git clone the
byaesn repository with the --recurse-submodules flag, that directory may be empty.
Running `git submodule update --init` should provide the required files.

This will be very slow on CPU.
"""
from ruamel.yaml import YAML
from pathlib import Path
import pickle

import numpy as np
import pytest
from scipy.stats import chi2

from bayesn.bayesn_model import SEDmodel

BASE_DIR: Path = Path(__file__).parent.parent.absolute()

def get_model_params_from_model(model: SEDmodel):
    """ Return shape (N_params,) array of scalar model parameters in the order
    L_Omega, W0, W1, sigma0, tauA, then (RV) or (mu_RV, sigma_RV) depending on RV_type
    """
    N_L = int(model.N_knots_sig*(model.N_knots_sig+1)/2)
    N_W = model.l_knots.shape[0]*model.tau_knots.shape[0]
    N_params = N_L + 2*N_W + 3 + int(model.sigma_R is not None)
    model_params = np.zeros(N_params)
    k = 0
    for i in range(model.N_knots_sig):
        for j in range(model.N_knots_sig):
            if j > i:
                continue
            model_params[k] = model.L_Sigma[i,j]
            k += 1
    for key in ("W0", "W1"):
        for i in range(N_W):
            model_params[k] = getattr(model, key)[i%model.l_knots.shape[0], i//model.l_knots.shape[0]]
            k += 1
    for param in ("sigma0", "tauA"):
        model_params[k] = getattr(model, param)
        k += 1
    if model.mu_R is not None:
        model_params[k] = float(model.mu_R)
        model_params[k+1] = float(model.sigma_R)
    else:
        model_params[k] = float(model.RV)
        model_params = model_params[:-1]
    return model_params

def get_model_params_from_chains(chains: dict, N_knots_sig: int=24, N_W: int=36):
    """ Return shape (N_params, N_chains*N_samples) array of scalar model parameters in the order
    L_Omega, W0, W1, sigma0, tauA, then (RV) or (mu_RV, sigma_RV) depending on RV_type
    """
    N_chains, N_samples = chains["Ds"].shape[:2]
    N_L = int(N_knots_sig*(N_knots_sig+1)/2)
    N_params = N_L + 2*N_W + 3 + int("sigma_R" in chains)
    model_params = np.zeros((N_params, N_chains*N_samples))
    sigeps_diag = np.zeros((N_chains, N_samples, N_knots_sig, N_knots_sig))
    for i in range(N_knots_sig):
        sigeps_diag[:,:,i,i] = chains["sigmaepsilon"][:,:,i]
    L_Sigma = np.matmul(sigeps_diag, chains["L_Omega"])
    k = 0
    for i in range(N_knots_sig):
        for j in range(N_knots_sig):
            if j > i:
                continue
            model_params[k] = L_Sigma[:,:,i,j].flatten()
            k += 1
    for key in ("W0", "W1"):
        for i in range(N_W):
            model_params[k] = chains[key][:,:,i].flatten()
            k += 1
    for param in ("sigma0", "tauA"):
        model_params[k] = chains[param].flatten()
        k += 1
    if "mu_R" in chains:
        model_params[k] = chains["mu_R"].flatten()
        model_params[k+1] = chains["sigma_R"].flatten()
    else:
        model_params[k] = chains["RV"].flatten()
        model_params = model_params[:-1]
    return model_params

def get_model_m_dists(chains: dict, model: SEDmodel):
    model_params = get_model_params_from_model(model=model)
    chain_params = get_model_params_from_chains(chains)
    diff = chain_params.mean(axis=1) - model_params
    precision = np.linalg.pinv(np.cov(chain_params))
    m_dist = diff @ precision @ diff
    return chi2.sf(m_dist, len(model_params))

@pytest.mark.slow
def test_train_T21(tmpdir: Path):
    yaml = YAML(typ="safe")
    with open(BASE_DIR / "example_T21_training.yaml", "r") as file:
        args = yaml.load(file)
    model = SEDmodel(
        load_model=args["load_model"],
        load_ext_rel=args["load_ext_rel"],
        apply_dovekie_mag_shifts=False,
    )
    args["data_root"] = str(BASE_DIR / args["data_root"])
    args["outputdir"] = tmpdir
    model._set_used_bands(bands=[f"{bp}_PS1" for bp in "griz"])
    model.run(args, {})

    with open(args["outputdir"] / "chains.pkl", "rb") as file:
        chains = pickle.load(file)

    assert get_model_m_dists(chains, model) > 0.9
