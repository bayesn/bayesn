from pathlib import Path
import pickle
from ruamel.yaml import YAML

import numpy as np
import pytest
from scipy.stats import chi2

from bayesn.bayesn_model import SEDmodel
from bayesn.utils import get_model_params_from_chains

BASE_DIR: Path = Path(__file__).parent.parent.absolute()

@pytest.mark.slow
def test_train_T21(tmpdir: Path):
    """ Slow end-to-end test of T21 training. This relies on the Foundation_DR1 submodule
    in tests/test_files/training_data being initialised. If you did not git clone the
    byaesn repository with the --recurse-submodules flag, that directory may be empty.
    Running `git submodule update --init` should provide the required files.

    This will be very slow on CPU but takes <15 m on 4 nvidia A100s.
    """
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
    model.run(args)

    with open(args["outputdir"] / "chains.pkl", "rb") as file:
        chains = pickle.load(file)

    model_params = model.get_model_params()
    chain_params = get_model_params_from_chains(chains)
    diff = chain_params.mean(axis=1) - model_params
    precision = np.linalg.pinv(np.cov(chain_params))
    m_dist = diff @ precision @ diff
    assert chi2.sf(m_dist, len(model_params)) > 0.99
