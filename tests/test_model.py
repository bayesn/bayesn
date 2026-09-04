import os
import copy
from pathlib import Path
import pickle
from ruamel.yaml import YAML
import shutil
import time
from typing import Any
from warnings import warn

from bayesn.bayesn_model import SEDmodel, default_kwargs
import argparse
import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.handlers import trace, seed
import pandas as pd
import pytest
try:
    import extinction
    ext_loaded = True
except ModuleNotFoundError:
    ext_loaded: bool = False
try:
    from dust_extinction import parameter_averages as de
    de_loaded: bool = True
except ModuleNotFoundError:
    de_loaded: bool = False

BASE_DIR: Path = Path(__file__).parent.parent.absolute()
TEST_DIR: Path = BASE_DIR / "tests/test_files"
PICKLE_DIR: Path = TEST_DIR / "pickles"
NON_EXISTENT_PATH: Path = TEST_DIR / "non_existent"
N_sn: int = 5
N_epochs: int = 10
rng_seed: int = 1
rng_key: jax._src.prng.PRNGKeyArray = jax.random.key(rng_seed)

def non_existent_check():
    if NON_EXISTENT_PATH.exists():
        raise FileExistsError(
            f"{NON_EXISTENT_PATH} exists, so this test cannot trigger the expected "
            "FileNotFoundError."
        )
################
### Fixtures ###
################
@pytest.fixture(scope="module")
def initial_args() -> dict:
    yaml = YAML(typ="safe")
    with open(TEST_DIR / "input.yaml", "r") as file:
        args = yaml.load(file)
    args["data_root"] = str(BASE_DIR / args["data_root"])
    return args

@pytest.fixture(scope="module")
def model(initial_args: dict) -> SEDmodel:
    model = SEDmodel(load_model=initial_args["load_model"], load_ext_rel=initial_args["load_ext_rel"], filter_yaml=None)
    return model

from unittest.mock import patch

@pytest.fixture(scope="module")
def custom_model(initial_args: dict, model: SEDmodel) -> SEDmodel:
    # copy of T21_model/BAYESN.YAML but with MUR=2.61 instead of RV=2.61
    def mock_load_hsiao(self, *args, **kwargs):
        for attr in ("min_hsiao_wave", "max_hsiao_wave", "hsiao_t", "hsiao_l", "hsiao_flux", "KD_t_hsiao", "J_l_T_hsiao", "hsiao_offset"):
            setattr(self, attr, getattr(model, attr))

    def mock_load_ext_rel(self, *args, **kwargs):
        self.mw_ext = model.mw_ext
        self.ext_rel = model.ext_rel

    with patch.object(SEDmodel, "_load_hsiao_template", mock_load_hsiao), \
         patch.object(SEDmodel, "load_ext_rel", mock_load_ext_rel), \
         patch.object(SEDmodel, "_load_dovekie_cov"):
        custom_model = SEDmodel(load_model=TEST_DIR / "test_model.yaml", load_ext_rel=initial_args["load_ext_rel"], filter_yaml=None)
    return custom_model

@pytest.fixture(scope="module")
def sample_model_parameters(model: SEDmodel) -> tuple[jax.Array, ...]:
    mu_R = 3.1
    sigma_R = 0.5
    theta = jax.random.normal(rng_key, (N_sn,))
    AV = jax.random.exponential(rng_key, (N_sn,))
    RV = mu_R + sigma_R*jax.random.normal(rng_key, (N_sn,))
    W0 = jax.random.normal(rng_key, (N_sn, *model.W0.shape))
    W1 = jax.random.normal(rng_key, (N_sn, *model.W1.shape))
    eps = jax.random.normal(rng_key, (N_sn, *model.W1.shape))
    eps = eps.at[:,0].set(0).at[:,-1].set(0)
    t = jax.random.uniform(rng_key, (N_epochs, N_sn))
    J_t = model.get_J_t(t)
    hsiao_interp = model.get_hsiao_interp(t)
    z = 0.1*jax.random.uniform(rng_key, (N_sn,))
    return theta, AV, W0, W1, eps, RV, t, J_t, hsiao_interp, z

@pytest.fixture(scope="module")
def loaded_model_and_args(initial_args: dict, model: SEDmodel) -> tuple[SEDmodel, dict]:
    cmd_args = {"input": str(TEST_DIR / "input.yaml")}
    # T21_mini_set.txt uses PS1 data.
    model._set_used_bands(bands=[f"{bp}_PS1" for bp in "griz"])
    args = model.parse_args(initial_args, cmd_args)
    model.process_dataset(args)
    return model, args


#############
### Tests ###
#############
class TestInit:
    expected_constants: tuple[tuple, ...] = (
        ("RV_MW", 3.1),
        ("sigma_pec", 150 / 3e5),
        ("trunc_val", 1.2),
        ("ZPT", 27.5),
        ("spectrum_bins", 300),
        ("band_oversampling", 51),
        ("max_redshift", 4),
    )

    def test_model_file_non_existent(self, initial_args: dict):
        non_existent_check()
        with pytest.raises(FileNotFoundError):
            model = SEDmodel(load_model=NON_EXISTENT_PATH, filter_yaml=None)

    def test_custom_model_file(self, initial_args: dict, model: SEDmodel, custom_model: SEDmodel):
        for attr in ("l_knots", "L_Sigma", "tau_knots", "W0", "W1"):
            assert (getattr(custom_model, attr) == getattr(model, attr)).all()
        for attr in ("M0", "sigma0", "tauA"):  # skipping RV because the custom model uses MUR
            assert getattr(custom_model, attr) == getattr(model, attr)

    def test_pop_rv_init(self, initial_args: dict, custom_model: SEDmodel):
        assert custom_model.RV_type == "pop"
        assert custom_model.mu_R == 2.61
        assert custom_model.sigma_R == 0.5


    @pytest.mark.parametrize("attr_name,value", expected_constants)
    def test_init_constants(self, model: SEDmodel, attr_name: str, value: int | float):
        assert getattr(model, attr_name) == value

    def test_odd_oversampling(self, model: SEDmodel):
        assert model.band_oversampling % 2 == 1

    def test_example_lc_exists(self, model: SEDmodel):
        assert Path(model.example_lc).exists()
    @pytest.mark.parametrize("attr_name", ("M0", "sigma0", "tauA", "mu_R", "sigma_R"))
    def test_0d_arrs(self, model: SEDmodel, attr_name: str):
        attr = getattr(model, attr_name, jnp.array(0))  # mu_R and sigma_R can be undef
        assert isinstance(attr, jax.Array)
        assert len(attr.shape) == 0
        assert not jnp.isnan(attr)

    @pytest.mark.parametrize("attr_name", ("l_knots", "tau_knots"))
    def test_1d_arrs(self, model: SEDmodel, attr_name: str):
        attr = getattr(model, attr_name)
        assert isinstance(attr, jax.Array)
        assert not jnp.isnan(attr).any()
        assert list(attr) == sorted(attr)  # monotonicity
        assert len(attr.shape) == 1  # 1d array
        assert attr.shape[0] == jnp.unique(attr).shape[0]  # uniqueness

    @pytest.mark.parametrize("attr_name", ("W0", "W1"))
    def test_W_arrs(self, model: SEDmodel, attr_name: str):
        attr = getattr(model, attr_name)
        assert not jnp.isnan(attr).any()
        assert attr.shape == (model.l_knots.shape[0], model.tau_knots.shape[0])

    def test_L_Sigma(self, model: SEDmodel):
        row, col = model.L_Sigma.nonzero()
        assert not jnp.isnan(model.L_Sigma).any()
        assert model.L_Sigma.shape == (model.N_knots_sig, model.N_knots_sig)
        assert max(col - row) <= 0  # L_Sigma is lower triangular
        assert all(jnp.diag(model.L_Sigma) > 0)  # Positive unique Cholesky decomposition

    @pytest.mark.parametrize("arr_name,shape", [("J_l_T", (300, 7)), ("KD_t", (6, 7))])
    def test_model_array(self, model: SEDmodel, arr_name: str, shape: tuple[int,int], rtol: float=0, atol: float =1e-15):
        test_obj = getattr(model, arr_name)
        assert test_obj.shape == shape
        with open(PICKLE_DIR / f"{arr_name}.pkl", "rb") as f:
            ref = pickle.load(f)
        assert jnp.isclose(test_obj, ref, rtol=rtol, atol=atol).all()

    def test_init_band_dicts(self, model: SEDmodel):
        assert model.band_dict == {"NULL_BAND": 0}
        assert model.zp_dict == {"NULL_BAND": 10}
        assert model.band_lim_dict == {"NULL_BAND": (model.hires_wave[0], model.hires_wave[-1])}


class TestHsiao:
    def test_hsiao_t(self, model):
        assert all(model.hsiao_t == np.arange(-20, 86, 1, dtype=float))

    def test_hsiao_l(self, model):
        assert all(model.hsiao_l == np.arange(1000, 25001, 10, dtype=float))

    @pytest.mark.parametrize("arr_name,shape",
        [
            ("hsiao_flux", (300, 106)),
            ("J_l_T_hsiao", (300, 2402)),
            ("KD_t_hsiao", (106, 107)),
        ]
    )
    def test_hsiao_array(self, model, arr_name, shape):
        test_obj = getattr(model, arr_name)
        assert test_obj.shape == shape
        with open(PICKLE_DIR / f"{arr_name}.pkl", "rb") as f:
            ref = pickle.load(f)
        assert jnp.isclose(test_obj, ref, rtol=1e-15, atol=1e-20).all()

class TestBandWeights:
    bp_caching_args: tuple[tuple, ...] = (
        ([], 1, False),  # fresh init with just NULL_BAND
        ([f"{bp}_PS1" for bp in "gr"], 3, False),
        ([f"{bp}_PS1" for bp in "griz"]+["Y_LSST"], 6, True),
        )
    set_used_bands_args: tuple[tuple, ...] = (
        ([f"{bp}_PS1" for bp in "ri"], False),
        ([f"{bp}_WFCAM" for bp in "YJH"], False),
        (None, True),
    )
    def test_filter_dict(self, model: SEDmodel):
        yaml = YAML(typ='safe')
        with open(BASE_DIR / "bayesn/bayesn-filters/filters.yaml", "r") as file:
            ref_dict = yaml.load(file)
        for bp in ref_dict["filters"]:
            ref_dict["filters"][bp]["path"] = str(Path(model.__root_dir__, "bayesn-filters", ref_dict["filters"][bp]["path"]))
        assert model.filter_dict["filters"] == ref_dict["filters"]

    def test_filter_dict_non_existent(self, model: SEDmodel):
        non_existent_check()
        model.filter_yaml = NON_EXISTENT_PATH
        with pytest.raises(FileNotFoundError):
            model._load_filter_dict()
        model.filter_yaml = None

    @pytest.mark.parametrize("filter_yaml", ("test_filter_std_root.yaml", "test_filter_filt_root.yaml"))
    def test_custom_filter_dict(self, model: SEDmodel, filter_yaml: str):
        model = copy.deepcopy(model)
        model.filter_yaml = str(TEST_DIR / filter_yaml)
        # filter.yaml file uses env variable
        old_env = os.environ.pop("BAYESN_TEST_VAR", None)
        with pytest.raises(FileNotFoundError, match="The environment variable"):
            model._load_filter_dict()
        shutil.copy(model.filter_dict["standards"]["vega"]["path"], TEST_DIR / "test_standard.fits")
        os.environ["BAYESN_TEST_VAR"] = str(TEST_DIR)
        test_dict = model._load_filter_dict()
        if old_env is None:
            os.environ.pop("BAYESN_TEST_VAR")
        else:
            os.environ["BAYESN_TEST_VAR"] = old_env
        (TEST_DIR / "test_standard.fits").unlink()
        for key in ("lam", "f_lam"):
            assert all(model.filter_dict["standards"]["vega"][key] == test_dict["standards"]["test_standard"][key])
        assert Path(test_dict["filters"].pop("test_filter")["path"]) == TEST_DIR / "non_existent"
        for filt in model.filter_dict["filters"]:
            assert model.filter_dict["filters"][filt] == test_dict["filters"][filt]

    def test_load_band_weights_no_file(self, model: SEDmodel):
        non_existent_check()
        with pytest.raises(FileNotFoundError):
            model._load_band_weights(bands_to_load=[], shift_file=NON_EXISTENT_PATH)
        model.filter_dict["filters"]["test_filter"] = {"path": str(NON_EXISTENT_PATH)}
        with pytest.raises(FileNotFoundError):
            model._load_band_weights(bands_to_load=["test_filter"])
        model.filter_dict["filters"].pop("test_filter")

    def test_load_band_weights_bp_not_found(self, model: SEDmodel):
        if "non_existent" in model.filter_dict["filters"]:
            raise ValueError(
                "'non_existent' is defined in the model's filter_dict, so this test "
                "cannot trigger the expected ValueError."
            )
        with pytest.raises(ValueError):
            model._load_band_weights(bands_to_load=["non_existent"])

    def test_load_band_weights_shift_file(self, model: SEDmodel):
        shift_file = TEST_DIR / "test_shift_file_PS1.dat"
        shift_df = pd.read_csv(shift_file, comment="#")
        shift_model = copy.deepcopy(model)
        model._load_band_weights(bands_to_load=[f"{bp}_PS1" for bp in "griz"], shift_file=None)
        shift_model._load_band_weights(
            bands_to_load=[f"{bp}_PS1" for bp in "griz"],
            shift_file=shift_file,
            apply_lam_shifts=True,
            apply_mag_shifts=True,
        )
        dlam = shift_df[shift_df["BAND"] == "g_PS1"]["LAM_SHIFT"].values[0]
        dmag = shift_df[shift_df["BAND"] == "r_PS1"]["MAG_SHIFT"].values[0]
        assert model.zp_dict["r_PS1"] == shift_model.zp_dict["r_PS1"] - dmag
        assert all(np.array(model.band_lim_dict["g_PS1"]) == np.array(shift_model.band_lim_dict["g_PS1"]) - dlam)
        model._init_band_weights()

    @pytest.mark.parametrize("bands_to_load,new_N_bands,clean", bp_caching_args)
    def test_bp_caching(self, model: SEDmodel, bands_to_load: list, new_N_bands: int, clean: bool):
        model._load_band_weights(bands_to_load)
        for model_dict in (model.band_dict, model.zp_dict, model.band_lim_dict):
            assert len(set(bands_to_load).difference(set(model_dict))) == 0
        assert model.band_interpolate_weights.shape == (new_N_bands, len(model.hires_wave))
        for arr in (model.zps, model.wave_sigmas):
            assert arr.shape == (new_N_bands,)
        assert model.calib_cov.shape == (new_N_bands, new_N_bands)
        if clean:
            model._init_band_weights()

    @pytest.mark.parametrize("bands,clean", set_used_bands_args)
    def test_set_used_bands(self, model: SEDmodel, bands: None | list, clean: bool):
        if bands is None:
            with pytest.warns(UserWarning):
                model._set_used_bands(bands=bands)
            bands = list(model.band_dict.keys())
        else:
            model._set_used_bands(bands=bands)
        if "NULL_BAND" not in bands:
            bands = ["NULL_BAND",] + list(bands)
        assert (model.used_band_inds == np.array([model.band_dict[bp] for bp in bands])).all()
        assert model.used_band_dict == dict(zip(model.used_band_inds, range(len(bands))))
        assert (model.used_zps == model.zps[model.used_band_inds]).all()
        assert (model.used_calib_cov == model.calib_cov[jnp.ix_(model.used_band_inds, model.used_band_inds)]).all()
        assert (model.used_calib_chcov == jnp.linalg.cholesky(model.used_calib_cov)).all()
        assert (model.used_wave_sigmas == model.wave_sigmas[model.used_band_inds]).all()
        if clean:
            model._init_band_weights()

    def test_calculate_band_weights(self, model: SEDmodel, sample_model_parameters: tuple[jax.Array, ...], rtol: float=1e-14, atol: float=1e-14):
        AV = sample_model_parameters[1]
        redshifts = sample_model_parameters[9]
        ebv = AV / model.RV_MW
        bands = [f"{bp}_PS1" for bp in "griz"]
        model._set_used_bands(bands=bands)
        band_weights = model._calculate_band_weights(redshifts, ebv, lam_shifts=0)
        with open(PICKLE_DIR / "T21_band_weights.pkl", "rb") as f:
            example_band_weights = pickle.load(f)
        # +1 for NULL_BAND
        assert band_weights.shape == (N_sn, len(model.model_wave), len(bands)+1)
        assert jnp.isclose(band_weights, example_band_weights, rtol=rtol, atol=atol).all()

class TestYaml:
    expected_values: dict = {
        "infer_dust_properties": [False, True, True],
        "train_new_model": [False, True, False],
        "fix_tmax": [False, True, True],
        "vary_redshift": [False, False, False],
        "muhat_err": [5, None, None],
        "data_type": ["flux", "mag", "flux"],
        "mu_R_min": [1.2, 1, 1.2],
        "mu_R_max": [6, 5, 6],
    }

    def test_parse_mode_fitting(self, initial_args: dict, model: SEDmodel):
        mode_args = copy.deepcopy(initial_args)
        mode_args["mode"] = "fitting"
        mode_args = model._parse_mode(mode_args)
        for key, val in self.expected_values.items():
            assert mode_args[key] == val[0]

    def test_parse_mode_conflicting_args(self, initial_args: dict, model: SEDmodel):
        mode_args = copy.deepcopy(initial_args)
        mode_args["mode"] = "fitting"
        mode_args["train_new_model"] = True
        with pytest.raises(ValueError):
            model._parse_mode(mode_args)

    def test_parse_mode_training(self, initial_args: dict, model: SEDmodel):
        for mode in (
            "training_popRv",
            "training_globalRv",
            "training_uniformRv",
        ):
            mode_args = copy.deepcopy(initial_args)
            mode_args["mode"] = mode
            if "rv_type" in mode_args:
                mode_args.pop("rv_type")
            mode_args = model._parse_mode(mode_args)
            for key, val in self.expected_values.items():
                assert mode_args[key] == val[1]
            assert mode_args["rv_type"] == mode.split('_')[1].replace("Rv", "")

    def test_parse_mode_dust(self, initial_args: dict, model: SEDmodel):
        for mode in (
            "dust",
            "dust_redshift",
            "dust_split_mag",
            "dust_split_sed",
        ):
            mode_args = copy.deepcopy(initial_args)
            mode_args["mode"] = mode
            if 'rv_type' in mode_args:
                mode_args.pop("rv_type")
            mode_args = model._parse_mode(mode_args)
            assert mode_args["rv_type"] == "pop"
            for key, val in self.expected_values.items():
                comparison = val[2]
                if key == "vary_redshift" and mode == "dust_redshift":
                    comparison = True
                assert mode_args[key] == comparison
            if "split_mag" in mode:
                assert mode_args["split_variant"] == "split_mag"
            elif "split_sed" in mode:
                assert mode_args["split_variant"] == "split_sed"

    def test_parse_mode_dust_rv_not_pop(self, initial_args: dict, model: SEDmodel):
        mode_args = copy.deepcopy(initial_args)
        mode_args["mode"] = "dust"
        mode_args["rv_type"] = "global"
        with pytest.raises(ValueError):
            model._parse_mode(mode_args)

    @pytest.mark.parametrize("mode", ("dust", "training_uniformRV"))
    def test_parse_mode_conflicting_rv_types(self, initial_args: dict, model: SEDmodel, mode: str):
        mode_args = copy.deepcopy(initial_args)
        mode_args["mode"] = mode
        mode_args["rv_type"] = "global"
        with pytest.raises(ValueError):
            model._parse_mode(mode_args)

    def test_parse_mode_no_rv_type(self, initial_args: dict, model: SEDmodel):
        mode_args = copy.deepcopy(initial_args)
        mode_args["mode"] = "fit_Rv"
        with pytest.raises(ValueError):
            model._parse_mode(mode_args)

    def test_parse_args(self, loaded_model_and_args: tuple[SEDmodel, dict]):
        # If regenerating the pickled results, remove your personal directory structure
        # before dumping with args["data_root"] = args["data_root"].lstrip(BASE_DIR)
        _, args = loaded_model_and_args
        with open(PICKLE_DIR / "full_args.pkl", "rb") as f:
            ref_args = pickle.load(f)
        ref_args["data_root"] = str(BASE_DIR / ref_args["data_root"])
        ref_args["outputdir"] = Path(ref_args["outputdir"])
        assert args == ref_args

class TestSpectra:
    def test_get_spectra(self, model: SEDmodel, sample_model_parameters: tuple[jax.Array, ...], rtol: float=1e-14):
        theta, AV, W0, W1, eps, RV, _, J_t, hsiao_interp, _ = sample_model_parameters
        spec = model._get_spectra(theta=theta, AV=AV, W0=W0, W1=W1, eps=eps, RV=RV, J_t=J_t, hsiao_interp=hsiao_interp)
        with open(PICKLE_DIR / "T21_spectra.pkl", "rb") as f:
            example_spec = pickle.load(f)
        assert jnp.isclose(spec, example_spec, rtol=rtol).all()


    class TestWrapper:
        def test_against_method(self, model: SEDmodel, sample_model_parameters: tuple[jax.Array, ...]):
            theta, AV, W0, W1, eps, RV, t, J_t, hsiao_interp, _ = sample_model_parameters
            ref_spec = model._get_spectra(theta, AV, W0, W1, eps, RV, J_t, hsiao_interp)
            test_spec = model.get_spectra(theta=theta, AV=AV, W0=W0, W1=W1, eps=eps, RV=RV, J_t=J_t, hsiao_interp=hsiao_interp)
            assert jnp.isclose(test_spec, ref_spec, atol=0).all()

        def test_no_eps(self, model: SEDmodel, sample_model_parameters: tuple[jax.Array, ...]):
            theta, AV, W0, W1, eps, RV, t, J_t, hsiao_interp, _ = sample_model_parameters
            with pytest.raises(TypeError):
                model.get_spectra(theta=theta, AV=AV, W0=W0, W1=W1, eps=None, RV=RV, J_t=J_t, hsiao_interp=hsiao_interp)

        def test_defaults(self, model:SEDmodel, sample_model_parameters: tuple[jax.Array, ...]):
            theta, AV, W0, W1, eps, RV, t, J_t, hsiao_interp, _ = sample_model_parameters
            test_spec = model.get_spectra(theta=theta, AV=AV, W0=None, W1=None, eps=eps, RV=RV, J_t=J_t, hsiao_interp=hsiao_interp)
            ref_spec = model.get_spectra(theta=theta, AV=AV, W0=model.W0, W1=model.W1, eps=eps, RV=RV, J_t=J_t, hsiao_interp=hsiao_interp)
            assert jnp.isclose(test_spec, ref_spec, atol=0).all()

        def test_no_t_no_Jt_hi(self, model:SEDmodel, sample_model_parameters: tuple[jax.Array, ...]):
            theta, AV, W0, W1, eps, RV, t, J_t, hsiao_interp, _ = sample_model_parameters
            with pytest.raises(TypeError):
                model.get_spectra(theta=theta, AV=AV, W0=W0, W1=W1, eps=eps, RV=RV)

        def test_too_many_dims_Jt_hi(self, model:SEDmodel, sample_model_parameters: tuple[jax.Array, ...]):
            theta, AV, W0, W1, eps, RV, t, J_t, hsiao_interp, _ = sample_model_parameters
            double_theta = jnp.stack([theta, theta])
            with pytest.raises(ValueError):
                model.get_spectra(theta=double_theta, AV=AV, W0=W0, W1=W1, eps=eps, RV=RV, J_t=J_t, hsiao_interp=hsiao_interp)

        def test_too_many_dims_t(self, model:SEDmodel, sample_model_parameters: tuple[jax.Array, ...]):
            theta, AV, W0, W1, eps, RV, t, J_t, hsiao_interp, _ = sample_model_parameters
            double_t = jnp.stack([t, t])
            with pytest.raises(ValueError):
                model.get_spectra(theta=theta, AV=AV, W0=W0, W1=W1, eps=eps, RV=RV, t=double_t)

        def test_1d_mismatch_Jt_hi(self, model:SEDmodel, sample_model_parameters: tuple[jax.Array, ...]):
            theta, AV, W0, W1, eps, RV, t, J_t, hsiao_interp, _ = sample_model_parameters
            with pytest.raises(ValueError):
                model.get_spectra(theta=theta[:-1], AV=AV, W0=W0, W1=W1, eps=eps, RV=RV, J_t=J_t, hsiao_interp=hsiao_interp)

        def test_1d_mismatch_t(self, model:SEDmodel, sample_model_parameters: tuple[jax.Array, ...]):
            theta, AV, W0, W1, eps, RV, t, J_t, hsiao_interp, _ = sample_model_parameters
            with pytest.raises(ValueError):
                model.get_spectra(theta=theta, AV=AV, W0=W0, W1=W1, eps=eps, RV=RV, t=t[:,:-1])

        def test_scalar_params_scalar_t(self, model:SEDmodel, sample_model_parameters: tuple[jax.Array, ...]):
            theta, AV, W0, W1, eps, RV, t, J_t, hsiao_interp, _ = sample_model_parameters
            test_spec = model.get_spectra(theta=theta[0], AV=AV[0], W0=W0[0], W1=W1[0], eps=eps[0], RV=RV[0], t=t[0,0])
            ref_spec = model.get_spectra(theta=theta, AV=AV, W0=W0, W1=W1, eps=eps, RV=RV, J_t=J_t, hsiao_interp=hsiao_interp)
            assert len(test_spec.shape) == 1
            assert jnp.isclose(test_spec, ref_spec[0,:,0], atol=0).all()

        def test_scalar_params_1d_t(self, model:SEDmodel, sample_model_parameters: tuple[jax.Array, ...]):
            theta, AV, W0, W1, eps, RV, t, J_t, hsiao_interp, _ = sample_model_parameters
            test_spec = model.get_spectra(theta=theta[0], AV=AV[0], W0=W0[0], W1=W1[0], eps=eps[0], RV=RV[0], t=t[:,0])
            ref_spec = model.get_spectra(theta=theta, AV=AV, W0=W0, W1=W1, eps=eps, RV=RV, t=t)
            assert len(test_spec.shape) == 2
            assert jnp.isclose(test_spec.transpose(1,0), ref_spec[0], atol=0).all()

        def test_1d_params_scalar_t(self, model:SEDmodel, sample_model_parameters: tuple[jax.Array, ...]):
            theta, AV, W0, W1, eps, RV, t, J_t, hsiao_interp, _ = sample_model_parameters
            test_spec = model.get_spectra(theta=theta, AV=AV, W0=W0, W1=W1, eps=eps, RV=RV, t=t[0,0])
            ref_spec = model.get_spectra(theta=theta, AV=AV, W0=W0, W1=W1, eps=eps, RV=RV, t=t)
            assert jnp.isclose(test_spec, ref_spec[:,:,0], atol=0)[0].all()
            assert not jnp.isclose(test_spec, ref_spec[:,:,0], atol=0)[1:].all()

        def test_incompatible_Jt_hi(self, model:SEDmodel, sample_model_parameters: tuple[jax.Array, ...]):
            theta, AV, W0, W1, eps, RV, t, J_t, hsiao_interp, _ = sample_model_parameters
            with pytest.raises(ValueError):
                model.get_spectra(theta=theta, AV=AV, W0=W0, W1=W1, eps=eps, RV=RV, J_t=J_t[:-1], hsiao_interp=hsiao_interp)

        def test_incompatible_params_Jt_hi(self, model:SEDmodel, sample_model_parameters: tuple[jax.Array, ...]):
            theta, AV, W0, W1, eps, RV, t, J_t, hsiao_interp, _ = sample_model_parameters
            with pytest.raises(ValueError):
                model.get_spectra(theta=theta, AV=AV, W0=W0, W1=W1, eps=eps, RV=RV, J_t=J_t[:-1], hsiao_interp=hsiao_interp[:,:,:-1])

        def test_scalars_and_Jt_hi(self, model:SEDmodel, sample_model_parameters: tuple[jax.Array, ...]):
            theta, AV, W0, W1, eps, RV, t, J_t, hsiao_interp, _ = sample_model_parameters
            sn0_spec = model.get_spectra(theta=theta[0], AV=AV[0], W0=W0[0], W1=W1[0], eps=eps[0], RV=RV[0], J_t=J_t[:1], hsiao_interp=hsiao_interp[:,:,:1])
            ref_spec = model.get_spectra(theta=theta, AV=AV, W0=W0, W1=W1, eps=eps, RV=RV, J_t=J_t, hsiao_interp=hsiao_interp)
            assert jnp.isclose(sn0_spec, ref_spec[:1], atol=0).all()
class TestFlux:
    @pytest.fixture(scope="module")
    def flux_batch_args(self, model: SEDmodel, sample_model_parameters: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        flux_model = copy.deepcopy(model)
        bands = [f"{bp}_PS1" for bp in "griz"]
        flux_model._set_used_bands(bands=bands)
        theta, AV, W0, W1, eps, RV, t, J_t, hsiao_interp, z = sample_model_parameters
        spectra = flux_model._get_spectra(theta, AV, W0, W1, eps, RV, J_t, hsiao_interp)
        ebv = AV / flux_model.RV_MW
        band_weights = flux_model._calculate_band_weights(z, ebv, lam_shifts=0)
        Ds = np.array(flux_model.cosmo.distmod(np.array(z)))
        # +1 to get past NULL_BAND
        band_indices = jax.random.randint(rng_key, shape=(N_epochs, N_sn), minval=0, maxval=len(bands))+1
        # Masking randomly
        mask = jnp.ones_like(band_indices)
        mask_start_idx = jax.random.randint(rng_key, shape=(N_sn,), minval=5, maxval=10)
        for i in range(N_sn):
            mask = mask.at[mask_start_idx[i]:,i].set(0)
        return flux_model, spectra, mask, Ds, z, ebv, band_indices, mask, RV, band_weights

    def test_get_flux_batch(self, flux_batch_args: tuple[Any, ...], rtol: float=1e-14):
        flux_model, spectra, mask, Ds, z, ebv, band_indices, mask, RV, band_weights = flux_batch_args
        flux = flux_model.get_flux_batch(
            model_spectra=spectra,
            M0=flux_model.M0,
            Ds=Ds,
            z=z,
            ebv=ebv,
            RV=RV,
            band_indices=band_indices,
            mask=mask,
            weights=band_weights,
            mag_shift=0,
            num_batch=N_sn
        )
        with open(PICKLE_DIR / "T21_flux.pkl", "rb") as f:
            example_flux = pickle.load(f)
        assert flux.shape == (N_epochs, N_sn)
        assert jnp.isclose(flux, example_flux, rtol=rtol).all()

    def test_get_flux_batch_w_lam_shift(self, flux_batch_args: tuple[Any, ...], rtol: float=1e-5):
        # Bigger rtol because there's a normalization via trapezoidal integration even with shifts of 0 A.
        flux_model, spectra, mask, Ds, z, ebv, band_indices, mask, RV, band_weights = flux_batch_args
        flux = flux_model.get_flux_batch(
            model_spectra=spectra,
            M0=flux_model.M0,
            Ds=Ds,
            z=z,
            ebv=ebv,
            RV=RV,
            band_indices=band_indices,
            mask=mask,
            weights=band_weights,
            lam_shift=jnp.zeros(5),
            mag_shift=jnp.zeros(5),
            num_batch=N_sn
        )
        with open(PICKLE_DIR / "T21_flux.pkl", "rb") as f:
            example_flux = pickle.load(f)
        assert flux.shape == (N_epochs, N_sn)
        assert jnp.isclose(flux, example_flux, rtol=rtol).all()

    def test_get_mag_batch(self, flux_batch_args: tuple[Any, ...], rtol: float=1e-14):
        flux_model, spectra, mask, Ds, z, ebv, band_indices, mask, RV, band_weights = flux_batch_args
        data = {}
        for key, fn in zip(("flux", "mag"), (flux_model.get_flux_batch, flux_model.get_mag_batch)):
            data[key] = fn(
                model_spectra=spectra,
                M0=flux_model.M0,
                Ds=Ds,
                z=z,
                ebv=ebv,
                RV=RV,
                band_indices=band_indices,
                mask=mask,
                weights=band_weights,
                mag_shift=0,
                num_batch=N_sn
            )
        converted_mag = flux_model.ZPT - 2.5*jnp.log10(data["flux"])
        converted_mag = converted_mag.at[jnp.where(data["flux"] == 0)].set(0)
        assert jnp.isclose(converted_mag, data["mag"], atol=0).all()

class TestModelTrace:
    # These tests are very sensitive to the current code structure.
    # Even changing the order of numpyro.sample statements will lead to test failure.
    # Generate new pkl traces with caution, and only after you believe things are good.

    # Instantiating a bunch of different model_kwarg dicts to compare traces under
    # different configurations. The tests need to span possible use cases rather than
    # comprise sensible use cases.
    variants = {
        "fix_theta": {"fix_theta": 0},
        "fix_AV": {"fix_AV": 0.5},
        "fix_tmax": {"fix_tmax": True},
        "training": {"train_new_model": True},
        "split_mag": {"split_variant": "split_mag"},
        "split_sed": {"split_variant": "split_sed"},
        "vary_filter_shifts": {"vary_filter_shifts": True},
        "vary_offsets": {"vary_offsets": True},
        "vary_redshift": {"infer_dust_properties": True, "vary_redshift": True},
        "uniform": {},
        "pop": {"infer_dust_properties": True},
        "global": {"infer_dust_properties": True},
        "mag": {"data_type": "mag"},
        "photoz": {"photoz": True},
    }
    RV_types = []
    for key in variants:
        if key == "uniform":
            RV_types.append("uniform")
        elif key in ("pop", "vary_redshift"):
            RV_types.append("pop")
        else:
            RV_types.append("global")

    @pytest.mark.parametrize("variant,RV_type", zip((variants.keys()), RV_types))
    def test_trace(self, loaded_model_and_args: tuple[SEDmodel, dict], variant: str, RV_type: str):
        model, args = loaded_model_and_args
        kwargs = copy.deepcopy(args)
        kwargs.update(self.variants[variant])
        model.RV_type = RV_type
        test_trace = trace(
                seed(model._model, jax.random.PRNGKey(0))
            ).get_trace(model.data, model.band_weights, **kwargs)
        with open(PICKLE_DIR / f"T21_trace.{variant}.pkl", "rb") as file:
            ref_trace = pickle.load(file)
        for test, ref in zip(test_trace.values(), ref_trace.values()):
            if test["type"] == "sample" and ref["type"] == "sample" and test["name"]:
                assert (np.array(test["kwargs"]["rng_key"]) == np.array(ref["kwargs"]["rng_key"])).all()
            assert (test["value"] == ref["value"]).all()
        model.RV_type = "global"

class TestUtils:
    def test_inv_band_dict(self, model: SEDmodel):
        inv_dict = model.inv_band_dict
        for key, val in model.band_dict.items():
            assert val in inv_dict and inv_dict[val] == key
