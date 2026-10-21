import inspect
from ruamel.yaml import YAML
from pathlib import Path
import pickle
import sys

import numpy as np
import numpy.testing as npt
import pytest
from scipy.stats import chi2

from bayesn.bayesn_model import SEDmodel, default_kwargs
from bayesn import run_bayesn

BASE_DIR: Path = Path(__file__).parent.parent.absolute()

def test_get_cmd_args_minimal():
    old_argv = sys.argv
    with pytest.raises(SystemExit):
        sys.argv = [old_argv[0]]
        cmd_args = run_bayesn.get_cmd_args()
    sys.argv = [old_argv[0], "test.yaml"]
    cmd_args = run_bayesn.get_cmd_args()
    assert cmd_args.input == "test.yaml"
    sys.argv = [old_argv[0], "input=test.yaml"]
    cmd_args = run_bayesn.get_cmd_args()
    assert cmd_args.input == "input=test.yaml"
    for arg in [
        "snana", "photoz", "fix_tmax", "fix_eps", "vary_redshift", "train_new_model",
        "infer_dust_properties", "vary_filter_shifts", "vary_offsets",
        "save_fit_errors", "save_summary",
    ]:
        assert getattr(cmd_args, arg) == False
    sys.argv = old_argv

def test_get_cmd_args_str():
    old_argv = sys.argv
    sys.argv = [old_argv[0], "test.yaml", "--load_model", "T21_model", "--load_ext_rel=F99"]
    cmd_args = run_bayesn.get_cmd_args()
    assert cmd_args.load_model == "T21_model"
    assert cmd_args.load_ext_rel == "F99"
    assert cmd_args.fit_method is None
    with pytest.raises(TypeError, match="'int' object is not subscriptable"):
        sys.argv = [sys.argv[0], "test.yaml", "--data_root", 12345]
        cmd_args = run_bayesn.get_cmd_args()
    sys.argv = old_argv

def test_get_cmd_args_numbers():
    old_argv = sys.argv
    sys.argv = [old_argv[0], "test.yaml", "--num_chains", "5", "--fix_theta=1", "--fix_AV=1e-1"]
    cmd_args = run_bayesn.get_cmd_args()
    assert cmd_args.num_chains == 5
    assert cmd_args.fix_theta == 1.0
    assert cmd_args.fix_AV == 0.1
    assert cmd_args.num_samples is None
    assert cmd_args.M_split is None
    with pytest.raises(SystemExit):
        sys.argv.append("--num_samples=3.0")
        cmd_args = run_bayesn.get_cmd_args()
    sys.argv = old_argv

def test_get_cmd_args_false():
    old_argv = sys.argv
    sys.argv = [old_argv[0], "test.yaml", "--snana"]
    cmd_args = run_bayesn.get_cmd_args()
    assert cmd_args.snana == True
    assert cmd_args.photoz == False
    with pytest.raises(SystemExit):
        sys.argv.append("--photoz=True")
        cmd_args = run_bayesn.get_cmd_args()
    sys.argv = old_argv

def test_get_cmd_args_lists():
    old_argv = sys.argv
    sys.argv = [old_argv[0], "test.yaml", "--l_knots", "1", "2", "3", "--tau_knots", "1e-1", "1e0", "1e1"]
    cmd_args = run_bayesn.get_cmd_args()
    npt.assert_equal(cmd_args.l_knots, [1, 2, 3])
    npt.assert_equal(cmd_args.tau_knots, [0.1, 1, 10])
    with pytest.raises(SystemExit):
        sys.argv = [old_argv[0], "test.yaml", "--l_knots", "1,2,3,4"]
        cmd_args = run_bayesn.get_cmd_args()
    with pytest.raises(SystemExit):
        sys.argv = [old_argv[0], "test.yaml", "--l_knots", "[1, 2, 3, 4]"]
        cmd_args = run_bayesn.get_cmd_args()
    sys.argv = old_argv

def test_get_input_args(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        run_bayesn.get_input_args(tmp_path / "NON_EXISTENT")
    input_args = run_bayesn.get_input_args(BASE_DIR / "bayesn/defaults.yaml")
    for key in input_args:  # default_kwargs has AV_dist.
        assert input_args[key] == default_kwargs[key]

def test_override(tmp_path: Path):
    old_argv = sys.argv
    sys.argv = [old_argv[0], "test.yaml", "--load_model=G25_model", "--map", str(tmp_path / "map.txt")]
    cmd_args = run_bayesn.get_cmd_args()
    with open(str(tmp_path / "map.txt"), "w") as f:
        f.write("g g_PS1\nr r_PS1")
    input_args = {"load_model": "T21_model", "map": {}}
    with pytest.warns(UserWarning, match="differs from the command line"):
        new_input = run_bayesn.override_input_args_w_cmd_args(input_args, cmd_args)
    assert new_input["load_model"] == cmd_args.load_model
    assert new_input["map"] == {"g": "g_PS1", "r": "r_PS1"}
    assert new_input["filter_yaml"] is None
    sys.argv = old_argv

def test_get_init_args():
    init_sig = inspect.signature(SEDmodel.__init__)
    input_args = {"load_model": "test_fake_model"}
    with pytest.warns(UserWarning, match="load_ext_rel not specified"):
        init_args = run_bayesn.get_init_args(input_args)
    assert init_args["load_model"] == "test_fake_model"
    for param in init_sig.parameters.values():
        if param.name in ("self", "load_model"):
            continue
        assert init_args[param.name] == init_sig.parameters[param.name].default

@pytest.mark.slow
def test_main(tmpdir: Path):
    old_argv = sys.argv
    sys.argv = [
        sys.argv[0],
        str(BASE_DIR / "tests/test_files/input.yaml"),
        "--num_warmup=5",
        "--num_samples=5",
        "--outputdir", str(tmpdir)
    ]
    with pytest.warns(UserWarning):
        run_bayesn.main()
    for fname in ("chains.pkl", "fit_summary.csv", "input.yaml", "output.FITRES.TEXT", "output.LCPLOT", "output.LCSUMMARY.TEXT"):
        assert (tmpdir / fname).exists()
    sys.argv = old_argv

