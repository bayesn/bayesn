#!/usr/bin/env python
"""Command-line interface for running BayeSN inference jobs."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any
from ruamel.yaml import YAML
from warnings import warn

import numpy as np

from bayesn import SEDmodel
from bayesn.utils import get_yaml_comments

def get_cmd_args() -> argparse.Namespace:
    """
    Parse command-line arguments for running BayeSN.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    defaults_yaml = Path(__file__).parent / "defaults.yaml"
    arg_help = get_yaml_comments(defaults_yaml)
    arg_help.update({
        "input": "Path to the input YAML configuration file.",
        "load_model": "Path or name of pre-trained BayeSN model.",
        "load_ext_rel": "Path or name of extinction relation.",
        "filters": "Path to filters YAML specification.",
        "private_data_path": "Comma separated paths to non-public directories.",
        "data_root": "The base directory for relative file paths specified in a data_table.",
        "data_table": "A space-separated file with at least a 'files' column full of paths to data files.",
        "version_photometry": "A string specifying a directory with a similarly named .LIST file.",
        "snana": "If True, look for the version_photometry directory assuming an SNANA-like file structure.",
    })

    parser = argparse.ArgumentParser(
        description=(
            "A wrapper script that will instantiate a BayeSN SEDmodel, process a "
            "specified dataset, and store the results in a specified output directory."
        )
    )
    parser.add_argument('input', type=str, help=arg_help.get("input"))
    # str args
    for arg in [
        "load_model", "load_ext_rel", "filters", "mode", "outputdir", "fit_method",
        "chain_method", "initialisation", "map", "data_root", "data_table",
        "version_photometry", "outfile_prefix", "keep_list", "peakmjd_key",
        "laplace_method", "data_type", "split_variant",
    ]:
        parser.add_argument(f"--{arg}", type=str, required=False, help=arg_help.get(arg))
    # int args
    for arg in [
        "num_chains", "num_warmup", "num_samples", "lm_maxiter", "zltn_particles",
        "num_zltn_iter", "batch_size", "sim_prescale", "num_lcplot", "RNG_seed",
        "file_format" # should these be here?
    ]:
        parser.add_argument(f"--{arg}", type=int, required=False, help=arg_help.get(arg))
    # float args
    for arg in [
        "error_floor", "fix_theta", "fix_AV", "uniform_RV_min", "uniform_RV_max",
        "mu_R_min", "mu_R_max", "sigma_sigma_R", "tau_z_min", "tau_z_max",
        "fix_dist_limit", "fix_dist_Ds_err", "M_split", "muhat_err", "lm_lam_init",
        "zltn_lr", "zltn_lr_final", "zlim",
    ]:
        parser.add_argument(f"--{arg}", type=float, required=False, help=arg_help.get(arg))
    # default False bool args
    for arg in [
        "snana", "photoz", "fix_tmax", "fix_eps", "vary_redshift", "train_new_model",
        "infer_dust_properties", "vary_filter_shifts", "vary_offsets",
        "save_fit_errors", "save_summary",
    ]:
        parser.add_argument(
            f"--{arg}", action="store_true", required=False, help=arg_help.get(arg)
        )
    # list of floats args
    for arg in ["l_knots", "tau_knots"]:
        parser.add_argument(
            f"--{arg}", type=float, required=False, nargs='*', help=arg_help[arg]
        )
    # list of strs args
    for arg in ["drop_bands", "private_data_path"]:
        parser.add_argument(f"--{arg}", type=str, required=False, nargs='*', help=arg_help[arg])

    parser.add_argument("--jobsplit", type=int, nargs=2, help=arg_help["jobsplit"])

    return parser.parse_args()

def get_input_args(input_yaml: str | Path) -> dict[str, Any]:
    """
    Load configuration arguments from an input YAML file.

    Parameters
    ----------
    input_yaml
        Path to the input YAML configuration file.

    Returns
    -------
    dict
        Configuration dictionary parsed from YAML.
    """
    if not Path(input_yaml).exists():
        raise FileNotFoundError(
            f"Input file {input_yaml} was not found, please provide the path to an "
            f"input yaml file or create an input.yaml in your current directory. "
        )
    yaml = YAML(typ="safe")
    with open(input_yaml, "r") as file:
        input_args = yaml.load(file)
    return input_args

def override_input_args_w_cmd_args(input_args: dict[str, Any], cmd_args: argparse.Namespace) -> dict[str, Any]:
    """
    Override configuration arguments with explicitly provided command-line arguments.

    Parameters
    ----------
    input_args
        Base configuration dictionary from YAML.
    cmd_args
        Parsed command-line arguments.

    Returns
    -------
    dict
        Updated configuration dictionary with command-line overrides applied.
    """
    for key, val in vars(cmd_args).items():
        if key == "input" or val is None:
            continue
        if isinstance(val, bool) and not val:
            # boolean flags will default to False if not included, which should not
            # override specified input_args.
            continue
        if key == "map" and isinstance(val, str):
            filt_map = np.loadtxt(cmd_args.map, dtype=str)
            val = {row[0]: row[1] for row in filt_map}
        if input_args.get(key, val) != val:
            warn(UserWarning(
                f"Input yaml contains {key}={input_args[key]}, which differs from the "
                f"command line argument {val}. The latter will be override the former."
            ))
        input_args[key] = val

    # If no default filters.yaml is specified in the input.yaml, use the cmd_arg value
    # (including default None)
    if "filters" not in input_args:
        input_args["filters"] = cmd_args.filters
    input_args["filter_yaml"] = input_args.pop("filters")
    return input_args

def get_init_args(input_args: dict[str, Any]) -> dict[str, Any]:
    """
    Extract initialization arguments for ``SEDmodel.__init__`` from configuration.

    Parameters
    ----------
    input_args
        Configuration dictionary.

    Returns
    -------
    dict
        Keyword arguments to pass to ``SEDmodel.__init__``.
    """
    init_sig = inspect.signature(SEDmodel.__init__)
    init_args = {}
    for param in init_sig.parameters.values():
        if param.name == "self":
            continue
        if param.name in ("load_model", "load_ext_rel") and param.name not in input_args:
            warn(UserWarning(
                f"{param.name} not specified in input yaml or cmd line. Assuming "
                f"default {param.name}={param.default}."
            ))
        init_args[param.name] = input_args.get(param.name, param.default)
    return init_args

def main() -> None:
    """CLI entry point for running BayeSN inference jobs."""
    cmd_args = get_cmd_args()
    input_args = get_input_args(cmd_args.input)
    input_args = override_input_args_w_cmd_args(input_args, cmd_args)
    init_args = get_init_args(input_args)
    model = SEDmodel(**init_args)
    model.run(input_args)

if __name__ == "__main__":
    main()
