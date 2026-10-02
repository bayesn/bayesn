#!/usr/bin/env python

import argparse
import inspect
import os

from ruamel.yaml import YAML

from bayesn import SEDmodel

def main():
    yaml = YAML(typ='safe')

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('--filters', type=str, required=False)
    parser.add_argument('--outputdir', type=str, required=False)
    parser.add_argument('--load_model', type=str, required=False)
    parser.add_argument('--load_ext_rel', type=str, required=False)
    parser.add_argument('--mode', type=str, required=False)
    parser.add_argument('--num_chains', type=int, required=False)
    parser.add_argument('--fit_method', type=str, required=False)
    parser.add_argument('--chain_method', type=str, required=False)
    parser.add_argument('--initialisation', type=str, required=False)
    parser.add_argument('--l_knots', type=float, required=False, nargs='*')
    parser.add_argument('--tau_knots', type=float, required=False, nargs='*')
    parser.add_argument('--map', type=str, required=False)
    parser.add_argument('--data_root', type=str, required=False)
    parser.add_argument('--data_table', type=str, required=False)
    parser.add_argument('--version_photometry', type=str, required=False)
    parser.add_argument('--drop_bands', type=str, required=False, nargs='*')
    parser.add_argument('--num_warmup', type=int, required=False)
    parser.add_argument('--num_samples', type=int, required=False)
    parser.add_argument('--snana', type=bool, required=False)
    parser.add_argument('--jobsplit', type=int, nargs=2)
    parser.add_argument('--outfile_prefix', type=str, required=False)
    parser.add_argument('--private_data_path', type=str, required=False, nargs='*')
    parser.add_argument('--sim_prescale', type=int, required=False)
    parser.add_argument('--photoz', action='store_true', default=None, required=False)
    parser.add_argument('--file_format', type=int, required=False)
    parser.add_argument('--save_fit_errors', type=int, required=False)
    parser.add_argument('--peakmjd_key', type=str, required=False)
    parser.add_argument('--keep_list', type=float, required=False)
    parser.add_argument('--error_floor', type=float, required=False)
    parser.add_argument('--batch_size', type=int, required=False)
    parser.add_argument('--num_lcplot', type=float, required=False)
    parser.add_argument('--save_summary', type=float, required=False)
    cmd_args = parser.parse_args()

    init_sig = inspect.signature(SEDmodel.__init__)

    if not os.path.exists(cmd_args.input):
        raise FileNotFoundError(f'Specified input file ({cmd_args.input}) was not found, please provide the path to an '
                                f'input yaml file or create an input.yaml in your current directory')
    with open(cmd_args.input, 'r') as file:
        args = yaml.load(file)

    # If no default model to load is specified in input.yaml or via command line, assume default
    if cmd_args.load_model is not None:
        if args.get("load_model", cmd_args.load_model) != cmd_args.load_model:
            print(
                f"Input yaml contains load_model={args['load_model']}, which differs "
                "from the command line arg {cmd_args.load_model}. The latter will "
                "override the former, so load_model will be {cmd_args.load_model}."
            )
        args["load_model"] = cmd_args.load_model
    elif "load_model" not in args:
        print(
            "load_model not specified in input yaml or cmd line. Assuming default "
            f"load_model={init_sig.parameters['load_model'].default}."
        )
    # Set default/override reddening law
    if cmd_args.load_ext_rel is not None:
        if args.get("load_ext_rel", cmd_args.load_ext_rel) != cmd_args.load_ext_rel:
            print(
                f"Input yaml contains load_ext_rel={args['load_ext_rel']}, which differs "
                "from the command line arg {cmd_args.load_ext_rel}. The latter will "
                "override the former, so load_ext_rel will be {cmd_args.load_ext_rel}."
            )
        args["load_ext_rel"] = cmd_args.load_ext_rel
    elif "load_ext_rel" not in args:
        print(
            "load_ext_rel not specified in input yaml or cmd line. Assuming default "
            f"load_ext_rel={init_sig.parameters['load_ext_rel'].default}."
        )

    # If no default filters.yaml is specified in input.yaml, use argparse value (including default of None)
    if "filters" not in args.keys():
        args["filters"] = cmd_args.filters
    args["filter_yaml"] = args.pop("filters")

    init_args = {}
    for param in init_sig.parameters.values():
        if param.name == "self":
            continue
        init_args[param.name] = args.get(param.name, param.default)

    model = SEDmodel(**init_args)
    model.run(args, cmd_args)

if __name__ == "__main__":
    main()
