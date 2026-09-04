import os
import copy
import pickle
from ruamel.yaml import YAML
import shutil
import time
from typing import Any
from warnings import warn

from bayesn.extinction_relations import DustExtRel
import argparse
import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
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

BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR: str = os.path.join(BASE_DIR, "tests", "test_files")
PICKLE_DIR: str = os.path.join(TEST_DIR, "pickles")
NON_EXISTENT_PATH: str = os.path.join("TEST_DIR", "non_existent")
N_sn: int = 5
N_epochs: int = 10
rng_seed: int = 1
rng_key: jax._src.prng.PRNGKeyArray = jax.random.key(rng_seed)

def non_existent_check():
    if os.path.exists(NON_EXISTENT_PATH):
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
    with open(os.path.join(TEST_DIR, "input.yaml"), "r") as file:
        args = yaml.load(file)
    args["data_root"] = os.path.join(BASE_DIR, args["data_root"])
    return args

@pytest.fixture(scope="module")
def F99() -> DustExtRel:
    return DustExtRel(name="F99", x_in="default", default_min_wave=3500, default_max_wave=9500)
#############
### Tests ###
#############
ext_args = (
    ("F99", 9e-15),
    ("C00", 3e-15),
)
de_args = (
    ("CCM89", 5e-12),
    ("O94", 6e-11),
    ("F04", 2e-14),
    ("VCG04", 3e-14),
    ("GCC09", 3e-14),
    ("M14", 2e-15),
    ("F19", 1e-14),
    ("D22", 5e-16),
    ("G23", 2e-9),
)

def test_custom_DER(F99: DustExtRel):
    test_der = DustExtRel(name=os.path.join(TEST_DIR, "test_ext_rel.yaml"))
    assert (F99.ax == test_der.ax).all()
    assert (F99.bx == test_der.bx).all()
    assert (F99.Jx == test_der.Jx).all()

def test_ext_rel_short_wl_coverage():
    with pytest.warns(UserWarning):
        DustExtRel(name=os.path.join(TEST_DIR, "test_ext_rel_short.yaml"))

def test_ext_rel_file_non_existent():
    non_existent_check()
    with pytest.raises(FileNotFoundError):
        DustExtRel(name=os.path.join(TEST_DIR, "non_existent"))

@pytest.mark.parametrize("name,atol", ext_args)
def test_vs_ext(name: str, atol: float):
    """
    dust_extinction and BayeSN differ in the type of cubic spline boundary conditions
    used in F99 and F19. BayeSN's spline math supports the natural boundary condition
    the scipy default used in F99 and F19 is the not-a-knot boundary conditions.
    The F99 paper uses natural boundaries and the F19 paper endorses the
    dust_extinction implementation.
    """
    ext_rel = DustExtRel(name=name, x_in="full_range")
    RVs = np.linspace(*ext_rel.rv_range, 30)
    test_axav = ext_rel._get_axav(RVs)
    x_exp = -1 if "inv" in ext_rel.units else 1
    match ext_rel.units.lower():
        case units if "micron" in units or "um" in units:
            x_mult = 1e4
        case units if "nanomet" in units or "nm" in units:
            x_mult = 10
        case units if "angstrom" in units or "aa" in units:
            x_mult = 1
    angstroms = x_mult*ext_rel.x**x_exp
    match name:
        case "F99":
            ref_der = extinction.fitzpatrick99
        case "C00":
            ref_der = extinction.calzetti00
        case _:
            raise NotImplementedError(
                f"{name} is not currently supported for comparisons with the "
                "extinction package."
            )
    ref_axav = np.zeros((30, len(angstroms)))
    for i, RV in enumerate(RVs):
        ref_axav[i] = ref_der(angstroms, 1, RV)
    assert jnp.isclose(test_axav, ref_axav, rtol=0, atol=atol).all()

@pytest.mark.parametrize("name,atol", de_args)
@pytest.mark.skipif(not de_loaded, reason="dust_extinction package not found.")
def test_vs_de(name: str, atol: float, RV: float = 3.1):
    """ Does not test the case where dust_extinction is defined over a wider domain
    than the BayeSN extinction relation.
    """
    ref_der = getattr(de, name)
    ext_rel = DustExtRel(name=name, x_in="full_range")
    RV_overlap = (max(ref_der.Rv_range[0], ext_rel.rv_range[0]), min(ref_der.Rv_range[1], ext_rel.rv_range[1]))
    x_exp = -1 if "inv" not in ext_rel.units else 1
    match ext_rel.units.lower():
        case units if "micron" in units or "um" in units:
            x_mult = 1
        case units if "nanomet" in units or "nm" in units:
            x_mult = 1e3
        case units if "angstrom" in units or "aa" in units:
            x_mult = 1e4
    inv_microns = x_mult*ext_rel.x**x_exp
    wave_overlap = (max(ref_der.x_range[0], min(inv_microns)), min(ref_der.x_range[1], max(inv_microns)))
    idx = jnp.where((ext_rel.x >= wave_overlap[0]) & (ext_rel.x <= wave_overlap[1]))
    RVs = np.linspace(*RV_overlap, 30)
    test_axav = ext_rel._get_axav(RVs)
    ref_axav = np.zeros((30, len(inv_microns[idx])))
    for i, RV in enumerate(RVs):
        ref_axav[i] = ref_der(RV)(inv_microns[idx]/u.micron)
    assert jnp.isclose(test_axav[:,idx[0]], ref_axav, rtol=0, atol=atol).all()


def test_getaxav_wrapper_RV_shape(F99: DustExtRel):
    with pytest.raises(ValueError):
        F99.get_axav(RV=jnp.full((2, 3), 3.1))

def test_getaxav_wrapper_extreme_RV(F99: DustExtRel):
    with pytest.warns(UserWarning):
        F99.get_axav(RV=1)
