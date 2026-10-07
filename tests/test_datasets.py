import copy
from numbers import Number
import os
from pathlib import Path
import pickle
from typing import Callable

from astropy.table import QTable
import pandas as pd
import pandas.testing as pdt
import pytest
import numpy as np
import numpy.testing as npt

from bayesn import io
from bayesn.constants import C_LIGHT
from bayesn.utils import assert_dicts_match, convert_z, mag_to_flux, flux_to_mag
from bayesn.datasets import (
    SNDataset,
    meta_names,
    all_meta_names,
    get_standard_name,
    get_SNANA_name,
    clean_sn_dict,
    clean_obs_df,
)

BASE_DIR: Path = Path(__file__).parent.parent.absolute()
TEST_DIR: Path = BASE_DIR / "tests/test_files"
NON_EXISTENT_PATH: Path = TEST_DIR / "non_existent"

def non_existent_check():
    if NON_EXISTENT_PATH.exists():
        raise FileExistsError(
            f"{NON_EXISTENT_PATH} exists, so this test cannot trigger the expected "
            "FileNotFoundError."
        )

def random_sn_dict(RNG_seed=0, N=1, sim=False) -> dict[str, str | Number]:
    rng = np.random.default_rng(RNG_seed)
    sn_dict = {
            "snid":             np.array([f"test{i}" for i in range(N)]),
            "field":            np.full(N, "test_field"),
            "idsurvey":         np.full(N, "test_survey"),
            "cutflag_snana":    np.full(N, "test_cut"),
            "ra":               rng.uniform(size=N)*360,
            "dec":              rng.uniform(size=N)*180-90,
            "peak_mjd":         rng.normal(5e4, 5, N),
            "sn_type":          np.full(N, 1),
            "z_helio":          rng.lognormal(np.log(3e-2), 0.1, N),
            "z_helio_err":      rng.lognormal(np.log(1e-4), 0.1, N),
            "z_cmb":            rng.lognormal(np.log(3e-2), 0.1, N),
            "z_cmb_err":        rng.lognormal(np.log(1e-4), 0.1, N),
            "z_hubble":         rng.lognormal(np.log(3e-2), 0.1, N),
            "z_hubble_err":     rng.lognormal(np.log(1e-4), 0.1, N),
            "mwebv":            rng.exponential(0.1, N),
            "mwebv_err":        rng.lognormal(np.log(1e-2), 0.1, N),
            "host_logmass":     rng.lognormal(np.log(10), 0.1, N),
            "host_logmass_err": rng.lognormal(0, 0.1, N),
            "vpec":             rng.normal(size=N)*150,
            "vpec_err":         rng.lognormal(np.log(100), size=N),
        }
    if sim:
        sn_dict.update({
                "sim_gentypes":      np.ones(N),
                "sim_template_ids":  np.zeros(N),
                "sim_libids":        rng.choice(100, size=N),
                "sim_redshift_cmbs": rng.lognormal(np.log(3e-2), 0.1, N),
                "sim_vpecs":         rng.normal(size=N)*150,
                "sim_dlmags":        rng.lognormal(np.log(35.5), 0.1, N),
                "sim_peakmjds":      rng.normal(5e4, 5, N),
                "sim_thetas":        rng.normal(size=N),
                "sim_AVs":           rng.exponential(0.1, N),
                "sim_RVs":           rng.uniform(1.2, 6, N),
            })
    return sn_dict

def random_obs_df(RNG_seed: int = 0, zp: Number = 27.5) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_seed)
    N_obs = rng.choice(15)
    mjd = rng.normal(5e4, 10, N_obs)
    flt = rng.choice(list("abcdefghi"), N_obs)
    flux = rng.lognormal(np.log(100), 1, N_obs)
    flux_err = rng.lognormal(np.log(10), 1, N_obs)
    mag, mag_err = flux_to_mag(flux, flux_err, zp=zp)
    obs_df = pd.DataFrame({
        "mjd": mjd, "flt": flt, "flux": flux, "flux_err": flux_err,
        "mag": mag, "mag_err": mag_err
    })
    obs_df["snid"] = f"test{RNG_seed}"
    obs_df["snid"] = obs_df["snid"].astype("category")
    return obs_df

def format_df(df: pd.DataFrame):
    df = df.sort_values(["snid", "flt", "mjd"]).reset_index(drop=True)
    df = df[["snid", "flt", "mjd", "flux", "flux_err", "mag", "mag_err"]]
    df["snid"] = df["snid"].astype("category")
    return df

@pytest.fixture
def sample_data_single_sn() -> tuple[dict[str, np.ndarray], pd.DataFrame, np.ndarray]:
    N_sn = 1
    sn_dict = random_sn_dict(RNG_seed=0, N=N_sn)
    sn_dict["test_key"] = np.arange(N_sn)
    obs_df = random_obs_df(RNG_seed=0)
    return sn_dict, obs_df

@pytest.fixture
def sample_data_two_sne() -> tuple[dict[str, np.ndarray], pd.DataFrame, np.ndarray]:
    N_sn = 2
    sn_dict = random_sn_dict(RNG_seed=0, N=N_sn)
    sn_dict["test_key"] = np.arange(N_sn)
    obs_dfs = [random_obs_df(RNG_seed=i) for i in range(N_sn)]
    obs_df = pd.concat(obs_dfs, ignore_index=True)
    return sn_dict, obs_df

@pytest.fixture
def sample_data_sim() -> tuple[dict[str, np.ndarray], pd.DataFrame, np.ndarray]:
    N_sn = 5
    sn_dict = random_sn_dict(RNG_seed=0, N=N_sn, sim=True)
    obs_dfs = [random_obs_df(RNG_seed=i) for i in range(N_sn)]
    obs_df = pd.concat(obs_dfs, ignore_index=True)
    return sn_dict, obs_df


def make_dataset(sn_dict: dict, obs_df: pd.DataFrame, sim: bool=False) -> SNDataset:
    return SNDataset(
        N_sn=len(sn_dict["snid"]),
        photometry=obs_df,
        sim=sim,
        other_metadata={k: np.array(v) for k, v in sn_dict.items() if k not in all_meta_names},
        **{k: np.array(v) for k, v in sn_dict.items() if k in all_meta_names}
    )

@pytest.fixture
def dataset_single_sn(sample_data_single_sn) -> SNDataset:
    return make_dataset(*sample_data_single_sn)

@pytest.fixture
def dataset_two_sne(sample_data_two_sne) -> SNDataset:
    return make_dataset(*sample_data_two_sne)

@pytest.fixture
def dataset_sim(sample_data_sim) -> SNDataset:
    return make_dataset(*sample_data_sim, sim=True)

class TestGlobals:
    def test_standard_SNANA_name_roundtrip(self):
        for name in all_meta_names:
            SNANA_name = get_SNANA_name(name)
            std_name = get_standard_name(SNANA_name)
            assert std_name == name
        with pytest.warns(UserWarning, match="Not sure"):
            assert get_SNANA_name("test_key") == "TEST_KEY"

    def test_clean_sn_dict_rename(self, sample_data_single_sn):
        ref_dict = sample_data_single_sn[0]
        test_dict = copy.deepcopy(ref_dict)
        for key in ("z_helio", "z_helio_err", "dec", "snid"):
            test_dict[get_SNANA_name(key)] = test_dict.pop(key)
        test_dict = clean_sn_dict(test_dict)
        assert_dicts_match(test_dict, sample_data_single_sn[0])

    def test_clean_sn_dict_0D(self, sample_data_single_sn):
        ref_dict = sample_data_single_sn[0]
        test_dict = copy.deepcopy(ref_dict)
        for key in ("sn_type", "mwebv", "mwebv_err", "ra", "field"):
            test_dict[key] = test_dict[key][0]
        test_dict = clean_sn_dict(test_dict)
        assert_dicts_match(test_dict, sample_data_single_sn[0])

    def test_clean_sn_dict_padding(self, sample_data_sim):
        ref_dict = sample_data_sim[0]
        test_dict = copy.deepcopy(ref_dict)
        test_dict.pop("peak_mjd")
        ref_dict["peak_mjd"] = np.full(len(ref_dict["snid"]), None)
        test_dict = clean_sn_dict(test_dict)
        assert_dicts_match(test_dict, sample_data_sim[0])

    def test_clean_obs_df_rename(self, sample_data_single_sn):
        snids = sample_data_single_sn[0]["snid"]
        ref_df = sample_data_single_sn[1]
        test_df = copy.deepcopy(ref_df)
        test_df.rename({"flux": get_SNANA_name("flux"), "flt": "BAND", "snid": "SNID"})
        test_df = clean_obs_df(test_df, snids, phot_idx=None)
        pdt.assert_frame_equal(format_df(test_df), format_df(ref_df))

    def test_clean_obs_df_add_snid_for_single(self, sample_data_single_sn):
        snids = sample_data_single_sn[0]["snid"]
        ref_df = sample_data_single_sn[1]
        test_df = copy.deepcopy(ref_df)
        test_df.pop("snid")
        test_df = clean_obs_df(test_df, snids, phot_idx=None)
        pdt.assert_frame_equal(format_df(test_df), format_df(ref_df))

    def test_clean_obs_df_add_snid_for_multi(self, sample_data_two_sne):
        snids = sample_data_two_sne[0]["snid"]
        ref_df = sample_data_two_sne[1]
        test_df = copy.deepcopy(ref_df)
        test_df.pop("snid")
        with pytest.raises(TypeError, match="phot_idx cannot be inferred"):
            test_df = clean_obs_df(test_df, snids, phot_idx=None)
        phot_idx = np.array([0, sum(ref_df["snid"] == "test0"), len(ref_df)])
        test_df = clean_obs_df(test_df, snids, phot_idx=phot_idx)
        pdt.assert_frame_equal(format_df(test_df), format_df(ref_df))

    def test_clean_obs_extra_col(self, sample_data_single_sn):
        snids = sample_data_single_sn[0]["snid"]
        ref_df = format_df(sample_data_single_sn[1])
        phase_col = np.arange(len(ref_df))
        # Inserting phase column before data columns
        ref_df["phase"] = phase_col
        ref_df = ref_df[["snid", "flt", "mjd", "phase", "flux", "flux_err", "mag", "mag_err"]]
        test_df = copy.deepcopy(ref_df)
        test_df.rename({"flux": get_SNANA_name("flux"), "flt": "BAND", "snid": "SNID"})
        test_df = clean_obs_df(test_df, snids, phot_idx=None)
        # Only checks snid, flt, mjd, flux+err, mag+err
        pdt.assert_frame_equal(format_df(test_df), format_df(ref_df))
        assert test_df.columns[-1] == "phase"  # should come after req and data columns
        pdt.assert_series_equal(test_df["phase"], ref_df["phase"])

    def test_clean_obs_df_empty(self, sample_data_single_sn):
        empty_df = pd.DataFrame()
        assert len(clean_obs_df(empty_df, sample_data_single_sn[0]["snid"])) == 0

class TestInit:
    def test_init_empty(self):
        ds = SNDataset()
        assert ds.N_sn == 0
        assert ds.sim is False
        npt.assert_equal(ds.phot_idx, np.array([0]))
        for attr in meta_names["str"] + meta_names["num"]:
            npt.assert_equal(getattr(ds, attr), np.array([]))
        for attr in meta_names["sim"]:
            assert getattr(ds, attr) is None

    def test_bad_init(self):
        with pytest.raises(AssertionError, match="snid"):
            # Fails when len(ds.snid) != N_sn
            ds = SNDataset(N_sn=1)
        with pytest.raises(AssertionError, match="field"):
            # Does snid instantiation, fails when len(ds.field) != N_sn (1)
            ds = SNDataset(N_sn=1, snid=np.array(["test"]))

    def test_init(self, sample_data_two_sne, dataset_two_sne):
        sn_dict, obs_df = sample_data_two_sne
        phot_idx = np.array([0, obs_df["snid"].value_counts()["test0"], len(obs_df)])
        obs_df = format_df(obs_df)
        assert dataset_two_sne.N_sn == 2
        for attr in all_meta_names:
            npt.assert_equal(getattr(dataset_two_sne, attr), sn_dict.get(attr))
        pdt.assert_frame_equal(dataset_two_sne.photometry, obs_df)

    def test_init_sim(self, sample_data_sim, dataset_sim):
        sn_dict = sample_data_sim[0]
        for attr in meta_names["sim"]:
            npt.assert_equal(getattr(dataset_sim, attr), sn_dict[attr])

    def test_init_0d_arrs(self, sample_data_single_sn, dataset_single_sn):
        sn_dict, obs_df = sample_data_single_sn
        test_ds = SNDataset(
            N_sn=1,
            photometry=obs_df,
            # init meta keys w/ scalars/strings instead of ArrayLikes
            **{key: val[0] for key, val in sn_dict.items() if key in all_meta_names}
        )
        assert test_ds == dataset_single_sn

    def test_eq_phot(self, dataset_two_sne):
        copied_ds = copy.deepcopy(dataset_two_sne)
        assert dataset_two_sne == copied_ds
        copied_ds.photometry.loc[5, "flux"] *= 2.
        assert dataset_two_sne != copied_ds

    def test_eq_meta(self, dataset_two_sne):
        copied_ds = copy.deepcopy(dataset_two_sne)
        assert dataset_two_sne == copied_ds
        copied_ds.z_helio[0] = 0.1
        assert dataset_two_sne != copied_ds

class TestAttributesProperties:
    def test_unique_bands(self, sample_data_two_sne, dataset_two_sne):
        obs_df = sample_data_two_sne[1]
        ref_bands = np.sort(obs_df["flt"].unique().astype(str))
        test_bands = np.sort(dataset_two_sne.unique_bands.astype(str))
        npt.assert_equal(ref_bands, test_bands)

    def test_metadata(self, sample_data_sim, dataset_sim):
        sn_dict = sample_data_sim[0]
        ref_meta = dict(zip(
            all_meta_names, [sn_dict.get(attr) for attr in all_meta_names]
        ))
        assert_dicts_match(ref_meta, dataset_sim.metadata)

class TestDataAddition:
    def test_append_new_single(self, sample_data_single_sn, dataset_single_sn):
        sn_dict, obs_df = sample_data_single_sn
        test_ds = SNDataset()
        test_ds._append_new(sn_dict, obs_df)
        test_ds._clean_photometry()
        assert test_ds == dataset_single_sn

    def test_append_new_single_missing_data(self, sample_data_single_sn, dataset_single_sn):
        test_ds = SNDataset()
        sn_dict, obs_df = sample_data_single_sn
        drop_keys = (
            "z_hubble", "z_hubble_err", "vpec", "vpec_err", "host_logmass",
            "host_logmass_err"
        )
        for key in drop_keys:
            sn_dict.pop(key)
        test_ds._append_new(sn_dict, obs_df)
        test_ds._clean_photometry()
        for attr in all_meta_names:
            if attr in drop_keys:
                assert getattr(test_ds, attr) == np.array([None])
            else:
                assert getattr(test_ds, attr) == sn_dict.get(attr)
        pdt.assert_frame_equal(test_ds.photometry, dataset_single_sn.photometry)

    def test_append_new_multi(self, sample_data_sim, dataset_sim):
        sn_dict, obs_df = sample_data_sim
        test_ds = SNDataset()
        test_ds._append_new(sn_dict, obs_df)
        test_ds._clean_photometry()
        assert test_ds == dataset_sim

    def test_append_duplicate_single(self, sample_data_single_sn, dataset_single_sn):
        sn_dict, obs_df = sample_data_single_sn
        new_df = copy.deepcopy(obs_df)
        # Buffering mjd to avoid potential data discrepancies.
        new_df["mjd"] += obs_df["mjd"].max() - new_df["mjd"].min() + 1
        expected_phot = format_df(pd.concat([obs_df, new_df]))
        dataset_single_sn._append_duplicate(sn_dict, new_df)
        assert_dicts_match(dataset_single_sn.metadata, sn_dict)
        pdt.assert_frame_equal(dataset_single_sn.photometry, expected_phot)

    def test_append_duplicate_phot_overlap(self, sample_data_single_sn, dataset_single_sn):
        sn_dict, obs_df = sample_data_single_sn
        test_ds = copy.deepcopy(dataset_single_sn)
        # 100% overlap changes nothing
        test_ds._append_duplicate(sn_dict, obs_df)
        test_ds._clean_photometry()
        pdt.assert_frame_equal(test_ds.photometry, test_ds.photometry)

        # mismatch between new and old data
        new_df = copy.deepcopy(obs_df)
        with pytest.raises(ValueError, match="There are discrepancies"):
            new_df["flux"] += 1  # mjd/filt match + discrepancies elsewhere
            dataset_single_sn._append_duplicate(sn_dict, new_df)

        # some new photometry
        new_df = copy.deepcopy(obs_df)
        # 0:2 grabs first three rows here
        new_df.loc[0:2, "mjd"] += new_df["mjd"].max() - new_df["mjd"].min()
        test_ds._append_duplicate(sn_dict, new_df)
        test_ds._clean_photometry()
        # 0:3 grabs first three rows here
        expected_phot = format_df(pd.concat([dataset_single_sn.photometry, new_df[0:3]]))
        assert_dicts_match(test_ds.metadata, dataset_single_sn.metadata)
        pdt.assert_frame_equal(test_ds.photometry, expected_phot)

    def test_append_duplicate_new_metadata(self, sample_data_single_sn, dataset_single_sn):
        sn_dict, obs_df = sample_data_single_sn
        dataset_single_sn.z_helio = np.array([None])
        dataset_single_sn.other_metadata = {}
        dataset_single_sn._append_duplicate(sn_dict, obs_df)
        npt.assert_equal(dataset_single_sn.z_helio, sn_dict["z_helio"])
        npt.assert_equal(dataset_single_sn.other_metadata["test_key"], sn_dict["test_key"])

    def test_append_duplicate_multi(self, sample_data_two_sne, dataset_two_sne):
        sn_dict, obs_df = sample_data_two_sne
        new_df = copy.deepcopy(obs_df)
        # Buffering mjd to avoid potential data discrepancies.
        new_df["mjd"] += obs_df["mjd"].max() - new_df["mjd"].min() + 1
        expected_phot = format_df(pd.concat([obs_df, new_df]))
        dataset_two_sne._append_duplicate(sn_dict, new_df)
        assert_dicts_match(dataset_two_sne.metadata, sn_dict)
        pdt.assert_frame_equal(dataset_two_sne.photometry, expected_phot)

    def test_append_empty(self):
        test_ds, ds_ref = [SNDataset() for _ in range(2)]
        test_ds.append(ds=None, sn_dict=None, obs_df=None)
        assert test_ds == ds_ref

    def test_append_dataset(self, sample_data_two_sne, dataset_two_sne):
        sn_dict, obs_df = sample_data_two_sne
        new_dict = copy.deepcopy(sn_dict)
        new_dict["snid"][0] = "test2"
        new_df = copy.deepcopy(obs_df)
        new_df["snid"] = new_df["snid"].cat.add_categories("test2")
        new_df.loc[new_df["snid"] == "test0", "snid"] = "test2"
        new_df["mjd"] += obs_df["mjd"].max() - new_df["mjd"].min() + 1
        ds_to_be_added = make_dataset(new_dict, new_df)
        test_ds = copy.deepcopy(dataset_two_sne)
        dataset_two_sne.append(sn_dict=new_dict, obs_df=new_df)
        test_ds.append(ds=ds_to_be_added)
        assert dataset_two_sne == test_ds

    def test_append_sn_dict_obs_df(self, sample_data_two_sne, dataset_two_sne):
        sn_dict, obs_df = sample_data_two_sne
        new_dict = copy.deepcopy(sn_dict)
        new_dict["snid"][0] = "test2"
        new_df = copy.deepcopy(obs_df)
        new_df["snid"] = new_df["snid"].cat.add_categories("test2")
        new_df.loc[new_df["snid"] == "test0", "snid"] = "test2"
        new_df["mjd"] += obs_df["mjd"].max() - new_df["mjd"].min() + 1
        test_ds = copy.deepcopy(dataset_two_sne)
        test_ds.append(sn_dict=new_dict, obs_df=new_df)
        dataset_two_sne.append(sn_dict=new_dict, obs_df=new_df)
        assert dataset_two_sne == test_ds

    def test_append_mismatch(self, sample_data_two_sne, dataset_two_sne):
        sn_dict, obs_df = sample_data_two_sne
        new_dict = copy.deepcopy(sn_dict)
        new_dict["snid"][0] = "test2"
        new_df = copy.deepcopy(obs_df)
        new_df["snid"] = new_df["snid"].cat.add_categories("test2")
        new_df.loc[new_df["snid"] == "test0", "snid"] = "test2"
        new_df["mjd"] += obs_df["mjd"].max() - new_df["mjd"].min() + 1
        ds_to_be_added = make_dataset(new_dict, copy.deepcopy(new_df))
        new_df["flux"] += 10
        with pytest.raises(ValueError, match="The provided arguments are not equiv"):
            dataset_two_sne.append(ds=ds_to_be_added, sn_dict=new_dict, obs_df=new_df)


    def test_append_missing_other_metadata(self, sample_data_two_sne, dataset_two_sne):
        sn_dict, obs_df = sample_data_two_sne
        new_dict = copy.deepcopy(sn_dict)
        new_dict["snid"][0] = "test2"
        new_df = copy.deepcopy(obs_df)
        new_df["snid"] = new_df["snid"].cat.add_categories("test2")
        new_df.loc[new_df["snid"] == "test0", "snid"] = "test2"
        new_df["mjd"] += obs_df["mjd"].max() - new_df["mjd"].min() + 1
        new_dict.pop("test_key")
        test_ds = copy.deepcopy(dataset_two_sne)
        test_ds.append(sn_dict=new_dict, obs_df=new_df)
        ref_test_key = np.append(dataset_two_sne.other_metadata["test_key"], None)
        npt.assert_equal(test_ds.other_metadata["test_key"], ref_test_key)

    def test_append_infer_phot_idx(self, sample_data_single_sn, dataset_single_sn):
        sn_dict, obs_df = sample_data_single_sn
        obs_df.pop("snid")
        test_ds = SNDataset()
        test_ds.append(sn_dict=sn_dict, obs_df=obs_df, phot_idx=None)
        dataset_single_sn.photometry = clean_obs_df(dataset_single_sn.photometry, sn_dict["snid"])
        assert test_ds == dataset_single_sn

class TestGetterMethods:
    def test_get_idx(self, dataset_two_sne):
        assert dataset_two_sne.get_idx("test0") == 0
        npt.assert_equal(dataset_two_sne.get_idx(["test0"]), np.array([0]))
        npt.assert_equal(dataset_two_sne.get_idx(["test1", "test0"]), np.array([1, 0]))

    def test_get_idx_not_found(self, dataset_two_sne):
        with pytest.raises(ValueError, match="snid missing not found."):
            dataset_two_sne.get_idx(snid="missing")

    def test_get_idx_dtype(self, dataset_two_sne):
        with pytest.raises(TypeError, match="snid of type <class 'NoneType'>"):
            dataset_two_sne.get_idx(snid=None)

    def test_parse_snid_idx(self, dataset_two_sne):
        assert dataset_two_sne._parse_snid_idx_args(idx=1) == 1
        assert dataset_two_sne._parse_snid_idx_args(snid="test1") == 1
        npt.assert_equal(dataset_two_sne._parse_snid_idx_args(snid=["test0", "test1"]), np.array([0, 1]))

    def test_parse_snid_idx_args_no_args(self, dataset_two_sne):
        with pytest.raises(ValueError, match="Either snid or idx should be specified."):
            dataset_two_sne._parse_snid_idx_args()

    def test_parse_snid_idx_args_diff_args(self, dataset_two_sne):
        with pytest.raises(ValueError, match="Either snid or idx should be specified, not both."):
            dataset_two_sne._parse_snid_idx_args(idx=0, snid="test1")

    def test_parse_snid_idx_matching_args(self, dataset_two_sne):
        assert dataset_two_sne._parse_snid_idx_args(idx=0, snid="test0") == 0

    def test_get_metadata_subset(self, sample_data_sim, dataset_two_sne, dataset_sim):
        sn_dict = sample_data_sim[0]
        ref_meta0 = {key: np.atleast_1d(val[0]) for key, val in sn_dict.items()}
        ref_meta10 = {key: np.array([val[4], val[2], val[3]]) for key, val in sn_dict.items()}
        assert_dicts_match(ref_meta0, dataset_sim.get_metadata_subset(idx=0))
        assert_dicts_match(ref_meta10, dataset_sim.get_metadata_subset(idx=[4, 2, 3]))
        no_sim_meta = dataset_two_sne.get_metadata_subset(idx=0)
        for attr in meta_names["sim"]:
            assert attr not in no_sim_meta

    def test_get_metadata_subset_empty(self, dataset_sim):
        ref_meta = dataset_sim.metadata
        test_meta = dataset_sim.get_metadata_subset()
        assert_dicts_match(ref_meta, test_meta)

    def test_get_phot_subset(self, sample_data_two_sne, dataset_two_sne):
        obs_df = sample_data_two_sne[1]
        df0 = format_df(obs_df[obs_df["snid"] == "test0"])
        df1 = format_df(obs_df[obs_df["snid"] == "test1"])
        switched_df = pd.concat([df1, df0], ignore_index=True)
        pdt.assert_frame_equal(dataset_two_sne.get_phot_subset(snid="test0"), df0)
        pdt.assert_frame_equal(dataset_two_sne.get_phot_subset(snid=["test1", "test0"]), switched_df)
class TestDataRemoval:
    def test_remove_sn(self, dataset_sim):
        ref_meta = dataset_sim.get_metadata_subset(idx=[0, 2, 4])
        ref_phot = dataset_sim.get_phot_subset(idx=[0, 2, 4])
        ref_phot_idx = dataset_sim.get_phot_idx_subset(idx=[0, 2, 4])
        with pytest.raises(IndexError, match="index 10 is out of bounds"):
            dataset_sim.remove_sn(idx=10)
        dataset_sim.remove_sn(snid=["test3", "test1"])
        assert dataset_sim.N_sn == 3
        assert_dicts_match(ref_meta, dataset_sim.metadata)
        pdt.assert_frame_equal(ref_phot, dataset_sim.photometry)
        # npt.assert_equal(ref_phot_idx, dataset_sim.phot_idx)

    def test_keep_according_to_list(self, dataset_sim):
        ref_meta = dataset_sim.get_metadata_subset(idx=[2, 3])
        ref_phot = dataset_sim.get_phot_subset(idx=[2, 3])
        dataset_sim.keep_according_to_list(["test2", "test3", "test10"])
        assert dataset_sim.N_sn == 2
        assert_dicts_match(ref_meta, dataset_sim.metadata)
        pdt.assert_frame_equal(ref_phot, dataset_sim.photometry)

    def test_remove_phot_idx(self, sample_data_sim, dataset_sim):
        # First object has more than 2 observations, so metadata shouldn't change.
        sn_dict, ref_phot = sample_data_sim
        ref_phot_idx = copy.deepcopy(dataset_sim.phot_idx)  # tied to dataset_sim so will change.
        ref_phot_idx[1:] -= 2
        ref_phot = format_df(ref_phot).drop(index=[0, 1]).reset_index(drop=True)
        dataset_sim.remove_phot_by_idx([0, 1])
        assert_dicts_match(sn_dict, dataset_sim.metadata)
        pdt.assert_frame_equal(ref_phot, dataset_sim.photometry)
        npt.assert_equal(ref_phot_idx, dataset_sim.phot_idx)

    def test_remove_phot_idx_drop_sn(self, sample_data_sim, dataset_sim):
        # Removing all photometry from first object should cause it to be dropped.
        sn_dict, ref_phot = sample_data_sim
        phot_idx = dataset_sim.phot_idx
        ref_meta = dataset_sim.get_metadata_subset(idx=np.arange(1, dataset_sim.N_sn))
        ref_phot_idx = phot_idx[1:] - phot_idx[1]
        ref_phot = format_df(ref_phot).drop(index=np.arange(phot_idx[1])).reset_index(drop=True)
        dataset_sim.remove_phot_by_idx(np.arange(phot_idx[1]))
        assert_dicts_match(ref_meta, dataset_sim.metadata)
        pdt.assert_frame_equal(ref_phot, dataset_sim.photometry)
        npt.assert_equal(ref_phot_idx, dataset_sim.phot_idx)

    def test_drop_bands(self, dataset_sim):
        unique_bands = dataset_sim.unique_bands
        original_length = len(dataset_sim.photometry)
        counts = [(dataset_sim.photometry["flt"] == b).sum() for b in unique_bands]
        with pytest.raises(TypeError, match="only list-like objects"):
            dataset_sim.drop_bands(unique_bands[0])
        dataset_sim.drop_bands([unique_bands[0], unique_bands[2]])
        assert unique_bands[0] not in dataset_sim.unique_bands
        assert unique_bands[2] not in dataset_sim.unique_bands
        assert len(dataset_sim.photometry) == original_length - counts[0] - counts[2]

    def test_drop_by_band_lims(self, dataset_sim):
        all_bands = dataset_sim.unique_bands
        original_length = len(dataset_sim.photometry)
        wave_min, wave_max = 2000, 9000
        # start band_lim dict with all bandpasses well within wave range.
        band_lim_dict = dict(zip(all_bands, [[4000, 7000] for _ in all_bands]))
        # Remove one band from lowest (highest) redshift due to red (blue) limit.
        # Other SNe should not be affected.
        zmin_idx, zmax_idx = dataset_sim.z_helio.argmin(), dataset_sim.z_helio.argmax()
        zmin_band, zmax_band = [
            dataset_sim.get_phot_subset(idx=idx)["flt"].unique()[0]
            for idx in (zmin_idx, zmax_idx)
        ]
        zmin_counts = (dataset_sim.get_phot_subset(idx=zmin_idx)["flt"] == zmin_band).sum()
        zmax_counts = (dataset_sim.get_phot_subset(idx=zmax_idx)["flt"] == zmax_band).sum()
        band_lim_dict[zmin_band][1] = wave_max*(1+dataset_sim.z_helio[zmin_idx]) + 1e-5
        band_lim_dict[zmax_band][0] = wave_min*(1+dataset_sim.z_helio[zmax_idx]) - 1e-5
        with pytest.raises(ValueError, match="The data contain a set of bandpasses not"):
            dataset_sim.drop_by_band_lims(
                band_lim_dict={}, wave_min=wave_min, wave_max=wave_max
            )
        dataset_sim.drop_by_band_lims(
            band_lim_dict=band_lim_dict, wave_min=wave_min, wave_max=wave_max
        )
        assert zmin_band not in dataset_sim.get_phot_subset(idx=zmin_idx)["flt"]
        assert zmax_band not in dataset_sim.get_phot_subset(idx=zmax_idx)["flt"]
        assert len(dataset_sim.photometry) == original_length - zmin_counts - zmax_counts

    def test_cut_by_meta_numeric(self, dataset_sim):
        test_ds = copy.deepcopy(dataset_sim)
        z = test_ds.z_cmb
        high_idx = np.where(z >= np.median(z))[0]
        ref_meta = test_ds.get_metadata_subset(idx=high_idx)
        ref_phot = test_ds.get_phot_subset(idx=high_idx)
        test_meta, test_phot = test_ds.cut_by_meta_numeric("z_cmb", "<", np.median(z), inplace=False)
        assert test_ds == dataset_sim
        assert_dicts_match(test_meta, ref_meta)
        pdt.assert_frame_equal(test_phot, ref_phot)
        test_ds.cut_by_meta_numeric("z_cmb", "<", np.median(z), inplace=True)
        assert test_ds != dataset_sim
        assert_dicts_match(test_ds.metadata, ref_meta)
        pdt.assert_frame_equal(test_ds.photometry, ref_phot)

    def test_cut_by_phot_numeric(self, dataset_sim):
        # Given unique minimum fluxes for all SNe, cutting fluxes >= the greatest
        # minimum flux should remove one SN from the dataset.
        test_ds = copy.deepcopy(dataset_sim)
        flux = test_ds.photometry["flux"]
        min_fluxes = [min(flux[test_ds.photometry["snid"] == f"test{i}"]) for i in range(test_ds.N_sn)]
        sn_to_be_removed = test_ds.snid[np.argmax(min_fluxes)]
        ref_meta = test_ds.get_metadata_subset(snid=[snid for snid in test_ds.snid if snid != sn_to_be_removed])
        ref_phot = test_ds.photometry[flux < max(min_fluxes)].reset_index(drop=True)
        test_meta, test_phot = test_ds.cut_by_phot_numeric("flux", ">=", max(min_fluxes), inplace=False)
        assert test_ds == dataset_sim
        assert_dicts_match(test_meta, ref_meta)
        pdt.assert_frame_equal(test_phot, ref_phot)
        test_ds.cut_by_phot_numeric("flux", ">=", max(min_fluxes), inplace=True)
        assert test_ds != dataset_sim
        assert_dicts_match(test_ds.metadata, ref_meta)
        pdt.assert_frame_equal(test_ds.photometry, ref_phot)

    def test_cut_by_meta_numeric_bad_col(self, dataset_sim):
        with pytest.raises(ValueError, match="foo not recognised."):
            dataset_sim.cut_by_meta_numeric("foo", "!=", 0)

    def test_cut_by_phot_numeric_bad_col(self, dataset_sim):
        with pytest.raises(ValueError, match="foo not recognised."):
            dataset_sim.cut_by_phot_numeric("foo", "<", 1)

class TestAstroGetter:
    def test_calculate_snrmaxes(self, dataset_two_sne):
        N_extra = 2
        for snid in dataset_two_sne.snid:
            phot = dataset_two_sne.get_phot_subset(snid=snid)
            phot["snr"] = phot["flux"] / phot["flux_err"]
            flts = phot.sort_values("snr", ascending=False)["flt"].unique()
            N_bands = len(flts)
            ref_snrmaxes = [phot[phot["flt"] == flt]["snr"].max() for flt in flts]
            ref_snrmaxes += [-99]*N_extra
            test_snrmaxes = dataset_two_sne.calculate_snrmaxes(snid=snid, N=N_bands+N_extra, default_value=-99)
            npt.assert_allclose(test_snrmaxes, ref_snrmaxes)
        with pytest.raises(AssertionError, match="Non-str snids are not supported."):
            dataset_two_sne.calculate_snrmaxes(snid=dataset_two_sne.snid)

    def test_estimate_tmax(self, dataset_two_sne):
        for snid in dataset_two_sne.snid:
            phot = dataset_two_sne.get_phot_subset(snid=snid)
            snr = phot["flux"] / phot["flux_err"]
            ref_tmax = np.average(phot["mjd"], weights=snr**2)
            test_tmax = dataset_two_sne.estimate_tmax(snid)
            npt.assert_allclose(test_tmax, ref_tmax)
        with pytest.raises(AssertionError, match="Non-str snids are not supported."):
            dataset_two_sne.estimate_tmax(snid=dataset_two_sne.snid)

    def test_calculate_rest_phases(self, dataset_two_sne):
        for idx, snid in enumerate(dataset_two_sne.snid):
            phot = dataset_two_sne.get_phot_subset(snid=snid)
            z_scaling = 1+dataset_two_sne.z_helio[idx]

            ref_phases0 = phot["mjd"]/z_scaling
            test_phases0 = dataset_two_sne.calculate_rest_phases(snid=snid, peak_mjd=0)
            npt.assert_allclose(test_phases0, ref_phases0)

            ref_phases = ref_phases0 - dataset_two_sne.peak_mjd[idx]/z_scaling
            test_phases = dataset_two_sne.calculate_rest_phases(snid=snid, peak_mjd=None)
            npt.assert_allclose(test_phases, ref_phases)
        with pytest.raises(AssertionError, match="Non-str snids are not supported."):
            dataset_two_sne.calculate_rest_phases(snid=dataset_two_sne.snid)

    def test_get_band_indices(self, dataset_two_sne):
        bands = dataset_two_sne.unique_bands
        default_band_dict = dict(zip(bands, range(1, len(bands)+1)))
        default_band_dict["NULL_BAND"] = 0
        ref_indices = np.array([default_band_dict[flt] for flt in dataset_two_sne.photometry["flt"]])
        test_indices = dataset_two_sne.get_band_indices()
        npt.assert_equal(test_indices, ref_indices)
        default_band_dict.pop(bands[0])
        with pytest.warns(UserWarning, match="The provided band_dict does not cover"):
            dataset_two_sne.get_band_indices(band_dict=default_band_dict)

class TestAstroSetter:
    def test_fill_out_redshifts_hub_to_hel(self, dataset_two_sne):
        N_sn, ra, dec, z_hubble, z_hubble_err, vpec, vpec_err = [
            getattr(dataset_two_sne, attr) for attr in
            ("N_sn", "ra", "dec", "z_hubble", "z_hubble_err", "vpec", "vpec_err")
        ]
        v = vpec / C_LIGHT
        dv = vpec_err / C_LIGHT
        z_cmb = (1 + z_hubble) / (1 + v) - 1
        dz_pv = z_hubble_err / (1 + v)
        z_dpv = -dv * (1 + z_hubble) / (1 + v)**2
        z_cmb_err = np.hypot(dz_pv, z_dpv)
        z_hel, z_hel_err = convert_z(
                z=z_cmb, ra=ra, dec=dec, z_in_type="cmb", z_err=z_cmb_err
        )
        for attr in ("z_helio", "z_helio_err", "z_cmb", "z_cmb_err"):
            setattr(dataset_two_sne, attr, np.full(N_sn, None))
        dataset_two_sne.fill_out_redshifts()
        npt.assert_allclose(dataset_two_sne.z_cmb.astype(float), z_cmb)
        npt.assert_allclose(dataset_two_sne.z_cmb_err.astype(float), z_cmb_err)
        npt.assert_allclose(dataset_two_sne.z_helio.astype(float), z_hel)
        npt.assert_allclose(dataset_two_sne.z_helio_err.astype(float), z_hel_err)

    def test_fill_out_redshifts_hel_to_hub(self, dataset_two_sne):
        N_sn, ra, dec, z_hel, z_hel_err, vpec, vpec_err = [
            getattr(dataset_two_sne, attr) for attr in
            ("N_sn", "ra", "dec", "z_helio", "z_helio_err", "vpec", "vpec_err")
        ]
        v = vpec / C_LIGHT
        dv = vpec_err / C_LIGHT
        z_cmb, z_cmb_err = convert_z(
                z=z_hel, ra=ra, dec=dec, z_in_type="hel", z_err=z_hel_err
        )
        z_hubble = (1 + z_cmb) * (1 + v) - 1
        dz_pv = z_cmb_err * (1 + v)
        z_dpv = dv * (1 + z_cmb)
        z_hubble_err = np.hypot(dz_pv, z_dpv)
        for attr in ("z_hubble", "z_hubble_err", "z_cmb", "z_cmb_err"):
            setattr(dataset_two_sne, attr, np.full(N_sn, None))
        dataset_two_sne.fill_out_redshifts()
        npt.assert_allclose(dataset_two_sne.z_cmb.astype(float), z_cmb)
        npt.assert_allclose(dataset_two_sne.z_cmb_err.astype(float), z_cmb_err)
        npt.assert_allclose(dataset_two_sne.z_hubble.astype(float), z_hubble)
        npt.assert_allclose(dataset_two_sne.z_hubble_err.astype(float), z_hubble_err)

    def test_fill_out_redshifts_no_overwrite(self, dataset_two_sne):
        N_sn, ra, dec, z_hel, z_hel_err, z_cmb_err, vpec = [
            getattr(dataset_two_sne, attr) for attr in
            ("N_sn", "ra", "dec", "z_helio", "z_helio_err", "z_cmb_err", "vpec")
        ]
        v = vpec / C_LIGHT
        z_cmb, other_z_cmb = convert_z(
                z=z_hel, ra=ra, dec=dec, z_in_type="hel", z_err=z_hel_err
        )
        z_hubble = (1 + z_cmb) * (1 + v) - 1
        z_hubble_err = z_cmb_err * (1 + v)   # for vpec_err = None
        for attr in ("z_hubble", "z_hubble_err", "z_cmb", "vpec_err"):
            setattr(dataset_two_sne, attr, np.full(N_sn, None))
        dataset_two_sne.fill_out_redshifts()
        npt.assert_allclose(dataset_two_sne.z_cmb.astype(float), z_cmb)
        npt.assert_allclose(dataset_two_sne.z_cmb_err.astype(float), z_cmb_err)
        npt.assert_allclose(dataset_two_sne.z_hubble.astype(float), z_hubble)
        npt.assert_allclose(dataset_two_sne.z_hubble_err.astype(float), z_hubble_err)

    def test_set_all_rest_phases(self, dataset_two_sne):
        test_ds = copy.deepcopy(dataset_two_sne)
        assert "phase" not in test_ds.photometry
        obs_peaks = np.concatenate([np.full(test_ds.N_obs[i], test_ds.peak_mjd[i]) for i in range(test_ds.N_sn)])
        obs_z_hels = np.concatenate([np.full(test_ds.N_obs[i], test_ds.z_helio[i]) for i in range(test_ds.N_sn)])
        test_ds.set_all_rest_phases()
        pdt.assert_series_equal(test_ds.photometry["mjd"], test_ds.photometry["phase"] * (1+obs_z_hels) + obs_peaks, check_names=False)

    def test_recalibrate_fluxcal_zpt(self, dataset_two_sne):
        test_ds = copy.deepcopy(dataset_two_sne)
        test_phot, ref_phot = test_ds.photometry, dataset_two_sne.photometry
        fluxes, mags = ["flux", "flux_err"], ["mag", "mag_err"]
        assert test_ds.fluxcal_zpt == 27.5
        # testing different data zero-points for test0 and test1, 28.5 and 27.5
        test_phot.loc[test_phot["snid"] == "test0", fluxes] *= 10**0.4
        test_ds.fluxcal_zpt = 32.5
        scaling = 100  # 5 mags
        test_ds.recalibrate_fluxcal_zpt()
        flux_ratios = test_phot[fluxes] / ref_phot[fluxes]
        # test0 and test1 both brought to 32.5, ratio with ref_phot (27.5) is 100
        npt.assert_allclose(flux_ratios.values.flatten(), 100)
        pdt.assert_frame_equal(test_phot[mags], ref_phot[mags])

    def test_recalibrate_fluxcal_zpt_all_negative(self, dataset_two_sne):
        test_ds = copy.deepcopy(dataset_two_sne)
        test_phot = test_ds.photometry
        fluxes, mags = ["flux", "flux_err"], ["mag", "mag_err"]
        test_phot["flux"] *= -1
        test_phot[["mag", "mag_err"]] = -99
        test_ds.fluxcal_zpt = 32.5
        ref_phot = copy.deepcopy(test_phot)
        test_ds.recalibrate_fluxcal_zpt()  # no-op when no positive fluxes
        pdt.assert_frame_equal(test_phot, ref_phot)

    def test_recalibrate_fluxcal_zpt_negatives_single_zpt(self, dataset_two_sne):
        test_ds = copy.deepcopy(dataset_two_sne)
        test_phot, ref_phot = test_ds.photometry, dataset_two_sne.photometry
        fluxes, mags = ["flux", "flux_err"], ["mag", "mag_err"]
        assert test_ds.fluxcal_zpt == 27.5
        # same setup as test_recalibrate_fluxcal_zpt, but with negatives for first flt
        # and test0 and test1 have the same zeropoint.
        flt_mask = test_phot["flt"] == test_ds.unique_bands[0]
        test_phot.loc[flt_mask, "flux"] *= -1
        test_phot.loc[flt_mask, mags] = -99
        test_ds.fluxcal_zpt = 32.5
        scaling = 100  # 5 mags
        test_ds.recalibrate_fluxcal_zpt()
        flux_ratios = test_phot[fluxes] / ref_phot[fluxes]
        npt.assert_allclose(flux_ratios[~flt_mask].values.flatten(), 100)
        # single zpt inferred for negative fluxes, ratios match with negative for flux
        npt.assert_allclose(flux_ratios.loc[flt_mask, "flux"].values.flatten(), -100)
        npt.assert_allclose(flux_ratios.loc[flt_mask, "flux_err"].values.flatten(), 100)
        npt.assert_allclose(test_phot.loc[flt_mask, mags].values.flatten(), -99)

    def test_recalibrate_fluxcal_zpt_negatives_multiple_zpts(self, dataset_two_sne):
        test_ds = copy.deepcopy(dataset_two_sne)
        test_phot, ref_phot = test_ds.photometry, dataset_two_sne.photometry
        fluxes, mags = ["flux", "flux_err"], ["mag", "mag_err"]
        assert test_ds.fluxcal_zpt == 27.5
        # same setup as test_recalibrate_fluxcal_zpt, but with negatives for first flt
        snid_mask = test_phot["snid"] == "test0"
        test_phot.loc[snid_mask, fluxes] *= 10**0.4
        flt_mask = test_phot["flt"] == test_ds.unique_bands[0]
        test_phot.loc[flt_mask, "flux"] *= -1
        test_phot.loc[flt_mask, mags] = -99
        test_ds.fluxcal_zpt = 32.5
        scaling = 100  # 5 mags
        test_ds.recalibrate_fluxcal_zpt()
        flux_ratios = test_phot[fluxes] / ref_phot[fluxes]
        npt.assert_allclose(flux_ratios[~flt_mask].values.flatten(), 100)
        npt.assert_allclose(test_phot.loc[flt_mask, mags].values.flatten(), -99)
        pdt.assert_frame_equal(test_phot.loc[~flt_mask, mags], ref_phot.loc[~flt_mask, mags])
        # Different zpts means zp for negative fluxes cannot be inferred
        # negative test0 photometry only changed by 1 mag
        npt.assert_allclose(flux_ratios.loc[flt_mask*snid_mask, "flux"].values.flatten(), -10**0.4)
        npt.assert_allclose(flux_ratios.loc[flt_mask*snid_mask, "flux_err"].values.flatten(), 10**0.4)
        # negative test1 photometry not changed at all
        npt.assert_allclose(flux_ratios.loc[flt_mask*~snid_mask, "flux"].values.flatten(), -1)
        npt.assert_allclose(flux_ratios.loc[flt_mask*~snid_mask, "flux_err"].values.flatten(), 1)

    def test_apply_filter_map(self, dataset_two_sne):
        test_ds = copy.deepcopy(dataset_two_sne)
        bands = test_ds.unique_bands
        test_ds.apply_filter_map(map_dict={bands[0]: "alpha", bands[1]: "beta"})
        mask = [col for col in test_ds.photometry.columns if col != "flt"]
        for i, band in enumerate(("alpha", "beta")):
            ref_df = dataset_two_sne.photometry.loc[dataset_two_sne.photometry["flt"] == bands[i]]
            test_df = test_ds.photometry.loc[test_ds.photometry["flt"] == band]
            pdt.assert_frame_equal(test_df[mask], ref_df[mask])

    def test_apply_error_floor(self, dataset_two_sne):
        test_ds = copy.deepcopy(dataset_two_sne)
        # no op for non-positive error floors.
        test_ds.apply_error_floor(0)
        pdt.assert_frame_equal(test_ds.photometry, dataset_two_sne.photometry)
        sorted_mag_errs = test_ds.photometry.mag_err.sort_values()
        floor = sorted_mag_errs.values[3]
        test_ds.apply_error_floor(floor)
        for i in range(3):
            idx = sorted_mag_errs.index[i]
            assert test_ds.photometry.mag_err[idx] == floor
            assert test_ds.photometry.flux_err[idx] == floor*np.log(10)/2.5*test_ds.photometry.flux[idx]

class TestFactoryMethods:
    @pytest.mark.parametrize("fname,file_format,read_fn", (("Foundation_DR1_2016W.txt", "SNANA", io.read_snana_ascii), ("CSP_SN2004dt.snpy", "snpy", io.read_snpy)))
    def test_from_one_ascii(self, fname: str, file_format: str, read_fn: Callable):
        path = TEST_DIR / f"training_data/{fname}"
        test_ds = SNDataset.from_ascii_files(path, file_format=file_format)
        sn_dict, obs_df = read_fn(path)
        sn_dict = clean_sn_dict(sn_dict)
        obs_df["snid"] = sn_dict["snid"][0]
        ref_ds = make_dataset(sn_dict, obs_df)
        ref_ds.fill_out_redshifts()
        ref_ds.set_all_rest_phases()
        assert test_ds == ref_ds

    def test_from_ascii_bad_format(self):
        path = TEST_DIR / "training_data/Foundation_DR1_2016W.txt"
        with pytest.raises(ValueError, match="file_format"):
            test_ds = SNDataset.from_ascii_files(path, file_format="unsupported_format")

    def test_from_ascii_wrong_peakmjd_key(self):
        path = TEST_DIR / "training_data/Foundation_DR1_2016W.txt"
        test_ds = SNDataset.from_ascii_files(path, file_format="SNANA", peakmjd_key=["missing_key",])

    def test_from_multi_ascii(self):
        paths = [TEST_DIR / f"training_data/{fname}" for fname in ("Foundation_DR1_2016W.txt", "CSP_SN2004dt.snpy")]
        test_ds = SNDataset.from_ascii_files(paths, file_format=["SNANA", "snpy"])
        sn_dict_0, obs_df_0 = io.read_snana_ascii(paths[0])
        sn_dict_1, obs_df_1 = io.read_snpy(paths[1])
        sn_dict_0 = clean_sn_dict(sn_dict_0)  # just need to clean for make_dataset
        ref_ds = make_dataset(sn_dict_0, obs_df_0)
        ref_ds.append(sn_dict=sn_dict_1, obs_df=obs_df_1)
        ref_ds.fill_out_redshifts()
        ref_ds.set_all_rest_phases()
        assert test_ds == ref_ds

    def test_from_ascii_overrides(self):
        paths = [TEST_DIR / f"training_data/Foundation_DR1_2016{name}.txt" for name in ("W", "afk")]
        ors = {"z_helio_err": 2e-5, "mwebv_err": [0.1, 0.2]}
        test_ds = SNDataset.from_ascii_files(paths, file_format="SNANA", overrides=ors)
        npt.assert_allclose(test_ds.z_helio_err, 2e-5)
        npt.assert_allclose(test_ds.mwebv_err, [0.1, 0.2])

    def test_from_table_file(self):
        table_path = TEST_DIR / "T21_mini_set.txt"
        test_ds = SNDataset.from_table_file(table_path, data_root=TEST_DIR)
        df = pd.read_csv(table_path, sep=r"\s+")
        N_sn = len(df["SNID"])
        overrides = {
            "peak_mjd": df["SEARCH_PEAKMJD"].values,
            "z_cmb": df["REDSHIFT_CMB"].values,
            "z_cmb_err": df["REDSHIFT_CMB_ERR"].values,
            # Propagating overriden redshifts
            "z_helio": np.full(N_sn, None),
            "z_helio_err": np.full(N_sn, None),
            "z_hubble": np.full(N_sn, None),
            "z_hubble_err": np.full(N_sn, None),
        }
        ref_ds = SNDataset.from_ascii_files(
            [TEST_DIR / path for path in df["files"]],
            file_format="SNANA",
            overrides=overrides
        )
        assert test_ds == ref_ds

    def test_from_table_file_bad_sn_list(self):
        table_path = TEST_DIR / "training_data/CSP_SN2004dt.snpy"
        with pytest.raises(ValueError, match="does not have a header row"):
            test_ds = SNDataset.from_table_file(table_path, data_root=TEST_DIR)

    def test_from_table_file_bad_formats(self):
        table_path = TEST_DIR / "T21_mini_set.txt"
        with pytest.raises(ValueError, match="file_format was provided"):
            test_ds = SNDataset.from_table_file(
                table_path, data_root=TEST_DIR, file_format=["snana", "snana"]
            )

    def test_from_snana_fits(self):
        path = TEST_DIR / "training_data/BAYESN_test_fits/BAYESN_test_fits_HEAD.FITS"
        test_ds = SNDataset.from_snana_fits(path, peakmjd_key="PEAKMJD")
        sn_dict, obs_df = io.read_snana_fits(path)
        sn_dict = clean_sn_dict(sn_dict)
        ref_ds = make_dataset(sn_dict, obs_df, sim=True)
        ref_ds.fill_out_redshifts()
        ref_ds.set_all_rest_phases()
        assert test_ds == ref_ds

    def test_from_snana_list_fits(self):
        fits_dir = TEST_DIR / f"training_data/BAYESN_test_fits/"
        test_ds = SNDataset.from_snana_list(fits_dir / "BAYESN_test_fits.LIST", data_root=fits_dir, peakmjd_key="PEAKMJD")
        ref_ds = SNDataset.from_snana_fits(fits_dir / "BAYESN_test_fits_HEAD.FITS", peakmjd_key="PEAKMJD")
        assert test_ds == ref_ds

    def test_from_snana_list_mixed(self):
        fits_dir = TEST_DIR / f"training_data/BAYESN_test_fits/"
        test_ds = SNDataset.from_snana_list(fits_dir / "BAYESN_test_mixed.LIST", data_root=fits_dir, peakmjd_key=["PEAKMJD", "SEARCH_PEAKMJD"])
        ref_ds = SNDataset.from_snana_fits(fits_dir / "BAYESN_test_fits_HEAD.FITS", peakmjd_key="PEAKMJD")
        ref_ds.append(SNDataset.from_ascii_files(fits_dir / "../Foundation_DR1_2016W.txt", file_format="SNANA", peakmjd_key="SEARCH_PEAKMJD"))
        assert test_ds == ref_ds

class TestDataProducts:
    def test_make_fitres_table(self, dataset_two_sne):
        pass
    def test_make_fitres_table_sim(self, dataset_sim):
        pass
    def test_cut_fitres_table(self):
        pass
    def test_make_lcplot_data(self):
        pass
    def test_make_bayesn_data(self):
        pass
