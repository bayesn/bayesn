from collections import OrderedDict as odict
import copy
from io import StringIO
import gzip
from numbers import Number
import os
from pathlib import Path
import pickle
import time

import pandas as pd
import pytest
import numpy as np
import sncosmo

from bayesn import io
from bayesn.utils import assert_dicts_match, mag_to_flux, flux_to_mag, get_MWEBV
from bayesn.datasets import SNDataset, meta_names, all_meta_names, clean_sn_dict, clean_obs_df

BASE_DIR = Path(__file__).parent.absolute()
TEST_DIR = Path(BASE_DIR, "test_files")
PICKLE_DIR = Path(TEST_DIR, "pickles")
READ_DTYPE = tuple[odict[str, str | Number], pd.DataFrame]

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
    return obs_df

def format_df(df: pd.DataFrame):
    df = df.sort_values(["snid", "flt", "mjd"]).reset_index(drop=True)
    return df[["snid", "flt", "mjd", "flux", "flux_err", "mag", "mag_err"]]

@pytest.fixture
def sample_data_single_sn() -> tuple[dict[str, np.ndarray], pd.DataFrame, np.ndarray]:
    N_sn = 1
    sn_dict = random_sn_dict(RNG_seed=0, N=N_sn)
    sn_dict["test_key"] = np.arange(N_sn)
    obs_df = random_obs_df(RNG_seed=0)
    phot_idx = np.array([0, len(obs_df)])
    return sn_dict, obs_df

@pytest.fixture
def sample_data_two_sne() -> tuple[dict[str, np.ndarray], pd.DataFrame, np.ndarray]:
    N_sn = 2
    sn_dict = random_sn_dict(RNG_seed=0, N=N_sn)
    sn_dict["test_key"] = np.arange(N_sn)
    obs_dfs = [random_obs_df(RNG_seed=i) for i in range(N_sn)]
    obs_df = pd.concat(obs_dfs, ignore_index=True)
    N_obs = np.array([len(df) for df in obs_dfs])
    phot_idx = np.append(0, np.cumsum(N_obs))
    return sn_dict, obs_df

@pytest.fixture
def sample_data_sim() -> tuple[dict[str, np.ndarray], pd.DataFrame, np.ndarray]:
    N_sn = 5
    sn_dict = random_sn_dict(RNG_seed=0, N=N_sn, sim=True)
    obs_dfs = [random_obs_df(RNG_seed=i) for i in range(N_sn)]
    obs_df = pd.concat(obs_dfs, ignore_index=True)
    N_obs = np.array([len(df) for df in obs_dfs])
    phot_idx = np.append(0, np.cumsum(N_obs))
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

class TestInit:
    def test_init_empty(self):
        ds = SNDataset()
        assert ds.N_sn == 0
        assert ds.sim is False
        np.testing.assert_equal(ds.phot_idx, np.array([0]))
        for attr in meta_names["str"] + meta_names["num"]:
            np.testing.assert_equal(getattr(ds, attr), np.array([]))
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
            np.testing.assert_equal(getattr(dataset_two_sne, attr), sn_dict.get(attr))
        pd.testing.assert_frame_equal(dataset_two_sne.photometry, obs_df)

    def test_init_sim(self, sample_data_sim, dataset_sim):
        sn_dict = sample_data_sim[0]
        for attr in meta_names["sim"]:
            np.testing.assert_equal(getattr(dataset_sim, attr), sn_dict[attr])

    def test_init_0d_arrs(self, sample_data_single_sn, dataset_single_sn):
        sn_dict, obs_df = sample_data_single_sn
        ds_test = SNDataset(
            N_sn=1,
            photometry=obs_df,
            # init meta keys w/ scalars/strings instead of ArrayLikes
            **{key: val[0] for key, val in sn_dict.items() if key in all_meta_names}
        )
        assert ds_test == dataset_single_sn

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
        np.testing.assert_equal(ref_bands, test_bands)

    def test_metadata(self, sample_data_sim, dataset_sim):
        sn_dict = sample_data_sim[0]
        ref_meta = dict(zip(
            all_meta_names, [sn_dict.get(attr) for attr in all_meta_names]
        ))
        assert_dicts_match(ref_meta, dataset_sim.metadata)

class TestDataAddition:
    def test_append_new_single(self, sample_data_single_sn, dataset_single_sn):
        sn_dict, obs_df = sample_data_single_sn
        ds_test = SNDataset()
        ds_test._append_new(sn_dict, obs_df)
        ds_test._clean_photometry()
        assert ds_test == dataset_single_sn

    def test_append_new_single_missing_data(self, sample_data_single_sn, dataset_single_sn):
        ds_test = SNDataset()
        sn_dict, obs_df = sample_data_single_sn
        drop_keys = (
            "z_hubble", "z_hubble_err", "vpec", "vpec_err", "host_logmass",
            "host_logmass_err"
        )
        for key in drop_keys:
            sn_dict.pop(key)
        ds_test._append_new(sn_dict, obs_df)
        ds_test._clean_photometry()
        for attr in all_meta_names:
            if attr in drop_keys:
                assert getattr(ds_test, attr) == np.array([None])
            else:
                assert getattr(ds_test, attr) == sn_dict.get(attr)
        pd.testing.assert_frame_equal(ds_test.photometry, dataset_single_sn.photometry)

    def test_append_new_multi(self, sample_data_two_sne, dataset_two_sne):
        sn_dict, obs_df = sample_data_two_sne
        ds_test = SNDataset()
        ds_test._append_new(sn_dict, obs_df)
        ds_test._clean_photometry()
        assert ds_test == dataset_two_sne

    def test_append_duplicate_single(self, sample_data_single_sn, dataset_single_sn):
        sn_dict, obs_df = sample_data_single_sn
        new_df = copy.deepcopy(obs_df)
        # Buffering mjd to avoid potential data discrepancies.
        new_df["mjd"] += obs_df["mjd"].max() - new_df["mjd"].min() + 1
        expected_phot = format_df(pd.concat([obs_df, new_df]))
        dataset_single_sn._append_duplicate(sn_dict, new_df)
        assert_dicts_match(dataset_single_sn.metadata, sn_dict)
        pd.testing.assert_frame_equal(dataset_single_sn.photometry, expected_phot)

    def test_append_duplicate_phot_overlap(self, sample_data_single_sn, dataset_single_sn):
        sn_dict, obs_df = sample_data_single_sn
        ds_test = copy.deepcopy(dataset_single_sn)
        # 100% overlap changes nothing
        ds_test._append_duplicate(sn_dict, obs_df)
        ds_test._clean_photometry()
        pd.testing.assert_frame_equal(ds_test.photometry, ds_test.photometry)

        # mismatch between new and old data
        new_df = copy.deepcopy(obs_df)
        with pytest.raises(ValueError, match="There are discrepancies"):
            new_df["flux"] += 1  # mjd/filt match + discrepancies elsewhere
            dataset_single_sn._append_duplicate(sn_dict, new_df)

        # some new photometry
        new_df = copy.deepcopy(obs_df)
        # 0:2 grabs first three rows here
        new_df.loc[0:2, "mjd"] += new_df["mjd"].max() - new_df["mjd"].min()
        ds_test._append_duplicate(sn_dict, new_df)
        ds_test._clean_photometry()
        # 0:3 grabs first three rows here
        expected_phot = format_df(pd.concat([dataset_single_sn.photometry, new_df[0:3]]))
        assert_dicts_match(ds_test.metadata, dataset_single_sn.metadata)
        pd.testing.assert_frame_equal(ds_test.photometry, expected_phot)

    def test_append_duplicate_multi(self, sample_data_two_sne, dataset_two_sne):
        sn_dict, obs_df = sample_data_two_sne
        new_df = copy.deepcopy(obs_df)
        # Buffering mjd to avoid potential data discrepancies.
        new_df["mjd"] += obs_df["mjd"].max() - new_df["mjd"].min() + 1
        expected_phot = format_df(pd.concat([obs_df, new_df]))
        dataset_two_sne._append_duplicate(sn_dict, new_df)
        assert_dicts_match(dataset_two_sne.metadata, sn_dict)
        pd.testing.assert_frame_equal(dataset_two_sne.photometry, expected_phot)

    def test_append_empty(self):
        ds_test, ds_ref = [SNDataset() for _ in range(2)]
        ds_test.append(ds=None, sn_dict=None, obs_df=None)
        assert ds_test == ds_ref

    def test_append_dataset(self, sample_data_two_sne, dataset_two_sne):
        sn_dict, obs_df = sample_data_two_sne
        new_dict = copy.deepcopy(sn_dict)
        new_dict["snid"][0] = "test2"
        new_df = copy.deepcopy(obs_df)
        new_df.loc[new_df["snid"] == "test0", "snid"] = "test2"
        new_df["mjd"] += obs_df["mjd"].max() - new_df["mjd"].min() + 1
        ds_to_be_added = make_dataset(new_dict, new_df)
        ds_test = copy.deepcopy(dataset_two_sne)
        dataset_two_sne.append(sn_dict=new_dict, obs_df=new_df)
        ds_test.append(ds=ds_to_be_added)
        assert dataset_two_sne == ds_test

    def test_append_sn_dict_obs_df(self, sample_data_two_sne, dataset_two_sne):
        sn_dict, obs_df = sample_data_two_sne
        new_dict = copy.deepcopy(sn_dict)
        new_dict["snid"][0] = "test2"
        new_df = copy.deepcopy(obs_df)
        new_df.loc[new_df["snid"] == "test0", "snid"] = "test2"
        new_df["mjd"] += obs_df["mjd"].max() - new_df["mjd"].min() + 1
        ds_test = copy.deepcopy(dataset_two_sne)
        ds_test.append(sn_dict=new_dict, obs_df=new_df)
        dataset_two_sne.append(sn_dict=new_dict, obs_df=new_df)
        assert dataset_two_sne == ds_test

    def test_append_mismatch(self, sample_data_two_sne, dataset_two_sne):
        sn_dict, obs_df = sample_data_two_sne
        new_dict = copy.deepcopy(sn_dict)
        new_dict["snid"][0] = "test2"
        new_df = copy.deepcopy(obs_df)
        new_df.loc[new_df["snid"] == "test0", "snid"] = "test2"
        new_df["mjd"] += obs_df["mjd"].max() - new_df["mjd"].min() + 1
        ds_to_be_added = make_dataset(new_dict, copy.deepcopy(new_df))
        new_df["flux"] += 10
        with pytest.raises(ValueError, match="The provided arguments are not equiv"):
            dataset_two_sne.append(ds=ds_to_be_added, sn_dict=new_dict, obs_df=new_df)


    def test_append_infer_phot_idx(self, sample_data_single_sn, dataset_single_sn):
        sn_dict, obs_df = sample_data_single_sn
        obs_df.pop("snid")
        ds_test = SNDataset()
        ds_test.append(sn_dict=sn_dict, obs_df=obs_df, phot_idx=None)
        dataset_single_sn.photometry = clean_obs_df(dataset_single_sn.photometry, sn_dict["snid"])
        assert ds_test == dataset_single_sn

class TestGetterMethods:
    def test_get_idx(self, dataset_two_sne):
        assert dataset_two_sne.get_idx("test0") == 0
        np.testing.assert_equal(dataset_two_sne.get_idx(["test0"]), np.array([0]))
        np.testing.assert_equal(dataset_two_sne.get_idx(["test1", "test0"]), np.array([1, 0]))

    def test_get_idx_not_found(self, dataset_two_sne):
        with pytest.raises(ValueError, match="snid missing not found."):
            dataset_two_sne.get_idx(snid="missing")

    def test_get_idx_dtype(self, dataset_two_sne):
        with pytest.raises(TypeError, match="snid of type <class 'NoneType'>"):
            dataset_two_sne.get_idx(snid=None)

    def test_parse_snid_idx(self, dataset_two_sne):
        assert dataset_two_sne._parse_snid_idx_args(idx=1) == 1
        assert dataset_two_sne._parse_snid_idx_args(snid="test1") == 1
        np.testing.assert_equal(dataset_two_sne._parse_snid_idx_args(snid=["test0", "test1"]), np.array([0, 1]))

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
        pd.testing.assert_frame_equal(dataset_two_sne.get_phot_subset(snid="test0"), df0)
        pd.testing.assert_frame_equal(dataset_two_sne.get_phot_subset(snid=["test1", "test0"]), switched_df)
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
        pd.testing.assert_frame_equal(ref_phot, dataset_sim.photometry)
        # np.testing.assert_equal(ref_phot_idx, dataset_sim.phot_idx)

    def test_keep_according_to_list(self, dataset_sim):
        ref_meta = dataset_sim.get_metadata_subset(idx=[2, 3])
        ref_phot = dataset_sim.get_phot_subset(idx=[2, 3])
        dataset_sim.keep_according_to_list(["test2", "test3", "test10"])
        assert dataset_sim.N_sn == 2
        assert_dicts_match(ref_meta, dataset_sim.metadata)
        pd.testing.assert_frame_equal(ref_phot, dataset_sim.photometry)

    def test_remove_phot_idx(self, sample_data_sim, dataset_sim):
        # First object has more than 2 observations, so metadata shouldn't change.
        sn_dict, ref_phot = sample_data_sim
        ref_phot_idx = copy.deepcopy(dataset_sim.phot_idx)  # tied to dataset_sim so will change.
        ref_phot_idx[1:] -= 2
        ref_phot = format_df(ref_phot).drop(index=[0, 1]).reset_index(drop=True)
        dataset_sim.remove_phot_by_idx([0, 1])
        assert_dicts_match(sn_dict, dataset_sim.metadata)
        pd.testing.assert_frame_equal(ref_phot, dataset_sim.photometry)
        np.testing.assert_equal(ref_phot_idx, dataset_sim.phot_idx)

    def test_remove_phot_idx_drop_sn(self, sample_data_sim, dataset_sim):
        # Removing all photometry from first object should cause it to be dropped.
        sn_dict, ref_phot = sample_data_sim
        phot_idx = dataset_sim.phot_idx
        ref_meta = dataset_sim.get_metadata_subset(idx=np.arange(1, dataset_sim.N_sn))
        ref_phot_idx = phot_idx[1:] - phot_idx[1]
        ref_phot = format_df(ref_phot).drop(index=np.arange(phot_idx[1])).reset_index(drop=True)
        dataset_sim.remove_phot_by_idx(np.arange(phot_idx[1]))
        assert_dicts_match(ref_meta, dataset_sim.metadata)
        pd.testing.assert_frame_equal(ref_phot, dataset_sim.photometry)
        np.testing.assert_equal(ref_phot_idx, dataset_sim.phot_idx)

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
        dataset_sim.drop_by_band_lims(
            band_lim_dict=band_lim_dict,
            wave_min=wave_min,
            wave_max=wave_max
        )
        assert zmin_band not in dataset_sim.get_phot_subset(idx=zmin_idx)["flt"]
        assert zmax_band not in dataset_sim.get_phot_subset(idx=zmax_idx)["flt"]
        assert len(dataset_sim.photometry) == original_length - zmin_counts - zmax_counts

    def test_cut_by_meta_numeric(self, dataset_sim):
        ds_ref = copy.deepcopy(dataset_sim)
        z = dataset_sim.z_cmb
        high_idx = np.where(z >= np.median(z))[0]
        ref_meta = dataset_sim.get_metadata_subset(idx=high_idx)
        ref_phot = dataset_sim.get_phot_subset(idx=high_idx)
        test_meta, test_phot, phot_idx = dataset_sim.cut_by_meta_numeric("z_cmb", "<", np.median(z), inplace=False)
        assert ds_ref == dataset_sim
        assert_dicts_match(ref_meta, test_meta)
        pd.testing.assert_frame_equal(ref_phot, test_phot)
        dataset_sim.cut_by_meta_numeric("z_cmb", "<", np.median(z), inplace=True)
        assert ds_ref != dataset_sim
        assert_dicts_match(ref_meta, dataset_sim.metadata)
        pd.testing.assert_frame_equal(ref_phot, dataset_sim.photometry)

    def test_cut_by_phot_numeric(self, dataset_sim):
        flux = dataset_sim.photometry["flux"]
        low, high = np.quantile(flux, [0.1, 0.9])
        ref_phot = dataset_sim.photometry[(flux > low) & (flux < high)].reset_index(drop=True)
        dataset_sim.cut_by_phot_numeric("flux", ">=", high)
        dataset_sim.cut_by_phot_numeric("flux", "<=", low)
        pd.testing.assert_frame_equal(ref_phot, dataset_sim.photometry)

    def test_cut_by_meta_numeric_bad_col(self, dataset_sim):
        with pytest.raises(ValueError, match="foo not recognised."):
            dataset_sim.cut_by_meta_numeric("foo", "!=", 0)

    def test_cut_by_phot_numeric_bad_col(self, dataset_sim):
        with pytest.raises(ValueError, match="foo not recognised."):
            dataset_sim.cut_by_phot_numeric("foo", "<", 1)

class TestAstroGetter:
    def test_calculate_snrmaxes(self):
        pass
    def test_calculate_rest_phases(self):
        pass
    def test_estimate_tmax(self):
        pass
    def test_get_band_indices(self):
        pass

class TestAstroSetter:
    def test_fill_out_redshifts(self):
        pass
    def test_set_all_rest_phases(self):
        pass
    def test_recalibrate_fluxcal_zpt(self):
        pass
    def test_apply_filter_map(self):
        pass
    def test_apply_error_floor(self):
        pass

class TestFactoryMethods:
    def test_from_ascii_files(self):
        pass
    def test_from_table_file(self):
        pass
    def test_from_snana_fits(self):
        pass
    def test_from_snana_list(self):
        pass

class TestDataProducts:
    def test_make_fitres_table(self):
        pass
    def test_cut_fitres_table(self):
        pass
    def test_make_lcplot_data(self):
        pass
    def test_make_bayesn_data(self):
        pass
