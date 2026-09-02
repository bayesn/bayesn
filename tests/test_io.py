from collections import OrderedDict as odict
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
from pandas.testing import assert_frame_equal

from bayesn import io
from bayesn.utils import assert_dicts_match, mag_to_flux, flux_to_mag, get_MWEBV

BASE_DIR = Path(__file__).parent.absolute()
TEST_DIR = BASE_DIR / "test_files"
PICKLE_DIR = TEST_DIR / "pickles"
READ_DTYPE = tuple[odict[str, str | Number], pd.DataFrame]

class TestRead:
    @pytest.mark.parametrize("file_path", ["CSPDR3_2004ef.dat", "CSPDR3_2004ef.dat.gz"])
    def test_read_spectrum(self, file_path: str):
        file_path = TEST_DIR / file_path
        res = io.read_snana_spectra(file_path)
        if str(file_path).endswith(".gz"):
            f = gzip.open(file_path, "rt")
        else:
            f = open(file_path, "r")
        lines = f.readlines()
        f.close()
        spec_ids, start_idx, end_idx = [[] for _ in range(3)]
        for line_num, line in enumerate(lines):
            if line.startswith("SPECTRUM_ID"):
                spec_id = int(line.split()[1])
                assert spec_id in res
                spec_ids.append(spec_id)
                start_idx.append(line_num+3)  # SPECTRUM_ID, SPECTRUM_MJD, SPECTRUM_NLAM
            elif line.endswith("SPECTRUM_END"):
                end_idx.append(line_num)
            elif line.startswith("VARNAMES_SPEC"):
                var_names = [key.strip("\n") for key in line.split()][1:]  # Skip "VARNAMES_SPEC:"

        # NLAM is not always accurate, e.g. for Dovekie's CSPDR3_2004ef.dat,
        # SPECTRUM_ID 2 has NLAM = 1873 but only 1869 rows
        for spec_idx, (start, end) in enumerate(zip(start_idx, end_idx)):
            assert len(res[spec_ids[spec_idx]]["FLAM"]) == end - start
            for split_idx, var_name in enumerate(var_names):
                split_idx += 1  # line includes "SPEC:" first
                res_vals = res[spec_ids[spec_idx]][var_name]
                read_vals = np.array([float(line.split()[split_idx].strip("\n")) for line in lines[start:end]])
                assert all(res_vals == read_vals)

    class TestReadSNANAAsciiMeta:
        def test_non_numeric_fields(self):
            raw = StringIO("SNID: 1\nIAUC: one2\nSURVEY: 002")
            meta = io.read_snana_ascii_meta(raw)
            assert isinstance(meta["SNID"], str)
            assert isinstance(meta["IAUC"], str)
            assert isinstance(meta["SURVEY"], str)
            assert meta["SNID"] == "1"
            assert meta["IAUC"] == "one2"
            assert meta["SURVEY"] == "002"

        def test_ignored_lines(self):
            raw = StringIO(
                "# Header comment\n"  # ignored by comment="#"
                "RA: 123.45 # in J2000\n"  # "in J2000" ignored because following "#"
                "IMPORTANT_KEY 25\n"  # ignored because no ":"
                "OBS: data data data\n"  # ignored by tablename="OBS"
                "END:\n"  # hard-coded (for now) to be ignored
            )
            meta = io.read_snana_ascii_meta(raw)
            assert meta == odict({"RA": 123.45})

        def test_float_regex(self):
            raw = StringIO(
                "RA: +45 and trailing text\n"        # +int
                "DECL: leading text and -12\n"       # -int
                "REDSHIFT_HELIO: 0.031 +/- 2e-3\n"   # unsigned float, eng not w/ -power
                "REDSHIFT_CMB: +0.03037 +- .002\n"   # +float and float starting with .
                "VPEC: 2.5e+2 plus or minus 1e2\n"   # eng not w/ + or unsigned power
                "REDSHIFT_HUBBLE: ~3 over 100\n"     # should be parsed as two floats
            )
            meta = io.read_snana_ascii_meta(raw)
            assert meta["RA"] == 45
            assert meta["DECL"] == -12
            assert meta["REDSHIFT_HELIO"] == 0.031
            assert meta["REDSHIFT_HELIO_ERR"] == 0.002
            assert meta["REDSHIFT_CMB"] == 0.03037
            assert meta["REDSHIFT_CMB_ERR"] == 0.002
            assert meta["VPEC"] == 250.
            assert meta["VPEC_ERR"] == 100.
            assert meta["REDSHIFT_HUBBLE"] == 3
            assert meta["REDSHIFT_HUBBLE_ERR"] == 100

        def test_stat_and_sys(self):
            raw = StringIO("REDSHIFT_CMB: 0.05 +/- 0.003 (stat) +/- 0.004 (sys)")
            meta = io.read_snana_ascii_meta(raw, stat_and_sys=True)
            assert meta["REDSHIFT_CMB"] == 0.05
            assert np.isclose(meta["REDSHIFT_CMB_ERR"], np.hypot(0.003, 0.004))

        def test_too_many_numbers_error(self):
            raw = StringIO("RA: 12:25:32.25")
            with pytest.raises(ValueError, match="Found 3 numeric strings"):
                io.read_snana_ascii_meta(raw, stat_and_sys=False)

    class TestReadSNANAAscii:
        @pytest.fixture(scope="class")
        def sample(self) -> READ_DTYPE:
            example_lc = Path(BASE_DIR.parent, "bayesn/data/example_lcs/Foundation_DR1_2016W.txt")
            return io.read_snana_ascii(example_lc)

        def test_schema(self, sample: READ_DTYPE):
            sn_dict, obs_df = sample
            assert set(io.obs_df_columns).issubset(obs_df.columns)
            assert "BAND" not in obs_df
            for key in io.sn_dict_keys:
                assert key in sn_dict

        def test_pogson_relation(self, sample: READ_DTYPE):
            obs_df = sample[1]
            expected_mag = 27.5 - 2.5 * np.log10(obs_df["flux"])
            np.testing.assert_allclose(obs_df["mag"], expected_mag)

        def test_mags_of_negative_fluxes_and_errs(self):
            raw_content = """NVAR: 4
                VARLIST: MJD FLT FLUXCAL FLUXCALERR
                OBS: 50000 g 100 10
                OBS: 50000 r 0 10
                OBS: 50000 i -10 10"""
            _, obs_df = io.read_snana_ascii(StringIO(raw_content))
            expected_mag = np.array([22.5, -99, -99])
            expected_mag_err = np.array([2.5*np.log10(1+10/100), -99, -99])
            np.testing.assert_allclose(obs_df["mag"].values.astype(float), expected_mag)
            np.testing.assert_allclose(obs_df["mag_err"].values.astype(float), expected_mag_err)
            with pytest.raises(ValueError, match="Negative flux errors"):
                io.read_snana_ascii(StringIO(raw_content + "\nOBS: 50001 g 10 -1"))

        def test_fluxcal_zpt_arg(self):
            raw_content = """NVAR: 4
                VARLIST: MJD FLT FLUXCAL FLUXCALERR
                OBS: 50000 g 100 10"""
            fluxcal_zpt = 31.4
            expected_mag, expected_mag_err = flux_to_mag(100, 10, zp=fluxcal_zpt)
            sn_dict, obs_df = io.read_snana_ascii(StringIO(raw_content), fluxcal_zpt=fluxcal_zpt)
            assert sn_dict["ZP_FLUXCAL"] == 31.4
            np.testing.assert_allclose(obs_df["mag"], expected_mag)
            np.testing.assert_allclose(obs_df["mag_err"], expected_mag_err)

        def test_BAND_to_FLT_rename(self):
            raw_content = """NVAR: 4
                VARLIST: MJD BAND FLUXCAL FLUXCALERR
                OBS: 50000 g 100 10"""
            _, obs_df = io.read_snana_ascii(StringIO(raw_content))
            assert "BAND" not in obs_df
            np.testing.assert_equal(obs_df["FLT"].values.astype(str), np.array(["g"]))

    def test_read_snpy(self, RNG_seed=0, N=5):
        rng = np.random.default_rng(RNG_seed)
        N_filt1 = rng.choice(N-1)+1
        N_filt2 = N - N_filt1
        snid = "sample_SN"
        z_helio = rng.lognormal(-4)
        ra = rng.uniform()*360
        dec = rng.uniform()*180 - 90
        mjd = np.arange(N) + 5e4
        mag = rng.normal(16, 1, N)
        mag_err = rng.lognormal(-2.5, 0.2, N)
        expected_flux, expected_flux_err = mag_to_flux(mag, mag_err, 27.5)
        filts = np.append(np.full(N_filt1, "filt1"), np.full(N_filt2, "filt2"))

        raw_str_list = [f"{snid} {z_helio} {ra} {dec}", "filter filt1"]
        for i in range(N_filt1):
            raw_str_list.append(f"{mjd[i]} {mag[i]} {mag_err[i]}")
        raw_str_list.append("filter filt2")
        for i in range(N_filt2):
            raw_str_list.append(f"{mjd[N_filt1+i]} {mag[N_filt1+i]} {mag_err[N_filt1+i]}")
        raw = "\n".join(raw_str_list)
        sn_dict, obs_df = io.read_snpy(StringIO(raw))
        # testing sn_dict
        assert sn_dict["SNID"] == "sample_SN"
        assert sn_dict["RA"] == ra
        assert sn_dict["DECL"] == dec
        assert sn_dict["REDSHIFT_HELIO"] == z_helio
        assert np.isclose(sn_dict["MWEBV"], get_MWEBV(ra, dec))
        for key in sn_dict:
            if key.endswith("ERR") or key in ("HOSTGAL_LOGMASS", "REDSHIFT_FINAL", "VPEC"):
                assert sn_dict[key] is None
        # testing obs_df
        np.testing.assert_allclose(obs_df["MJD"].values.astype(float), mjd)
        np.testing.assert_allclose(obs_df["mag"].values.astype(float), mag)
        np.testing.assert_allclose(obs_df["mag_err"].values.astype(float), mag_err)
        np.testing.assert_allclose(obs_df["flux"].values.astype(float), expected_flux)
        np.testing.assert_allclose(obs_df["flux_err"].values.astype(float), expected_flux_err)
        np.testing.assert_equal(obs_df["FLT"].values.astype(str), filts)

    def test_read_snana_fits(self):
        sn_dict, obs_df = io.read_snana_fits(Path(TEST_DIR, "training_data", "BAYESN_test_fits", "BAYESN_test_fits_HEAD.FITS"))
        with open(PICKLE_DIR / "snana_fits.pkl", "rb") as f:
            ref_sn_dict, ref_obs_df = pickle.load(f)
        assert_dicts_match(sn_dict, ref_sn_dict, flag_missing_data=True, rtol=1e-5, atol=1e-8)
        for key in ref_obs_df:
            assert (obs_df[key] == ref_obs_df[key]).all()
        assert "snid" in obs_df

class TestWrite:
    def test_write_from_sn_dict_obs_df(self):
        example_lc = Path(BASE_DIR.parent, "bayesn/data/example_lcs/Foundation_DR1_2016W.txt")
        orig_dict, orig_df = io.read_snana_ascii(example_lc)
        filename = io._write_snana_lcfile(
            output_dir=TEST_DIR,
            snname=orig_dict["SNID"],
            sn_dict=orig_dict,
            obs_df=orig_df,
        )
        filename = Path(TEST_DIR, filename)
        rec_dict, rec_df = io.read_snana_ascii(filename)
        os.remove(filename)
        if "FILTERS" in rec_dict:
            rec_dict["FILTERS"] = rec_dict["FILTERS"].replace(",", "")
        assert_dicts_match(orig_dict, rec_dict)
        compare_cols = ["MJD", "FLT", "flux", "flux_err", "mag", "mag_err"]
        assert_frame_equal(orig_df[compare_cols], rec_df[compare_cols])

    def test_wrapper_w_mags(self):
        mjd = np.array([53000.0, 53002.0])
        flt = np.array(["g", "r"])
        mag = np.array([17.5, 18.0])
        mag_err = np.array([0.05, 0.06])
        expected_flux, expected_flux_err = mag_to_flux(mag, mag_err, 27.5)
        tmax=53001
        z_helio=0.03
        z_cmb=0.03
        ebv_mw=0.012
        ra=98.765
        dec=12.34

        filename = io.write_snana_lcfile(
            output_dir=TEST_DIR,
            snname="SN_test",
            mjd=mjd,
            flt=flt,
            mag=mag,
            mag_err=mag_err,
            tmax=tmax,
            z_helio=z_helio,
            z_cmb=z_cmb,
            ebv_mw=ebv_mw,
            ra=ra,
            dec=dec,
        )
        filename = Path(TEST_DIR, filename)
        rec_dict, rec_df = io.read_snana_ascii(filename)
        os.remove(filename)
        assert rec_dict["RA"] == ra
        assert rec_dict["DECL"] == dec
        assert rec_dict["SEARCH_PEAKMJD"] == tmax
        assert rec_dict["REDSHIFT_HELIO"] == z_helio
        assert rec_dict["REDSHIFT_CMB"] == z_cmb
        assert rec_dict["MWEBV"] == ebv_mw
        np.testing.assert_allclose(rec_df["mag"].values, mag, rtol=1e-6)
        np.testing.assert_allclose(rec_df["mag_err"].values, mag_err, rtol=1e-6)
        np.testing.assert_allclose(rec_df["flux"].values, expected_flux, rtol=1e-6)
        np.testing.assert_allclose(rec_df["flux_err"].values, expected_flux_err, rtol=1e-6)

    def test_wrapper_w_flux(self):
        mjd = np.array([53000.0, 53002.0])
        flt = np.array(["g", "r"])
        flux = np.array([100, 200])
        flux_err = np.array([11, 15])
        expected_mag, expected_mag_err = flux_to_mag(flux, flux_err, 27.5)
        tmax=53001
        z_helio=0.03
        z_cmb=0.03
        ebv_mw=0.012
        ra=98.765
        dec=12.34

        filename = io.write_snana_lcfile(
            output_dir=TEST_DIR,
            snname="SN_test",
            mjd=mjd,
            flt=flt,
            flux=flux,
            flux_err=flux_err,
            tmax=tmax,
            z_helio=z_helio,
            z_cmb=z_cmb,
            ebv_mw=ebv_mw,
            ra=ra,
            dec=dec,
        )
        filename = Path(TEST_DIR, filename)
        rec_dict, rec_df = io.read_snana_ascii(filename)
        os.remove(filename)
        assert rec_dict["RA"] == ra
        assert rec_dict["DECL"] == dec
        assert rec_dict["SEARCH_PEAKMJD"] == tmax
        assert rec_dict["REDSHIFT_HELIO"] == z_helio
        assert rec_dict["REDSHIFT_CMB"] == z_cmb
        assert rec_dict["MWEBV"] == ebv_mw
        np.testing.assert_allclose(rec_df["flux"].values, flux, rtol=1e-6)
        np.testing.assert_allclose(rec_df["flux_err"].values, flux_err, rtol=1e-6)
        np.testing.assert_allclose(rec_df["mag"].values, expected_mag, rtol=1e-6)
        np.testing.assert_allclose(rec_df["mag_err"].values, expected_mag_err, rtol=1e-6)

    def test_write_uneven_arrays(self):
        N = 100
        rng = np.random.default_rng(0)
        with pytest.raises(TypeError, match="mul got incompatible shapes"):
            io.write_snana_lcfile(
                output_dir=TEST_DIR,
                snname="short_mag",
                mjd=np.linspace(0, 10, N),
                flt=rng.choice(["g", "r", "i", "z"], N),
                mag=rng.normal(18, 2, N-1),  # triggers Value Error
                mag_err=rng.lognormal(-1, 0.1, N),
                tmax=5,
                z_helio=0.05,
                z_cmb=0.05,
                z_cmb_err=1e-5,
                ebv_mw=0.1,
            )
