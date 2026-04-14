import os
from collections.abc import Iterable
from zoneinfo import ZoneInfo
import logging
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
import importlib.resources
from toolz import keyfilter
import jstyleson as json
from addict import Dict as adict
from operator import itemgetter
from toolz import valfilter
import parse
from scipy.interpolate import griddata

import trosat.sunpos as sp

logger = logging.getLogger(__name__)

EPOCH_JD_2000_0 = np.datetime64("2000-01-01T12:00")
def to_datetime64(time, epoch=EPOCH_JD_2000_0):
    """
    Convert various representations of time to datetime64.

    Parameters
    ----------
    time : list, ndarray, or scalar of type float, datetime or datetime64
        A representation of time. If float, interpreted as Julian date.
    epoch : np.datetime64, default JD2000.0
        The epoch to use for the calculation

    Returns
    -------
    datetime64 or ndarray of datetime64
    """
    jd = sp.to_julday(time, epoch=epoch)
    jdms = np.int64(86_400_000*jd)
    return (epoch + jdms.astype('timedelta64[ms]')).astype("datetime64[ns]")

def tz_offset(zone: str, tz_reference: dt.datetime | None = None) -> int:
    if tz_reference is None:
        tz_reference = dt.datetime.now(dt.timezone.utc)
    elif tz_reference.tzinfo is None:
        tz_reference = tz_reference.replace(tzinfo=dt.timezone.utc)
    
    tz_target = ZoneInfo(zone)
    offset = tz_reference.astimezone(tz_target).utcoffset()
    return int(offset.total_seconds())

def offset_hhmm(seconds: int) -> str:
    sign = "+" if seconds >= 0 else "-"
    secs = abs(seconds)
    hh, mm = divmod(secs // 60, 60)
    return f"{sign}{hh:02d}:{mm:02d}"

def dt64_add_tz_offset(x, zone: str):
    offset = np.timedelta64(tz_offset(zone),'s')
    if isinstance(x, Iterable):
        if len(x) == 0:
            return np.array([]).astype("datetime64[us]")
        # handle if x is nested list
        if isinstance(x[0], Iterable):
            dt64 = []
            for xi in x:
                dt64.append([ np.datetime64(pd.to_datetime(t).tz_localize(None),"us") for t in xi])
            dt64 = np.array(dt64)
        else:
            dt64 = np.array([ np.datetime64(pd.to_datetime(t).tz_localize(None),"us") for t in x])
    else:
        dt64 = np.datetime64(pd.to_datetime(x).tz_localize(None),"us")

    return dt64 + offset

def dt64_sub_tz_offset(x, zone: str):
    offset = np.timedelta64(tz_offset(zone),'s')
    if isinstance(x, Iterable):
        if len(x) == 0:
            return np.array([]).astype("datetime64[us]")
        # handle if x is nested list
        if isinstance(x[0], Iterable):
            dt64 = []
            for xi in x:
                dt64.append([ np.datetime64(pd.to_datetime(t).tz_localize(None),"us") for t in xi])
            dt64 = np.array(dt64)
        else:
            dt64 = np.array([ np.datetime64(pd.to_datetime(t).tz_localize(None),"us") for t in x])
    else:
        dt64 = np.datetime64(pd.to_datetime(x).tz_localize(None),"us")

    return dt64 - offset

def round_to(base, x):
    """ Round x to a given base
    """
    return base * np.round(x/base, 0)

def read_json(fpath: str, *, object_hook: type = adict, cls = None) -> dict:
    """ Parse json file to python dict.
    """
    with open(fpath,"r") as f:
        js = json.load(f, object_hook=object_hook, cls=cls)
    return js

def pick(whitelist: list[str], d: dict) -> dict:
    """ Keep only whitelisted keys from input dict.
    """
    return keyfilter(lambda k: k in whitelist, d)

def omit(blacklist: list[str], d: dict) -> dict:
    """ Omit blacklisted keys from input dict.
    """
    return keyfilter(lambda k: k not in blacklist, d)

def get_var_attrs(d: dict) -> dict:
    """
    Parse cf-compliance dictionary.

    Parameters
    ----------
    d: dict
        Dict parsed from cf-meta json.

    Returns
    -------
    dict
        Dict with netcdf attributes for each variable.
    """
    get_vars = itemgetter("variables")
    get_attrs = itemgetter("attributes")
    vattrs = {k: get_attrs(v) for k,v in get_vars(d).items()}
    for k,v in get_vars(d).items():
        vattrs[k].update({
            "dtype": v["type"],
            "gzip":True,
            "complevel":6
        })
    return vattrs

def get_attrs_enc(d : dict) -> (dict,dict):
    """ Split variable attributes in attributes and encoding-attributes.
    """
    _enc_attrs = {
        "scale_factor",
        "add_offset",
        "_FillValue",
        "dtype",
        "zlib",
        "gzip",
        "complevel",
        "calendar",
    }
    # extract variable attributes
    vattrs = {k: omit(_enc_attrs, v) for k, v in d.items()}
    # extract variable encoding
    vencode = {k: pick(_enc_attrs, v) for k, v in d.items()}
    return vattrs, vencode

def get_default_config():
    """
    Get shrad default config
    """
    fn_config = os.path.join(
        importlib.resources.files("shrad"),
        "conf/shrad_config.json"
    )
    default_config = read_json(fn_config)

    # expand default file paths
    for key in default_config:
        if key.startswith("file"):
            default_config.update({
                key: os.path.join(
                    importlib.resources.files("shrad"),
                    default_config[key]
                )
            })
    return default_config

def merge_config(config):
    """
    Merge config dictionary with taro default config
    """
    default_config = get_default_config()
    if isinstance(config,str):
        config = read_json(config)
    if config is None:
        config = default_config
    else:
        config = {**default_config, **config}
    return config

def init_logger(config=None):
    """
    Initialize Logging based on taro config
    """
    config = merge_config(config)
    fname = os.path.abspath(config["file_log"])
    print(fname)
    # logging setup
    logging.basicConfig(
        filename=fname,
        encoding='utf-8',
        level=logging.DEBUG,
        format='%(asctime)s %(name)s %(levelname)s:%(message)s',
    )
    logging.captureWarnings(True)


def get_pfx_time_from_raw_input(input_files, config=None):
    """Parsing Date,Time and campaign/prefix from raw GUVis files (config["fname_raw"]).
    The file prefix is identified for later use for for all dataset files and to identify ancillary data.
    """
    config = merge_config(config)

    input_dates = []
    input_pfxs = []
    for fn in input_files:
        fname_info = parse.parse(
                    config["fname_raw"],
                    os.path.basename(fn)
                ).named
        input_dates.append(fname_info["dt"])
        input_pfxs.append(fname_info["campaign"])

    pfx = np.unique(input_pfxs)
    if len(pfx) != 1:
        raise ValueError("Input files should always have the same prefix")
    pfx = str(pfx[0])
    return pfx, input_dates

def get_calibration_factor(date, config=None):
    """
    Retrieve the corrected calibration factor for GUVis from
    GUVis_calibrations.json

    Parameters
    ----------
    date: numpy.datetime64
        Day of the data
    file: str
        Path to the calibration file (.json)

    Returns
    -------
    calib_ds: xarray.Dataset
        variables:
            centroid_wvl: nm
                the centre wavelength of the spectral response
            calibration_factor: V / (uW cm-2 nm-1)
                drift corrected calibration factor
            calibration_factor_stored: V / (uW cm-2 nm-1)
                calibration factor stored by Biospherical calibration procedure in the instrument storage
            signal_noise_ratio:
                Signal/Noise ration retrieved from the Biospherical calibration certificate.
        coords:
            channel: nm
                Name of the spectral channel of the GUVis
    """
    config = merge_config(config=config)
    date = to_datetime64(date).astype("datetime64[D]").astype(int)
    calibrations = read_json(config["file_calibration"])

    channel = calibrations['_CHANNEL']
    cwvl = calibrations['_CENTROID_WVL']
    cdates = list(calibrations['calibration'].keys())
    cdates = np.array(cdates, dtype='datetime64[D]').astype(int)
    values = []
    snrs = []
    stored = []
    for c in calibrations['calibration'].keys():
        val = np.array(calibrations['calibration'][c]['calibF'])
        val[val is None] = np.nan
        snr = np.array(calibrations['calibration'][c]['SNR'])
        snr[snr is None] = np.nan
        stored.append(calibrations['calibration'][c]['stored'])
        if len(values) == 0:
            values = np.array(val)
            snrs = np.array(snr)
        else:
            values = np.vstack((values, val))
            snrs = np.vstack((snrs, snr))
    stored = np.array(stored, dtype=bool)

    si = np.argsort(cdates)
    cdates = cdates[si]
    values = np.array(values[si, :], dtype=float)
    snrs = np.array(snrs[si, :], dtype=float)
    stored = stored[si]

    # fill nan values with interpolatet values
    for i in range(len(values[0, :])):
        mask = np.isnan(values[:, i])
        values[mask, i] = np.interp(np.flatnonzero(mask),
                                    np.flatnonzero(~mask),
                                    values[~mask, i])
        mask = np.isnan(snrs[:, i])
        snrs[mask, i] = np.interp(np.flatnonzero(mask),
                                  np.flatnonzero(~mask),
                                  snrs[~mask, i])

    # interpolation linear between the closest two calibrations
    # to correct calibration assuming a linear drift
    ca = griddata(cdates, values, date, method='linear')
    snr = griddata(cdates, snrs, date, method='linear')
    if np.all(np.isnan(ca)):
        ca = griddata(cdates, values, date, method='nearest')
        snr = griddata(cdates, snrs, date, method='nearest')

    # return cdates,stored,date,values
    # stored calibration in GUVis uLogger
    cs = values[stored, :][np.searchsorted(cdates[stored], date) - 1, :]

    calib_ds = xr.Dataset({'centroid_wvl': ('channel', cwvl),
                           'calibration_factor': ('channel', ca),
                           'calibration_factor_stored': ('channel', cs),
                           'signal_noise_ratio': ('channel', snr)},
                          coords={'channel': ('channel', channel)})

    return calib_ds