import os
import re
import gzip
import shutil
import click
import logging
import pandas as pd
import numpy as np
import xarray as xr

import shrad.utils

logger = logging.getLogger(__name__)

def ident_header(filepath):
    with open(filepath,"r",encoding="ISO-8859-1") as txt:
        header = True
        lheader = 0
        while header:
            line = txt.readline()
            if lheader==0 and not line.startswith("Start of Header"):
                break
            if line.startswith("End of Header"):
                break
            lheader+=1
        headerline = txt.readline()
    rows_to_skip = lheader+1
    return headerline, rows_to_skip

def read_last_line(filepath, chunk_size=1024):
    """Read the last line of a file efficiently."""
    try:
        with open(filepath, 'rb') as f:
            # Start from the end of the file
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()
            
            if file_size == 0:
                return None
            
            # Read backwards in chunks
            chunk = b''
            offset = min(chunk_size, file_size)
            
            while True:
                f.seek(file_size - offset)
                chunk = f.read(offset)
                
                # Count newlines in chunk
                newline_count = chunk.count(b'\n')
                
                # require at least two line breaks in case file ends on empty line
                if newline_count >= 2 or offset >= file_size:
                    break
                
                offset += chunk_size
                offset = min(offset, file_size)
            
            # Find the last newline position
            last_newline = len(chunk)
            # skip empty line
            while last_newline >= len(chunk)-1:
                last_newline = chunk[:last_newline].rfind(b'\n')

            if last_newline != -1:
                last_line = chunk[last_newline + 1:]
            else:
                last_line = chunk
            
            if len(last_line)==0:
                chunk[:]
            
            return last_line.decode('utf-8', errors='replace').strip()
    except FileNotFoundError:
        return None

def get_datetime_cols(header):
    datetime_cols = [i for i,var in enumerate(header.split(',')) if var.startswith("DateTime")]
    return datetime_cols


def gzip_raw(fname, keep=False):
    with open(fname, 'rb') as f_in:
        with gzip.open(fname+".gz", 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    if not keep:
        os.remove(fname)

def get_flx_vars(vars):
    def get_vars_from_pfx(vars,pfx):
        flxvars = []
        for var in vars:
            try:
                # strip prefix and "_corr" for corrected channels
                int(var.strip(pfx).strip("_corr"))
            except:
                continue
            flxvars.append(var)
        return flxvars

    # Try with uLogger prefix
    pfx = "Es"
    flxvars = get_vars_from_pfx(vars,pfx=pfx)
    if len(flxvars)!=0:
        return pfx, flxvars
    
    # Try with uProfile prefix
    pfx = "Ed0"
    flxvars = get_vars_from_pfx(vars,pfx=pfx)
    if len(flxvars)!=0:
        return pfx, flxvars
    
    # No Channels detected
    raise ValueError("No Flux variables detected")


def load_rawdata_and_combine(input_files,config=None):
    """ Reading raw GUVis files and combine to one dataframe
    """
    config = shrad.utils.merge_config(config=config)


    complete_df = None
    count_raw = 0
    total_size = 0.

    item_show_func = lambda a: f"{os.path.basename(a)} {fsize:.2f}MB" if a is not None else ""
    logger.info(f"Parsing {len(input_files)} raw files.")
    with click.progressbar(input_files, label='Load raw files:', item_show_func=item_show_func) as files:
        for i,fname in enumerate(files):
            fsize = os.path.getsize(fname) / 1024 ** 2
            if fsize < 0.1:
                logger.info((f'Skip file {fname} of size {fsize:.2f}MB - less or no data.'))
                continue
            df = pd.read_csv(fname, sep=',', encoding="ISO-8859-1")
            # df = pd.read_csv(fname, sep=',')
            count_raw += 1
            total_size += fsize
            if complete_df is None:
                complete_df = df.copy()
            else:
                complete_df = pd.concat([complete_df, df], ignore_index=True)

    if complete_df is None:
        # no files, or all files are empty
        logger.warning("No raw files are loaded.")
        return False

    logger.info("Start homogenizing the raw dataset.")
    # homogenize  dataframe
    complete_df.drop_duplicates(subset='DateTimeUTCISO', keep='first', inplace=True)
    complete_df.reset_index(drop=True, inplace=True)
    # remove the unit appendix in standard raw csv data of GUVis        
    keys_rename = {}
    keys_units = {}
    for k in complete_df.keys():
        ksplit = k.split(' ', 1)
        ksplit += [''] * (2 - len(ksplit))
        keys_rename.update({k: ksplit[0]})
        keys_units.update({ksplit[0]:ksplit[1]})
    complete_df = complete_df.rename(keys_rename, axis='columns')

    # to xarray dataset
    ds = xr.Dataset.from_dataframe(complete_df)

    # parse datetime for date objects
    for key in ds:
        if not key.startswith("DateTime"):
            continue
        datetime = pd.to_datetime(ds[key].values)
        ds[key].values = datetime

    ds = ds.rename_vars({'DateTimeUTCISO': 'time'})
    ds = ds.swap_dims({'index': 'time'})
    ds = ds.reset_coords(names=['index'], drop=True)
    
    # drop unnecessary variables
    # drop duplicate or local time variables
    ds = ds.drop_vars([key for key in ds if key.startswith("DateTime")])
    # drop milliseconds as already included in datetimeUTCISO
    ds = ds.drop_vars([key for key in ds if key.startswith("Millisecond")])

    # add units
    for key in ds:
        ds[key].attrs.update({"units":keys_units[key]})

    # Bug correction for uLogger version < 1.0.24
    if ("BioGpsTime" in list(ds.keys())) and (ds.time.values[0] < np.datetime64("2016-04-01")):
        lat = ds.BioGpsLatitude.values
        ni = lat < 0
        lat[ni] = np.floor(lat[ni]) + 1. - (lat[ni] - np.floor(lat[ni]))
        ds.BioGpsLatitude.values = lat
    
    logger.info(f"Done loading {count_raw} raw files of total size {total_size:.2f}MB.")
    return ds


def calibrate_raw(ds,config=None):
    config = shrad.utils.merge_config(config=config)
    calib_ds = shrad.utils.get_calibration_factor(date=ds.time.values[0], config=config)

    channel_pfx, flxvars = get_flx_vars(ds)

    for key in ds:
        if not key in flxvars:
            # skip non flux measurements
            continue
        
        channel = key.strip(channel_pfx).strip("_corr")

        # check if calibration is stored for this channel
        stored_calib = False if re.match(".*V", ds[key].attrs["units"]) else True
        stored_calib = False if int(channel)==0 else stored_calib # broadband is never stored

        # get calibration factors
        cds = calib_ds.sel(channel=int(channel))
        cs = cds.calibration_factor_stored.values if stored_calib else 1.
        ca = cds.calibration_factor.values

        # calibrate
        ds[key].values = ds[key].values * cs / ca

        # uW cm-2 -> W m-2
        ds[key].values = ds[key].values * 1e-2
        
        # update units
        unit = "W m-2 nm-1" if channel!=0 else "W m-2"
        ds[key].attrs.update({
            "units": unit
        })
        
    return ds

def make_nc():
    # Make nice dataset with attributes
    dsa = xr.Dataset()
    dsa = dsa.assign_coords({'time': ('time', ds.time.data)})

    channels = calib_ds.channel.data
    channel_idx = np.where(channels != 0)  # skipping broadband for now
    # add channels as dimension
    key = 'wavelength'
    dsa = dsa.assign_coords({'wavelength': ('ch', channels[channel_idx])})
    dsa[key].attrs.update({'standard_name': CONFIG['CF Standard Names'][key],
                           'units': CONFIG['CF Units'][key]})

    # add specified centroid wavelength of each spectral channel
    key = 'centroid_wavelength'
    dsa = dsa.assign({key: ('ch', calib_ds.centroid_wvl.data[channel_idx])})
    dsa[key].attrs.update({'standard_name': CONFIG['CF Standard Names'][key],
                           'units': CONFIG['CF Units'][key]})

    # add measurements of spectral flux of all spectral channels
    for i, ch in enumerate(dsa.wavelength.values):
        if i == 0:
            rad = ds[f'Es{ch}'].data
        else:
            rad = np.vstack((rad, ds[f'Es{ch}'].data))
    rad = rad.T
    # convert units: uW cm-2 nm-1 -> W m-2 nm
    rad *= 1e-2

    key = 'spectral_flux'
    dsa = dsa.assign({key: (('time', 'ch'), rad)})
    dsa[key].attrs.update({'standard_name': CONFIG['CF Standard Names'][key],
                           'units': CONFIG['CF Units'][key]})

    # if available, add the broadband measurements too
    if 'Es0' in ds.keys():
        key = 'broadband_flux'
        bb_flux = ds['Es0'].data
        bb_flux *= 1e-2 # uW cm-2 -> W m-2
        dsa = dsa.assign({key: ('time', bb_flux)})
        dsa[key].attrs.update({'standard_name': CONFIG['CF Standard Names'][key],
                               'units': CONFIG['CF Units'][key],
                               'notes': 'Measured with the GUVis Radiometer'})

    for key in ['EsRoll',
                'EsPitch',
                'BioShadeAngle',
                'BioShadeMode',
                'BioGpsLongitude',
                'BioGpsLatitude',
                'EsTemp',
                'SolarAzimuthAngle',
                'SolarZenithAngle']:
        if key in ds.keys():
            dsa = dsa.assign({key: ('time', ds[key].data)})
            if key in CONFIG['NC Variables Map'].keys():
                cfgkey = CONFIG['NC Variables Map'][key]
            else:
                cfgkey = key
            dsa[key].attrs.update({'standard_name': CONFIG['CF Standard Names'][cfgkey],
                                   'units': CONFIG['CF Units'][cfgkey],
                                   'notes': f'Obtained from GUVis raw data'})
    for key in ['EsPitch','EsRoll']:
        ds[key].attrs.update({'coordinatesys':'guvis-aligned'})

    if verbose:
        prints("... done", lvl=lvl)
    return dsa