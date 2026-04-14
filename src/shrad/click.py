import shutil
import os.path
import warnings
import click
import logging
import parse
import numpy as np
import importlib.resources
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt

import shrad
import shrad.utils
import shrad.futils
#import shrad.shcalc



mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)
parse_logger = logging.getLogger("parse")
parse_logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = fn_config = os.path.join(
        importlib.resources.files("shrad"),
        "conf/shrad_config.json"
    )


# initialize commandline interface
@click.version_option()
@click.group("shrad")
def cli():
    pass

@cli.group("raw")
def cli_raw():
    pass

@cli_raw.command("compress")
@click.argument("input_files", nargs=-1)
@click.option("--config", "-c", type=click.Path(dir_okay=False, exists=True),
              help="Config file - will merge and override the default config.")
@click.option("--today/--no-today", default=False, help="Include/exclude data from today. The default is False.")
@click.option("--keep/--no-keep", default=False, help="Keep original file. The default is False.")
def gzip_raw( input_files: str, config, today:bool, keep:bool):
    """Compress raw files, exclude today and remove original file by default.
    """
    config = shrad.utils.merge_config(config)
    shrad.utils.init_logger(config)

    logger.info("Call taro.futils.gzip_raw")
    dt_today = dt.date.today()
    with click.progressbar(input_files, label='Zipping:', item_show_func=lambda a: os.path.basename(a) if a is not None else "") as files:
        for fn in files:
            basename = os.path.basename(fn)
            if fn.endswith(".gz"):
                logger.info(f"Skipping {basename} - is already zipped.")
                continue
            try:
                finfo = parse.parse(config["fname_raw"], basename).named
            except:
                logger.warning(f"Skipping {basename} - doesn't fit the config 'fname_raw' pattern.")
                continue
            if finfo["dt"]==dt_today and not today:
                logger.info(f"Skipping {basename} - not today.")
                continue
            shrad.futils.gzip_raw(fname=fn,keep=keep)

pass_ns = click.make_pass_decorator(dict, ensure=True)
@cli_raw.group("process", chain=True)
@click.argument("input_files", nargs=-1)
@click.option("--config", "-c", type=click.Path(dir_okay=False, exists=True),
              help="Config file - will merge and override the default config.")
@pass_ns
def cli_raw_process(ns, input_files, output_path, config):
    config = shrad.utils.merge_config(config=config)
    shrad.utils.init_logger(config)

    # consider only files that fit in config filename pattern
    logger.info("Scanning files for raw processing.")
    raw_files = []
    raw_dates = []
    for fn in input_files:
        basename = os.path.basename(fn)
        try:
            finfo = parse.parse(config["fname_raw"], basename).named
        except:
            logger.warning(f"Skipping {basename} - doesn't fit the config 'fname_raw' pattern.")
            continue
        raw_files.append(os.path.abspath(fn))
        raw_dates.append(finfo["dt"])

    isort = np.argsort(raw_dates)
    raw_dates = np.array(raw_dates)[isort]
    raw_files = np.array(raw_files)[isort]

    ns.update({
        "input_files": raw_files,
        "input_dates": raw_dates,
        "config": config
    })

@cli_raw_process.command("l1a")
@click.argument("output_path", nargs=1)
@click.option("--config", "-c", type=click.Path(dir_okay=False, exists=True),
              help="Config file - will merge and override the default config.")
@pass_ns
def raw2l1a(ns, output_path: str, config):
    config = shrad.utils.merge_config(config=config)
    if "config" in ns:
        # override and merge with ns config
        config = {**ns["config"], **config}
    input_files = ns["input_files"]

    ### TODO: interate over unique days
    with click.progressbar(input_files, label='Processing to l1a:', item_show_func=lambda a: os.path.basename(a) if a is not None else "") as files:
            for fn in files:
                fname_info = parse.parse(
                    config["fname_raw"],
                    os.path.basename(fn)
                ).named

    

    

