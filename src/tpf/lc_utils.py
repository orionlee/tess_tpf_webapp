import re
import warnings

from astropy.time import Time
import astropy.units as u
from astropy.table import Table

import numpy as np

import lightkurve as lk

from lightkurve.interact_sky_providers.skypatrol2 import get_lightcurve


def read_ztf_csv(
    url=None,
    time_column="hjd",
    time_format="jd",
    time_scale="utc",
    flux_column="mag",
    flux_err_column="magerr",
    mask_func=lambda lc: lc["catflags"] != 0,
):
    """Return ZTF Archive lightcurve files in IPAC Table csv.

    Parameters
    ----------
    mask_func : function, optional
        a function that returns a boolean mask given a `Lightcurve` object
        of the data. Cadences with `True` will be masked out.
        Pass `None` to disable masking.
        The default is to exclude cadences where `catflags` is not 0, the
        guideline for VSX submission.
        https://www.aavso.org/vsx/index.php?view=about.notice
    """
    # Note: First tried to read ZTF's ipac table .tbl, but got
    #   TypeError: converter type does not match column type
    # due to: https://github.com/astropy/astropy/issues/15989

    def get_required_column(tab, colname):
        if colname not in tab.colnames:
            raise ValueError(f"Required column {colname} is not found")
        return tab[colname]

    def filter_rows_with_invalid_time(tab, colname):
        # Some times the time value is nan for unknown reason. E.g.,
        # https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID=848110200002484&COLLECTION=ztf_dr20&FORMAT=csv
        # some hjd values are nan, even though there is mjd value
        filtered = tab[np.isfinite(tab[colname])]
        num_rows_filtered = len(tab) - len(filtered)
        if num_rows_filtered > 0:
            warnings.warn(f"{num_rows_filtered} skipped because they do not have valid time values.", lk.LightkurveWarning)
        return filtered

    tab = Table.read(
        url,
        format="ascii.csv",
        converters={
            "oid": np.int64,
            "expid": np.int64,
            "filefracday": np.int64,
        },
    )

    tab = filter_rows_with_invalid_time(tab, time_column)

    time = get_required_column(tab, time_column)
    time = Time(time, format=time_format, scale=time_scale)
    flux = get_required_column(tab, flux_column)
    flux_err = get_required_column(tab, flux_err_column)

    lc = lk.LightCurve(
        time=time,
        flux=flux,
        flux_err=flux_err,
        data=tab,
    )

    # assign units
    for col in ["flux", "flux_err", flux_column, flux_err_column, "limitmag"]:
        if col in lc.colnames:
            lc[col] = lc[col] * u.mag

    lc.meta.update(
        {
            "FILEURL": url,
            "FLUX_ORIGIN": flux_column,
            "TIME_ORIGIN": time_column,
        }
    )

    oid_match = re.search(r"https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves.+ID=(\d+)", url)
    if oid_match is not None:
        id = f"ZTF OID {oid_match[1]}"  # include data release number too?
        lc.meta["OBJECT"] = id
        lc.meta["LABEL"] = id

    if mask_func is not None:
        mask = mask_func(lc)
        lc = lc[~mask]

    return lc


def read_lc(url):
    """Read a lightcurve data supported by the webapp"""
    asas_sn_id = get_asas_sn_id(url)
    if asas_sn_id is not None:
        lc = get_lightcurve(asas_sn_id)
        return lc
    else:  # assumed to be an URL for ZTF CSV for now
        lc = read_ztf_csv(url)

        if "filtercode" in lc.colnames:  # include ZTF filter in label for title
            filter_str = ",".join(np.unique(lc["filtercode"]))
            lc.label += f" ({filter_str})"
        return lc


def get_asas_sn_id(url):
    # SkyPatrol v2 URL is http, but I include https to be more flexible
    match = re.match(r"^https?://asas-sn.ifa.hawaii.edu/skypatrol/objects/(\d+)", url)
    if match is not None:
        return int(match[1])

    return None


def guess_lc_source(url):
    if get_asas_sn_id(url) is not None:
        return "skypatrol2"
    elif url.startswith("https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"):
        # in theory, ZTF CSV can come from other URLs, but this is the form to be used in practice.
        return "ztf"
    else:
        return None
