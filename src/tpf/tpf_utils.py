import logging
import warnings

import astropy.units as u
import numpy as np

import lightkurve as lk
from .lk_patch.interact import _create_background_task

log = logging.getLogger(__name__)


async def _do_download_tesscut(sr):
    cutout_size = (11, 11)  # OPEN: would be too small for bright stars
    # TODO: query TIC catalog to backfill proper motion / TESSMAG
    # (used by the webapp)
    tpf = sr[-1].download(cutout_size=cutout_size)

    try:
        # tweaks to make it look like SPOC-produced TPF
        tic = int(sr.target_name[-1].replace("TIC", ""))
        tpf.hdu[0].header["TICID"] = tic
        tpf.hdu[0].header["LABEL"] = f"TIC {tic}"
        tpf.hdu[0].header["OBJECT"] = f"TIC {tic}"
        tpf.meta = lk.targetpixelfile.HduToMetaMapping(tpf.hdu[0])
    except Exception as e:
        warnings.warn("Unexpected error in extracting TIC from TessCut SearchResult. TIC will not be shown." f" Error: {e}")
    return tpf


async def get_tpf(tic, sector, msg_label):
    # suppress the unnecessary logging error messages "No data found for target ..." from search
    # they just pollute the log output in an webapp environment
    search_log = lk.search.log
    error_original = search_log.error

    def error_ignore_no_data(msg, *args, **kwargs):
        if str(msg).startswith("No data found for target "):
            return
        error_original(msg, *args, **kwargs)

    search_log.error = error_ignore_no_data
    try:
        return await _do_get_tpf(tic, sector, msg_label)
    finally:
        search_log.error = error_original


async def _do_get_tpf(tic, sector, msg_label):
    def do_search_tpf():
        sr = lk.search_targetpixelfile(f"TIC{tic}", mission="TESS", sector=sector)
        if len(sr) > 1:
            # exclude fast cadence data (20s), TPFs with fast cadence always has 2 min cadence counterparts
            # for the use case here, the fast cadence data is irrelevant. It'd just make the processing slower.
            sr = sr[sr.exptime > 60 * u.s]
        return sr

    def do_search_tesscut():
        # TODO: query tesspoint to avoid returning sectors where the target is not on science pixels
        # e.g, TIC 160193537, sector 74, TessCut can return a TPF but the target is not on science pixels
        sr = lk.search_tesscut(f"TIC{tic}", sector=sector)
        return sr

    # Search TPF and TessCut in parallel, to speed up for cases for TessCut is to be used
    # at the expanse of extra TessCut search for cases TPF is to be used
    sr_tpf_task = _create_background_task(do_search_tpf)
    sr_tc_task = _create_background_task(do_search_tesscut)

    sr = await sr_tpf_task
    if len(sr) > 0:
        # case downloading a TPF
        sr_tc_task.cancel()  # no longer needed
        tpf = sr[-1].download()
        return tpf, sr

    sr = await sr_tc_task
    if len(sr) > 0:
        # case downloading a TessCut
        log.debug(f"No TPF found for {msg_label}. Use TessCut.")
        tpf = await _do_download_tesscut(sr)
        return tpf, sr

    # case no TPF nor TessCut
    return None, None


def is_tesscut(tpf):
    return "astrocut" == tpf.meta.get("CREATOR")


def cutout_by_range(tpf, aperture_mask, col_range, row_range):
    img_shape = tpf.flux[0].shape

    # handle cases the given range is outside the TPF (zoomed out)
    col_start = col_range[0] if col_range[0] >= 0 else 0
    col_end = col_range[1] if col_range[1] <= img_shape[1] else img_shape[1]
    row_start = row_range[0] if row_range[0] >= 0 else 0
    row_end = row_range[1] if row_range[1] <= img_shape[0] else img_shape[0]
    col_range, row_range = (col_start, col_end), (row_start, row_end)

    center = (
        np.ceil((col_range[1] - col_range[0]) / 2) + col_range[0],
        np.ceil((row_range[1] - row_range[0]) / 2) + row_range[0],
    )
    size = (col_range[1] - col_range[0], row_range[1] - row_range[0])
    if col_range[0] == 0 and row_range[0] == 0 and size == (img_shape[1], img_shape[0]):
        return tpf, aperture_mask  # no cutout is needed (full range)
    # else cutout is needed
    tpf_cut = tpf.cutout(center=center, size=size)
    aperture_mask_cut = aperture_mask[
        row_range[0] : row_range[1], col_range[0] : col_range[1]  # noqa: E203 (use black format for :)
    ]
    return tpf_cut, aperture_mask_cut
