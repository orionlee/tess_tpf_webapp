import logging
import warnings

import astropy.units as u

import lightkurve as lk
from lightkurve.interact import _create_background_task

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
        warnings.warn(
            "Unexpected error in extracting TIC from TessCut SearchResult. TIC will not be shown."
            f" Error: {e}"
        )
    return tpf


async def get_tpf(tic, sector, msg_label):
    def do_search_tpf():
        sr = lk.search_targetpixelfile(f"TIC{tic}", mission="TESS", sector=sector)
        if len(sr) > 1:
            # exclude fast cadence data (20s), TPFs with fast cadence always has 2 min cadence counterparts
            # for the use case here, the fast cadence data is irrelevant. It'd just make the processing slower.
            sr = sr[sr.exptime > 60 *u.s]
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
    return "astrocut" == tpf.meta.get('CREATOR')
