import logging
import os
import warnings

from retry import retry

from astropy.table import Table
import astropy.units as u
import numpy as np

import lightkurve as lk
from .lk_patch.interact import _get_corrected_coordinate, _create_background_task
from .lk_patch.timed import timed


log = logging.getLogger(__name__)

if os.environ.get("MAST_ENABLE_CLOUD_DATASET", True):
    # Prefer AWS to download MAST products (LCs, TPFs, etc.). Search is still on MAST
    # (requires boto3)
    from astroquery.mast import Observations

    Observations.enable_cloud_dataset()


LK_SEARCH_NUM_RETRIES = os.environ.get("LK_SEARCH_NUM_RETRIES", 4)

# use tpf_utils.log instance so that retry warning would show up in gcloud log

@retry(IOError, tries=LK_SEARCH_NUM_RETRIES, delay=0.5, backoff=2, jitter=(0, 0.5), logger=log)
def search_targetpixelfile(*args, **kwargs):
    return lk.search_targetpixelfile(*args, **kwargs)


@retry(IOError, tries=LK_SEARCH_NUM_RETRIES, delay=0.5, backoff=2, jitter=(0, 0.5), logger=log)
def search_tesscut(*args, **kwargs):
    return lk.search_tesscut(*args, **kwargs)


def fast_search_tesscut(target, sector):
    """Return a `SearchResult` for TessCut for a given target / sector quickly,
    bypassing querying MAST.
    This function assumes the target (typically a TIC) and sector are valid.
    """
    if target is None:
        raise ValueError("fast_search_tesscut(): parameter tic must not be None.")

    # sector gets validated in get_exptime()

    def get_exptime(sector):
        if not isinstance(sector, int):
            raise ValueError("fast_search_tesscut(): parameter sector must be int.")
        if sector < 1:
            raise ValueError("fast_search_tesscut(): invalid sector: {sector}.")
        if sector < 27:
            return 1426  # 30-min cadence in cycles 1 - 2
        if sector < 56:
            return 475  # 10-min cadence in cycles 3 - 4
        return 158  # 200-sec cadence in cycles 5+

    exptime = get_exptime(sector)
    # sector time, it's not really used in the use case
    # so I just hardcode a time that it earlier than TESS mission (~ year 1995) to signify it's a dummy value
    t_min = 50000

    tab_str = f"""\
    description,mission,target_name,targetid,t_min,exptime,productFilename,provenance_name,author,distance,sequence_number,project,obs_collection
    TESS FFI Cutout(sector {sector}),TESS Sector {sector:02d},{target},{target},{t_min},{exptime},TESSCut,TESSCut,TESSCut,0.0,{sector},TESS,TESS"""  # noqa: E501

    return lk.SearchResult(Table.read(tab_str, format="ascii.csv"))


@timed()
def _do_download_spoc_tpf(sr, download_kwargs=None):
    if download_kwargs is None:
        download_kwargs = dict()
    return sr[-1].download(**download_kwargs)


@timed()
def _do_download_tesscut(sr, download_kwargs=None):
    # TODO: query TIC catalog to backfill proper motion / TESSMAG (or use patched lk.search)
    # (used by the webapp)
    if download_kwargs is None:
        download_kwargs = dict()
    if download_kwargs.get("cutout_size", None) is None:
        download_kwargs = download_kwargs.copy()
        download_kwargs["cutout_size"] = (11, 11)  # OPEN: would be too small for bright stars

    tpf = sr[-1].download(**download_kwargs)

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


async def get_tpf(tic, sector, msg_label, search_sources=None, search_kwargs=None, download_kwargs=None):
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
        return await _do_get_tpf(
            tic, sector, msg_label, search_sources=search_sources, search_kwargs=search_kwargs, download_kwargs=download_kwargs
        )
    finally:
        search_log.error = error_original


async def _do_get_tpf(tic, sector, msg_label, search_sources=None, search_kwargs=None, download_kwargs=None):
    if search_sources is None or search_sources == "all":
        to_search_tpf, to_search_tesscut = True, True
    elif search_sources == "tpf":
        to_search_tpf, to_search_tesscut = True, False
    elif search_sources == "tesscut":
        to_search_tpf, to_search_tesscut = False, True
    else:
        raise ValueError(f"Invalid search_sources: {search_sources}")

    @timed()
    def do_search_tpf():
        if not to_search_tpf:
            log.debug("do_search_tpf() skipped due to user override")
            return lk.SearchResult()

        search_kwargs_actual = search_kwargs if search_kwargs is not None else dict()

        sr = search_targetpixelfile(f"TIC{tic}", mission="TESS", sector=sector, **search_kwargs_actual)
        if len(sr) > 1 and len(search_kwargs_actual) < 1:
            # by default, exclude fast cadence data (20s), TPFs with fast cadence always has 2 min cadence counterparts
            # for the use case here, the fast cadence data is irrelevant. It'd just make the processing slower.
            #
            # if user specifies `search_kwargs`, no default exclusion is done,
            # to allow users to get fast cadence data, e.g., by specifying exptime
            sr = sr[sr.exptime > 60 * u.s]
        return sr

    @timed()
    def do_search_tesscut():
        if not to_search_tesscut:
            log.debug("do_search_tesscut() skipped due to user override")
            return lk.SearchResult()
        sr = search_tesscut(f"TIC{tic}", sector=sector)
        return sr

    @timed()
    def do_fast_search_tesscut():
        # timing it is somewhat useless as it's almost instantaneous
        # but it makes the timing log more uniform.
        if not to_search_tesscut:
            log.debug("do_fast_search_tesscut() skipped due to user override")
            return lk.SearchResult()
        sr = fast_search_tesscut(f"TIC{tic}", sector=sector)
        return sr

    # Search TPF and TessCut in parallel, to speed up for cases for TessCut is to be used
    # at the expanse of extra TessCut search for cases TPF is to be used
    sr_tpf_task = _create_background_task(do_search_tpf)
    if sector is not None:
        # case sector specified, bypass querying MAST to speed things up.
        sr_tc_task = _create_background_task(do_fast_search_tesscut)
    else:
        sr_tc_task = _create_background_task(do_search_tesscut)

    sr = await sr_tpf_task
    if len(sr) > 0:
        # case downloading a TPF
        sr_tc_task.cancel()  # no longer needed
        tpf = _do_download_spoc_tpf(sr, download_kwargs)
        return tpf, sr

    sr = await sr_tc_task
    if len(sr) > 0:
        # case downloading a TessCut
        log.debug(f"No TPF found for {msg_label}. Use TessCut.")
        try:
            tpf = _do_download_tesscut(sr, download_kwargs)
            return tpf, sr
        except lk.search.SearchError as e:
            if "index 0 is out of bounds" in str(e):
                # special case the error is likely because there is no TessCut in the specified sector.
                #
                # root cause: the shortcut do_fast_search_tesscut() will return SearchResult
                # object for the given sector even if there is no data.
                # When download is attempted, it would fail with an IndexError
                # in SearchResult._fetch_tesscut_path(), line:
                #    sector_name = sec[sec["sector"] == sector]["sectorName"][0]
                # The `IndexError: index 0 is out of bounds for axis 0 with size 0`
                # is then wrapped as a SearchError.
                #
                # Here we try to catch this special case, and treat it as no TPF found,
                # which also reduces log pollution.
                log.debug(f"TPF likely not found for {msg_label} in  TessCut either. {e}")
                return None, None
            else:
                raise e

    # case no TPF nor TessCut
    return None, None


def is_tesscut(tpf):
    return "astrocut" == tpf.meta.get("CREATOR")


def has_non_science_pixels(tpf):
    # see figure 4.3 of https://archive.stsci.edu/missions/tess/doc/TESS_Instrument_Handbook_v0.1.pdf
    # or https://heasarc.gsfc.nasa.gov/docs/tess/data-products.html#full-frame-images
    return (
        tpf.column < 45  # virtual pixels to the left
        or tpf.column + tpf.shape[2] > 2092  # virtual pixels to the right
        or tpf.row + tpf.shape[1] > 2048  # virtual pixels above
        or tpf.row < 1  # virtual pixels below (Should not happen, but keep it here for just in case)
    )


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


def create_mask_for_target(tpf, mask_shape="1pixel"):
    """Create a mask of the pixel the target is located.
    A 1-pixel mask is returned by default.
    """
    t_ra, t_dec, _ = _get_corrected_coordinate(tpf)
    pix_x, pix_y = tpf.wcs.all_world2pix([(t_ra, t_dec)], 0)[0]
    # + 0.5: the pixel coordinate refers to the center of the pixel
    #        e.g., for y=2.7, visually it's on y=3, as y=2 really covers [1.5, 2.5]
    xx, yy = int(pix_x + 0.5), int(pix_y + 0.5)
    data = tpf.flux[0]
    mask = np.full(data.shape, False)
    if mask_shape == "1pixel":
        mask[yy][xx] = True
    elif mask_shape == "3x3":
        # Make sure the 3x3 patch does not leave the TPF bounds
        # based on lightkurve.utils.centroid_quadratic()
        # https://github.com/lightkurve/lightkurve/blob/main/src/lightkurve/utils.py
        if yy < 1:
            yy = 1
        if xx < 1:
            xx = 1
        if yy > (data.shape[0] - 2):
            yy = data.shape[0] - 2
        if xx > (data.shape[1] - 2):
            xx = data.shape[1] - 2

        mask[yy - 1 : yy + 2, xx - 1 : xx + 2] = True  # noqa: E203
    else:
        raise ValueError(f"Unsupported mask_shape parameter value: '{mask_shape}'")
    return mask


def create_background_mask_by_threshold(tpf, exclude_target_pixels=True):
    """Create a rough background mask."""
    # based on:
    # https://github.com/lightkurve/lightkurve/blob/main/docs/source/tutorials/2-creating-light-curves/2-1-cutting-out-tpfs.ipynb
    background_mask_initial = ~tpf.create_threshold_mask(threshold=0.001, reference_pixel=None)
    if not exclude_target_pixels:
        return background_mask_initial

    target_mask = create_mask_for_target(tpf, mask_shape="3x3")
    background_mask = background_mask_initial & ~target_mask
    if background_mask.sum() < 1:
        background_mask = background_mask_initial
        warnings.warn(
            "0 pixel in background mask after excluding those around the target. "
            f"Revert to the background mask without the exclusion, with {background_mask.sum()} pixels."
        )
    return background_mask


def create_background_per_pixel_lc(tpf, exclude_target_pixels=True):
    """Helper for a rough background subtraction, used in TessCut TPFs."""
    # based on:
    # https://github.com/lightkurve/lightkurve/blob/main/docs/source/tutorials/2-creating-light-curves/2-1-cutting-out-tpfs.ipynb
    background_mask = create_background_mask_by_threshold(tpf, exclude_target_pixels=exclude_target_pixels)
    n_background_pixels = background_mask.sum()
    background_lc_per_pixel = tpf.to_lightcurve(aperture_mask=background_mask) / n_background_pixels
    return background_lc_per_pixel


def subtract_background(lc, tpf):
    """Subtract a (rough) background from the lightcurve, used in TessCut TPFs.
    The rough background is created from `create_background_per_pixel_lc()`
    """
    return lc - create_background_per_pixel_lc(tpf) * lc.meta["APERTURE_MASK"].sum()
