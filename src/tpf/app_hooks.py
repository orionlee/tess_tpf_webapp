import logging
import os
import sys
import time
import threading
import traceback

import psutil

import lightkurve as lk
import astropy


log = logging.getLogger(__name__)
_FILECACHE_TTL_MINS = -1


def set_log_level_from_env():
    # use Python standard string constant in
    #  https://docs.python.org/3/howto/logging.html
    level_str = os.environ.get("TESS_TPF_WEBAPP_LOGLEVEL", None)
    if level_str:
        log.setLevel(level_str)
    return level_str


def get_last_line_of_tb(tb):
    fs = traceback.extract_tb(tb)[-1]
    # summarize the last line of the traceback to a single line
    return f'File "{fs.filename}", line {fs.lineno}, in {fs.name}: {fs.line}'


def _get_open_files():
    try:
        return [f.path for f in psutil.Process().open_files()]
    except Exception as e:
        # we do not want any error to stop the overall process
        if isinstance(e, IndexError):
            # intermittently psutil's `open_files()` fails in gcloud with IndexError
            # log it separately to avoid polluting the log
            tb = sys.exc_info()[2]
            log.debug(
                (
                    f"_get_open_files() failed with (likely intermittent) error {type(e).__name__} {e}. Return empty list. "
                    f"{get_last_line_of_tb(tb)}"
                )
            )
        else:
            # other unexpected errors that might need further investigation
            log.error(
                f"_get_open_files() failed unexpectedly with error of type {type(e).__name__}. Return empty list",
                exc_info=True,
            )
        return []


def _safe_remove_file(filepath):
    try:
        os.remove(filepath)
    except Exception as e:
        # we don't want the failure to stop the overall process
        log.warning(f"_safe_remove_file() failed unexpectedly for '{filepath}'.", e)


def _rm_files_by_atime(basedir, atime_threshold, dry_run=False):
    proc_open_files = _get_open_files()

    total_num_files = 0
    files_removed = []
    for dirpath, _, filenames in os.walk(basedir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            atime = os.path.getatime(filepath)
            total_num_files += 1
            if atime < atime_threshold:
                if dry_run:
                    pass
                elif filepath in proc_open_files:
                    # the intent is to keep TPF FITS files that are still opened / used by bokeh
                    log.debug(
                        f"_rm_files_by_atime() retained file '{filepath}' because " "it is still being opened by the process."
                    )
                else:
                    # pass all check. remove it
                    _safe_remove_file(filepath)
                    files_removed.append(filepath)
    return files_removed, total_num_files


def _get_ttl_mins_from_env():
    ttl = -1
    ttl_str = os.environ.get("TESS_TPF_FILECACHE_TTL", "-1")
    try:
        ttl = int(ttl_str)
    except Exception as e:
        log.warning(f"Invalid value for env var `TESS_TPF_FILECACHE_TTL`: {ttl_str}, Use -1. ", e)

    if ttl < 0 and os.environ.get("K_SERVICE", None) is not None:
        # we always apply a TTL in Google Cloud Run, as the file system is memory,
        # so we must periodically clear cache to avoid out of memory error.
        # See https://cloud.google.com/run/docs/container-contract#filesystem
        #
        # The var `K_SERVICE` is defined in:
        # https://cloud.google.com/run/docs/container-contract#env-vars
        ttl = 15
        log.debug(f"In Google Cloud Run, tess-tpf file cache. No value specified. Fallbacks to the default {ttl}")
    return ttl


_LOCK_CLEAR_OLD_CACHE_ENTRY = threading.Lock()


def _clear_old_cache_entry(ttl_mins):
    def log_clear_file_cache_results(msg_prefix, files_removed, total_num_files):
        files_removed_abbrev = [os.path.basename(f) for f in files_removed]
        log.debug(f"{msg_prefix}: removed {len(files_removed)} / {total_num_files} . {files_removed_abbrev}")

    if ttl_mins < 0:
        return

    # use a lock prevent unwanted side effects,
    # in case it's invoked almost simultaneously by different bokeh sessions
    with _LOCK_CLEAR_OLD_CACHE_ENTRY:
        atime_threshold = time.time() - ttl_mins * 60

        atime_threshold_str = time.strftime(
            "%Y-%m-%d %H:%M:%S %z",
            time.localtime(atime_threshold),
        )
        log.debug(f"_clear_old_cache_entry(): ttl={ttl_mins}, or threshold={atime_threshold_str}")

        lk_files_removed, lk_total_num_files = _rm_files_by_atime(lk.config.get_cache_dir(), atime_threshold)
        log_clear_file_cache_results("Lightkurve cache", lk_files_removed, lk_total_num_files)
        ap_files_removed, ap_total_num_files = _rm_files_by_atime(astropy.config.get_cache_dir(), atime_threshold)
        log_clear_file_cache_results("astropy    cache", ap_files_removed, ap_total_num_files)
        # OPEN: clear in-memory cache from lk.search_targetpixelfile, lk.search_tesscut
        # - no easy way to do it other than clear all, given
        #   it's configured with unlimited cache with no TTL at lightkurve level
        #   (using `memoization` package)
        # - for now leave them alone, as they're rather small


def on_server_loaded(server_context):
    global _FILECACHE_TTL_MINS
    set_log_level_from_env()
    _FILECACHE_TTL_MINS = _get_ttl_mins_from_env()
    log.debug(f"on_server_loaded(): TPF File Cache TTL (min): {_FILECACHE_TTL_MINS}")


def on_session_destroyed(session_context):
    log.debug(f"on_session_destroyed() for '{session_context.id}'")
    _clear_old_cache_entry(_FILECACHE_TTL_MINS)
