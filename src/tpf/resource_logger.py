import logging
import os
import linecache
import tracemalloc

import psutil

log = logging.getLogger(__name__)

snapshot_old = None

LOG_TOP_MALLOC_LIMIT = 5

LOG_MUMPY_DIFF = True

_cur_process = psutil.Process()
_sys_mem_old = None
_proc_mem_old = None


def get_mem_info_text():
    global _sys_mem_old, _proc_mem_old

    def get_val_in_mb_and_delta(cur, old, attr, label):
        cur_val = getattr(cur, attr) / 1024 / 1024
        if old is None:
            return f"{label}: {cur_val:.0f}"
        old_val = getattr(old, attr) / 1024 / 1024
        delta = cur_val - old_val
        return f"{label}: {cur_val:.0f} ({delta:.1f})"

    sys_mem = psutil.virtual_memory()
    proc_mem = _cur_process.memory_full_info()
    msg = ""

    msg += get_val_in_mb_and_delta(sys_mem, _sys_mem_old, "available", "SysMemAvailable")
    msg += " , "
    msg += get_val_in_mb_and_delta(proc_mem, _proc_mem_old, "rss", "ProcMemRSS")
    msg += " , "
    msg += get_val_in_mb_and_delta(proc_mem, _proc_mem_old, "uss", "ProcMemUSS")
    msg += " , "
    msg += get_val_in_mb_and_delta(proc_mem, _proc_mem_old, "vms", "ProcMemVMS")
    _sys_mem_old = sys_mem
    _proc_mem_old = proc_mem
    return msg


def get_file_info_text():
    p = _cur_process  # shorthand
    if hasattr(p, "num_fds"):
        msg = f"num_fds: {p.num_fds()}"
    elif hasattr(p, "num_handles"):
        msg = f"num_handles: {p.num_handles}"  # Windows
    else:  # should not happen
        msg = ""

    try:
        # to track if TPF FITS files remain open after it's done
        num_open_fits = len([po for po in p.open_files() if ".fits" in po.path])
        msg += f" , num_open_FITS: {num_open_fits}"
    except Exception as e:
        # in gcloud, it intermnittently causes `IndexError: list index out of range``
        # From .../psutil/_pslinux.py", line 2256, in open_files
        # flags = int(f.readline().split()[1], 8)
        msg += f", num_open_FITS: <{type(e).__name__}>"

    return msg


def log_resource_info(msg_prefix):
    # log various system resources info,
    # to triage apparent memory leak in GCloud deployment

    if log.level < logging.DEBUG:  # run only if logging level is at least debug
        return

    log.debug(
        (
            f"[RsrcLog] {msg_prefix: <26} {get_mem_info_text()}"
            f" ; num_threads: {_cur_process.num_threads()}"
            f" ; {get_file_info_text()}"
        )
    )

    log_top_malloc(limit=LOG_TOP_MALLOC_LIMIT)

    if LOG_MUMPY_DIFF is True:
        log_mumppy_diff()


def _do_log_top_malloc(stats, limit):
    # based on https://stackoverflow.com/a/45679009
    # TODO: replace print with log
    print("Top %s lines" % limit)
    for index, stat in enumerate(stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filepath = os.sep.join(frame.filename.split(os.sep)[:-2])
        stat_abbrev = f"{stat}".replace(filepath, "")
        print(f"{index: <2}: {stat_abbrev}")
        # filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        # print("#%s: %s:%s: %.1f KiB"
        #       % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print("    %s" % line)

    other = stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def log_top_malloc(key_type="lineno", limit=5):
    global snapshot_old

    if not tracemalloc.is_tracing():
        return

    snapshot = tracemalloc.take_snapshot()
    snapshot = snapshot.filter_traces(
        (
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<frozen importlib._bootstrap_external>"),
            tracemalloc.Filter(False, "<unknown>"),
            tracemalloc.Filter(False, tracemalloc.__file__),
        )
    )
    if snapshot_old is None:
        top_stats = snapshot.statistics(key_type)
        _do_log_top_malloc(top_stats, limit=limit)
    else:
        stats_diff = snapshot.compare_to(snapshot_old, key_type)
        _do_log_top_malloc(stats_diff, limit=limit)
    snapshot_old = snapshot


def start_tracemalloc():
    # Or set env var PYTHONTRACEMALLOC to 1
    # see: https://docs.python.org/3/library/tracemalloc.html
    tracemalloc.start()


def stop_tracemalloc():
    tracemalloc.stop()


mumpy_tracker = None


def log_mumppy_diff():
    """Generate a summary of memory difference each time it's invoked."""
    # pip install pympler
    # https://pympler.readthedocs.io/en/latest/muppy.html#muppy
    from pympler import tracker

    global mumpy_tracker

    if mumpy_tracker is None:
        mumpy_tracker = tracker.SummaryTracker()
    print("Memory Usage Diff:")
    mumpy_tracker.print_diff()
    print("", flush=True)
