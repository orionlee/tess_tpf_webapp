import logging
import os
import linecache
import tracemalloc


log = logging.getLogger(__name__)

snapshot_old = None

LOG_TOP_MALLOC_LIMIT = 5


def log_resource_info(msg_prefix):
    # log various system resources info,
    # to triage apparent memory leak in GCloud deployment
    import threading

    def get_memory_available():
        # based on: https://stackoverflow.com/a/41125461
        with open("/proc/meminfo", "r") as mem:
            free_memory = 0
            for i in mem:
                sline = i.split()
                # probably needed for older system without MemAvailable
                # if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                if str(sline[0]) == "MemAvailable:":
                    free_memory += int(sline[1])
            return free_memory

    def get_num_active_threads():
        return threading.active_count()

    if os.name != "posix":  # memory usage only works on Linux-like system
        return
    if log.level < logging.DEBUG:  # run only if logging level is at least debug
        return

    log.debug(
        (
            f"[MemLog] {msg_prefix: <26} MemAvailable: {get_memory_available()}"
            f" ; Num. Active threads: {get_num_active_threads()}"
        )
    )
    log_top_malloc(limit=LOG_TOP_MALLOC_LIMIT)


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
