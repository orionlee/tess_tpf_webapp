import logging
import time

log = logging.getLogger(__name__)


def timed(msg_prefix=None, logger=log):
    """This decorator prints the execution time as DEBUG logging output for the decorated function."""

    def timed_inner(func):
        msg_prefix_final = f"{func.__name__}()" if msg_prefix is None else msg_prefix

        def wrapper(*args, **kwargs):
            return timed_call(func, args, kwargs, msg_prefix=msg_prefix_final, logger=logger)

        return wrapper

    return timed_inner


def timed_call(func, func_args=None, func_kwargs=None, msg_prefix=None, logger=log):
    msg_prefix_final = f"{func.__name__}()" if msg_prefix is None else msg_prefix

    args = func_args if func_args else list()
    kwargs = func_kwargs if func_kwargs else dict()

    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    logger.debug(f"{msg_prefix_final} execution time: {end - start:.2f}s")
    return result
