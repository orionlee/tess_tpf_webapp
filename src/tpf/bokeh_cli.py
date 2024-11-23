import logging

log = logging.getLogger(__name__)


def do_bokeh_serve():
    """A thin wrapper over `bokeh` cli to catch the common `Token is expired.` error.

    In practice, the error seems to happen with some stale browser sessions,
    and is not a cause of concern.
    The wrapper here catches it to reduce the noises in server logs.
    """
    from bokeh.__main__ import main
    from bokeh.protocol.exceptions import ProtocolError

    try:
        main()
    except ProtocolError as pe:
        if "Token is expired" in str(pe):
            log.warning(str(pe))
        else:
            raise pe
    except Exception as e:
        raise e


if __name__ == "__main__":
    do_bokeh_serve()
