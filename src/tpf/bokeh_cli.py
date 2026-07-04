import logging
import os
import sys

log = logging.getLogger(__name__)


def _run_bokeh_cli_with_env_vars():
    from bokeh.command.bootstrap import main as _main

    argv = sys.argv.copy()
    # a. handle $PORT
    if "--port" not in argv:
        port = os.environ.get("PORT", None)
        if port is not None:
            argv.append("--port")
            argv.append(port)
    else:
        # honor the port in the command line
        pass

    # b. extra arguments from $BOKEH_EXTRA_ARGS
    extra_args = [
        s.strip()
        for s in os.environ.get("BOKEH_EXTRA_ARGS", "").split(" ")
        if len(s.strip()) > 0
    ]
    argv = argv + extra_args

    print("Run bokeh with:", *argv)
    _main(argv)


def run_bokeh_cli():
    """A thin wrapper over `bokeh` CLI for :
    1. Handle arguments from env variables PORT (Google Cloud Run convention)
       and BOKEH_EXTRA_ARGS (tess_tpf_webapp specific extension)
       These are needed for cases that bokeh is run from a distroless Docker image.
       An distroless image has no shell, and thus env variables cannot be be supplied
       as command line arguments, e.g., for `bokeh serve --port $PORT`, the PORT
       variable is not evaluated.

    2. Catch the common `Token is expired.` errors and report them as warnings.
       In practice, the error seems to happen with some stale browser sessions,
       and is not a cause of concern.
       The wrapper here catches it to reduce the noises in server logs.
    """
    from bokeh.protocol.exceptions import ProtocolError

    try:
        # 1. Handle bokeh arguments from env var
        _run_bokeh_cli_with_env_vars()
    except ProtocolError as pe:
        # 2. Handle the common `Token is expired.` errors
        if "Token is expired" in str(pe):
            log.warning(str(pe))
        else:
            raise pe
    except Exception as e:
        raise e


if __name__ == "__main__":
    run_bokeh_cli()
