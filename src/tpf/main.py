import logging
import os
from functools import cache

from bokeh.document import without_document_lock
from bokeh.events import DocumentReady
from bokeh.layouts import column, row
from bokeh.models import (
    CustomJS,
    Div,
)
from bokeh.plotting import curdoc

from .bokeh_utils import suppress_bokeh_default_reconnect_and_ui
from .tpf_inspect import create_app_body_ui_from_tpf, progressive_plot_catalogs
from .tpf_utils import get_tpf


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__  # type: ignore
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter Notebook or JupyterLab
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal-based IPython
        else:
            return False  # Other shells
    except NameError:
        return False  # Standard Python interpreter or other env


# for matplotlib, use non-interactive backend, to avoid
# UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail
# using the standard tk backend could cause fatal error during GC:
#   ...
#   Exception ignored while calling deallocator <function Image.__del__ at 0x000001AD2D1CFED0>:
#   ...
#   RuntimeError: main thread is not in main loop
#   Tcl_AsyncDelete: async handler deleted by the wrong thread
if not is_notebook():
    import matplotlib

    matplotlib.use("Agg")  # for png export; must be called before importing pyplot
# else case in jupyter notebook, use the backend as-is, generally inline

if os.environ.get("MAST_ENABLE_CLOUD_DATASET", True):
    # Prefer AWS to download MAST products (LCs, TPFs, etc.). Search is still on MAST
    # (requires boto3)
    from astroquery.mast import Observations

    Observations.enable_cloud_dataset()

log = logging.getLogger(__name__)


def set_log_level_from_env():
    from .lk_patch.interact import log as interact_log
    from .tpf_inspect import log as tpf_inspect_log
    from .tpf_utils import log as tpf_utils_log

    # use Python standard string constant in
    #  https://docs.python.org/3/howto/logging.html
    level_str = os.environ.get("TESS_TPF_WEBAPP_LOGLEVEL", None)
    if level_str:
        log.setLevel(level_str)
        interact_log.setLevel(level_str)
        tpf_inspect_log.setLevel(level_str)
        tpf_utils_log.setLevel(level_str)
    return level_str


def set_log_timed_from_env():
    from .lk_patch.timed import log as timed_log

    timed_str = os.environ.get("TESS_TPF_WEBAPP_LOG_TIMED", "false")
    log_timed = timed_str.lower() == "true"
    if log_timed:
        timed_log.setLevel("DEBUG")
    return log_timed


def create_data_cache_dirs_from_env():
    """Create data cache directories specified in the environment if specified.
    The use case is to create the directories on a
    fresh ephemeral disk (without the needed directories)
    in Google Cloud Run environment.

    https://docs.cloud.google.com/run/docs/configuring/services/ephemeral-disk
    """
    # the logic is invoked only when specified in the tess-tpf specific env var,
    # as the generic XDG_CACHE_HOME may be specified for other reasons.
    if "true" != os.environ.get("TESS_TPF_CREATE_CACHE_DIRS", "").lower():
        return None

    cache_basedir = os.environ.get("XDG_CACHE_HOME", None)
    if cache_basedir is None or cache_basedir == "":
        return None

    # lightkurve / astropy behaviors are that if the expected subdir exists,
    # it will be used as the cache dir, otherwise the default dir will be used
    # Hence we need to ensure the expected dirs have been created
    os.makedirs(f"{cache_basedir}/lightkurve", exist_ok=True)  # for TPF FIT files
    os.makedirs(f"{cache_basedir}/astropy", exist_ok=True)  # for astroquery

    return cache_basedir


@cache
def get_build_sha():
    from pathlib import Path

    build_fname = Path(__file__).parent / "build.txt"
    try:
        with build_fname.open() as f:
            return f.readline().strip()
    except FileNotFoundError:
        # in dev mode from source, build.txt is not generated
        return ""
    except Exception as e:
        log.error(f"get_build_sha(): Unexpected error, return empty string. {e}")
        return ""


def get_build_sha_short():
    return get_build_sha()[:8]


def _screenshot_js_codes():
    # the javascript codes that support taking screenshots (using dom-to-image-more library)
    # At UI level, it adds a button at the bottom left of the screen.
    return """
    console.debug('JS codes for screenshots...');
    (function() {
      'use strict';
      console.debug('custom js codes added');
      function saveBlob(blob, filename) {
        // Create a local URL pointing to the memory blob
        const url = URL.createObjectURL(blob);

        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        link.click();

        // Free memory after the download triggers
        URL.revokeObjectURL(url);
        link.remove();
      }

      function createScreenshotBlob(ctr) {
        return domtoimage.toBlob(ctr, {bgcolor: 'white'});
      }

      function saveScreenshot(ctr, filename) {
        return createScreenshotBlob(ctr).then((blob) => {
          saveBlob(blob, filename);
        });
      }

      function copyScreenshotToClipboard(ctr) {
        return createScreenshotBlob(ctr).then((blob) => {
          navigator.clipboard.write([
            new ClipboardItem({
              [blob.type]: blob
            })
          ]).then(() => {
            console.debug('blob copied to clipboard');
          })
        });
      }

      function getMainCtr() {
        // the app's main container
        // MUST NOT return the .shadowRoot. as domtoimage would not work then (complaining that the node is not attached)
        return document.querySelector("body > div > div.bk-Row").shadowRoot.querySelector("div.bk-Column:nth-of-type(2)").shadowRoot.querySelector('div.bk-Column');
      }

      function getSkyViewCtr() {
        return getMainCtr().shadowRoot.querySelector('div.bk-Column:nth-of-type(1)');
      }

      function getTPFInspectCtr() {
        return getMainCtr().shadowRoot.querySelector('div.bk-Column:nth-of-type(3)');
      }

      function getExternalLCCtr() {
        return getMainCtr().shadowRoot.querySelector('div.bk-Column:nth-of-type(4)');
      }

      //
      // The UI elements
      //

      function showTakeScreenshotProgressUI() {
        document.body.insertAdjacentHTML('beforeend', `
    <div id="screenshot-ui-progres-ctr" style="position: fixed;top: 10vh;left: 20vw;padding: 16px 32px;z-index: 9999;background-color: rgba(255, 255, 225, 0.9);border: 1px solid black;border-radius: 12px;">
    Taking Screenshot ...
    </div>
    `);
      }

      function removeTakeScreenshotProgressUI() {
        document.getElementById('screenshot-ui-progres-ctr')?.remove();
      }


      function showSaveScreenShotPopIn() {
        // https://github.com/1904labs/dom-to-image-more
        if (document.getElementById('script#dtim-js') == null) {
          const jsURL = 'https://cdn.jsdelivr.net/npm/dom-to-image-more@3.10.0/dist/dom-to-image-more.min.js';
          const jsEl = document.createElement('script');
          jsEl.id = 'dtim-js';
          jsEl.src = jsURL;
          document.head.appendChild(jsEl);
        }

        const saveSVG = `
<svg xmlns="http://www.w3.org/2000/svg" style="height: 1.2em;width: 1.2em;vertical-align: middle;" viewBox="0 -960 960 960" fill="#1f1f1f">
    <path d="M480-336 288-528l51-51 105 105v-342h72v342l105-105 51 51-192 192ZM263.72-192Q234-192 213-213.15T192-264v-72h72v72h432v-72h72v72q0 29.7-21.16 50.85Q725.68-192 695.96-192H263.72Z"></path>
</svg>`;

        const copySVG = `
<svg xmlns="http://www.w3.org/2000/svg" style="height: 1.2em;width: 1.2em;vertical-align: middle;" viewBox="0 -960 960 960" fill="#1f1f1f">
    <path d="M360-240q-29.7 0-50.85-21.15Q288-282.3 288-312v-480q0-29.7 21.15-50.85Q330.3-864 360-864h384q29.7 0 50.85 21.15Q816-821.7 816-792v480q0 29.7-21.15 50.85Q773.7-240 744-240H360Zm0-72h384v-480H360v480ZM216-96q-29.7 0-50.85-21.15Q144-138.3 144-168v-552h72v552h456v72H216Zm144-216v-480 480Z"/>
</svg>
`;
        document.body.insertAdjacentHTML('beforeend', `
    <div id="screenshot-ui-ctr">
      <style>
        #screenshot-ui-ctr {
          position:fixed;top: 10vh;left: 20vw;z-index:9999;padding: 8px 32px 16px 32px;background-color: rgba(255, 255, 225, 0.9);border: 1px solid #333;border-radius: 12px;
        }
        #screenshot-ui-ctr  ul {
          padding-left: 0; /* remove bullets indentation */
        }
        #screenshot-ui-ctr li {
          list-style-type: none;
        }
    </style>
    <h4>Screenshot</h4>
        <ul>
            <li><input type="radio" name="ss_opts" value="all" checked>All</li>
            <li><input type="radio" name="ss_opts" value="skyview">Skyview</li>
            <li><input type="radio" name="ss_opts" value="tpf">Pixels Inspection</li>
            <li><input type="radio" name="ss_opts" value="extLC">External Lightcurve</li>
        </ul>

      <button id="screenshot-save-ctl">${saveSVG}Save</button>&emsp;
      <button id="screenshot-copy-ctl">${copySVG}Copy</button>&emsp;
      <button id="screenshot-cancel-ctl">Cancel</button>
    </div>
    `);

        function getContainerToTakeScreenshot(ctrStr) {
          if (ctrStr == 'all') {
            return getMainCtr();
          }
          if (ctrStr == 'skyview') {
            return getSkyViewCtr();
          }
          if (ctrStr == 'tpf') {
            return getTPFInspectCtr();
          }
          if (ctrStr == 'extLC') {
            return getExternalLCCtr();
          }
          return null; // should never happen
        }

        document.getElementById('screenshot-save-ctl').onclick = (evt) => {
          const popInCtr = document.getElementById('screenshot-ui-ctr');
          const ctrStr = popInCtr.querySelector('input[name="ss_opts"]:checked').value;
          const ctr = getContainerToTakeScreenshot(ctrStr);
          console.debug('ctr:', ctr);

          const [,tic] = location.search.match(/[?&]tic=([^&]+)/) || [''];
          const [,sector] = location.search.match(/[?&]sector=([^&]*)/) || [''];
          const filename = `ttpi_screenshot_tic${tic}_s${sector}_${ctrStr}.png`;

          document.getElementById('screenshot-ui-ctr').remove();
          showTakeScreenshotProgressUI();
          saveScreenshot(ctr, filename).then(removeTakeScreenshotProgressUI);
        };

        document.getElementById('screenshot-copy-ctl').onclick = (evt) => {
          const popInCtr = document.getElementById('screenshot-ui-ctr');
          const ctrStr = popInCtr.querySelector('input[name="ss_opts"]:checked').value;
          const ctr = getContainerToTakeScreenshot(ctrStr);
          console.debug('ctr:', ctr);

          document.getElementById('screenshot-ui-ctr').remove();
          showTakeScreenshotProgressUI();
          copyScreenshotToClipboard(ctr).then(removeTakeScreenshotProgressUI);
        };

        document.getElementById('screenshot-cancel-ctl').onclick = (evt) => {
          document.getElementById('screenshot-ui-ctr').remove();
        };
      }

      function initSaveScreenShotUI() {
        document.body.insertAdjacentHTML('beforeend', `
     <div style="position: fixed; left: 16px; bottom: 36px; z-index:999;">
       <button id="screenshot-ctl">Screenshot...</button>
     </div>
    `);
        document.getElementById('screenshot-ctl').onclick = showSaveScreenShotPopIn;
      }
      initSaveScreenShotUI();

    })();
"""


def create_search_form(tic, sector, magnitude_limit):
    def to_str(val):
        if val is None:
            return ""
        else:
            return str(val)

    # return a plain html search form, instead of using bokeh widgets
    #
    # HTML search form has the advantage of being completely stateless,
    # not relying on communicating with server (via WebSocket).
    # So for cases such as deploying in serverless environments such as Google Cloud Run,
    # if the server instance has been shutdown due to idle policy,
    # - plain html form would still work, as it will create a new HTTP request.
    # - bokeh widget / WebSocket based form would not work, as it
    #   relies on connecting to the server instance that has been shutdown.

    # put css text into its constant string so that curly braces
    # will not be misinterpreted as f-string substitution
    css_text = """
    <style>
        #search-form-ctr {
            padding-left: 10px;
            padding-right: 16px;
        }
        #search-form-ctr input {
            padding: 4px;
            margin-bottom: 10px;
        }
        #ext-links-ctr {
            margin-top: 1em;
            padding-left: 10px;
            padding-right: 16px;
        }
        #ext-links-ctr li {
            margin-left: -16px;
        }
        footer {
            margin-top: 3em;
            font-size: 90%;
            padding-left: 10px;
        }
        footer details {
            margin-top: 0.75em;
        }
        footer li {
            margin-left: -16px;
        }
    </style>
"""

    search_form_html = f"""
<div id="search-form-ctr">
    <form>
        TIC *<br>
        <input name="tic" value="{to_str(tic)}" accesskey="/"><br>
        Sector<br>
        <input name="sector" value="{to_str(sector)}" placeholder="optional, latest if not specified"><br>
        mag. limit<br>
        <input name="magnitude_limit" value="{to_str(magnitude_limit)}" placeholder="optional, Tmag + 7 if not specified"><br>
        <input type="submit" value="Show">
    </form>
</div>
"""

    # include external links if a TIC is specified
    is_tic_specified = tic is not None and len(str(tic).strip()) > 0

    ext_links_html = ""
    if is_tic_specified:
        ext_links_html = f"""
<div id="ext-links-ctr">
    <p>See also:
        <svg style="height: 1em; width: 1em;" class="svg-inline--fa fa-external-link-alt fa-w-18" aria-hidden="true" data-prefix="fas" data-icon="external-link-alt" role="img" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 576 512" data-fa-i2svg=""><path fill="currentColor" d="M576 24v127.984c0 21.461-25.96 31.98-40.971 16.971l-35.707-35.709-243.523 243.523c-9.373 9.373-24.568 9.373-33.941 0l-22.627-22.627c-9.373-9.373-9.373-24.569 0-33.941L442.756 76.676l-35.703-35.705C391.982 25.9 402.656 0 424.024 0H552c13.255 0 24 10.745 24 24zM407.029 270.794l-16 16A23.999 23.999 0 0 0 384 303.765V448H64V128h264a24.003 24.003 0 0 0 16.97-7.029l16-16C376.089 89.851 365.381 64 344 64H48C21.49 64 0 85.49 0 112v352c0 26.51 21.49 48 48 48h352c26.51 0 48-21.49 48-48V287.764c0-21.382-25.852-32.09-40.971-16.97z"></path></svg>
    </p>
    <ul>
        <li><a href="https://exofop.ipac.caltech.edu/tess/target.php?id={tic}"
               target="_blank">ExoFOP</a></li>
        <li><a href="https://heasarc.gsfc.nasa.gov/wsgi-scripts/TESS/TESS-point_Web_Tool/TESS-point_Web_Tool/wtv_v2.0.py/TICID_result/ticid={tic}"
               target="_blank">Sector Visibility</a></li>
    </ul>
</div>
"""

    footer_html = f"""
<footer>
    Build:
    <a target="_blank" href="https://github.com/orionlee/tess_tpf_webapp/commit/{get_build_sha()}"
        >{get_build_sha_short()}</a><br>
    <a href="https://github.com/orionlee/tess_tpf_webapp" target="_blank">Issues / Sources</a>
    <details>
        <summary>Data sources</summary>
        <ul>
            <li>TESS Pixels <a href="https://archive.stsci.edu/missions-and-data/tess" target="_blank">MAST</a>
                , <a href="https://mast.stsci.edu/tesscut/" target="_blank">TessCut</a></li>
            <li>TIC on <a href="https://vizier.u-strasbg.fr/viz-bin/VizieR?-source=IV/39/tic82" target="_blank">Vizier</a></li>
            <li>Gaia DR3 on <a href="https://cdsarc.cds.unistra.fr/viz-bin/cat/I/355" target="_blank">Vizier</a></li>
            <li><a href="https://irsa.ipac.caltech.edu/Missions/ztf.html" target="_blank">ZTF</a> Archive</li>
            <li>ASAS-SN <a href="http://asas-sn.ifa.hawaii.edu/skypatrol/" target="_blank">Sky Patrol V2</a></li>
            <li>AAVSO <a href="https://www.aavso.org/vsx/" target="_blank">VSX</a></li>
        </ul>
    </details>
</footer>
"""
    return column(
        Div(
            text=f"""
{css_text}
{search_form_html}
{ext_links_html}
{footer_html}
""",
        ),
        name="app_search",
    )


def create_app_ui_container():
    ui_layout = row(
        column(name="app_left"),  # for search form
        column(name="app_main"),
        name="app_ctr",
    )

    return ui_layout


def get_default_catalogs_from_env():
    catalogs_str = os.environ.get("TESS_TPF_WEBAPP_CATALOGS", "")
    if catalogs_str.strip() == "":
        catalogs_str = "skypatrol2,ztf,vsx,gaiadr3_tic"
    return [cat.strip() for cat in catalogs_str.split(",")]


def add_connection_lost_ui(doc):
    # UI to notify users when the websocket connection to the server is lost
    # thus losing all server-side based interactive features

    # https://docs.bokeh.org/en/latest/docs/examples/interaction/js_callbacks/doc_js_events.html

    js_connection_lost = CustomJS(
        code="""
document.body.insertAdjacentHTML("afterbegin", `
<div id="banner_ctr" style="font-size: 1.1rem; padding: 10px; padding-left: 5vw;
    background-color: rgba(255, 0, 0, 0.7); color: white; font-weight: bold;">
Lost the connection to the server. You'd need to reload the page for some interactive functions.
</div>
`);
"""
    )
    doc.js_on_event("connection_lost", js_connection_lost)


def create_in_progress_msg_html(message):
    in_progress_style = """
<style>
@keyframes shimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }
.in-progress-box {
    padding: 16px;
    background: linear-gradient(90deg, #fbfbfb 25%, #ececec 50%, #fbfbfb 75%);
    background-size: 200% 100%;
    animation: shimmer 5s infinite linear;
    border-radius: 6px; border: 1px solid #e1e1e1;
    display: flex; align-items: center; justify-content: center;
    font-weight: bold;
}
</style>
"""
    return f"""
{in_progress_style}
<div class="in-progress-box">{message}</div>
"""


async def search_and_download_tpf_with_ui(tic, sector):
    # returning a (TPF, err_msg) tuple

    if sector is not None:
        msg_label = f"TIC {tic} sector {sector}"
    else:
        msg_label = f"TIC {tic}"

    # log the beginning of search/download TPF to clearly see if it takes a long time
    log.debug(f"Search and download TPF for {msg_label}.")

    try:
        # mark_tpf_accessed=True to facilitate LRU-like file cache cleaning done in app_hooks.py
        tpf, _ = await get_tpf(tic, sector, msg_label, mark_tpf_accessed=True)

        if tpf is None:
            err_msg = f"Cannot find Pixel data for {msg_label}"
            return None, err_msg
    except Exception as e:
        if isinstance(e, IOError):
            # usually some issues in network or MAST server, nothing can be done on our end
            warn_msg = (
                f"IOError (likely intermittent) of type {type(e).__name__} in "
                f"creating Inspector for TIC {tic}, sector {sector}"
            )
            log.warning(warn_msg)
            err_msg = (
                f"Network or MAST Server Error in creating Inspector. {type(e).__name__}: {e}.<br>"
                "Reload the page after a while to see if the issue is resolved."
            )
        else:
            # unexpected errors that might mean bugs on our end.
            log.error(
                f"Error of type {type(e).__name__} in creating Inspector for TIC {tic}, sector {sector}",
                exc_info=True,
            )
            err_msg = f"Error in creating Inspector. {type(e).__name__}: {e}"
        return None, err_msg

    # case tpf has been downloaded successfully
    return tpf, None


def show_app(tic, sector, magnitude_limit=None):

    #
    # 1. First create the skeleton UI
    #
    doc = curdoc()

    ui_ctr = create_app_ui_container()
    ui_left = ui_ctr.select_one({"name": "app_left"})
    ui_left.children = [create_search_form(tic, sector, magnitude_limit)]
    ui_main = ui_ctr.select_one({"name": "app_main"})
    doc.add_root(ui_ctr)

    suppress_bokeh_default_reconnect_and_ui(doc)

    # convert (potential) textual inputs to typed value
    try:
        tic = None if tic is None or tic == "" else int(tic)
        sector = None if sector is None or sector == "" else int(sector)
        magnitude_limit = (
            None
            if magnitude_limit is None or magnitude_limit == ""
            else float(magnitude_limit)
        )
    except Exception as err:
        ui_main.children = [Div(text=f"Invalid Parameter. Error: {err}")]
        return

    # case no TIC, just the search form
    if tic is None:
        ui_main.children = [
            column(
                Div(
                    text="""
<h3>TESS Target Pixels Inspector</h3>
<p>Inspect TESS satellite pixels data for a given target by TIC, e.g., 86263325, 400621146.</p>
"""
                )
            )
        ]
        return

    # case a non-empty TIC, search and download the TPF

    #  Screenshot Javascript logic
    #    the js codes are evaluated only at DocumentReady event.
    #    I can't make it work at other events.
    doc.js_on_event(DocumentReady, CustomJS(code=_screenshot_js_codes()))

    # the UI for monitoring WebSocket connection is only relevant when there is a TIC
    add_connection_lost_ui(doc)

    if sector is None:
        in_progress_msg = f"Search and download TPF for TIC {tic} ..."
    else:
        in_progress_msg = f"Search and download TPF for TIC {tic}, sector {sector} ..."
    in_progress_msg = create_in_progress_msg_html(in_progress_msg)
    ui_main.children = [Div(text=in_progress_msg)]  # to be replaced with actual UI

    #
    # 2. Then, fill  the skeleton UI with the actual UI for the tpf
    #
    # The functions for TPF download / rendering (2 parts)
    # 2a. actually rendering, to be run in main,  with bokeh doc locked
    async def do_render_app_body_in_main(tpf, err_msg):
        # to be run in the main thread that updates the bokeh doc
        if err_msg is not None:
            ui_main.children = [Div(text=err_msg)]
            return

        ui_body, catalog_plot_fns = await create_app_body_ui_from_tpf(
            doc,
            tpf,
            magnitude_limit=None,
            catalogs=get_default_catalogs_from_env(),
        )
        ui_main.children = [ui_body]  # replace the skeleton UI
        progressive_plot_catalogs(doc, catalog_plot_fns)

    # 2b. TPF download to be run in a background, without boekh doc locked;
    #     allowing the above skeleton UI be rendered to the user ASAP
    @without_document_lock
    async def do_create_app_body_ui_in_background():
        # do the search and download TPF in a background thread
        tpf, err_msg = await search_and_download_tpf_with_ui(tic, sector)

        # CRITICAL STEP: Use add_next_tick_callback on the saved 'doc' reference.
        # This pushes data back to the main thread securely without locking the engine.
        doc.add_next_tick_callback(lambda: do_render_app_body_in_main(tpf, err_msg))

    # Fill in the skeleton UI in the background, allowing the
    # initialization logic exits instantly, pushing the skeleton UI to the client.
    # - do the TPF search/download in the background with add_timeout_callback()
    # - update the bokeh doc (the UI) synchronously, with the doc object locked.
    # - the pattern is the same as async plotting of catalog stars on TPF pixels in
    #   lk_patch.interact.async_parse_and_add_catalogs_figure_elements()
    doc.add_timeout_callback(do_create_app_body_ui_in_background, 0)


#
# Webapp entry Point logic
#


def get_arg_as_int(args, arg_name, default_val=None):
    try:
        val = int(args.get(arg_name)[0])
    except:  # noqa: E722
        val = default_val
    return val


def get_arg_as_float(args, arg_name, default_val=None):
    try:
        val = float(args.get(arg_name)[0])
    except:  # noqa: E722
        val = default_val
    return val


if __name__.startswith("bokeh_app_"):  # invoked from `bokeh serve`
    set_log_level_from_env()
    set_log_timed_from_env()
    create_data_cache_dirs_from_env()

    # debug codes to ensure custom MAST timeout is applied
    from astroquery.mast import Observations

    log.debug(f"MAST Timeout: {Observations._portal_api_connection.TIMEOUT}")

    args = curdoc().session_context.request.arguments
    tic = get_arg_as_int(args, "tic", None)  # default value for sample
    sector = get_arg_as_int(args, "sector", None)
    magnitude_limit = get_arg_as_float(args, "magnitude_limit", None)
    # log bokeh session ID to make the log easier to correlate but logs generated by bokeh server
    session_id = curdoc().session_context.id
    log.debug(
        f"Parameters: , {tic}, {sector}, {magnitude_limit} ; {args} . session '{session_id}'"
    )

    curdoc().title = "TESS Target Pixels Inspector"
    show_app(tic, sector, magnitude_limit)
