from functools import cache, lru_cache
import logging
import os
import warnings

from astropy.time import Time
import astropy.units as u
import astroquery.vizier as vizier

import numpy as np

import lightkurve as lk
from .lk_patch.interact import (
    show_skyview_widget,
    prepare_lightcurve_datasource,
    make_lightcurve_figure_elements,
    show_interact_widget,
)
from .ext_gaia_tic import ExtendedGaiaDR3TICInteractSkyCatalogProvider
from .tpf_utils import (
    get_tpf,
    is_tesscut,
    has_non_science_pixels,
    cutout_by_range,
    create_mask_for_target,
    create_background_per_pixel_lc,
)
from .lc_utils import read_lc, guess_lc_source
from .resource_logger import log_resource_info

import bokeh
from bokeh.layouts import row, column
from bokeh.models import (
    Button,
    Div,
    TextInput,
    Select,
    CustomJS,
    NumeralTickFormatter,
    ColorBar,
    LinearColorMapper,
    Checkbox,
    Spacer,
    Tooltip,
    InlineStyleSheet,
    HelpButton,
)
from bokeh.models.dom import HTML
from bokeh.plotting import curdoc


LOG_TPF_REFERRERS = True

log = logging.getLogger(__name__)


def set_log_level_from_env():
    from .tpf_utils import log as tpf_utils_log
    from .lk_patch.interact import log as interact_log
    from .resource_logger import log as resource_log

    # use Python standard string constant in
    #  https://docs.python.org/3/howto/logging.html
    level_str = os.environ.get("TESS_TPF_WEBAPP_LOGLEVEL", None)
    if level_str:
        log.setLevel(level_str)
        tpf_utils_log.setLevel(level_str)
        interact_log.setLevel(level_str)
        resource_log.setLevel(level_str)
    return level_str


def set_log_timed_from_env():
    from .lk_patch.timed import log as timed_log

    timed_str = os.environ.get("TESS_TPF_WEBAPP_LOG_TIMED", "false")
    log_timed = timed_str.lower() == "true"
    if log_timed:
        timed_log.setLevel("DEBUG")
    return log_timed


def get_value_in_float(input: TextInput, default=None):
    val = input.value
    try:
        val = float(val) if val != "" else default
    except:  # noqa: E722
        # Non float value, treat it as default
        val = default
    return val


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


@lru_cache(20)  # in a single session, users generally will inspect a handful of LCs at most.
def _do_read_lc(url):
    """A wrapper of `read_lc()` that supports caching of lightcurves, so that
    if an users switches back and forth between a few of them
    in a session, there'd be no need to repeatedly get the data
    remotely again (from ZTF, SkyPatrol, etc.)
    """
    return read_lc(url)


def _replace_or_append_by_name(ui_container, new_model):
    old_model = ui_container.select_one({"name": new_model.name})
    if old_model is not None:
        # https://discourse.bokeh.org/t/clearing-plot-or-removing-all-glyphs/6792/6
        idx = ui_container.children.index(old_model)
        ui_container.children[idx] = new_model
    else:
        ui_container.children.append(new_model)


def make_lc_fig(url, period=None, epoch=None, epoch_format=None, use_cmap_for_folded=False):
    log.info(f"Plot LC: {url}, period={period}, epoch={epoch}, epoch_format={epoch_format}, use_cmap={use_cmap_for_folded}")
    try:
        lc = _do_read_lc(url)

        if period is not None:
            if epoch is not None:
                if epoch_format is None or epoch_format == "btjd":
                    epoch_time = Time(epoch, format="btjd")
                elif epoch_format == "hjd":
                    epoch_time = Time(epoch, format="jd", scale="utc")
                else:
                    raise ValueError(f"Invalid epoch_format: {epoch_format}")
            else:
                epoch_time = None
            lc = lc.fold(period=period, epoch_time=epoch_time, normalize_phase=True)
            lc.label += f", P = {period} d"

        #  hack: cadenceno and quality columns expected by prepare_lightcurve_datasource()
        lc["quality"] = np.zeros_like(lc.flux, dtype=int)
        lc["cadenceno"] = lc["quality"]
        lc_source = prepare_lightcurve_datasource(lc)
        if isinstance(lc, lk.FoldedLightCurve) and use_cmap_for_folded:
            lc_source.data["time_original"] = lc.time_original.value

        def ylim_func(lc):
            # hide extreme outliers, and add margin to top/bottom
            # high cutoff at 99.9%, to optimize for transits / eclipses cases
            #  (so as to preserve dips in considering outliers)
            low, high = np.nanpercentile(lc.flux.value, (1, 99.9))
            margin = 0.10 * (high - low)
            return (low - margin, high + margin)

        fig_lc, vertical_line = make_lightcurve_figure_elements(lc, lc_source, ylim_func=ylim_func)
        fig_lc.name = "lc_fig"
        # Customize the plot
        vertical_line.visible = False
        if isinstance(lc, lk.FoldedLightCurve):
            fig_lc.xaxis.axis_label = "Phase"
        else:
            fig_lc.xaxis.axis_label = f"Time [{lc.time.format.upper()}]"
            # show HJD time in plain numbers, rather than in scientific notation.
            fig_lc.xaxis.formatter = NumeralTickFormatter(format="0,0.00")

        if lc.flux.unit is u.mag:
            fig_lc.yaxis.axis_label = "Mag"
            # flip y-axis as it's in magnitude (smaller value at top)
            ystart, yend = fig_lc.y_range.start, fig_lc.y_range.end
            fig_lc.y_range.start, fig_lc.y_range.end = yend, ystart

        # make the plot scatter like instead of lines
        # hack: assume the renderers are in specific order
        #       can be avoided if the renderers have name when they are created.
        # r_lc_step = [r for r in fig_lc.renderers if r.name == "lc_step"][0]
        r_lc_step = fig_lc.renderers[0]
        r_lc_step.visible = False

        # r_lc_circle = [r for r in fig_lc.renderers if r.name == "lc_circle"][0]
        r_lc_circle = fig_lc.renderers[1]
        r_lc_circle.glyph.fill_color = "gray"
        r_lc_circle.glyph.fill_alpha = 1.0
        r_lc_circle.nonselection_glyph.fill_color = "gray"
        r_lc_circle.nonselection_glyph.fill_alpha = 1.0
        if isinstance(lc, lk.FoldedLightCurve) and use_cmap_for_folded:
            # for phase plot, add color to circles to signify time
            time_cmap = LinearColorMapper(
                palette="Viridis256", low=min(lc_source.data["time_original"]), high=max(lc_source.data["time_original"])
            )
            r_lc_circle.glyph.fill_color = dict(field="time_original", transform=time_cmap)
            r_lc_circle.nonselection_glyph.fill_color = dict(field="time_original", transform=time_cmap)
            color_bar = ColorBar(color_mapper=time_cmap, location=(0, 0), formatter=NumeralTickFormatter(format="0,0"))
            fig_lc.add_layout(color_bar, "right")
            fig_lc.width += 100  # extra horizontal space for the color bar
        elif "phot_filter" in lc.colnames:
            # case ASAS-SN lightcurve, no colormap used, mimic color scheme used by SkyPatrol v2 website
            def _to_color(filter):
                if filter == "V":
                    return "teal"
                elif filter == "g":
                    return "mediumblue"
                else:  # last resort
                    return "gray"

            lc_source.data["fill_color"] = [_to_color(f) for f in lc["phot_filter"]]
            r_lc_circle.glyph.fill_color = "fill_color"
            r_lc_circle.nonselection_glyph.fill_color = "fill_color"

        # enable box_zoom_tool by default
        box_zoom_tools = [t for t in fig_lc.toolbar.tools if isinstance(t, bokeh.models.BoxZoomTool)]
        fig_lc.toolbar.active_drag = box_zoom_tools[0] if len(box_zoom_tools) > 0 else "auto"

        return fig_lc
    except Exception as e:
        if isinstance(e, IOError):
            # usually some issues in network or ZTF server, nothing can be done on our end
            log.warning(f"IOError (likely intermittent) of type {type(e).__name__} in loading ZTF lc: {url}")
        else:
            # other unexpected errors that might mean bugs on our end.
            log.error(f"Error of type {type(e).__name__} in loading lc: {url}", exc_info=True)
        # traceback.print_exc()  # for server side debug
        err_msg = f"Error in loading lightcurve. {type(e).__name__}: {e}"
        if guess_lc_source(url) is None:
            # helpful in cases users copy-and-paste a wrong link
            err_msg += (
                "<br>The URL does not seem to be an ASAS-SN SkyPatrol v2 URL "
                "(starts with <code>http://asas-sn.ifa.hawaii.edu/skypatrol/objects/</code>), "
                "<br>or a valid ZTF lightcurve URL "
                "(starts with <code>https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves</code>)."
            )
        return Div(text=err_msg, name="lc_fig")


def create_lc_viewer_ui():
    in_url = TextInput(
        width=600,
        placeholder=(
            "ZTF Lightcurve CSV URL (the LC link to the right of ZTF OID), or ASAS-SN SkyPatrol v2 URL (the SkyPatrol v2 link)"
        ),
        # value="https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID=660106400019009&COLLECTION=ztf_dr21&FORMAT=csv",  # noqa: E501  # TST
    )

    in_period = TextInput(
        width=100,
        placeholder="optional",
        # value="1.651",  # TST
    )

    in_epoch = TextInput(width=120, placeholder="optional")  # slightly wider, to (long) accommodate JD value

    in_epoch_format = Select(options=[("btjd", "BTJD"), ("hjd", "HJD")], value="btjd")

    btn_period_half = Button(label="1/2 P")
    btn_period_double = Button(label="2x P")

    in_use_cmap_for_folded = Checkbox(label="Use color map to show time in phase plot", active=False)
    btn_plot = Button(label="Plot", button_type="primary")

    ui_layout = column(
        Div(text="<hr>"),  # a spacer
        Div(text="<h3>Lightcurve<span style='font-weight: normal; font-size: 90%;'> from ZTF or SkyPatrol v2</span></h3>"),
        row(Div(text="URL *"), in_url),
        row(
            Div(text="Period (d)"),
            in_period,
            Div(text="epoch"),
            in_epoch,
            in_epoch_format,
            btn_period_half,
            btn_period_double,
        ),
        row(btn_plot, in_use_cmap_for_folded),
        name="lc_viewer_ctl_ctr",
    )

    # add interactivity
    def add_lc_fig():
        url = in_url.value

        period = get_value_in_float(in_period, None)
        if period is not None and period <= 0:
            # ignore non-positive period
            # so users could switch between folded and non-folded LC
            # by adding a minus sign to the period, without completely losing the value.
            period = None

        epoch = get_value_in_float(in_epoch, None)
        epoch_format = in_epoch_format.value

        use_cmap_for_folded = in_use_cmap_for_folded.active
        fig = make_lc_fig(url, period, epoch, epoch_format, use_cmap_for_folded)

        # add the plot (replacing existing plot, if any)
        _replace_or_append_by_name(ui_layout, fig)

    def add_lc_fig_with_msg():
        # immediately show a message, as the actual plotting would take time
        msg_ui = Div(text="Plotting...", name="lc_fig")
        _replace_or_append_by_name(ui_layout, msg_ui)
        curdoc().add_next_tick_callback(add_lc_fig)

    btn_plot.on_click(add_lc_fig_with_msg)

    def do_change_period(factor):
        try:
            period = float(in_period.value)
        except Exception:
            # No-op for case no period (empty string), or somehow an invalid number is used
            return

        # change the period and  re-plot
        in_period.value = str(period * factor)
        add_lc_fig_with_msg()

    btn_period_half.on_click(lambda: do_change_period(0.5))
    btn_period_double.on_click(lambda: do_change_period(2))

    return ui_layout


def show_tpf_orientation_html(tpf):
    """ "Helper to visualize the TPF's orientation in the sky.
    Long arm is north, short arm with arrow is east.
    """
    coord_bottom_left = tpf.wcs.pixel_to_world(0, 0)
    #     coord_upper_right = tpf.wcs.pixel_to_world(tpf.shape[2] - 1, tpf.shape[1] - 1)
    coord_upper_left = tpf.wcs.pixel_to_world(0, tpf.shape[2] - 1)
    deg_from_north = coord_bottom_left.position_angle(coord_upper_left).to(u.deg).value

    return f"""
<div style="position: relative; margin-left: 16px;height: 64px;">
    <div title="Long arm: North; Short arm with arrow: East"
         style="display: inline-block; max-width: 64px;font-size: 32px;margin: 16px;\
transform: rotate({-deg_from_north}deg);transform-origin: left; cursor:pointer;">↳</div>
    <div style="display: inline-block;">Orientation<br>Long arm: north; short arm: east.</div>
</div>"""


def create_skyview_metadata_ui(tpf, ztf_search_radius, ztf_ngoodobsrel_min, skypatrol2_search_radius):
    if tpf is None:
        return Div(name="skyview_metadata", text="")

    tpf_author_str = "TessCut" if is_tesscut(tpf) else "SPOC"
    cur_time_relative = tpf.time[0].value - tpf.meta.get("TSTART", np.nan)
    open_attr, unreliable_pixels_warn_msg = "", ""
    if has_non_science_pixels(tpf):
        open_attr = "open"  # to ensure the warnings can be spotted without user opening the details
        unreliable_pixels_warn_msg = """
<span style="background-color: yellow; padding-left: 4px; padding-right: 4px;">Warning:</span>
Some of the pixels are not science pixels.
See section 4.1.3 in
<a href="https://archive.stsci.edu/missions/tess/doc/TESS_Instrument_Handbook_v0.1.pdf"
   target="_blank">TESS Instrument Handbook</a>,
<br>
or Collateral (Non-science) Pixels section in
<a href="https://heasarc.gsfc.nasa.gov/docs/tess/data-products.html#collapseCollateral"
   target="_blank">TESS Data Products Information at GSFC</a>.
<br>
"""
    # extra margin-top below is a hack. Otherwise, the UI will bleed into skyview widget above it.
    return Div(
        name="skyview_metadata",
        text=f"""
<div id="skyview_metadata_ctr" style="margin-top: 30px;">
    <details {open_attr}>
        <summary>Pixels source: {tpf_author_str}</summary>
        Brightness at {tpf.time.format.upper()} {tpf.time[0].value:.2f} ({cur_time_relative:.2f} d in the sector)<br>
        {unreliable_pixels_warn_msg}
        {show_tpf_orientation_html(tpf)}
        <br>
        ZTF search radius: {ztf_search_radius}, min. # good observations: {ztf_ngoodobsrel_min} ;
        ASAS-SN SkyPatrol v2 search radius: {skypatrol2_search_radius}
        <br>
    </details>
</div>
""",
    )


def export_plt_fig_as_data_uri(fig, close_fig=True):
    import base64
    from io import BytesIO
    import matplotlib.pyplot as plt

    try:
        buf = BytesIO()  # a temporary buffer
        fig.savefig(buf, format="png")
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        uri = f"data:image/png;base64,{data}"
        return uri
    finally:
        # to avoid memory leaks,
        # as figures created by pyplot are kept in memory by default
        # see: https://matplotlib.org/stable/gallery/user_interfaces/web_application_server_sgskip.html
        if close_fig:
            # https://stackoverflow.com/a/49748374
            fig.clear()  # fig.clf()
            plt.close(fig)


def create_tpf_interact_ui(tpf, fig_tpf_skyview=None, hide_save_to_local_btn=True):
    btn_inspect = Button(label="Inspect", button_type="primary")
    btn_help_inspect = HelpButton(
        tooltip=Tooltip(
            content=HTML(
                """
Construct a lightcurve from selected pixels.<br>
Shift-Click to add to the selections. Ctrl-Shift-Click to remove from the selections.
"""
            ),
            position="right",
        ),
        button_type="light",  # make help UI standout less
        stylesheets=[
            InlineStyleSheet(  # make help UI to be closer to the inspect button
                css="button { margin-left: -6px !important; padding-left: 0px !important; }"
            )
        ],
    )

    in_flux_normalized = Checkbox(label="normalized", active=True)
    # optional for Tesscut, an approximate background subtracted for quick look
    in_bkg_subtraction = Checkbox(label="background subtracted", visible=is_tesscut(tpf), active=False)
    in_ymin = TextInput(width=100, placeholder="optional")
    in_ymax = TextInput(width=100, placeholder="optional")

    centroid_link_html = f"""\
&emsp;<a href="https://transit-vetting.streamlit.app/?tic={tpf.meta.get("TICID")}&sec={tpf.meta.get("SECTOR")}"
   style="font-size: 120%;" title="Webapp to identify centroids" target="_blank">centroid analysis</a>
"""
    ui_layout = column(
        Div(text="<hr>"),  # a spacer
        Div(text="<h3>Pixels Inspection</h3>"),
        row(
            btn_inspect,
            btn_help_inspect,
            Spacer(width=10),
            in_flux_normalized,
            in_bkg_subtraction,
            Spacer(width=20),
            Div(text="y min."),
            in_ymin,
            Div(text="y max."),
            in_ymax,
            Div(text=centroid_link_html),
        ),
        name="tpf_interact_ctr",
    )

    # add interactivity
    async def add_tpf_interact_fig():

        log.info(f"Plot tpf interact: {tpf}, sector={tpf.meta.get('SECTOR')}, TessCut={is_tesscut(tpf)}")

        # provide background LC via a memoized function so that
        # 1. it won't be unnecessarily created, and 2. if it's needed, it'll be created once only.
        @cache
        def get_bkg_per_pixel_lc():
            """Helper for a rough background subtraction, used in TessCut TPFs."""
            return create_background_per_pixel_lc(tpf)

        # flux: either normalized or raw e-/s
        def transform_func(lc):
            if in_bkg_subtraction.active:
                lc = lc - get_bkg_per_pixel_lc() * lc.meta["APERTURE_MASK"].sum()
            return lc.normalize() if in_flux_normalized.active else lc

        # a reference to the Lc plot to be created below, used by ylim_func()
        fig_lc = None

        def ylim_func(lc):
            # note: if `fig_lc` has been created,
            # use it to restrict the x-range of the LC
            # so that when one zooms in, it'll automatically scale
            # the y-range to the date range specified,
            # avoiding cases that extreme outliers (e.g.,) scattered light
            # continue to affect the default y-range once zoomed in
            if fig_lc is not None:
                lc_trunc = lc.truncate(fig_lc.x_range.start, fig_lc.x_range.end)
                if len(lc_trunc) > 0:
                    lc = lc_trunc

            # default ymin / ymax based on the range of flux in the LC (possibly zoomed in),
            # padding with some extra spacing
            ymin = np.nanmin(lc.flux).value
            ymax = np.nanmax(lc.flux).value
            ymax += np.abs(ymax - ymin) * 0.1
            ymin -= np.abs(ymax - ymin) * 0.1

            # override ymin / ymax if user specifies explicit values
            ymin = get_value_in_float(in_ymin, ymin)
            ymax = get_value_in_float(in_ymax, ymax)

            unit = lc.flux.unit
            return (ymin * unit, ymax * unit)

        # for TessCut, set the initial aperture mask to be the single pixel where the target is located
        # the behavior would be more consistent than using the threshold mask,
        # which might not be referring to the target at all.
        aperture_mask = create_mask_for_target(tpf) if is_tesscut(tpf) else tpf.pipeline_mask

        # ^^^ for transform_func() and ylim_func()
        #     read users input during execution (inside function body)
        #     so that if users changes the value, they will be
        #     reflected when users changes the LC by selecting different pixels
        #     (without the need to click inspect button again)

        # a more friendly fits filename than the default, useful in notebook env
        exported_filename = (
            f'tess-tic{tpf.meta.get("TICID")}-s{tpf.meta.get("SECTOR", 0):04}'
            f"-{tpf.flux.shape[1]}x{tpf.flux.shape[2]}_astrocut-custom-lc.fits"
        )
        create_tpf_interact_ui_func, interact_mask = show_interact_widget(
            tpf,
            ylim_func=ylim_func,
            transform_func=transform_func,
            aperture_mask=aperture_mask,
            return_type="doc_init_fn",
            also_return_selection_mask=True,
            exported_filename=exported_filename,
        )

        ui_body = await create_tpf_interact_ui_func()
        ui_body.name = "tpf_interact_fig"

        # enable box_zoom_tool by default
        try:
            # fig_lc is also used by ylim_func above
            fig_lc = ui_body.select_one({"name": "fig_lc"})
            box_zoom_tools = [t for t in fig_lc.toolbar.tools if isinstance(t, bokeh.models.BoxZoomTool)]
            fig_lc.toolbar.active_drag = box_zoom_tools[0] if len(box_zoom_tools) > 0 else "auto"
        except Exception as e:
            warnings.warn(f"Failed to enable box zoom as default for tpf.interact() LC viewer. {e}")

        # use the x/y zoom range of the SkyView TPF as the default, if available
        if fig_tpf_skyview is not None:
            fig_tpf = ui_body.select_one({"name": "fig_tpf"})
            fig_tpf.x_range = fig_tpf_skyview.x_range.clone()
            fig_tpf.y_range = fig_tpf_skyview.y_range.clone()

        # hide "Save lightcurve" button (not applicable in Web UI)
        # OPEN: we might present a new button to let users download the LC from the webapp
        if hide_save_to_local_btn:
            btn_save_lc = ui_body.select_one({"label": "Save Lightcurve"})
            if btn_save_lc is not None:
                btn_save_lc.visible = False

        # add UI and functions for per-pixel plot
        btn_plot_per_pixels = Button(label="Per-Pixel Plot", button_type="success")  # "success" to signify secondary
        out_plot_per_pixels = Div(text="", name="per_pixel_plot_fig")

        ui_body.children.append(row(btn_plot_per_pixels, out_plot_per_pixels))

        def plot_per_pixels():
            import matplotlib.pyplot as plt

            # cut tpf to the time range in `interact` figure
            xstart, xend = fig_lc.x_range.start, fig_lc.x_range.end
            log.info(
                f"Plot per pixels: {tpf}, sector={tpf.meta.get('SECTOR')}, TessCut={is_tesscut(tpf)}"
                f", time_range={xend - xstart:.1f}d"
            )

            tpf_trunc = tpf
            tpf_trunc = tpf_trunc[tpf_trunc.time.value >= xstart]
            tpf_trunc = tpf_trunc[tpf_trunc.time.value <= xend]

            # cut tpf to the pixel range (columns/rows) in `interact` figure
            fig_tpf = ui_body.select_one({"name": "fig_tpf"})
            row_range = round(fig_tpf.y_range.start - (tpf.row - 0.5)), round(fig_tpf.y_range.end - (tpf.row - 0.5))
            col_range = round(fig_tpf.x_range.start - (tpf.column - 0.5)), round(fig_tpf.x_range.end - (tpf.column - 0.5))
            tpf_trunc, aperture_mask_trunc = cutout_by_range(tpf_trunc, interact_mask.copy(), col_range, row_range)

            @cache
            def get_bkg_per_pixel_lc_trunc():
                return get_bkg_per_pixel_lc().truncate(xstart, xend)

            def corrector_func(lc):
                if in_bkg_subtraction.active:
                    return lc - get_bkg_per_pixel_lc_trunc()
                else:
                    return lc

            pixel_size_inches = 0.6
            # make each pixel larger if the cutout is zoomed-in
            if tpf_trunc.flux[0].shape[0] < 6 or tpf_trunc.flux[0].shape[1] < 6:
                pixel_size_inches = 1.5
            elif tpf_trunc.flux[0].shape[0] < 8 or tpf_trunc.flux[0].shape[1] < 8:
                pixel_size_inches = 1.1

            # scale marker size based on time range
            # (a proxy of number of dots to show)
            def get_default_marker_size_for_pixels_plot(tpf, pixel_size_inches):
                plot_duration = (tpf.time.max() - tpf.time.min()).value
                scale = 2.0 / plot_duration
                return pixel_size_inches * scale

            markersize = round(get_default_marker_size_for_pixels_plot(tpf_trunc, pixel_size_inches), 1)
            if markersize < 0.05:
                markersize = 0.05

            shape = tpf_trunc.flux[0].shape
            fig = plt.figure(figsize=(shape[1] * pixel_size_inches, shape[0] * pixel_size_inches))
            try:
                ax = tpf_trunc.plot_pixels(
                    ax=fig.gca(),
                    aperture_mask=aperture_mask_trunc,
                    corrector_func=corrector_func,
                    show_flux=True,
                    markersize=markersize,
                    # OPEN: add corrector_func to obey ylim_func?!
                )
                ax.set_title(
                    (
                        f"TIC {tpf_trunc.meta.get('TICID')}, "
                        f"Sector {tpf_trunc.meta.get('SECTOR')}, "
                        f"Camera {tpf_trunc.meta.get('CAMERA')}.{tpf_trunc.meta.get('CCD')}, "
                        f"{tpf_trunc.time.min().value:.2f} - {tpf_trunc.time.max().value:.2f} [{tpf_trunc.time.format.upper()}] "
                    ),
                    fontsize=12,
                )

                img_html = f'<img src="{export_plt_fig_as_data_uri(fig)}" />'
                out_plot_per_pixels.text = img_html
            finally:
                # release figure memory (last resort).
                # Normally it should have been already been done in export_plt_fig_as_data_uri
                plt.close()

        def plot_per_pixels_with_msg():
            msg = "Creating per-pixel plot..."
            xduration = fig_lc.x_range.end - fig_lc.x_range.start
            if xduration > 5:
                msg += f"""
<br> <span style="background-color: yellow;">Note:</span> Plotting over a long time range of {xduration:.1f} days.
Consider to shorten the range.
<br> A plot over a long time range is less legible, and it takes longer to create one.
"""
            out_plot_per_pixels.text = msg
            curdoc().add_next_tick_callback(plot_per_pixels)

        btn_plot_per_pixels.on_click(plot_per_pixels_with_msg)

        # interact() plot done
        # add the plot (replacing existing plot, if any)
        _replace_or_append_by_name(ui_layout, ui_body)

    def add_tpf_interact_fig_with_msg():
        # immediately show a message, as the actual plotting would take time
        msg_ui = Div(text="Plotting...", name="tpf_interact_fig")
        _replace_or_append_by_name(ui_layout, msg_ui)
        curdoc().add_next_tick_callback(add_tpf_interact_fig)

    btn_inspect.on_click(add_tpf_interact_fig_with_msg)

    return ui_layout


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
    return column(
        Div(
            text=f"""
{css_text}
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


async def create_app_body_ui(doc, tic, sector, magnitude_limit=None):
    # if True:  # for dev purpose only
    #     return column(create_lc_viewer_ui())

    # convert (potential) textual inputs to typed value
    try:
        tic = None if tic is None or tic == "" else int(tic)
        sector = None if sector is None or sector == "" else int(sector)
        magnitude_limit = None if magnitude_limit is None or magnitude_limit == "" else float(magnitude_limit)
    except Exception as err:
        return Div(text=f"<h3>Skyview</h3> Invalid Parameter. Error: {err}", name="skyview"), None

    if tic is None:
        return (
            column(
                Div(text="<h3>Skyview</h3>", name="skyview"),
                Div(text="<h3>Pixels Inspection</h3>", name="tpf_interact_ctr"),
                Div(text="<h3>Lightcurve</h3>", name="lc_viewer"),
            ),
            None,
        )

    if sector is not None:
        msg_label = f"TIC {tic} sector {sector}"
    else:
        msg_label = f"TIC {tic}"

    # log the beginning of search/download TPF to clearly see if it takes a long time
    log.debug(f"Search and download TPF for TIC {tic}, sector {sector}.")
    tpf, sr = await get_tpf(tic, sector, msg_label)

    if tpf is None:
        log.debug(f"Cannot find TPF or TESSCut for {msg_label}. No plot to be made.")
        return Div(text=f"<h3>SkyView</h3> Cannot find Pixel data for {msg_label}", name="skyview"), None

    # set at info level, as it might be useful to gather statistics on the type of tpfs being plotted ()
    log.info(f"Plot Skyview: {tpf}, sector={tpf.meta.get('SECTOR')}, exptime={sr.exptime[-1]}, TessCut={is_tesscut(tpf)}")
    if LOG_TPF_REFERRERS:
        doc.tpf = tpf  # keep tpf in UI session to test if it's kept beyond the session
    return await create_app_body_ui_from_tpf(doc, tpf, magnitude_limit=magnitude_limit)


def get_default_catalogs_from_env():
    catalogs_str = os.environ.get("TESS_TPF_WEBAPP_CATALOGS", "")
    if catalogs_str.strip() == "":
        catalogs_str = "gaiadr3_tic,skypatrol2,ztf,vsx"
    return [cat.strip() for cat in catalogs_str.split(",")]


async def create_app_body_ui_from_tpf(doc, tpf, magnitude_limit=None, catalogs=None):
    if magnitude_limit is None:
        # supply default
        magnitude_limit = tpf.meta.get("TESSMAG", 0)
        if magnitude_limit == 0:
            # handle case TESSMAG header is missing, or is explicitly 0 (from TessCut)
            magnitude_limit = 18
        else:
            magnitude_limit += 7

    # truncate the TPF to avoid showing the pixel brightness
    # at the beginning of observation,
    # as it's often not representative
    # (due to scatter light at the beginning of a sector)
    # Show brightness around day 3 (arbitrarily) instead.
    if tpf.time.max() - tpf.time.min() > 3 * u.day:
        tpf_skyview = tpf[tpf.time > tpf.time.min() + 3 * u.day]
    else:
        tpf_skyview = tpf

    # a catalog can be disabled by excluding it in the catalogs param
    # OPEN: consider to make the default configurable from an environment variable
    # so that a catalog can be disabled after deployment
    if catalogs is None:
        catalogs = get_default_catalogs_from_env()

    # the catalog strings are mapped to
    # the catalog name / class and parameters

    vizier_server = vizier.conf.server
    ztf_search_radius = 90 * u.arcsec
    ztf_ngoodobsrel_min = 200
    skypatrol2_search_radius = 90 * u.arcsec

    catalogs_with_params_map = dict(
        gaiadr3_tic=(
            ExtendedGaiaDR3TICInteractSkyCatalogProvider,
            dict(
                extra_cols_in_detail_view={
                    "BP-RP": "BP-RP",
                    "RUWE": "RUWE",
                    "sepsi": "sepsi",
                    "e_RV": "e_RV (km/s)",
                    "IPDfmp": "IPDfmp",
                },
                url_templates=dict(
                    # custom columns
                    gaiadr3_main_url=f"https://{vizier_server}/viz-bin/VizieR-4?-ref=VIZ6578bb1b54eda&-to=-4b&-from=-4&-this=-4&%2F%2Fsource=I%2F355%2Fgaiadr3&%2F%2Ftables=I%2F355%2Fgaiadr3&%2F%2Ftables=I%2F355%2Fparamp&-out.max=50&%2F%2FCDSportal=http%3A%2F%2Fcdsportal.u-strasbg.fr%2FStoreVizierData.html&-out.form=HTML+Table&%2F%2Foutaddvalue=default&-order=I&-oc.form=sexa&-out.src=I%2F355%2Fgaiadr3%2CI%2F355%2Fparamp&-nav=cat%3AI%2F355%26tab%3A%7BI%2F355%2Fgaiadr3%7D%26tab%3A%7BI%2F355%2Fparamp%7D%26key%3Asource%3DI%2F355%2Fgaiadr3%26HTTPPRM%3A&-c=&-c.eq=J2000&-c.r=++2&-c.u=arcmin&-c.geom=r&-source=&-x.rs=10&-source=I%2F355%2Fgaiadr3+I%2F355%2Fparamp&-out.orig=standard&-out=RA_ICRS&-out=DE_ICRS&-out=Source&Source=%s&-out=Plx&-out=PM&-out=pmRA&-out=pmDE&-out=sepsi&-out=IPDfmp&-out=RUWE&-out=Dup&-out=Gmag&-out=BPmag&-out=RPmag&-out=BP-RP&-out=RV&-out=e_RV&-out=VarFlag&-out=NSS&-out=XPcont&-out=XPsamp&-out=RVS&-out=EpochPh&-out=EpochRV&-out=MCMCGSP&-out=MCMCMSC&-out=Teff&-out=logg&-out=%5BFe%2FH%5D&-out=Dist&-out=A0&-out=HIP&-out=PS1&-out=SDSS13&-out=SKYM2&-out=TYC2&-out=URAT1&-out=AllWISE&-out=APASS9&-out=GSC23&-out=RAVE5&-out=2MASS&-out=RAVE6&-out=RAJ2000&-out=DEJ2000&-out=Pstar&-out=PWD&-out=Pbin&-out=ABP&-out=ARP&-out=GMAG&-out=Rad&-out=SpType-ELS&-out=Rad-Flame&-out=Lum-Flame&-out=Mass-Flame&-out=Age-Flame&-out=Flags-Flame&-out=Evol&-out=z-Flame&-meta.ucd=0&-meta=0&-usenav=1&-bmark=GET",  # noqa: E501
                    # include classification score, EB parameters
                    gaiadr3_var_url=f"https://{vizier_server}/viz-bin/VizieR-4?-ref=VIZ65ac1f481b91d6&-to=-4b&-from=-3&-this=-4&%2F%2Fsource=%2BI%2F358%2Fvarisum%2BI%2F358%2Fvclassre%2BI%2F358%2Fveb%2BI%2F358%2Fvcc%2BI%2F358%2Fvst&%2F%2Fc=06%3A59%3A36.3+%2B23%3A28%3A51.14&%2F%2Ftables=I%2F358%2Fvarisum&%2F%2Ftables=I%2F358%2Fvclassre&%2F%2Ftables=I%2F358%2Fvcc&%2F%2Ftables=I%2F358%2Fveb&%2F%2Ftables=I%2F358%2Fvst&-out.max=50&%2F%2FCDSportal=http%3A%2F%2Fcdsportal.u-strasbg.fr%2FStoreVizierData.html&-out.form=HTML+Table&-out.add=_r&%2F%2Foutaddvalue=default&-sort=_r&-order=I&-oc.form=sexa&-out.src=I%2F358%2Fvarisum%2CI%2F358%2Fvclassre%2CI%2F358%2Fveb%2CI%2F358%2Fvcc%2CI%2F358%2Fvst&-nav=cat%3AI%2F358%26tab%3A%7BI%2F358%2Fvarisum%7D%26tab%3A%7BI%2F358%2Fvclassre%7D%26tab%3A%7BI%2F358%2Fvcc%7D%26tab%3A%7BI%2F358%2Fveb%7D%26tab%3A%7BI%2F358%2Fvst%7D%26key%3Asource%3D%2BI%2F358%2Fvarisum%2BI%2F358%2Fvclassre%2BI%2F358%2Fveb%2BI%2F358%2Fvcc%2BI%2F358%2Fvst%26key%3Ac%3D06%3A59%3A36.3+%2B23%3A28%3A51.14%26pos%3A06%3A59%3A36.3+%2B23%3A28%3A51.14%28+60+arcsec%29%26HTTPPRM%3A&-c=&-c.eq=J2000&-c.r=+60&-c.u=arcsec&-c.geom=r&-source=&-x.rs=10&-source=I%2F358%2Fvarisum+I%2F358%2Fvclassre+I%2F358%2Fveb+I%2F358%2Fvcc+I%2F358%2Fvst&-out.orig=standard&-out=Source&Source=%s&-out=RA_ICRS&-out=DE_ICRS&-out=TimeG&-out=DurG&-out=Gmagmean&-out=TimeBP&-out=DurBP&-out=BPmagmean&-out=TimeRP&-out=DurRP&-out=RPmagmean&-out=VCR&-out=VRRLyr&-out=VCep&-out=VPN&-out=VST&-out=VLPV&-out=VEB&-out=VRM&-out=VMSO&-out=VAGN&-out=Vmicro&-out=VCC&-out=SolID&-out=Classifier&-out=Class&-out=ClassSc&-out=Rank&-out=TimeRef&-out=Freq&-out=magModRef&-out=PhaseGauss1&-out=sigPhaseGauss1&-out=DepthGauss1&-out=PhaseGauss2&-out=sigPhaseGauss2&-out=DepthGauss2&-out=AmpCHP&-out=PhaseCHP&-out=ModelType&-out=Nparam&-out=rchi2&-out=PhaseE1&-out=DurE1&-out=DepthE1&-out=PhaseE2&-out=DurE2&-out=DepthE2&-out=Per&-out=T0G&-out=T0BP&-out=T0RP&-out=HG0&-out=HG1&-out=HG2&-out=HG3&-out=HG4&-out=HG5&-out=HBP0&-out=HBP1&-out=HBP2&-out=HBP3&-out=HBP4&-out=HBP5&-out=HRP0&-out=HRP1&-out=HRP2&-out=HRP3&-out=HRP4&-out=HRP5&-out=Gmodmean&-out=BPmodmean&-out=RPmodmean&-out=Mratiomin&-out=alpha&-out=Ampl&-out=NfoVTrans&-out=FoVAbbemean&-out=NTimeScale&-out=TimeScale&-out=Variogram&-meta.ucd=2&-meta=1&-meta.foot=1&-usenav=1&-bmark=GET",  # noqa: E501
                ),
                target_tic=tpf.meta.get("TICID", None),
            ),
        ),
        skypatrol2=("skypatrol2", dict(radius=skypatrol2_search_radius)),
        ztf=("ztf", dict(radius=ztf_search_radius, ngoodobsrel_min=ztf_ngoodobsrel_min)),
        vsx="vsx",
    )

    catalogs_with_params = [catalogs_with_params_map[k] for k in catalogs]

    with warnings.catch_warnings():
        # Ignore warnings about no PM correction (happened to TessCut data)
        warnings.filterwarnings("ignore", message="Proper motion correction cannot", category=lk.LightkurveWarning)

        create_skyview_ui = show_skyview_widget(
            tpf_skyview,
            aperture_mask=tpf.pipeline_mask,
            magnitude_limit=magnitude_limit,
            catalogs=catalogs_with_params,
            return_type="doc_init_fn",
        )

        skyview_ui, catalog_plot_fns = await create_skyview_ui(doc)
        # return the UI, and the catalog_plot_fns (for progressive plot of the catalog data asynchronously)
        return (
            column(
                skyview_ui,
                create_skyview_metadata_ui(
                    tpf,
                    ztf_search_radius=ztf_search_radius,
                    ztf_ngoodobsrel_min=ztf_ngoodobsrel_min,
                    skypatrol2_search_radius=skypatrol2_search_radius,
                ),
                create_tpf_interact_ui(tpf, fig_tpf_skyview=skyview_ui.select_one({"name": "fig_tpf_skyview"})),
                create_lc_viewer_ui(),
                # the name is used to signify an interactive UI is returned
                # (as opposed to the UI with a dummy UI or error message in the boundary conditions)
                name="app_body_interactive",
            ),
            catalog_plot_fns,
        )


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


def _progressive_plot_catalogs(doc, catalog_plot_fns):
    # Use timeout to make the catalog data plot shown progressively
    # by running them in the background.
    if catalog_plot_fns is None:
        return
    for fn in catalog_plot_fns:
        doc.add_timeout_callback(fn, 0)


def log_tpf_referrers_on_session_destroyed(session_context):
    import datetime
    import gc

    def print_a_ref(ref, expected_referrer):
        if ref is expected_referrer:
            print("[Expected]", type(ref))
        elif isinstance(ref, dict) and ref.get("__name__") is not None:
            print(f"dict[__name=={ref['__name__']}] ({len(ref)})")
        elif isinstance(ref, dict) and len(ref) > 5:
            print(f"dict ({len(ref)}) w/keys: {ref.keys()}")
        elif isinstance(ref, dict):
            print(ref)
        else:
            print(type(ref))
            # print(ref)

    def show_referrers(obj, expected_referrer, label):
        refs = gc.get_referrers(obj)
        print(f"Referrers of {label} ({len(refs)}):")
        for r in refs:
            print_a_ref(r, expected_referrer)

    # main logic
    print(  # somehow log object cannot not be found
        f"DBG {datetime.datetime.now()} in on_session_destroyed(). "
        f"Num. bokeh sessions: {len(session_context.server_context.sessions)}"
    )
    # for a in dir(session_context):
    #     print("    ", a)
    # print("._document: ", session_context._document)
    if hasattr(session_context._document, "tpf"):
        print(f"DBG2: tpf ({session_context._document.tpf}) present. Check its referrers, then do GC ...")
        # the tpf is expected to be held by curdoc. Any additional referrer could be hints of memory leaks.
        show_referrers(session_context._document.tpf, session_context._document, "TPF")
        session_context._document.tpf = None
        gc.collect()

    print("", flush=True)


def show_app(tic, sector, magnitude_limit=None):

    async def create_app_ui(doc):
        ui_ctr = create_app_ui_container()
        ui_left = ui_ctr.select_one({"name": "app_left"})
        ui_left.children = [create_search_form(tic, sector, magnitude_limit)]

        ui_main = ui_ctr.select_one({"name": "app_main"})
        try:
            ui_body, catalog_plot_fns = await create_app_body_ui(doc, tic, sector, magnitude_limit=magnitude_limit)
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
                    f"Error of type {type(e).__name__} in creating Inspector for TIC {tic}, sector {sector}", exc_info=True
                )
                err_msg = f"Error in creating Inspector. {type(e).__name__}: {e}"
            ui_body = Div(text=err_msg)
            catalog_plot_fns = None

        ui_main.children = [ui_body]
        doc.add_root(ui_ctr)
        _progressive_plot_catalogs(doc, catalog_plot_fns)
        if ui_body.name == "app_body_interactive":
            # the UI for monitoring WebSocket connection is only relevant
            # in the normal case that interactive widgets are to be shown.
            add_connection_lost_ui(doc)

    #
    # the actual entry point
    #
    doc = curdoc()
    if LOG_TPF_REFERRERS:
        doc.on_session_destroyed(log_tpf_referrers_on_session_destroyed)
    doc.add_next_tick_callback(lambda: create_app_ui(doc))


# BEGIN Jupyter notebook helpers, not used by the web app
#


def show_in_notebook(ui, notebook_url="localhost:8888"):
    """Helper to show the Bokeh UI element in  a Jupyter notebook.

    It is not used by the webapp. It's a helper for users to reuse
    Bokeh UI Elements of the webapp in a notebook, e.g.,
    to show the UI of ``create_app_body_ui_from_tpf()``.
    """

    from bokeh.io import show, output_notebook

    def do_show(doc):
        if callable(ui):  # case actually a function
            ui(doc)
        else:
            doc.add_root(ui)

    output_notebook(verbose=False, hide_banner=True)
    return show(do_show, notebook_url=notebook_url)


def show_in_notebook_app_body_ui_from_tpf(tpf, magnitude_limit=None, catalogs=None, notebook_url="localhost:8888"):
    """Helper for use in Jupyter notebook to show `create_app_body_ui_from_tpf()`."""

    def create_app_body_ui_from_tpf_wrapper(doc):
        async def do_create():
            ui, catalog_plot_fns = await create_app_body_ui_from_tpf(
                doc, tpf, catalogs=catalogs, magnitude_limit=magnitude_limit
            )
            doc.add_root(ui)
            _progressive_plot_catalogs(doc, catalog_plot_fns)

        doc.add_next_tick_callback(do_create)

    return show_in_notebook(create_app_body_ui_from_tpf_wrapper, notebook_url=notebook_url)


#
# END   Jupyter notebook helpers, not used by the web app


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

    # debug codes to ensure custom MAST timeout is applied
    from astroquery.mast import Observations

    log.debug(f"MAST Timeout: {Observations._portal_api_connection.TIMEOUT}")

    args = curdoc().session_context.request.arguments
    tic = get_arg_as_int(args, "tic", None)  # default value for sample
    sector = get_arg_as_int(args, "sector", None)
    magnitude_limit = get_arg_as_float(args, "magnitude_limit", None)
    # log bokeh session ID to make the log easier to correlate but logs generated by bokeh server
    session_id = curdoc().session_context.id
    log.debug(f"Parameters: , {tic}, {sector}, {magnitude_limit} ; {args} . session '{session_id}'")
    log_resource_info(f"main(tic={tic})")

    curdoc().title = "TESS Target Pixels Inspector"
    show_app(tic, sector, magnitude_limit)
