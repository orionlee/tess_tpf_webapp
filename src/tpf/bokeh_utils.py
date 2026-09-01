from bokeh.layouts import column
from bokeh.models import CustomJS, Div, LayoutDOM, TextInput, Toggle


def get_value_in_float(input: TextInput, default=None):
    val = input.value
    try:
        val = float(val) if val != "" else default
    except:  # noqa: E722
        # Non float value, treat it as default
        val = default
    return val


def replace_or_append_by_name(ui_container, new_model):
    old_model = ui_container.select_one({"name": new_model.name})
    if old_model is not None:
        # https://discourse.bokeh.org/t/clearing-plot-or-removing-all-glyphs/6792/6
        idx = ui_container.children.index(old_model)
        ui_container.children[idx] = new_model
    else:
        ui_container.children.append(new_model)


def create_collapsible_section(
    title: str,
    content_model: LayoutDOM,
    title_tag: str="h3",
    title_style_expand: str="",
    title_style_collapse: str="",
) -> LayoutDOM:
    """
    Creates a collapsible component for the given Bokeh model.
    """

    # example more elaborate title styles
    # title_style_expand="background-color: #007bff; color: white;"
    # title_style_collapse="background-color: #6c757d; color: white;"

    # Implementation: use a transparent Toggle to control expand/collapse states

    # 1. Stylized visual header
    header_div = Div(
        text=f"""
        <{title_tag} style="{title_style_expand}">▼ {title}</{title_tag}>
        """,
        sizing_mode="stretch_width",
        margin=(0, 0, 0, 0),
    )

    # 2. Assign bottom spacing and visibility attributes directly to the user's Bokeh model
    content_model.visible = True
    # Ensure there is clean spacing below the content block when expanded
    if content_model.margin:
        content_model.margin = (
            content_model.margin[0],
            content_model.margin[1],
            15,
            content_model.margin[3],
        )
    else:
        content_model.margin = (0, 0, 15, 0)

    # 3. Transparent toggle overlay
    overlay_toggle = Toggle(
        active=True,
        sizing_mode="stretch_width",
        margin=(-43, 0, 0, 0),  # Pulls the interaction box on top of the header div
        styles={"opacity": "0", "height": "43px", "cursor": "pointer"},
    )

    # 4. JavaScript Callback links state across python server sessions
    toggle_js = CustomJS(
        args=dict(
            content=content_model,
            header=header_div,
            title=title,
            title_tag=title_tag,
            title_style_expand=title_style_expand,
            title_style_collapse=title_style_collapse,
        ),
        code="""
        // Sync visibility directly with toggle state
        content.visible = cb_obj.active;

        // Dynamically alter layout typography and arrow icons based on active states
        if (cb_obj.active) {
            header.text = `<${title_tag} style="${title_style_expand}">▼ ${title}</${title_tag}>`;
        } else {
            header.text = `<${title_tag} style="${title_style_collapse}">▶ ${title}</${title_tag}>`;
        }
        """,
    )
    overlay_toggle.js_on_change("active", toggle_js)

    return column(
        header_div, overlay_toggle, content_model, sizing_mode="stretch_width"
    )


def suppress_bokeh_default_reconnect_and_ui(doc):
    # starting in bokeh v3.8, bokeh by default tries to reconnect lost session (and notify the user with UI)
    # the new logic is not helpful in the main deployment scenario (webapp and Jupyter notebook)
    if hasattr(doc, "config"):
        doc.config.reconnect_session = False
        doc.config.notify_connection_status = False
