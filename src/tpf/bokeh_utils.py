from bokeh.models import TextInput


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


def suppress_bokeh_default_reconnect_and_ui(doc):
    # starting in bokeh v3.8, bokeh by default tries to reconnect lost session (and notify the user with UI)
    # the new logic is not helpful in the main deployment scenario (webapp and Jupyter notebook)
    if hasattr(doc, "config"):
        doc.config.reconnect_session = False
        doc.config.notify_connection_status = False
