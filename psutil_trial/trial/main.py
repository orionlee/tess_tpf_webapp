from bokeh.layouts import row, column
from bokeh.models import Div

from bokeh.plotting import curdoc


import psutil
import os
import traceback


def get_fdinfo_content(p: psutil.Process) -> str:
    content = ""

    # based on:
    # https://github.com/giampaolo/psutil/blob/ea5b55605f857affa4e65fa27eb80f4f2bfebd63/psutil/_pslinux.py#L2224
    files = os.listdir(f"{psutil._common.get_procfs_path()}/{p.pid}/fd")
    content += f"fd files: {files}"
    for fd in files:
        file = f"{psutil._common.get_procfs_path()}/{p.pid}/fd/{fd}"
        try:
            path = psutil._pslinux.readlink(file)
        except Exception as e:
            content += f"\nfile: {file} -- error {type(e).__name__}: {e}"
            continue

        if path.startswith("/") and psutil._common.isfile_strict(path):
            content += f"\nfile: {file}"
            file = f"{psutil._common.get_procfs_path()}/{p.pid}/fdinfo/{fd}"
            # # emulate _pslinux module
            # with psutil._common.open_binary(file) as f:
            #     content += f"\nfd = {fd} , fdinfo = {file} , filepath = {path}"
            #     content += f"\n{f.readline()}"  # L2255
            #     content += f"\n{f.readline()}"  # L2256
            # just read the entire file
            with open(file, "r") as f:
                content += f"\nfd = {fd} , fdinfo = {file} , filepath = {path}\n'''\n"
                content += f.read()
                content += "\n'''"
        else:
            content += f"\nfile: {file} -- path: {path}"

            content += "\n"
    return content


def get_psutil_result():
    msg = f"psutil version: {psutil.__version__}"
    try:
        p = psutil.Process()

        # based on:
        # https://github.com/giampaolo/psutil/blob/release-7.0.0/psutil/_pslinux.py#L2226
        fd_dir = f"{psutil._common.get_procfs_path()}/{p.pid}/fd"
        msg += f"\nfd_dir = {fd_dir}"

        fd_content = get_fdinfo_content(p)
        msg += f"\nfdinfo content:\n{fd_content}"

        # the actual call that raises error in Google Cloud Run
        msg += "\n------"
        res = p.open_files()
        msg += f"\nlen(process.open_files()) = {len(res)}"
        msg += f"\n{res}"

    except Exception as e:
        traceback_str = traceback.format_exc()
        msg += f"\nerror: {type(e).__name__} {e}\n{traceback_str}"

    return msg


def test_bokeh_server(doc):
    async def async_create_interact_ui(doc):
        psutil_result = get_psutil_result()
        return Div(
            text=f"""
<h2>psutil output</h2>
<pre><code>{psutil_result}</code></result>
"""
        )

    def create_interact_ui(doc):
        async def do_create_ui():
            fig = await async_create_interact_ui(doc)
            doc.add_root(column([fig]))

        doc.add_next_tick_callback(do_create_ui)

    create_interact_ui(doc)


test_bokeh_server(curdoc())
