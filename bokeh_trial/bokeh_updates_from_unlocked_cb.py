# Adapted from:
# https://docs.bokeh.org/en/latest/docs/user_guide/server/app.html#updating-from-unlocked-callbacks
#
# progressive update using `doc.add_timeout_callback(),  @without_document_lock, + doc.add_next_tick_callback()`
# - the basis of the version used in interact_sky() impl
#

import asyncio
from functools import partial
import time

import bokeh
from bokeh.document import without_document_lock
from bokeh.models import Range1d, ColumnDataSource
from bokeh.layouts import layout
from bokeh.plotting import curdoc, figure

# print("bokeh v:", bokeh.__version__)


def test_bokeh_server(doc, request_id):
    def create_plot_src_func(doc, fig, label, delay_sec, color):
        # simulate fetching data remotely that takes time
        def get_remote_data():
            time.sleep(delay_sec)
            source = ColumnDataSource(
                data=dict(
                    x=[delay_sec],
                    y=[delay_sec],
                )
            )
            return source

        # start fetching remote data right away in background
        data_task = asyncio.create_task(asyncio.to_thread(get_remote_data))

        async def do_plot_locked(source):
            print(f"DBG [{request_id}]  do_plot_locked() for ", label)
            fig.scatter(source=source, size=10 + (delay_sec**2) * 3, marker="square", fill_color=color)

        # return an async function that will get the remote data and plot them
        @without_document_lock
        async def do_plot_unlocked():
            source = await data_task
            print(f"DBG [{request_id}]  do_plot_unlocked() source ready for ", label)
            doc.add_next_tick_callback(partial(do_plot_locked, source=source))

        return do_plot_unlocked

    async def async_create_interact_ui(doc):
        print(f"DBG [{request_id}]  async_create_interact_ui() start")
        time.sleep(10)  # simulate it takes time to create the base plot
        fig = figure(
            title="Test Bokeh Progressive Plot",
        )

        fig.x_range = Range1d(0, 10)
        fig.y_range = Range1d(0, 10)

        # approximate progressive by ordering the function in the expected execution time
        plot_fns = [
            # put the slowest up front, and it won't block the rendering of the faster sources
            create_plot_src_func(doc, fig, "slowest", 7, "red"),
            create_plot_src_func(doc, fig, "fast", 2, "blue"),
            create_plot_src_func(doc, fig, "slow", 4, "green"),
        ]
        return fig, plot_fns

    def create_interact_ui(doc):
        async def do_create_ui():
            fig, plot_src_fns = await async_create_interact_ui(doc)
            # first show the base figure to users
            doc.add_root(
                layout(
                    [
                        [
                            fig,
                        ]
                    ]
                )
            )
            # then progressively show data as they become available
            for fn in plot_src_fns:
                doc.add_timeout_callback(fn, 0)  # Just need it run in the background

        doc.add_next_tick_callback(do_create_ui)

    create_interact_ui(doc)


request_id = int(time.time())
test_bokeh_server(curdoc(), request_id)
