import os

from flask import Flask
from flask import request, redirect

#
# Redirecting traffic from legacy
#   https://tess-skyview-fqhnyorhza-uw.a.run.app/
# to
#   https://tess-tpf-fqhnyorhza-uw.a.run.app/
#

app = Flask(__name__)

STATUS_CODE = int(os.environ.get("TESS_TPF_REDIRECT_STATUS_CODE", "302"))

DEST_URL_BASE = os.environ.get(
    "TESS_TPF_REDIRECT_DEST_URL_BASE",
    "https://tess-tpf-fqhnyorhza-uw.a.run.app/tpf",
)


@app.route('/')
@app.route('/skyview')
def do_redirect():
    qs = request.query_string.decode()
    if qs is None or qs == "":
        dest_url = DEST_URL_BASE
    else:
        dest_url = f"{DEST_URL_BASE}?{qs}"
    # print("DBG: ", dest_url)
    return redirect(dest_url, STATUS_CODE)


if __name__ == "__main__":
    app.run(
        debug=True,
        use_reloader=False,  # disable hot reload and avoid the extra watcher process
        use_evalex=False,  # disable werkzeug's web-based  interactive debugger on error page
        host=os.environ.get("TESS_TPF_REDIRECT_HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 8080)),
    )
