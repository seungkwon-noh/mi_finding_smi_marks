from __future__ import annotations

import os

from flask import Flask, request
from waitress import serve

from mi_finding_smi_marks import handler

app = Flask(__name__)


@app.route("/", defaults={"path": ""}, methods=["POST"])
@app.route("/<path:path>", methods=["POST"])
def call_handler(path: str):
    """Expose the same raw-body contract used by the OpenFaaS watchdog."""

    del path
    return handler.handle(request.get_data(as_text=True))


if __name__ == "__main__":
    serve(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "7184")),
    )
