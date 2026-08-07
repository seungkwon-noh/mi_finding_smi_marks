from __future__ import annotations

from flask import make_response

import http_test


def test_http_test_forwards_raw_body_to_handler(monkeypatch) -> None:
    received: list[str] = []

    def fake_handle(body: str):
        received.append(body)
        return make_response({"success": True, "message": "1,2"}, 200)

    monkeypatch.setattr(http_test.handler, "handle", fake_handle)
    response = http_test.app.test_client().post(
        "/function/mi-finding-smi-marks",
        data='{"mode":"review"}',
        content_type="application/json",
    )

    assert response.status_code == 200
    assert response.get_json() == {"success": True, "message": "1,2"}
    assert received == ['{"mode":"review"}']
