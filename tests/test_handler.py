import json

from mi_finding.handler import handle


def test_handler_rejects_missing_image() -> None:
    response, status = handle(json.dumps({"product": "P", "layer": "1"}))
    assert status == 400
    assert response["success"] is False
