from __future__ import annotations

import base64
import json

import numpy as np
from conftest import FakeMinio, encode_image
from flask import Flask

import mi_finding_smi_marks.handler as handler_module
from mi_finding_smi_marks.handler import handle
from mi_finding_smi_marks.util_functions import (
    FindingResult,
    MatchCandidate,
    MatchMetrics,
)


def response_json(request: dict[str, object], minio: FakeMinio):
    app = Flask(__name__)
    with app.app_context():
        response = handle(json.dumps(request), minio)
        return response.status_code, response.get_json()


def test_review_request_returns_template_center() -> None:
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, (220, 260, 3), dtype=np.uint8)
    template = rng.integers(0, 256, (80, 60, 3), dtype=np.uint8)
    x, y = 91, 73
    image[y : y + 80, x : x + 60] = template
    minio = FakeMinio({"MI/GA_TEMPLATE/P_01_button.png": template})

    status, body = response_json(
        {
            "image": encode_image(image),
            "product": "P",
            "layer": "01",
            "eqpid": "EQP01",
            "mode": "review",
        },
        minio,
    )

    assert status == 200
    assert body == {"success": True, "message": f"{x + 30},{y + 40}"}


def test_missing_product_template_is_a_business_failure() -> None:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    status, body = response_json(
        {
            "image": encode_image(image),
            "product": "UNKNOWN",
            "layer": "01",
            "mode": "review",
        },
        FakeMinio({}),
    )

    assert status == 200
    assert body == {"success": False, "message": "-1,-1"}


def test_review_falls_back_to_all_images_when_product_layer_has_no_template() -> None:
    rng = np.random.default_rng(37)
    image = rng.integers(0, 256, (220, 260, 3), dtype=np.uint8)
    template = rng.integers(0, 256, (60, 54, 3), dtype=np.uint8)
    x, y = 103, 81
    image[y : y + 60, x : x + 54] = template
    minio = FakeMinio({"MI/GA_TEMPLATE/OTHER_99_button.png": template})

    status, body = response_json(
        {
            "image": encode_image(image),
            "product": "UNKNOWN",
            "layer": "01",
            "recipe": "RCP01",
            "mode": "review",
        },
        minio,
    )

    assert status == 200
    assert body == {"success": True, "message": f"{x + 27},{y + 30}"}
    assert minio.listed_prefixes == [
        "MI/GA_TEMPLATE/UNKNOWN_01",
        "MI/GA_TEMPLATE/",
    ]


def test_review_fallback_excludes_popup_templates() -> None:
    rng = np.random.default_rng(39)
    image = rng.integers(0, 256, (180, 220, 3), dtype=np.uint8)
    popup_template = rng.integers(0, 256, (50, 48, 3), dtype=np.uint8)
    image[61:111, 93:141] = popup_template
    minio = FakeMinio(
        {"MI/GA_TEMPLATE/popup_on_target_false_positive.png": popup_template}
    )

    status, body = response_json(
        {
            "image": encode_image(image),
            "product": "UNKNOWN",
            "layer": "01",
            "mode": "review",
        },
        minio,
    )

    assert status == 200
    assert body == {"success": False, "message": "-1,-1"}
    assert minio.listed_prefixes == [
        "MI/GA_TEMPLATE/UNKNOWN_01",
        "MI/GA_TEMPLATE/",
    ]


def test_review_does_not_load_all_images_when_specific_templates_exist() -> None:
    rng = np.random.default_rng(41)
    image = rng.integers(0, 256, (160, 190, 3), dtype=np.uint8)
    template = rng.integers(0, 256, (40, 44, 3), dtype=np.uint8)
    image[55:95, 71:115] = template
    minio = FakeMinio({"MI/GA_TEMPLATE/PART_01_button.png": template})

    status, body = response_json(
        {
            "image": encode_image(image),
            "product": "PART",
            "layer": "01",
            "mode": "review",
        },
        minio,
    )

    assert status == 200
    assert body["success"] is True
    assert minio.listed_prefixes == ["MI/GA_TEMPLATE/PART_01"]


def test_popup_keeps_the_successful_button_in_its_original_position() -> None:
    rng = np.random.default_rng(29)
    image = rng.integers(0, 256, (180, 220, 3), dtype=np.uint8)
    template = rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)
    x, y = 121, 67
    image[y : y + 40, x : x + 50] = template
    minio = FakeMinio({"MI/GA_TEMPLATE/popup_next_site_01.png": template})

    status, body = response_json(
        {"image": encode_image(image), "eqpid": "EQP01", "mode": "popup"},
        minio,
    )

    assert status == 200
    assert body == {
        "success": True,
        "message": f"(-1,-1), ({x + 25},{y + 20})",
    }
    assert minio.listed_prefixes == ["MI/GA_TEMPLATE/popup"]


def test_invalid_request_returns_http_400() -> None:
    app = Flask(__name__)
    with app.app_context():
        response = handle(json.dumps({"product": "P", "layer": "01"}), FakeMinio({}))

    assert response.status_code == 400
    assert response.get_json()["success"] is False
    assert response.get_json()["message"] == "-1,-1"


def test_failed_diagnostic_candidate_is_emailed_but_response_stays_failed(
    monkeypatch,
) -> None:
    image = np.full((100, 120, 3), 80, dtype=np.uint8)
    template = np.full((30, 40, 3), 160, dtype=np.uint8)
    candidate = MatchCandidate(
        score=0.796,
        method="RAW",
        stage="full",
        ratio=1.0,
        top_left=(25, 20),
        matched_top_left=(25, 20),
        width=40,
        height=30,
        metrics=MatchMetrics(0.3, 0.5, 0.2, 0.1, 0.7, 45.0),
        template_name="MI/GA_TEMPLATE/P_01_button.png",
    )
    diagnostic = FindingResult(
        False,
        "no_candidate_passed_thresholds",
        candidate,
        attempts=3,
    )
    sent: dict[str, object] = {}

    class FakePostResponse:
        status_code = 200

    def fake_post(*, url, headers, data, timeout):
        sent.update(url=url, headers=headers, data=data, timeout=timeout)
        return FakePostResponse()

    monkeypatch.setenv("MI_MATCH_EMAIL_URL", "http://mail.internal/send")
    monkeypatch.setenv(
        "MI_MATCH_EMAIL_RECIPIENTS", "first@example.com, second@example.com"
    )
    monkeypatch.setattr(handler_module, "_find_review", lambda *_: diagnostic)
    monkeypatch.setattr(handler_module.requests, "post", fake_post)

    status, body = response_json(
        {
            "image": encode_image(image),
            "product": "P",
            "layer": "01",
            "eqpid": "EQP01",
            "mode": "review",
        },
        FakeMinio({"MI/GA_TEMPLATE/P_01_button.png": template}),
    )

    assert status == 200
    assert body == {"success": False, "message": "-1,-1"}
    assert sent["url"] == "http://mail.internal/send"
    assert sent["timeout"] == 20.0
    payload = json.loads(str(sent["data"]))
    assert payload["Recipient"] == ["first@example.com", "second@example.com"]
    report = json.loads(payload["JsonString"])
    assert report["img_info"] == "EQP01_P_01"
    assert base64.b64decode(report["image"]).startswith(b"\x89PNG")


def test_email_failure_does_not_change_successful_faas_response(monkeypatch) -> None:
    rng = np.random.default_rng(47)
    image = rng.integers(0, 256, (160, 190, 3), dtype=np.uint8)
    template = rng.integers(0, 256, (40, 44, 3), dtype=np.uint8)
    x, y = 71, 55
    image[y : y + 40, x : x + 44] = template

    def failing_post(**_):
        raise TimeoutError("mail service unavailable")

    monkeypatch.setenv("MI_MATCH_EMAIL_URL", "http://mail.internal/send")
    monkeypatch.setenv("MI_MATCH_EMAIL_RECIPIENTS", "first@example.com")
    monkeypatch.setattr(handler_module.requests, "post", failing_post)

    status, body = response_json(
        {
            "image": encode_image(image),
            "product": "P",
            "layer": "01",
            "eqpid": "EQP01",
            "mode": "review",
        },
        FakeMinio({"MI/GA_TEMPLATE/P_01_button.png": template}),
    )

    assert status == 200
    assert body == {"success": True, "message": f"{x + 22},{y + 20}"}
