from __future__ import annotations

import json

import numpy as np
from conftest import FakeMinio, encode_image
from flask import Flask

from mi_finding_smi_marks.handler import handle


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
