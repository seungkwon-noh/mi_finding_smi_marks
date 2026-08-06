from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .core import TemplateFinder
from .io import decode_base64_image, load_templates

FAIL_COORDINATE = "-1,-1"


def _payload(req: str | bytes | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(req, Mapping):
        return dict(req)
    if isinstance(req, bytes):
        req = req.decode("utf-8")
    parsed = json.loads(req)
    if not isinstance(parsed, dict):
        raise TypeError("request JSON must be an object")
    return parsed


def handle(
    req: str | bytes | Mapping[str, Any], template_root: str | Path | None = None
) -> tuple[dict[str, Any], int]:
    """Framework-neutral replacement for the original FaaS handler.

    A Flask/OpenFaaS adapter can pass this result to ``make_response(*handle(req))``.
    """

    try:
        data = _payload(req)
        image = decode_base64_image(str(data["image"]))
        root = Path(template_root or os.environ.get("MI_TEMPLATE_ROOT", "templates"))
        mode = str(data.get("mode", "normal"))

        if mode == "popup":
            coordinates: list[str] = []
            details: list[dict[str, Any]] = []
            for prefix in ("popup_on_target", "popup_next_site"):
                result = TemplateFinder().find(image, load_templates(root, prefix))
                coordinates.append(
                    ",".join(map(str, result.candidate.center))
                    if result.success and result.candidate
                    else FAIL_COORDINATE
                )
                details.append(result.to_dict())
            success = any(coord != FAIL_COORDINATE for coord in coordinates)
            message = ", ".join(f"({coord})" for coord in coordinates)
            return {"success": success, "message": message, "details": details}, 200

        prefix = f"{data['product']}_{data['layer']}"
        templates = load_templates(root, prefix)
        result = TemplateFinder().find(image, templates)
        coordinate = (
            ",".join(map(str, result.candidate.center))
            if result.success and result.candidate
            else FAIL_COORDINATE
        )
        return {
            "success": result.success,
            "message": coordinate,
            "details": result.to_dict(),
        }, 200
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"success": False, "message": FAIL_COORDINATE, "error": str(exc)}, 400
