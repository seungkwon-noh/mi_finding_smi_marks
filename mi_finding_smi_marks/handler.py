from __future__ import annotations

import base64
import json
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from flask import make_response

from .riselog import getMyLogger
from .util_functions import (
    FindingResult,
    MatchCandidate,
    MatchingConfig,
    apply_clahe_image,
    candidate_from_match,
    decode_base64_image,
    full_template,
    generate_edge_templates,
    match_template_for_popup,
    match_template_logic,
    process_image_with_detection,
    remove_green_mark,
    select_full_candidate,
    select_partial_candidate,
)

logger = getMyLogger()

FAIL_COORDINATE = "-1,-1"
DEFAULT_BUCKET = "static"
DEFAULT_TEMPLATE_PREFIX = "MI/GA_TEMPLATE/"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
EMAIL_CONTENT_TYPE = "application/json; charset=utf-8"

_MINIO_CLIENT: Any | None = None
_MINIO_LOCK = threading.Lock()


@dataclass(frozen=True)
class TemplateAsset:
    name: str
    image: np.ndarray


@dataclass(frozen=True)
class MatchMethod:
    name: str
    image: np.ndarray
    gray: np.ndarray
    apply_clahe: bool = False
    image_gray_is_clahe: bool = False


def _payload(req: str | bytes | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(req, Mapping):
        return dict(req)
    if isinstance(req, bytes):
        req = req.decode("utf-8")
    if not isinstance(req, str):
        raise TypeError("request body must be JSON text")
    parsed = json.loads(req)
    if not isinstance(parsed, dict):
        raise TypeError("request JSON must be an object")
    return parsed


def _required_text(data: Mapping[str, Any], key: str) -> str:
    if key not in data:
        raise ValueError(f"missing required field: {key}")
    value = str(data[key]).strip()
    if not value:
        raise ValueError(f"field must not be empty: {key}")
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _secret_or_env(env_name: str, secret_name: str) -> str | None:
    value = os.environ.get(env_name)
    if value:
        return value.strip()
    secret_path = Path("/var/openfaas/secrets") / secret_name
    try:
        secret_value = secret_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return secret_value or None


def get_minio_client() -> Any:
    """Create one process-wide client and reuse it across warm FaaS invocations."""

    global _MINIO_CLIENT
    if _MINIO_CLIENT is not None:
        return _MINIO_CLIENT
    with _MINIO_LOCK:
        if _MINIO_CLIENT is not None:
            return _MINIO_CLIENT

        endpoint = os.environ.get("MINIO_ENDPOINT", "").strip()
        access_key = _secret_or_env("MINIO_ACCESS_KEY", "mi-minio-access-key")
        secret_key = _secret_or_env("MINIO_SECRET_KEY", "mi-minio-secret-key")
        if not endpoint:
            raise RuntimeError("MINIO_ENDPOINT is not configured")
        if not access_key or not secret_key:
            raise RuntimeError("MinIO access key or secret key is not configured")

        from minio import Minio

        _MINIO_CLIENT = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=_env_bool("MINIO_SECURE", False),
        )
        return _MINIO_CLIENT


def _template_prefix() -> str:
    prefix = os.environ.get("MINIO_TEMPLATE_PREFIX", DEFAULT_TEMPLATE_PREFIX).strip()
    return prefix if prefix.endswith("/") else f"{prefix}/"


def _load_templates(
    minio_client: Any,
    prefix: str,
    *,
    excluded_names: tuple[str, ...] = (),
) -> list[TemplateAsset]:
    bucket = os.environ.get("MINIO_BUCKET", DEFAULT_BUCKET).strip() or DEFAULT_BUCKET
    excluded_names = tuple(name.lower() for name in excluded_names if name)
    object_names = sorted(
        obj.object_name
        for obj in minio_client.list_objects(bucket, prefix=prefix, recursive=True)
        if getattr(obj, "object_name", "").lower().endswith(IMAGE_SUFFIXES)
        and not any(
            name in getattr(obj, "object_name", "").lower() for name in excluded_names
        )
    )

    templates: list[TemplateAsset] = []
    for object_name in object_names:
        response = minio_client.get_object(bucket, object_name)
        try:
            raw = response.read()
        finally:
            response.close()
            response.release_conn()
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            logger.warning(f"template_decode_failed object={object_name}")
            continue
        templates.append(TemplateAsset(object_name, image))
    return templates


def _load_templates_with_fallback(
    minio_client: Any,
    prefix: str,
    base_prefix: str,
    *,
    excluded_names: tuple[str, ...] = (),
) -> tuple[list[TemplateAsset], bool]:
    """Load the requested template group, or every template when it is absent.

    The fallback intentionally mirrors the original operating code from the shared
    mi_finding conversation. It is based on an empty template lookup, not on a low
    visual matching score.
    """

    assets = _load_templates(
        minio_client,
        prefix,
        excluded_names=excluded_names,
    )
    if assets:
        return assets, False
    return (
        _load_templates(
            minio_client,
            base_prefix,
            excluded_names=excluded_names,
        ),
        True,
    )


def _match_methods(image: np.ndarray) -> list[MatchMethod]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_clahe = apply_clahe_image(gray)
    without_marker = remove_green_mark(image)
    return [
        MatchMethod("RAW", image, gray),
        MatchMethod(
            "CLAHE_ONLY",
            image,
            gray_clahe,
            apply_clahe=True,
            image_gray_is_clahe=True,
        ),
        MatchMethod(
            "REMOVE_MARKER_ONLY",
            without_marker,
            cv2.cvtColor(without_marker, cv2.COLOR_BGR2GRAY),
        ),
    ]


def _candidate(
    *,
    image_shape: tuple[int, ...],
    asset: TemplateAsset,
    edge: Any,
    method: MatchMethod,
    config: MatchingConfig,
) -> MatchCandidate | None:
    match = match_template_logic(
        method.image,
        method.gray,
        edge.image,
        edge.ratio,
        apply_clahe=method.apply_clahe,
        image_gray_is_clahe=method.image_gray_is_clahe,
        config=config,
    )
    return candidate_from_match(
        match,
        edge,
        image_shape=image_shape,
        method=method.name,
        template_name=asset.name,
    )


def _find_review(
    image: np.ndarray,
    assets: list[TemplateAsset],
    config: MatchingConfig,
) -> FindingResult:
    if not assets:
        return FindingResult(False, "template_not_found")

    methods = _match_methods(image)
    full_candidates: list[MatchCandidate] = []
    attempts = 0
    for asset in assets:
        edge = full_template(asset.image)
        for method in methods:
            attempts += 1
            candidate = _candidate(
                image_shape=image.shape,
                asset=asset,
                edge=edge,
                method=method,
                config=config,
            )
            if candidate is not None:
                full_candidates.append(candidate)

    selected, reason = select_full_candidate(full_candidates, config)
    if selected is not None:
        return FindingResult(True, reason, selected, attempts)

    all_candidates = list(full_candidates)
    for ratio in config.partial_ratios:
        ratio_candidates: list[MatchCandidate] = []
        for asset in assets:
            for edge in generate_edge_templates(asset.image, ratio):
                for method in methods:
                    attempts += 1
                    candidate = _candidate(
                        image_shape=image.shape,
                        asset=asset,
                        edge=edge,
                        method=method,
                        config=config,
                    )
                    if candidate is not None:
                        ratio_candidates.append(candidate)

        all_candidates.extend(ratio_candidates)
        selected_partial = select_partial_candidate(ratio_candidates, config)
        if selected_partial is not None:
            return FindingResult(True, "partial_pass", selected_partial, attempts)

    diagnostic = (
        max(all_candidates, key=lambda item: item.score) if all_candidates else None
    )
    return FindingResult(False, "no_candidate_passed_thresholds", diagnostic, attempts)


def _find_popup_coordinate(
    image: np.ndarray,
    assets: list[TemplateAsset],
    config: MatchingConfig,
) -> tuple[str, float, str | None]:
    if not assets:
        return FAIL_COORDINATE, -1.0, None
    without_marker = remove_green_mark(image)
    gray = cv2.cvtColor(without_marker, cv2.COLOR_BGR2GRAY)
    best: tuple[float, tuple[int, int], int, int, str] | None = None
    for asset in assets:
        top_left, roi, score = match_template_for_popup(
            without_marker, gray, asset.image
        )
        if roi.size == 0:
            continue
        height, width = asset.image.shape[:2]
        current = (score, top_left, width, height, asset.name)
        if best is None or current[0] > best[0]:
            best = current

    if best is None or best[0] <= config.popup_min_score:
        return FAIL_COORDINATE, best[0] if best else -1.0, best[4] if best else None
    score, top_left, width, height, template_name = best
    coordinate = f"{top_left[0] + width // 2},{top_left[1] + height // 2}"
    return coordinate, score, template_name


def _response_body(result: FindingResult) -> dict[str, object]:
    coordinate = (
        f"{result.candidate.center[0]},{result.candidate.center[1]}"
        if result.success and result.candidate
        else FAIL_COORDINATE
    )
    body: dict[str, object] = {"success": result.success, "message": coordinate}
    if _env_bool("MI_INCLUDE_DETAILS", False):
        body["details"] = result.to_dict()
    return body


def _email_recipients() -> list[str]:
    return [
        address.strip()
        for address in os.environ.get("MI_MATCH_EMAIL_RECIPIENTS", "").split(",")
        if address.strip()
    ]


def _report_panel(image: np.ndarray | None, title: str) -> np.ndarray:
    panel_width, panel_height = 360, 300
    panel = np.full((panel_height + 38, panel_width, 3), 245, dtype=np.uint8)
    cv2.putText(
        panel,
        title,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    if image is None or image.size == 0:
        return panel
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    height, width = image.shape[:2]
    scale = min(panel_width / width, panel_height / height)
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    y = 38 + (panel_height - resized.shape[0]) // 2
    x = (panel_width - resized.shape[1]) // 2
    panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return panel


def _build_match_report_image(
    image: np.ndarray,
    assets: list[TemplateAsset],
    result: FindingResult,
    name: str,
) -> bytes | None:
    candidate = result.candidate
    if candidate is None:
        return None

    x, y = candidate.top_left
    width, height = candidate.width, candidate.height
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(image.shape[1], x + width), min(image.shape[0], y + height)
    roi = image[y1:y2, x1:x2]
    boxed = image.copy()
    cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 255, 0), 3)
    template = next(
        (asset.image for asset in assets if asset.name == candidate.template_name),
        None,
    )

    panels = cv2.hconcat(
        [
            _report_panel(boxed, "Final Match (Box)"),
            _report_panel(roi, "Extracted ROI"),
            _report_panel(template, f"Template ({candidate.edge})"),
        ]
    )
    metrics = candidate.metrics
    status_color = (20, 20, 20) if result.success else (0, 140, 255)
    header = np.full((112, panels.shape[1], 3), 255, dtype=np.uint8)
    lines = (
        f"{name} | Success:{result.success} | Reason:{result.reason}",
        f"Method:{candidate.method} Stage:{candidate.stage} "
        f"Ratio:{candidate.ratio:.2f} Edge:{candidate.edge}",
        f"Score:{candidate.score:.3f} VarRatio:{metrics.variance_ratio:.3f} "
        f"SSIM:{metrics.ssim:.3f} Hist:{metrics.color_hist_similarity:.4f} "
        f"NMI:{metrics.nmi:.4f} BD:{metrics.bhattacharyya_distance:.4f} "
        f"AD:{metrics.average_color_distance:.2f}",
    )
    for index, line in enumerate(lines):
        cv2.putText(
            header,
            line,
            (12, 28 + index * 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            status_color,
            2 if index == 0 else 1,
            cv2.LINE_AA,
        )
    ok, encoded = cv2.imencode(".png", cv2.vconcat([header, panels]))
    return encoded.tobytes() if ok else None


def _send_match_email(
    image: np.ndarray,
    assets: list[TemplateAsset],
    result: FindingResult,
    eqpid: str,
    product: str,
    layer: str,
) -> None:
    """Send a diagnostic candidate without changing the business response."""

    url = os.environ.get("MI_MATCH_EMAIL_URL", "").strip()
    recipients = _email_recipients()
    if not url or not recipients:
        return
    if result.candidate is None:
        logger.warning("email_skip candidate=None")
        return

    name = f"{eqpid}_{product}_{layer}"
    try:
        image_bytes = _build_match_report_image(image, assets, result, name)
        if not image_bytes:
            logger.error("email_report_encode_failed")
            return
        payload = {
            "Group": os.environ.get("MI_MATCH_EMAIL_GROUP", "MI").strip() or "MI",
            "Project": os.environ.get("MI_MATCH_EMAIL_PROJECT", "find_smi").strip()
            or "find_smi",
            "JsonString": json.dumps(
                {
                    "img_info": name,
                    "image": base64.b64encode(image_bytes).decode("ascii"),
                }
            ),
            "Recipient": recipients,
        }
        response = requests.post(
            url=url,
            headers={"Content-Type": EMAIL_CONTENT_TYPE},
            data=json.dumps(payload),
            timeout=float(os.environ.get("MI_MATCH_EMAIL_TIMEOUT", "20")),
        )
        logger.info(f"email status={response.status_code}")
    except Exception as exc:
        logger.error(f"email_send_failed error={exc}")


def handle(req: str | bytes | Mapping[str, Any], minio_client: Any | None = None):
    """OpenFaaS entry point called by ``/home/app/index.py`` for each POST."""

    try:
        data = _payload(req)
        image = decode_base64_image(_required_text(data, "image"))
        mode = str(data.get("mode", "review")).strip().lower() or "review"
        eqpid = str(data.get("eqpid", "unknown"))
        client = minio_client or get_minio_client()
        config = MatchingConfig()
        base_prefix = _template_prefix()

        if mode == "popup":
            popup_assets, used_fallback = _load_templates_with_fallback(
                client,
                f"{base_prefix}popup",
                base_prefix,
            )
            coordinates: list[str] = []
            popup_logs: list[dict[str, object]] = []
            for suffix in ("popup_on_target", "popup_next_site"):
                prefix = f"{base_prefix}{suffix}"
                assets = [
                    asset for asset in popup_assets if asset.name.startswith(prefix)
                ]
                coordinate, score, template_name = _find_popup_coordinate(
                    image, assets, config
                )
                coordinates.append(coordinate)
                popup_logs.append(
                    {
                        "button": suffix,
                        "score": score,
                        "template": template_name,
                        "coordinate": coordinate,
                    }
                )

            success = any(item != FAIL_COORDINATE for item in coordinates)
            logger.info(
                json.dumps(
                    {
                        "eqpid": eqpid,
                        "mode": mode,
                        "template_scope": "all" if used_fallback else "popup",
                        "template_count": len(popup_assets),
                        "popup": popup_logs,
                    },
                    ensure_ascii=False,
                )
            )
            if not success:
                return make_response(
                    {"success": False, "message": FAIL_COORDINATE}, 200
                )
            message = ", ".join(f"({item})" for item in coordinates)
            return make_response({"success": True, "message": message}, 200)

        product = _required_text(data, "product")
        layer = _required_text(data, "layer")
        recipe = str(data.get("recipe", ""))
        prefix = f"{base_prefix}{product}_{layer}"
        assets, used_fallback = _load_templates_with_fallback(
            client,
            prefix,
            base_prefix,
            excluded_names=("popup",),
        )
        prepared_image = process_image_with_detection(image)
        result = _find_review(prepared_image, assets, config)
        logger.info(
            json.dumps(
                {
                    "eqpid": eqpid,
                    "mode": mode,
                    "product": product,
                    "layer": layer,
                    "recipe": recipe,
                    "template_scope": "all" if used_fallback else "product_layer",
                    "template_count": len(assets),
                    "result": result.to_dict(),
                },
                ensure_ascii=False,
            )
        )
        _send_match_email(prepared_image, assets, result, eqpid, product, layer)
        return make_response(_response_body(result), 200)

    except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError) as exc:
        logger.warning(f"invalid_request error={exc}")
        return make_response(
            {"success": False, "message": FAIL_COORDINATE, "error": str(exc)},
            400,
        )
    except Exception as exc:  # OpenFaaS must receive a response instead of a crash.
        logger.exception(f"function_failed error={exc}")
        return make_response(
            {
                "success": False,
                "message": FAIL_COORDINATE,
                "error": "internal_error",
            },
            500,
        )
