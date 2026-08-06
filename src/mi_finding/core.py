from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

from .decision import (
    auxiliary_failures,
    select_full_candidate,
    select_partial_candidate,
)
from .metrics import calculate_metrics
from .models import FindingResult, MatchCandidate, MatchingConfig
from .preprocessing import (
    EdgeTemplate,
    apply_clahe,
    generate_edge_templates,
    remove_green_mark,
    validate_bgr_image,
)


@dataclass(frozen=True)
class _Method:
    name: str
    color: np.ndarray
    gray: np.ndarray
    template_clahe: bool = False


class TemplateFinder:
    def __init__(self, config: MatchingConfig | None = None) -> None:
        self.config = config or MatchingConfig()

    def find(self, image: np.ndarray, templates: Iterable[np.ndarray]) -> FindingResult:
        validate_bgr_image(image)
        template_list = list(templates)
        if not template_list:
            return FindingResult(False, "no_templates")
        for index, template in enumerate(template_list):
            validate_bgr_image(template, f"template[{index}]")

        methods = self._methods(image)
        full_candidates: list[MatchCandidate] = []
        rejected: list[dict[str, object]] = []
        attempts = 0

        for template_index, template in enumerate(template_list):
            for method in methods:
                attempts += 1
                candidate = self._match_full(
                    image_shape=image.shape,
                    template=template,
                    template_index=template_index,
                    method=method,
                )
                if candidate is not None:
                    full_candidates.append(candidate)
                    rejected.append(
                        {
                            "stage": "full",
                            "score": float(candidate.score),
                            "method": candidate.method,
                            "template_index": candidate.template_index,
                            "auxiliary_failures": auxiliary_failures(
                                candidate, self.config, partial=False
                            ),
                        }
                    )

        selected, reason = select_full_candidate(full_candidates, self.config)
        if selected is not None:
            return FindingResult(
                True,
                reason,
                selected,
                attempts,
                {"full_candidates": len(full_candidates)},
            )

        all_candidates = list(full_candidates)
        for ratio in self.config.partial_ratios:
            ratio_candidates: list[MatchCandidate] = []
            for template_index, template in enumerate(template_list):
                for edge_template in generate_edge_templates(template, ratio):
                    for method in methods:
                        attempts += 1
                        candidate = self._match_partial(
                            image_shape=image.shape,
                            edge_template=edge_template,
                            template_index=template_index,
                            method=method,
                        )
                        if candidate is not None:
                            ratio_candidates.append(candidate)
                            rejected.append(
                                {
                                    "stage": "partial",
                                    "ratio": ratio,
                                    "edge": candidate.edge,
                                    "score": float(candidate.score),
                                    "method": candidate.method,
                                    "template_index": candidate.template_index,
                                    "auxiliary_failures": auxiliary_failures(
                                        candidate, self.config, partial=True
                                    ),
                                }
                            )

            all_candidates.extend(ratio_candidates)
            selected_partial = select_partial_candidate(ratio_candidates, self.config)
            if selected_partial is not None:
                return FindingResult(
                    True,
                    "partial_pass",
                    selected_partial,
                    attempts,
                    {
                        "full_candidates": len(full_candidates),
                        "partial_ratio": ratio,
                        "partial_candidates": len(ratio_candidates),
                    },
                )

        diagnostic_best = (
            max(all_candidates, key=lambda item: item.score) if all_candidates else None
        )
        return FindingResult(
            False,
            "no_candidate_passed_thresholds",
            diagnostic_best,
            attempts,
            {"rejected_candidates": rejected},
        )

    def _methods(self, image: np.ndarray) -> list[_Method]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        methods = [_Method("RAW", image, gray)]
        methods.append(
            _Method(
                "CLAHE_ONLY",
                image,
                apply_clahe(
                    gray,
                    self.config.clahe_clip_limit,
                    self.config.clahe_grid_size,
                ),
                template_clahe=True,
            )
        )
        if self.config.remove_green_marker:
            without_marker = remove_green_mark(image)
            methods.append(
                _Method(
                    "REMOVE_MARKER_ONLY",
                    without_marker,
                    cv2.cvtColor(without_marker, cv2.COLOR_BGR2GRAY),
                )
            )
        return methods

    def _match_full(
        self,
        *,
        image_shape: tuple[int, ...],
        template: np.ndarray,
        template_index: int,
        method: _Method,
    ) -> MatchCandidate | None:
        height, width = template.shape[:2]
        edge = EdgeTemplate(template, "full", 1.0, 0, 0, width, height)
        return self._match(image_shape, edge, template_index, method, "full")

    def _match_partial(
        self,
        *,
        image_shape: tuple[int, ...],
        edge_template: EdgeTemplate,
        template_index: int,
        method: _Method,
    ) -> MatchCandidate | None:
        return self._match(
            image_shape, edge_template, template_index, method, "partial"
        )

    def _match(
        self,
        image_shape: tuple[int, ...],
        edge_template: EdgeTemplate,
        template_index: int,
        method: _Method,
        stage: str,
    ) -> MatchCandidate | None:
        template = edge_template.image
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        if method.template_clahe:
            template_gray = apply_clahe(
                template_gray,
                self.config.clahe_clip_limit,
                self.config.clahe_grid_size,
            )

        image_height, image_width = method.gray.shape[:2]
        template_height, template_width = template_gray.shape[:2]
        if template_height > image_height or template_width > image_width:
            return None
        if float(template_gray.std()) <= 1e-8:
            return None

        response = cv2.matchTemplate(method.gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, score, _, match_location = cv2.minMaxLoc(response)
        matched_x, matched_y = match_location
        roi = method.color[
            matched_y : matched_y + template_height,
            matched_x : matched_x + template_width,
        ]
        if roi.size == 0 or roi.shape != template.shape:
            return None

        full_x = matched_x - edge_template.offset_x
        full_y = matched_y - edge_template.offset_y
        full_right = full_x + edge_template.full_width
        full_bottom = full_y + edge_template.full_height
        if (
            full_x < 0
            or full_y < 0
            or full_right > image_width
            or full_bottom > image_height
        ):
            return None

        metrics = calculate_metrics(roi, template)
        return MatchCandidate(
            score=float(score),
            method=method.name,
            stage="partial" if stage == "partial" else "full",
            ratio=edge_template.ratio,
            top_left=(full_x, full_y),
            width=edge_template.full_width,
            height=edge_template.full_height,
            metrics=metrics,
            template_index=template_index,
            edge=edge_template.edge,
            matched_top_left=(matched_x, matched_y),
            roi=roi,
            template=template,
        )
