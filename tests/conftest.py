from __future__ import annotations

import base64
from dataclasses import dataclass

import cv2
import numpy as np


def encode_image(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return base64.b64encode(encoded.tobytes()).decode("ascii")


@dataclass(frozen=True)
class FakeObject:
    object_name: str


class FakeResponse:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.closed = False
        self.released = False

    def read(self) -> bytes:
        return self.data

    def close(self) -> None:
        self.closed = True

    def release_conn(self) -> None:
        self.released = True


class FakeMinio:
    def __init__(self, images: dict[str, np.ndarray]) -> None:
        self.objects: dict[str, bytes] = {}
        self.listed_prefixes: list[str] = []
        for name, image in images.items():
            ok, encoded = cv2.imencode(".png", image)
            assert ok
            self.objects[name] = encoded.tobytes()

    def list_objects(self, bucket: str, *, prefix: str, recursive: bool):
        assert bucket == "static"
        assert recursive is True
        self.listed_prefixes.append(prefix)
        return [
            FakeObject(name) for name in sorted(self.objects) if name.startswith(prefix)
        ]

    def get_object(self, bucket: str, name: str) -> FakeResponse:
        assert bucket == "static"
        return FakeResponse(self.objects[name])
