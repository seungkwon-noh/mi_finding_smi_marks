from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import TemplateFinder
from .io import load_templates, read_image, write_image
from .visualize import annotate_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mi-finding", description="Find the best template location in an image."
    )
    parser.add_argument("image", help="target image path")
    parser.add_argument("templates", help="directory containing template images")
    parser.add_argument(
        "--prefix", default="", help="only load template names with this prefix"
    )
    parser.add_argument("--output", help="write an annotated result image")
    parser.add_argument(
        "--json", dest="json_path", help="write the JSON result to a file"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image = read_image(args.image)
    templates = load_templates(args.templates, args.prefix)
    result = TemplateFinder().find(image, templates)
    payload = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    print(payload)

    if args.output:
        write_image(args.output, annotate_result(image, result))
    if args.json_path:
        json_path = Path(args.json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(payload + "\n", encoding="utf-8")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
