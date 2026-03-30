from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader


DEFAULT_INPUT_DIR = Path("ingestion/pdfs")
DEFAULT_OUTPUT_DIR = Path("ingestion/extracted_images")


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    collapsed = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_only).strip("_").lower()
    return collapsed or "pdf"


@dataclass(slots=True)
class ExtractionCandidate:
    page_number: int
    image_name: str
    image: Image.Image
    sha256: str

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract manometry exam images embedded in PDF reports.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing exam PDFs. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where extracted images will be saved. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        action="append",
        default=[],
        help="Specific PDF to process. Can be passed multiple times.",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep repeated copies of the same embedded image instead of deduplicating by content hash.",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=350,
        help="Minimum width in pixels for an extracted exam image.",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=500,
        help="Minimum height in pixels for an extracted exam image.",
    )
    parser.add_argument(
        "--min-portrait-ratio",
        type=float,
        default=1.15,
        help="Minimum height/width ratio used to filter portrait exam images.",
    )
    return parser


def iter_target_pdfs(input_dir: Path, explicit_pdfs: list[Path]) -> list[Path]:
    if explicit_pdfs:
        return [pdf if pdf.is_absolute() else pdf for pdf in explicit_pdfs]
    return sorted(input_dir.glob("*.pdf"))


def open_embedded_image(image_bytes: bytes) -> Image.Image:
    with Image.open(BytesIO(image_bytes)) as image:
        return image.convert("RGB")


def looks_like_exam_image(
    image: Image.Image,
    *,
    min_width: int,
    min_height: int,
    min_portrait_ratio: float,
) -> bool:
    if image.width < min_width or image.height < min_height:
        return False
    if (image.height / image.width) < min_portrait_ratio:
        return False
    return True


def collect_candidates(
    pdf_path: Path,
    *,
    min_width: int,
    min_height: int,
    min_portrait_ratio: float,
) -> list[ExtractionCandidate]:
    reader = PdfReader(str(pdf_path))
    candidates: list[ExtractionCandidate] = []

    for page_number, page in enumerate(reader.pages, start=1):
        for page_image in page.images:
            try:
                image = open_embedded_image(page_image.data)
            except UnidentifiedImageError:
                continue

            if not looks_like_exam_image(
                image,
                min_width=min_width,
                min_height=min_height,
                min_portrait_ratio=min_portrait_ratio,
            ):
                continue

            sha256 = hashlib.sha256(image.tobytes()).hexdigest()
            candidates.append(
                ExtractionCandidate(
                    page_number=page_number,
                    image_name=page_image.name,
                    image=image,
                    sha256=sha256,
                )
            )

    return candidates


def save_candidates(
    pdf_path: Path,
    candidates: list[ExtractionCandidate],
    *,
    output_dir: Path,
    keep_duplicates: bool,
) -> dict[str, object]:
    pdf_slug = slugify(pdf_path.stem)
    pdf_output_dir = output_dir / pdf_slug
    pdf_output_dir.mkdir(parents=True, exist_ok=True)

    manifest_images: list[dict[str, object]] = []
    seen_hashes: dict[str, str] = {}
    saved_count = 0
    duplicate_count = 0

    for index, candidate in enumerate(candidates, start=1):
        image_slug = slugify(Path(candidate.image_name).stem)

        if not keep_duplicates and candidate.sha256 in seen_hashes:
            duplicate_count += 1
            manifest_images.append(
                {
                    "page": candidate.page_number,
                    "source_image_name": candidate.image_name,
                    "width": candidate.width,
                    "height": candidate.height,
                    "sha256": candidate.sha256,
                    "duplicate_of": seen_hashes[candidate.sha256],
                }
            )
            continue

        filename = f"image_{index:03d}_p{candidate.page_number:02d}_{image_slug}.png"
        destination = pdf_output_dir / filename
        candidate.image.save(destination, format="PNG")
        seen_hashes[candidate.sha256] = filename
        saved_count += 1
        manifest_images.append(
            {
                "page": candidate.page_number,
                "source_image_name": candidate.image_name,
                "width": candidate.width,
                "height": candidate.height,
                "sha256": candidate.sha256,
                "saved_as": filename,
            }
        )

    manifest = {
        "pdf": str(pdf_path),
        "output_dir": str(pdf_output_dir),
        "saved_images": saved_count,
        "duplicates_skipped": duplicate_count,
        "images": manifest_images,
    }
    (pdf_output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    args = build_parser().parse_args()

    pdf_paths = iter_target_pdfs(args.input_dir, args.pdf)
    if not pdf_paths:
        print("No PDF files found to process.", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            print(f"[missing] {pdf_path}", file=sys.stderr)
            failures += 1
            continue

        try:
            candidates = collect_candidates(
                pdf_path,
                min_width=args.min_width,
                min_height=args.min_height,
                min_portrait_ratio=args.min_portrait_ratio,
            )
            manifest = save_candidates(
                pdf_path,
                candidates,
                output_dir=args.output_dir,
                keep_duplicates=args.keep_duplicates,
            )
        except Exception as exc:
            print(f"[error] {pdf_path}: {exc}", file=sys.stderr)
            failures += 1
            continue

        print(
            f"[ok] {pdf_path.name}: {manifest['saved_images']} image(s) saved"
            f" to {manifest['output_dir']} ({manifest['duplicates_skipped']} duplicate(s) skipped)"
        )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
